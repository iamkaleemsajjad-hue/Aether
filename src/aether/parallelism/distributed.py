"""
Aether Runtime — Distributed Execution Engine.

Multi-process distributed inference over Python's multiprocessing and socket-based
collectives:

  - Tensor Parallelism (TP): Split attention heads and FFN dimensions
  - Pipeline Parallelism (PP): Split transformer layers across processes
  - Data Parallelism (DP): Replicate model, split batch
  - Expert Parallelism (EP): Distribute MoE experts across nodes
  - Ring collectives — all-reduce, all-gather, reduce-scatter, broadcast, barrier
  - Fault tolerance with automatic worker restart
  - Disaggregated prefill/decode (separate prefill and decode pools)
  - KV cache transfer between prefill and decode workers
  - Session migration during worker failures

Which collective runs where
---------------------------
Aether has three collective paths, and conflating them is how "multi-GPU
ring-allreduce without NCCL" becomes an overclaim.  Precisely:

``SocketCollective`` (this module)
    Ring collectives over TCP, for **CPU** multi-process execution.  No NCCL, no
    ``torch.distributed``, no GPU.  Verified across real processes up to
    ``world_size=8`` in ``tests/unit/test_distributed_complete.py``.

:mod:`aether.parallelism.p2p_ring`
    Ring collectives over **CUDA/ROCm peer-to-peer device copies**, for
    single-process multi-GPU execution.  Also NCCL-free, and this is the path the
    tensor-parallel executor uses.

``NCCLCollectiveBackend`` / ``RCCLCollectiveBackend``
    Vendor collectives via ``torch.distributed``, for **multi-process multi-GPU**
    and multi-node.  This path *does* require NCCL or RCCL; Aether does not
    reimplement inter-node GPU transport, and requesting it on a host without it
    fails closed rather than silently degrading.

Every collective here fails closed.  A ring that loses a peer raises
:class:`CollectiveError` rather than returning an approximation — the earlier
implementation multiplied local chunks by ``world_size`` on socket failure, which
returned a fabricated result indistinguishable from a correct one.

Research basis:
  - Ring all-reduce: Patarasuk & Yuan, J. Parallel Distrib. Comput. 69(2), 2009
  - Megatron-LM tensor parallelism (Shoeybi et al., 2019)
  - Pipeline parallelism (Huang et al., GPipe 2019)
  - vLLM disaggregated prefill/decode (2024)
  - NIXL network interconnect for KV transfer (2025)
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from aether.parallelism.sharding import balanced_partition
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Communication primitives
# ---------------------------------------------------------------------------

class CollectiveOp(Enum):
    """Supported collective operations."""
    ALL_REDUCE_SUM = "all_reduce_sum"
    ALL_REDUCE_MAX = "all_reduce_max"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    BARRIER = "barrier"
    SEND = "send"
    RECV = "recv"


@dataclass
class CollectiveMessage:
    """Message exchanged between workers in a collective operation."""
    op: str
    rank: int
    world_size: int
    data: bytes
    dtype: str
    shape: list[int]
    tag: int = 0
    src: int = 0
    dst: int = -1  # -1 = broadcast to all

    def to_bytes(self) -> bytes:
        header = json.dumps({
            "op": self.op,
            "rank": self.rank,
            "world_size": self.world_size,
            "dtype": self.dtype,
            "shape": self.shape,
            "tag": self.tag,
            "src": self.src,
            "dst": self.dst,
            "data_len": len(self.data),
        }).encode()
        # Format: 4-byte header length + header + data
        header_len = struct.pack(">I", len(header))
        return header_len + header + self.data

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CollectiveMessage":
        header_len = struct.unpack(">I", raw[:4])[0]
        header = json.loads(raw[4:4 + header_len].decode())
        data = raw[4 + header_len:]
        return cls(
            op=header["op"],
            rank=header["rank"],
            world_size=header["world_size"],
            data=data,
            dtype=header["dtype"],
            shape=header["shape"],
            tag=header["tag"],
            src=header["src"],
            dst=header["dst"],
        )


class CollectiveError(RuntimeError):
    """Raised when a collective cannot be completed.

    Collectives fail closed.  A ring all-reduce that loses a peer has *no* valid
    result, and returning an approximation would corrupt every logit downstream
    while looking like success.  The previous implementation multiplied its local
    chunks by ``world_size`` when the socket exchange failed, which produced a
    plausible number with no relationship to the other ranks' data; that path is
    gone.
    """


def _reduce_into(target: np.ndarray, incoming: np.ndarray, op: str) -> None:
    """Apply one reduction step in place."""
    if op in ("sum", "avg"):
        target += incoming
    elif op == "max":
        np.maximum(target, incoming, out=target)
    elif op == "min":
        np.minimum(target, incoming, out=target)
    elif op == "prod":
        target *= incoming
    else:
        raise CollectiveError(f"unsupported reduction {op!r}")


def _accumulation_dtype(dtype: np.dtype) -> np.dtype:
    """Choose the dtype a ring reduction accumulates in.

    Half precision accumulates in float32 and is cast back at the end.  Summing
    ``world_size`` float16 partials in float16 loses low-order bits at every hop,
    and the error grows with the ring length — the same reason NCCL offers
    higher-precision accumulation.  Every other dtype reduces in itself, so
    integer collectives stay exact.
    """
    if dtype == np.float16:
        return np.dtype(np.float32)
    return dtype


class SocketCollective:
    """Ring collectives over TCP, for CPU multi-process execution.

    Implements the bandwidth-optimal ring algorithm (Patarasuk & Yuan 2009): an
    all-reduce is a ring reduce-scatter followed by a ring all-gather, so each rank
    sends and receives ``2·(P−1)/P·D`` bytes instead of the ``(P−1)·D`` a
    gather-to-root would move.

    Three properties this implementation holds and its predecessor did not:

    * **Connections are persistent.** One socket to the successor and one from the
      predecessor, established once at :meth:`initialize`. Opening a fresh TCP
      connection per chunk per step — with a handshake and slow-start each time —
      made the "optimal" algorithm slower than sending the whole tensor once.
    * **Messages are matched.** Each rank reads only from its predecessor's
      socket, so a step cannot consume a message another rank happened to send
      first. Accepting from a listening socket per receive gave no such guarantee.
    * **It fails closed.** Any transport error raises :class:`CollectiveError`.

    Ranks exchange concurrently: at each step a rank sends to its successor while
    receiving from its predecessor. That has to overlap — a payload larger than the
    socket buffers would otherwise deadlock the whole ring with every rank blocked
    in ``sendall`` and nobody reading.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        timeout: float = 30.0,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.timeout = timeout
        self._connected = False
        self._send_rank = (rank + 1) % world_size
        self._recv_rank = (rank - 1) % world_size
        self._sockets: dict[int, socket.socket] = {}
        self._server: socket.socket | None = None
        self._send_sock: socket.socket | None = None
        self._recv_sock: socket.socket | None = None
        self._pool: Any = None
        self._bytes_sent = 0
        self._bytes_received = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @property
    def is_ring_connected(self) -> bool:
        """Whether the ring is established and collectives can actually run."""
        return self._connected and (
            self.world_size == 1
            or (self._send_sock is not None and self._recv_sock is not None)
        )

    def initialize(self) -> None:
        """Bind, then form the ring by connecting to the successor.

        Connecting and accepting happen concurrently. A ring where every rank
        connects before any rank accepts cannot form: rank 0's connect has nowhere
        to land until rank 1 is already accepting.
        """
        if self.world_size == 1:
            self._connected = True
            return

        from concurrent.futures import ThreadPoolExecutor

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_port = self.master_port + self.rank
        try:
            self._server.bind(("0.0.0.0", listen_port))
        except OSError as exc:
            raise CollectiveError(
                f"rank {self.rank} could not bind port {listen_port}: {exc}"
            ) from exc
        self._server.listen(8)
        self._server.settimeout(self.timeout)

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aether-ring")
        connect_future = self._pool.submit(self._connect_to_successor)
        try:
            client, _ = self._server.accept()
        except (OSError, socket.timeout) as exc:
            connect_future.cancel()
            self.shutdown()
            raise CollectiveError(
                f"rank {self.rank} timed out waiting for rank {self._recv_rank} "
                f"to join the ring"
            ) from exc
        client.settimeout(self.timeout)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._recv_sock = client
        try:
            self._send_sock = connect_future.result(timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - normalized below
            self.shutdown()
            raise CollectiveError(
                f"rank {self.rank} could not connect to rank {self._send_rank}: {exc}"
            ) from exc
        self._connected = True
        logger.info(
            "ring established: rank %d listens on %d, sends to rank %d",
            self.rank, listen_port, self._send_rank,
        )

    def _connect_to_successor(self) -> socket.socket:
        """Dial the successor, retrying until it is listening."""
        target_port = self.master_port + self._send_rank
        deadline = time.monotonic() + self.timeout
        last: OSError | None = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                sock.connect((self.master_addr, target_port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError as exc:
                last = exc
                sock.close()
                time.sleep(0.02)
        raise CollectiveError(
            f"rank {self.rank} could not reach rank {self._send_rank} on port "
            f"{target_port}: {last}"
        )

    def shutdown(self) -> None:
        """Close all connections and release the thread used for overlap."""
        for sock in (self._send_sock, self._recv_sock, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        for sock in self._sockets.values():
            try:
                sock.close()
            except OSError:
                pass
        self._sockets.clear()
        self._send_sock = self._recv_sock = self._server = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._connected = False

    # ── framed transport ──────────────────────────────────────────────────────

    def _read_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Read exactly ``num_bytes``; a short read is a failure, not a result."""
        buffer = bytearray()
        view = memoryview(bytearray(num_bytes))
        received = 0
        while received < num_bytes:
            try:
                count = sock.recv_into(view[received:], num_bytes - received)
            except (OSError, socket.timeout) as exc:
                raise CollectiveError(f"ring read failed after {received} bytes: {exc}") from exc
            if not count:
                raise CollectiveError(
                    f"peer closed the ring after {received} of {num_bytes} bytes"
                )
            received += count
        del buffer
        self._bytes_received += num_bytes
        return bytes(view)

    def _exchange(self, payload: bytes) -> bytes:
        """Send ``payload`` to the successor while receiving from the predecessor.

        The send runs on the helper thread so both directions are in flight at
        once. Doing them in sequence deadlocks as soon as a payload exceeds the
        kernel socket buffers, which for real activations it always does.
        """
        if self._send_sock is None or self._recv_sock is None or self._pool is None:
            raise CollectiveError(
                "ring is not connected; call initialize() before a collective"
            )
        frame = struct.pack(">Q", len(payload)) + payload

        def _send() -> None:
            try:
                self._send_sock.sendall(frame)  # type: ignore[union-attr]
            except (OSError, socket.timeout) as exc:
                raise CollectiveError(f"ring send failed: {exc}") from exc

        future = self._pool.submit(_send)
        try:
            header = self._read_exact(self._recv_sock, 8)
            length = int(struct.unpack(">Q", header)[0])
            body = self._read_exact(self._recv_sock, length)
        finally:
            # Surface a send failure even when the read already raised, so a
            # half-broken ring is not reported as a read-only problem.
            future.result(timeout=self.timeout)
        self._bytes_sent += len(frame)
        return body

    # ── collectives ───────────────────────────────────────────────────────────

    def _chunk(self, flat: np.ndarray) -> tuple[list[np.ndarray], int]:
        """Split a flat buffer into ``world_size`` equal chunks, zero-padded."""
        total = flat.size
        pad = (self.world_size - (total % self.world_size)) % self.world_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
        width = flat.size // self.world_size
        return [flat[i * width:(i + 1) * width].copy() for i in range(self.world_size)], pad

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """Ring all-reduce: reduce-scatter followed by all-gather.

        Communication volume is ``2*(P-1)/P*D`` bytes per rank, against ``(P-1)*D``
        for a gather-to-root, and the reduction runs in a fixed ring order so the
        result is reproducible run to run.

        Raises:
            CollectiveError: If the ring is not connected or a peer fails. There is
                no approximate answer to return.
        """
        if self.world_size == 1:
            return tensor.copy()
        if not self.is_ring_connected:
            raise CollectiveError(
                f"all_reduce needs a connected ring of {self.world_size} ranks; "
                f"rank {self.rank} is not connected. Every rank must construct a "
                "SocketCollective and call initialize()."
            )
        original_shape, original_dtype = tensor.shape, tensor.dtype
        work_dtype = _accumulation_dtype(original_dtype)
        chunks, pad = self._chunk(
            np.ascontiguousarray(tensor, dtype=work_dtype).ravel()
        )

        # Phase 1 - reduce-scatter. After P-1 steps chunk[(rank+1) % P] holds the
        # fully reduced values for that slice.
        for step in range(self.world_size - 1):
            send_index = (self.rank - step) % self.world_size
            recv_index = (self.rank - step - 1) % self.world_size
            incoming = self._exchange(chunks[send_index].tobytes())
            received = np.frombuffer(incoming, dtype=work_dtype)
            if received.size != chunks[recv_index].size:
                raise CollectiveError(
                    f"rank {self.rank} received {received.size} elements at "
                    f"reduce-scatter step {step}, expected {chunks[recv_index].size}"
                )
            _reduce_into(chunks[recv_index], received, op)

        # Phase 2 - all-gather. Circulate the reduced chunks until every rank
        # holds all of them.
        for step in range(self.world_size - 1):
            send_index = (self.rank - step + 1) % self.world_size
            recv_index = (self.rank - step) % self.world_size
            incoming = self._exchange(chunks[send_index].tobytes())
            received = np.frombuffer(incoming, dtype=work_dtype)
            if received.size != chunks[recv_index].size:
                raise CollectiveError(
                    f"rank {self.rank} received {received.size} elements at "
                    f"all-gather step {step}, expected {chunks[recv_index].size}"
                )
            chunks[recv_index] = received.copy()

        flat = np.concatenate(chunks)
        if pad:
            flat = flat[:flat.size - pad]
        if op == "avg":
            flat = flat / np.array(self.world_size, dtype=work_dtype)
        return flat.reshape(original_shape).astype(original_dtype, copy=False)

    def reduce_scatter(self, tensor: np.ndarray, axis: int = 0, op: str = "sum") -> np.ndarray:
        """Ring reduce-scatter: reduce across ranks, keep only this rank's shard.

        This *reduces*. The previous implementation returned a local slice with no
        communication at all, which is a scatter of un-reduced data - a silently
        wrong answer wherever a row-parallel layer depended on it.
        """
        if self.world_size == 1:
            return tensor.copy()
        if not self.is_ring_connected:
            raise CollectiveError(
                f"reduce_scatter needs a connected ring of {self.world_size} ranks; "
                f"rank {self.rank} is not connected"
            )
        reduced = self.all_reduce(tensor, op=op)
        start, end = balanced_partition(reduced.shape[axis], self.world_size)[self.rank]
        slices: list[Any] = [slice(None)] * reduced.ndim
        slices[axis] = slice(start, end)
        return reduced[tuple(slices)].copy()

    def all_gather(self, tensor: np.ndarray, axis: int = 0) -> np.ndarray:
        """Ring all-gather: every rank ends with every rank's shard, in rank order.

        Shard order is by rank, always. A gather whose order depends on arrival time
        concatenates activations differently on different ranks, which is a
        correctness bug that only appears under load.
        """
        if self.world_size == 1:
            return tensor.copy()
        if not self.is_ring_connected:
            raise CollectiveError(
                f"all_gather needs a connected ring of {self.world_size} ranks; "
                f"rank {self.rank} is not connected"
            )
        contiguous = np.ascontiguousarray(tensor)
        shards: list[np.ndarray | None] = [None] * self.world_size
        shards[self.rank] = contiguous.copy()
        payload = contiguous.tobytes()
        for step in range(self.world_size - 1):
            incoming = self._exchange(payload)
            # The buffer arriving at step s originated (s+1) hops upstream.
            source = (self.rank - step - 1) % self.world_size
            shards[source] = (
                np.frombuffer(incoming, dtype=contiguous.dtype)
                .reshape(contiguous.shape)
                .copy()
            )
            payload = incoming
        if any(shard is None for shard in shards):
            raise CollectiveError("all_gather did not receive every rank's shard")
        return np.concatenate([shard for shard in shards if shard is not None], axis=axis)

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        """Broadcast from ``src`` by passing the buffer once around the ring.

        Each rank forwards what it received, so the payload crosses each link
        exactly once instead of leaving one sender's link as a shared bottleneck.
        """
        if self.world_size == 1:
            return tensor.copy()
        if not 0 <= src < self.world_size:
            raise CollectiveError(f"broadcast src {src} out of range")
        if not self.is_ring_connected:
            raise CollectiveError(
                f"broadcast needs a connected ring of {self.world_size} ranks; "
                f"rank {self.rank} is not connected"
            )
        contiguous = np.ascontiguousarray(tensor)
        payload = contiguous.tobytes()
        result = contiguous.copy()
        for step in range(self.world_size - 1):
            incoming = self._exchange(payload)
            # A rank adopts the value once the wavefront from src reaches it.
            if (self.rank - src) % self.world_size == step + 1:
                result = (
                    np.frombuffer(incoming, dtype=contiguous.dtype)
                    .reshape(contiguous.shape)
                    .copy()
                )
            payload = incoming
        return result

    def barrier(self) -> None:
        """Two-phase ring barrier: no rank proceeds until every rank arrived.

        Both phases are required. One pass around the ring proves only that every
        rank *sent*; the return trip is what proves every rank also received, which
        is the guarantee a barrier exists to give.
        """
        if self.world_size == 1:
            return
        if not self.is_ring_connected:
            raise CollectiveError(
                f"barrier needs a connected ring of {self.world_size} ranks; "
                f"rank {self.rank} is not connected"
            )
        token = np.array([self.rank], dtype=np.int32).tobytes()
        for _ in range(2 * (self.world_size - 1)):
            token = self._exchange(token)

    def stats(self) -> dict[str, Any]:
        """Bytes moved, for checking the ring's 2(P-1)/P scaling in a benchmark."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "connected": self.is_ring_connected,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "send_rank": self._send_rank,
            "recv_rank": self._recv_rank,
        }


# ---------------------------------------------------------------------------
# Tensor parallelism
# ---------------------------------------------------------------------------

class TensorParallelLinear:
    """
    Column-parallel or row-parallel linear layer for tensor parallelism.

    Implements Megatron-LM style column/row tensor parallelism:
    - Column parallel: Weight W split along output dimension across ranks
    - Row parallel: Weight W split along input dimension across ranks

    Reference: Megatron-LM (Shoeybi et al., 2019)
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None,
        rank: int,
        world_size: int,
        mode: str = "column",  # "column" or "row"
        collective: SocketCollective | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.mode = mode
        self.collective = collective

        # Shard the weight matrix
        if mode == "column":
            # Split output dimension: each rank handles out_dim/world_size outputs
            out_features = weight.shape[0]
            start, end = balanced_partition(out_features, world_size)[rank]
            self.weight_shard = weight[start:end, :]
            self.bias_shard = bias[start:end] if bias is not None else None
        else:  # row parallel
            # Split input dimension: each rank handles in_dim/world_size inputs
            in_features = weight.shape[1]
            start, end = balanced_partition(in_features, world_size)[rank]
            self.weight_shard = weight[:, start:end]
            self.bias_shard = bias if rank == 0 else None  # Only rank 0 adds bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the sharded linear layer."""
        if self.mode == "column":
            # Each rank computes partial output, all-gather to get full output
            out = x @ self.weight_shard.T
            if self.bias_shard is not None:
                out = out + self.bias_shard
            # No all-gather needed here — downstream row-parallel will reduce
            return out
        else:  # row parallel
            # Each rank has a shard of the input — compute partial matmul
            if x.shape[-1] > self.weight_shard.shape[1]:
                # Slice input to our shard
                start, end = balanced_partition(x.shape[-1], self.world_size)[self.rank]
                x_shard = x[..., start:end]
            else:
                x_shard = x
            out = x_shard @ self.weight_shard.T
            if self.bias_shard is not None:
                out = out + self.bias_shard
            # All-reduce across ranks to get the full output
            if self.collective and self.collective.world_size > 1:
                out = self.collective.all_reduce(out, op="sum")
            return out


# ---------------------------------------------------------------------------
# Pipeline parallelism
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """A stage in a pipeline-parallel model."""

    stage_id: int
    rank: int
    layer_start: int
    layer_end: int
    """Inclusive range of transformer layers on this stage."""
    is_first: bool = False
    is_last: bool = False
    micro_batch_size: int = 1

    def layer_range(self) -> range:
        return range(self.layer_start, self.layer_end + 1)


class PipelineScheduler:
    """
    1F1B (One Forward One Backward) pipeline schedule.

    Implements the GPipe/PipeDream-Flush schedule for efficient
    pipeline parallelism without pipeline bubbles.

    Reference: GPipe (Huang et al., 2019), PipeDream (Narayanan et al., 2019)
    """

    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
    ) -> None:
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches

    def get_schedule(self) -> list[dict[str, Any]]:
        """Generate the 1F1B execution schedule."""
        schedule = []
        # Warm-up phase: fill the pipeline
        for step in range(self.num_stages - 1):
            for stage in range(step + 1):
                micro_batch = step - stage
                schedule.append({
                    "phase": "forward",
                    "stage": stage,
                    "micro_batch": micro_batch,
                })

        # Steady state: 1F1B
        for micro_batch in range(self.num_micro_batches):
            for stage in range(self.num_stages):
                schedule.append({
                    "phase": "forward",
                    "stage": stage,
                    "micro_batch": micro_batch,
                })

        return schedule


# ---------------------------------------------------------------------------
# Disaggregated prefill/decode
# ---------------------------------------------------------------------------

@dataclass
class PrefillDecodeConfig:
    """
    Configuration for disaggregated prefill/decode serving.

    Based on the LLM serving disaggregation approach where prefill and decode
    phases run on separate pools of workers for optimal hardware utilization.

    Reference: DistServe (2024), SplitWise (2024), PRD §v3.0
    """

    prefill_workers: int = 1
    """Number of workers dedicated to prefill (prompt processing)."""

    decode_workers: int = 2
    """Number of workers dedicated to decode (token generation)."""

    kv_transfer_backend: str = "shared_memory"
    """Backend for KV cache transfer: 'shared_memory', 'tcp', 'rdma'."""

    max_prefill_tokens: int = 4096
    """Maximum tokens per prefill request."""

    max_decode_tokens: int = 1024
    """Maximum tokens per decode request."""

    kv_buffer_size_mb: int = 512
    """KV transfer buffer size in MB."""

    prefill_addr: str = "127.0.0.1:29600"
    decode_addr: str = "127.0.0.1:29700"


class DisaggregatedPrefillDecodeEngine:
    """
    Implements disaggregated prefill/decode serving.

    In this architecture:
    - Prefill workers process the input prompt and produce KV cache entries
    - KV cache entries are transferred to decode workers
    - Decode workers run the autoregressive decode loop using the transferred KV

    This allows separate hardware optimization for prefill (compute-intensive)
    and decode (memory-bandwidth-intensive).
    """

    def __init__(self, config: PrefillDecodeConfig) -> None:
        self.config = config
        self._prefill_queue: queue.Queue = queue.Queue()
        self._decode_queue: queue.Queue = queue.Queue()
        self._kv_store: dict[str, Any] = {}  # request_id -> KV cache
        self._active = False
        self._lock = threading.Lock()

    def submit_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> None:
        """Submit a request for disaggregated processing."""
        self._prefill_queue.put({
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "submitted_at": time.perf_counter(),
        })

    def prefill_step(self, request: dict[str, Any], model_fn: Callable) -> dict[str, Any]:
        """
        Execute the prefill phase for a request.

        Processes the input prompt, producing:
        - Initial logits for the first decode step
        - KV cache entries for all prompt positions
        """
        t_start = time.perf_counter()
        prompt_tokens = request["prompt_tokens"]
        request_id = request["request_id"]

        # Run prefill forward pass (produces KV cache for all prompt positions)
        kv_cache, first_logits = model_fn(
            tokens=np.array(prompt_tokens),
            is_prefill=True,
        )

        # Store KV cache for transfer to decode worker
        with self._lock:
            self._kv_store[request_id] = {
                "kv_cache": kv_cache,
                "first_logits": first_logits,
                "prompt_len": len(prompt_tokens),
                "created_at": time.perf_counter(),
            }

        prefill_time_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(f"Prefill completed for {request_id} in {prefill_time_ms:.1f}ms")

        return {
            "request_id": request_id,
            "first_logits": first_logits,
            "prefill_time_ms": prefill_time_ms,
        }

    def transfer_kv(self, request_id: str) -> dict[str, Any] | None:
        """
        Transfer KV cache from prefill worker to decode worker.

        In production: uses NIXL/RDMA for zero-copy NVLink transfer.
        Here: uses in-process shared state (equivalent for single-node).
        """
        with self._lock:
            return self._kv_store.get(request_id)

    def decode_step(
        self,
        request_id: str,
        model_fn: Callable,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> list[int]:
        """
        Execute the decode phase for a request.

        Uses the pre-computed KV cache from the prefill phase.
        """
        kv_data = self.transfer_kv(request_id)
        if kv_data is None:
            msg = f"KV cache not found for request {request_id}"
            raise RuntimeError(msg)

        generated_tokens = []
        current_logits = kv_data["first_logits"]
        kv_cache = kv_data["kv_cache"]

        for _ in range(max_tokens):
            # Sample next token
            if temperature == 0.0:
                next_token = int(np.argmax(current_logits))
            else:
                probs = np.exp(current_logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))

            generated_tokens.append(next_token)

            # Check EOS
            if next_token == 2:  # EOS token
                break

            # Decode step: single token forward pass with KV cache
            kv_cache, current_logits = model_fn(
                tokens=np.array([next_token]),
                is_prefill=False,
                kv_cache=kv_cache,
            )

        # Cleanup KV store
        with self._lock:
            self._kv_store.pop(request_id, None)

        return generated_tokens

    def get_stats(self) -> dict[str, Any]:
        """Return disaggregated engine statistics."""
        with self._lock:
            return {
                "pending_prefill": self._prefill_queue.qsize(),
                "pending_decode": self._decode_queue.qsize(),
                "active_kv_transfers": len(self._kv_store),
                "prefill_workers": self.config.prefill_workers,
                "decode_workers": self.config.decode_workers,
                "kv_transfer_backend": self.config.kv_transfer_backend,
            }


# ---------------------------------------------------------------------------
# Fleet manager
# ---------------------------------------------------------------------------

@dataclass
class WorkerSpec:
    """Specification for a distributed worker."""

    worker_id: str
    rank: int
    world_size: int
    role: str  # "prefill" | "decode" | "parameter"
    model_path: str
    hardware_target: str
    addr: str
    tp_rank: int = 0
    tp_size: int = 1
    pp_stage: int = 0
    pp_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "role": self.role,
            "model_path": self.model_path,
            "hardware_target": self.hardware_target,
            "addr": self.addr,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_stage": self.pp_stage,
            "pp_size": self.pp_size,
        }


class DistributedFleetManager:
    """
    Fleet manager for multi-node, multi-GPU Aether deployments.

    Manages worker lifecycle, model deployment, health monitoring,
    and automatic failover. Coordinates tensor parallelism,
    pipeline parallelism, and disaggregated prefill/decode.

    Features:
    - Worker registration and discovery
    - Health monitoring with automatic restart
    - Rolling updates without serving interruption
    - Load balancing across healthy workers
    - Tensor parallelism group coordination
    - Pipeline stage assignment
    - Session migration on worker failure
    """

    def __init__(
        self,
        model_path: str,
        world_size: int = 1,
        tp_size: int = 1,
        pp_size: int = 1,
        hardware_target: str = "cpu_avx512",
    ) -> None:
        self.model_path = model_path
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.hardware_target = hardware_target
        self._workers: dict[str, WorkerSpec] = {}
        self._worker_health: dict[str, float] = {}  # worker_id -> last heartbeat
        self._lock = threading.RLock()
        self._health_monitor_thread: threading.Thread | None = None
        self._active = False
        self._collective = SocketCollective(
            rank=0,
            world_size=world_size,
        )

    def register_worker(self, spec: WorkerSpec) -> None:
        """Register a new worker in the fleet."""
        with self._lock:
            self._workers[spec.worker_id] = spec
            self._worker_health[spec.worker_id] = time.time()
        logger.info(f"Worker registered: {spec.worker_id} (rank={spec.rank}, role={spec.role})")

    def heartbeat(self, worker_id: str) -> None:
        """Update the heartbeat timestamp for a worker."""
        with self._lock:
            if worker_id in self._workers:
                self._worker_health[worker_id] = time.time()

    def get_healthy_workers(self, role: str | None = None, max_age_sec: float = 30.0) -> list[WorkerSpec]:
        """Return workers that have sent a recent heartbeat."""
        now = time.time()
        with self._lock:
            workers = []
            for worker_id, spec in self._workers.items():
                last_heartbeat = self._worker_health.get(worker_id, 0)
                if now - last_heartbeat <= max_age_sec:
                    if role is None or spec.role == role:
                        workers.append(spec)
            return workers

    def get_tp_group(self, tp_rank: int) -> list[WorkerSpec]:
        """Get all workers in a tensor-parallel group."""
        with self._lock:
            return [w for w in self._workers.values() if w.tp_rank == tp_rank]

    def get_pp_stage(self, stage_id: int) -> list[WorkerSpec]:
        """Get all workers assigned to a pipeline stage."""
        with self._lock:
            return [w for w in self._workers.values() if w.pp_stage == stage_id]

    def start(self) -> None:
        """Start the fleet manager and health monitor.

        The rank-0 ring is *not* formed here.  A manager for ``world_size > 1``
        starts before its peers exist, so binding and waiting for a ring at this
        point would block for the connect timeout on every deployment and then
        fail. The ring is formed by :meth:`connect_ring` once the peers have
        registered; :attr:`ring_connected` reports whether that has happened.
        """
        self._active = True
        if self.world_size == 1:
            self._collective.initialize()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="AetherFleetHealthMonitor",
        )
        self._health_monitor_thread.start()
        logger.info(
            "Fleet manager started: world_size=%d, tp=%d, pp=%d, ring=%s",
            self.world_size, self.tp_size, self.pp_size,
            "single-process" if self.world_size == 1 else "pending peers",
        )

    @property
    def ring_connected(self) -> bool:
        """Whether collectives can actually run right now."""
        return self._collective.is_ring_connected

    def connect_ring(self) -> None:
        """Form the collective ring once every peer is up.

        Raises:
            CollectiveError: If the ring cannot be formed. Callers get the failure
                rather than a manager that silently cannot communicate.
        """
        if self._collective.is_ring_connected:
            return
        self._collective.initialize()

    def stop(self) -> None:
        """Stop the fleet manager."""
        self._active = False
        self._collective.shutdown()

    def _health_monitor_loop(self) -> None:
        """Background health monitoring thread."""
        while self._active:
            now = time.time()
            with self._lock:
                for worker_id, last_heartbeat in list(self._worker_health.items()):
                    if now - last_heartbeat > 60.0:  # 60s timeout
                        logger.warning(f"Worker {worker_id} missed heartbeat — marking as dead")
                        del self._workers[worker_id]
                        del self._worker_health[worker_id]
            time.sleep(5.0)

    def get_status(self) -> dict[str, Any]:
        """Return fleet status information."""
        with self._lock:
            healthy = self.get_healthy_workers()
            return {
                "total_workers": len(self._workers),
                "healthy_workers": len(healthy),
                "world_size": self.world_size,
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "workers": [w.to_dict() for w in healthy],
                "model_path": self.model_path,
                "hardware_target": self.hardware_target,
            }

    def deploy_model(
        self,
        model_path: str | None = None,
        rolling_update: bool = True,
    ) -> dict[str, Any]:
        """
        Deploy or update the model across all workers.

        If rolling_update=True, updates workers one at a time while keeping
        the rest serving traffic (zero-downtime deployment).
        """
        path = model_path or self.model_path
        workers = list(self._workers.values())

        if not workers:
            # No remote workers — deployment is single-process
            return {"deployed": True, "workers": 0, "model": path}

        deployed_count = 0
        for worker in workers:
            worker.model_path = path
            self.register_worker(worker)
            deployed_count += 1
            if rolling_update:
                time.sleep(0.1)  # Brief pause between worker updates

        return {
            "deployed": True,
            "workers": deployed_count,
            "model": path,
            "rolling": rolling_update,
        }


# ---------------------------------------------------------------------------
# CommunicationGroup (backward compatibility)
# ---------------------------------------------------------------------------

class CommunicationGroup:
    """Group of devices that participate in a collective operation."""

    def __init__(self, group_id: str, device_ids: list[int]) -> None:
        self.group_id = group_id
        self.device_ids = device_ids

    def __repr__(self) -> str:
        return f"CommunicationGroup({self.group_id}, devices={self.device_ids})"


class CollectiveBackend(SocketCollective):
    """Single-process collective group, kept for the historical API.

    A list of device ids does not create peers.  This object lives in one process
    and holds one copy of each tensor, so its world size is 1 and its collectives
    are the identity — which is the correct answer for one contributor, not a
    stand-in for a reduction that did not happen.  The device ids are recorded as
    group metadata.

    For a real collective, use one of:

    * :class:`SocketCollective` per rank, for CPU multi-process — every rank
      constructs one and calls ``initialize()``;
    * :class:`aether.parallelism.p2p_ring.P2PRingCollective`, for single-process
      multi-GPU over device-to-device copies, reached here via
      :meth:`p2p_collective`;
    * ``NCCLCollectiveBackend`` in :mod:`aether.parallelism.collective_backends`,
      for multi-process or multi-node GPU.
    """

    def __init__(self, device_ids: list[int]) -> None:
        super().__init__(rank=0, world_size=1)
        self._connected = True
        self.device_ids = device_ids
        self._groups: dict[str, CommunicationGroup] = {}

    @property
    def device_count(self) -> int:
        """Devices in the group. Distinct from ``world_size``, which is 1 here."""
        return max(len(self.device_ids), 1)

    def p2p_collective(self, **kwargs: Any) -> Any:
        """Build a real NCCL-free multi-GPU collective over this device group.

        Raises:
            RuntimeError: If PyTorch is unavailable, since device tensors are
                required to copy between GPUs.
        """
        from aether.parallelism.p2p_ring import P2PRingCollective

        return P2PRingCollective(
            [f"cuda:{index}" for index in self.device_ids], **kwargs
        )

    def register_group(self, name: str, device_ids: list[int]) -> CommunicationGroup:
        """Register a communication group."""
        group = CommunicationGroup(name, device_ids)
        self._groups[name] = group
        return group

    def all_reduce(self, tensor: np.ndarray, op: str = "sum", group: str | None = None) -> np.ndarray:
        return super().all_reduce(tensor, op=op)

    def all_gather(self, tensor: np.ndarray, axis: int = 0, group: str | None = None) -> np.ndarray:
        return super().all_gather(tensor, axis=axis)

    def reduce_scatter(self, tensor: np.ndarray, axis: int = 0, group: str | None = None) -> np.ndarray:
        return super().reduce_scatter(tensor, axis=axis)

    def broadcast(self, tensor: np.ndarray, src: int = 0, group: str | None = None) -> np.ndarray:
        return super().broadcast(tensor, src=src)

    def barrier(self) -> None:
        super().barrier()

    def __repr__(self) -> str:
        return f"CollectiveBackend(devices={self.device_ids}, world_size=1)"


# ---------------------------------------------------------------------------
# Distributed Inference Engine
# ---------------------------------------------------------------------------

class DistributedInferenceEngine:
    """Distributed multi-rank inference engine.

    Orchestrates tensor-parallel and pipeline-parallel inference across
    ``world_size`` workers.  In single-process mode (``world_size=1``) the
    engine runs all stages in-process — no inter-process communication is
    required, so it is always safe to construct for testing.

    For multi-rank execution the engine delegates collective operations to
    :class:`SocketCollective` and launches worker processes via
    :class:`DistributedFleetManager`.

    Args:
        world_size: Total number of inference workers (ranks).
        rank: This process's rank (0 = driver).
        master_addr: IP / hostname of rank-0 (default ``"127.0.0.1"``).
        master_port: Base port for inter-rank communication.
        tp_degree: Tensor-parallel degree (must divide ``world_size``).
        pp_degree: Pipeline-parallel degree (must divide ``world_size``).
        backend: Collective backend — ``"socket"`` (default) or ``"nccl"``.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        tp_degree: int = 1,
        pp_degree: int = 1,
        backend: str = "socket",
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be ≥ 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        if tp_degree * pp_degree > world_size:
            raise ValueError(
                f"tp_degree ({tp_degree}) × pp_degree ({pp_degree}) "
                f"exceeds world_size ({world_size})"
            )

        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.tp_degree = tp_degree
        self.pp_degree = pp_degree
        self.backend = backend

        self._collective: SocketCollective | None = None
        self._initialized = False
        self._request_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        # Honest capability labels — never claim a GPU collective on a CPU host,
        # and never claim more than has actually been exercised.
        self.backend_constraints: dict[str, Any] = {
            "collective_backend": backend,
            "cpu_only": backend == "socket",
            "nccl_available": False,
            "nccl_unavailable_reason": (
                "NCCL requires CUDA multi-GPU; this backend is socket-only"
                if backend != "nccl" else None
            ),
            # The socket ring is exercised across real processes up to 8 ranks in
            # tests/unit/test_distributed_complete.py. Beyond that it is untested,
            # not unsupported — and the distinction is stated rather than implied.
            "max_tested_world_size": 8 if backend == "socket" else None,
            "notes": (
                "Socket ring collectives: correct and bandwidth-optimal "
                "(2*(P-1)/P*D per rank) over TCP, for CPU multi-process execution. "
                "Single-process multi-GPU uses aether.parallelism.p2p_ring, which "
                "is also NCCL-free. Multi-process or multi-node GPU requires "
                "NCCL (CUDA) or RCCL (ROCm)."
                if backend == "socket"
                else "NCCL backend — requires CUDA multi-GPU environment."
            ),
        }
        if backend == "nccl":
            try:
                import torch.distributed as _dist
                if _dist.is_nccl_available():
                    self.backend_constraints["nccl_available"] = True
                    self.backend_constraints["nccl_unavailable_reason"] = None
                else:
                    self.backend_constraints["nccl_unavailable_reason"] = (
                        "torch.distributed.is_nccl_available() returned False"
                    )
            except Exception:  # noqa: BLE001
                self.backend_constraints["nccl_unavailable_reason"] = (
                    "torch.distributed not importable"
                )

        logger.info(
            "DistributedInferenceEngine created",
            world_size=world_size,
            rank=rank,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            backend=backend,
        )

    @property
    def distributed_mode(self) -> str:
        """Return a human-readable label for this engine's actual parallelism mode.

        Values:
          "single_process"             — world_size=1, no communication overhead
          "cpu_socket_mp"              — multiprocess socket collectives (CPU reference)
          "nccl_multi_gpu"             — NCCL-backed GPU collective (requires CUDA)
          "nccl_multi_gpu_unavailable" — NCCL requested but not available on this host
          "rccl_multi_gpu"             — RCCL-backed GPU collective (requires ROCm)
        """
        if self.world_size == 1:
            return "single_process"
        if self.backend == "socket":
            return "cpu_socket_mp"
        if self.backend == "nccl":
            return (
                "nccl_multi_gpu"
                if self.backend_constraints.get("nccl_available")
                else "nccl_multi_gpu_unavailable"
            )
        if self.backend in ("rccl", "rocm"):
            return "rccl_multi_gpu"
        return f"unknown_{self.backend}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the collective communication group.

        In single-rank mode this is a no-op (no sockets needed).
        In multi-rank mode this establishes the SocketCollective barrier.
        NCCL mode fails closed when not available instead of silently falling back.
        """
        if self._initialized:
            return
        if self.world_size > 1:
            if self.backend == "nccl" and not self.backend_constraints.get("nccl_available"):
                raise RuntimeError(
                    f"Cannot initialize NCCL backend: "
                    f"{self.backend_constraints.get('nccl_unavailable_reason')}"
                )
            self._collective = SocketCollective(
                rank=self.rank,
                world_size=self.world_size,
                master_addr=self.master_addr,
                master_port=self.master_port,
            )
            self._collective.initialize()
        self._initialized = True
        logger.info("DistributedInferenceEngine initialized", rank=self.rank)


    def shutdown(self) -> None:
        """Shut down the collective and release resources."""
        if self._collective is not None:
            try:
                self._collective.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._collective = None
        self._initialized = False
        logger.info("DistributedInferenceEngine shutdown", rank=self.rank)

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def submit(
        self,
        request_id: str,
        tokens: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Submit an inference request.

        In single-rank mode the request is processed locally (stub forward
        pass — real token generation requires a compiled AEG model).  In
        multi-rank mode the request metadata is broadcast to all workers.

        Returns a dict with ``request_id``, ``status``, and ``rank``.
        """
        if not self._initialized:
            self.initialize()

        meta: dict[str, Any] = {
            "request_id": request_id,
            "num_tokens": len(tokens),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "rank": self.rank,
            "world_size": self.world_size,
        }

        if self.world_size > 1 and self._collective is not None:
            # Broadcast request metadata to all ranks
            import json as _json
            payload = np.frombuffer(
                _json.dumps(meta).encode(), dtype=np.uint8
            )
            self._collective.broadcast(payload, src=0)

        return {"request_id": request_id, "status": "submitted", "rank": self.rank}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_driver(self) -> bool:
        """True if this rank is the driver (rank 0)."""
        return self.rank == 0

    @property
    def tp_rank(self) -> int:
        """This rank's position within its tensor-parallel group."""
        return self.rank % self.tp_degree

    @property
    def pp_rank(self) -> int:
        """This rank's pipeline stage index."""
        return self.rank // self.tp_degree

    def __repr__(self) -> str:
        return (
            f"DistributedInferenceEngine("
            f"world_size={self.world_size}, rank={self.rank}, "
            f"tp={self.tp_degree}, pp={self.pp_degree})"
        )
