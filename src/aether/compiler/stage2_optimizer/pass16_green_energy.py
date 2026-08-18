from __future__ import annotations
import json
import math
import time
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_GRID_CARBON_INTENSITY = {
    "us-west": 82.0, "us-east": 318.0, "us-southeast": 425.0,
    "eu-north": 28.0, "eu-west": 156.0, "eu-central": 412.0,
    "ap-east": 487.0, "ap-south": 512.0, "me-gulf": 620.0,
    "au-east": 590.0, "cn-north": 634.0, "cn-south": 380.0,
}
_GPU_MEMORY_BANDWIDTH_GB_S = {
    "cuda_sm70": 900.0, "cuda_sm80": 2000.0, "cuda_sm90": 3350.0,
    "cuda_sm100": 8000.0, "cuda_sm120": 22000.0, "default": 1600.0,
}
_GPU_TDP_WATTS = {
    "cuda_sm70": 300.0, "cuda_sm80": 400.0, "cuda_sm90": 700.0,
    "cuda_sm100": 1000.0, "cuda_sm120": 1200.0, "cpu": 280.0, "default": 400.0,
}


class GreenEnergyCompilationPass(BasePass):
    name = "green_energy_compilation"
    description = "Embed carbon profile, DVFS breakpoints, and operator energy costs into AEG metadata for R7."

    def run(self, graph, architecture, config):
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})
        if not config.enable_green_energy:
            return graph, report
        try:
            region = config.green_carbon_region
            tdp_cap = config.green_target_tdp_watts
            targets = config.get_targets()
            primary_target = targets[0] if targets else "default"
            logger.info("Pass 16: Computing green energy profile for region=%r, target=%r.", region, primary_target)
            carbon_gco2_per_kwh = _GRID_CARBON_INTENSITY.get(region, _GRID_CARBON_INTENSITY["us-west"])
            hw_tdp = _GPU_TDP_WATTS.get(primary_target, _GPU_TDP_WATTS["default"])
            effective_tdp = min(tdp_cap, hw_tdp) if tdp_cap is not None else hw_tdp
            mem_bw = _GPU_MEMORY_BANDWIDTH_GB_S.get(primary_target, _GPU_MEMORY_BANDWIDTH_GB_S["default"])
            dvfs_hints = _compute_dvfs_hints(graph, mem_bw, effective_tdp)
            operator_energy = _compute_operator_energy_table(graph, mem_bw, effective_tdp)
            estimated_savings = _estimate_energy_savings(dvfs_hints, effective_tdp)
            n_hints_emitted = _emit_dvfs_annotations(graph, dvfs_hints)
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                _write_green_profile(graph.output_dir, region, carbon_gco2_per_kwh, effective_tdp, hw_tdp, mem_bw, dvfs_hints, operator_energy)
            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "carbon_region": region, "carbon_intensity_gco2_kwh": carbon_gco2_per_kwh,
                "hardware_tdp_w": hw_tdp, "effective_tdp_cap_w": effective_tdp,
                "memory_bandwidth_gb_s": mem_bw, "dvfs_hints_emitted": n_hints_emitted,
                "estimated_energy_savings_pct": round(estimated_savings * 100, 1),
            }
            logger.info("Pass 16 complete: %.0f gCO2/kWh, %.0fW TDP, %d hints, ~%.0f%% savings. Elapsed: %.3fs.",
                carbon_gco2_per_kwh, effective_tdp, n_hints_emitted, estimated_savings * 100, elapsed)
        except Exception as exc:
            logger.warning("Pass 16 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)
        return graph, report


def _compute_dvfs_hints(graph, mem_bw_gb_s, tdp_w):
    hints = []
    op_list = _iter_ops(graph)
    MEM_BOUND = frozenset({"embedding","softmax","layernorm","rmsnorm","attention",
        "kv_cache_read","kv_cache_write","aeg.semantic_kv_compress","aeg.kv_share_ref","gather","scatter"})
    for op in op_list:
        op_type = _get_op_type(op)
        is_mem = op_type.lower() in MEM_BOUND or _is_memory_bound(op)
        freq_mhz = int(_max_gpu_freq_mhz() * 0.60) if is_mem else _max_gpu_freq_mhz()
        hints.append({
            "op_id": str(getattr(op, "id", getattr(op, "name", str(len(hints))))),
            "op_type": op_type, "is_memory_bound": is_mem,
            "freq_mhz": freq_mhz, "voltage_mv": _freq_to_voltage(freq_mhz),
            "power_savings_pct": 35.0 if is_mem else 0.0,
        })
    return hints


def _compute_operator_energy_table(graph, mem_bw_gb_s, tdp_w):
    return {
        "linear": tdp_w * 0.3e-3 * 1000,
        "attention": tdp_w * 0.4 * ((4 * 4096 * 2048 * 1e6) / (mem_bw_gb_s * 1e9)) * 1000,
        "rmsnorm": tdp_w * 0.2 * ((4 * 4096 * 1e6) / (mem_bw_gb_s * 1e9)) * 1000,
        "layernorm": tdp_w * 0.2 * ((4 * 4096 * 1e6) / (mem_bw_gb_s * 1e9)) * 1000,
        "embedding": tdp_w * 0.2 * ((4 * 4096 * 1e6) / (mem_bw_gb_s * 1e9)) * 1000,
        "softmax": tdp_w * 0.2 * ((4 * 4096 * 1e6) / (mem_bw_gb_s * 1e9)) * 1000,
        "moe_dispatch": tdp_w * 0.3e-3 * 1000 * 0.3,
    }


def _estimate_energy_savings(hints, tdp_w):
    if not hints:
        return 0.0
    return min(0.48, sum(h["power_savings_pct"] for h in hints) / (len(hints) * 100))


def _emit_dvfs_annotations(graph, dvfs_hints):
    n = 0
    for hint in dvfs_hints:
        ann = {"opcode": "aeg.dvfs_hint", "op_id": hint["op_id"],
               "freq_mhz": hint["freq_mhz"], "voltage_mv": hint["voltage_mv"]}
        if hasattr(graph, "add_dvfs_hint"):
            graph.add_dvfs_hint(hint["op_id"], hint["freq_mhz"], hint["voltage_mv"])
            n += 1
        elif hasattr(graph, "metadata"):
            graph.metadata.setdefault("dvfs_hints", []).append(ann)
            n += 1
    return n


def _write_green_profile(output_dir, region, carbon_gco2_per_kwh, effective_tdp, hw_tdp, mem_bw_gb_s, dvfs_hints, operator_energy):
    import pathlib as _pl
    meta_dir = _pl.Path(output_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "format": "aether_green_profile_v1", "carbon_region": region,
        "carbon_intensity_gco2_per_kwh": carbon_gco2_per_kwh,
        "hardware_tdp_w": hw_tdp, "effective_tdp_cap_w": effective_tdp,
        "memory_bandwidth_gb_s": mem_bw_gb_s, "dvfs_hints": dvfs_hints,
        "operator_energy_mj_per_1m_tokens": operator_energy,
    }
    (meta_dir / "green_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.debug("Wrote green profile: %s", meta_dir / "green_profile.json")


def _max_gpu_freq_mhz():
    return 1980


def _freq_to_voltage(freq_mhz):
    f_max = _max_gpu_freq_mhz()
    return int(750 + (1100 - 750) * (freq_mhz / f_max))


def _iter_ops(graph):
    if hasattr(graph, "iter_nodes"):
        return list(graph.iter_nodes())
    if hasattr(graph, "nodes"):
        return list(graph.nodes)
    if hasattr(graph, "__iter__"):
        return list(graph)
    return []


def _get_op_type(op):
    for attr in ("op_type", "type", "name", "opcode"):
        val = getattr(op, attr, None)
        if val:
            return str(val)
    return "unknown"


def _is_memory_bound(op):
    op_type = _get_op_type(op).lower()
    return any(kw in op_type for kw in {"norm","embed","gather","softmax","relu","gelu","silu","kv"})
