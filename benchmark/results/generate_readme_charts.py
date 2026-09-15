"""
Generate unified multi-engine, multi-model benchmark charts for README.md.
Compares:
  - Aether Runtime
  - HuggingFace Transformers
  - PyTorch Native
Across 3 architectures:
  - SmolLM2-135M-Instruct
  - GPTNeo350M-Instruct-SFT
  - Qwen3-0.6B

Outputs:
  1. cross_engine_throughput_comparison.png (Batch 1 interactive + Batch 16 peak throughput)
  2. cross_engine_latency_comparison.png (End-to-end request latency, lower is better)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Dark Theme & Styling ──────────────────────────────────────────────────
BG_COLOR       = "#0D1117"   # GitHub Canvas Dark
AXES_BG        = "#161B22"   # GitHub Subdued Box
GRID_COLOR     = "#30363D"   # Subtle divider
TEXT_COLOR     = "#E6EDF3"   # Primary text
MUTED_TEXT     = "#8B949E"   # Muted subtext
BORDER_COLOR   = "#30363D"

# Engine Brand Colors
AETHER_COLOR   = "#7C65FF"   # Signature Violet / Indigo
TRANS_COLOR    = "#FF6B6B"   # Coral Red (Transformers)
TORCH_COLOR    = "#FFA657"   # Warm Amber (PyTorch Native)
ACCENT_GREEN   = "#3FB950"   # Speedup callout badge

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    AXES_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  AXES_BG,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Segoe UI", "DejaVu Sans", "Helvetica Neue", "Arial"],
    "font.size":         10,
})

# ─── Benchmark Data ──────────────────────────────────────────────────────────
models = ["SmolLM2-135M\n(135M params)", "GPTNeo-350M\n(456M params)", "Qwen3-0.6B\n(752M params)"]
x = np.arange(len(models))
bar_width = 0.25

# Batch 1 (Interactive Throughput in tok/s, prompt=256, output=128)
tps_b1_aether = [46.18, 110.77, 48.21]
tps_b1_trans  = [27.06, 41.27, 23.01]
tps_b1_torch  = [27.32, 40.06, 21.93]

# Batch 16 (Peak Batched Throughput in tok/s, prompt=256, output=128)
tps_b16_aether = [674.99, 1562.72, 273.35]
tps_b16_trans  = [413.00, 489.17, 241.03]
tps_b16_torch  = [410.20, 476.01, 239.23]

# End-to-End Latency in seconds (Batch 1, prompt=256, output=128)
lat_aether = [2.772, 1.156, 2.655]
lat_trans  = [4.729, 3.102, 5.563]
lat_torch  = [4.686, 3.195, 5.836]


def generate_throughput_chart():
    """Generates dual-panel Output Tokens Per Second chart with clean spacing and no overlaps."""
    # Generous figure size to allow ample vertical breathing room
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6), dpi=300)
    
    # top=0.73 gives plenty of room above subplots for Suptitle, Subtitle, and Legend
    fig.subplots_adjust(wspace=0.24, top=0.71, bottom=0.12, left=0.07, right=0.96)

    # ── Panel 1: Single-Request / Batch 1 ──────────────────────────────────────
    r1 = x - bar_width
    r2 = x
    r3 = x + bar_width

    ax1.bar(r1, tps_b1_aether, width=bar_width, label="Aether Runtime", color=AETHER_COLOR, edgecolor="#9F8CFF", linewidth=1.2, zorder=3)
    ax1.bar(r2, tps_b1_trans,  width=bar_width, label="HF Transformers", color=TRANS_COLOR,  edgecolor="#FF8E8E", linewidth=1.0, zorder=3)
    ax1.bar(r3, tps_b1_torch,  width=bar_width, label="PyTorch Native",  color=TORCH_COLOR,  edgecolor="#FFBE82", linewidth=1.0, zorder=3)

    ax1.set_title("Single-Request Interactive Throughput (Batch 1)\n▲ Higher is Better", fontsize=12, fontweight="bold", pad=14, color="#FFFFFF")
    ax1.set_ylabel("Output Tokens / Second (tok/s)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10.5)
    ax1.set_ylim(0, 160)  # Generous headroom for labels + badges
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotations on bars for Panel 1
    for i in range(len(models)):
        v = tps_b1_aether[i]
        speedup = v / max(tps_b1_trans[i], tps_b1_torch[i])
        # Direct value label
        ax1.text(r1[i], v + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#FFFFFF")
        # Speedup badge above Aether
        ax1.text(r1[i], v + 16.0, f"{speedup:.2f}x\n(+{(speedup-1)*100:.0f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                 color=ACCENT_GREEN, bbox=dict(boxstyle="round,pad=0.25", fc="#04210E", ec=ACCENT_GREEN, lw=0.9))

        # Transformers bar
        ax1.text(r2[i], tps_b1_trans[i] + 2.0, f"{tps_b1_trans[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
        # PyTorch bar
        ax1.text(r3[i], tps_b1_torch[i] + 2.0, f"{tps_b1_torch[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)

    # ── Panel 2: Batched Peak / Batch 16 ──────────────────────────────────────
    ax2.bar(r1, tps_b16_aether, width=bar_width, label="Aether Runtime", color=AETHER_COLOR, edgecolor="#9F8CFF", linewidth=1.2, zorder=3)
    ax2.bar(r2, tps_b16_trans,  width=bar_width, label="HF Transformers", color=TRANS_COLOR,  edgecolor="#FF8E8E", linewidth=1.0, zorder=3)
    ax2.bar(r3, tps_b16_torch,  width=bar_width, label="PyTorch Native",  color=TORCH_COLOR,  edgecolor="#FFBE82", linewidth=1.0, zorder=3)

    ax2.set_title("Peak Batched Serving Throughput (Batch 16)\n▲ Higher is Better", fontsize=12, fontweight="bold", pad=14, color="#FFFFFF")
    ax2.set_ylabel("Output Tokens / Second (tok/s)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10.5)
    ax2.set_ylim(0, 2200)  # Generous headroom for labels + badges
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotations on bars for Panel 2
    for i in range(len(models)):
        v = tps_b16_aether[i]
        speedup = v / max(tps_b16_trans[i], tps_b16_torch[i])
        ax2.text(r1[i], v + 35, f"{v:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#FFFFFF")
        # Speedup badge
        ax2.text(r1[i], v + 210, f"{speedup:.2f}x\n(+{(speedup-1)*100:.0f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                 color=ACCENT_GREEN, bbox=dict(boxstyle="round,pad=0.25", fc="#04210E", ec=ACCENT_GREEN, lw=0.9))

        ax2.text(r2[i], tps_b16_trans[i] + 30, f"{tps_b16_trans[i]:,.0f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
        ax2.text(r3[i], tps_b16_torch[i] + 30, f"{tps_b16_torch[i]:,.0f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)

    # ── Clean Top Header (Titles & Legend Completely Separated) ───────────────
    # 1. Main Title at y = 0.95
    fig.text(0.5, 0.95, "Aether Runtime vs Competitors — Output Tokens Per Second Across 3 Architectures",
             fontsize=15, fontweight="bold", ha="center", va="top", color="#FFFFFF")
    # 2. Hardware / Testbed Subtitle at y = 0.90
    fig.text(0.5, 0.90, "Hardware: 2× NVIDIA Tesla T4 (FP16 Native Tensor-Core Execution) · Kaggle Suite v2.0.0",
             fontsize=11, ha="center", va="top", color=MUTED_TEXT)

    # 3. Legend placed at y = 0.82 (safely above the subplots top=0.71 and subplot titles)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=3,
               frameon=True, fontsize=11, facecolor="#161B22", edgecolor=BORDER_COLOR)

    out_path = os.path.join(OUTPUT_DIR, "cross_engine_throughput_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated throughput comparison: {out_path}")
    return out_path


def generate_latency_chart():
    """Generates End-to-End Request Latency comparison chart with clean spacing and no overlaps."""
    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=300)
    fig.subplots_adjust(top=0.73, bottom=0.12, left=0.08, right=0.95)

    r1 = x - bar_width
    r2 = x
    r3 = x + bar_width

    ax.bar(r1, lat_aether, width=bar_width, label="Aether Runtime (Fastest)", color=AETHER_COLOR, edgecolor="#9F8CFF", linewidth=1.2, zorder=3)
    ax.bar(r2, lat_trans,  width=bar_width, label="HF Transformers", color=TRANS_COLOR,  edgecolor="#FF8E8E", linewidth=1.0, zorder=3)
    ax.bar(r3, lat_torch,  width=bar_width, label="PyTorch Native",  color=TORCH_COLOR,  edgecolor="#FFBE82", linewidth=1.0, zorder=3)

    ax.set_title("Single-Request End-to-End Latency (Batch 1, Prompt 256 / Output 128 Tokens)\n▼ Lower is Better — Decisive Latency Reduction Across All 3 Architectures",
                 fontsize=12, fontweight="bold", pad=14, color="#FFFFFF")
    ax.set_ylabel("End-to-End Request Latency (Seconds)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0, 8.2)  # Generous headroom for labels + badges
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotations on bars
    for i in range(len(models)):
        va = lat_aether[i]
        vt = lat_trans[i]
        vp = lat_torch[i]
        v_comp = min(vt, vp) # closest competitor

        reduction = ((v_comp - va) / v_comp) * 100
        time_saved = v_comp - va

        # Aether bar value
        ax.text(r1[i], va + 0.12, f"{va:.3f}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#FFFFFF")
        # Latency cut badge above Aether
        ax.text(r1[i], va + 0.75, f"-{reduction:.1f}%\n({time_saved:.2f}s saved)", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=ACCENT_GREEN, bbox=dict(boxstyle="round,pad=0.25", fc="#04210E", ec=ACCENT_GREEN, lw=0.9))

        # Transformers bar
        ax.text(r2[i], vt + 0.12, f"{vt:.3f}s", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

        # PyTorch Native bar
        ax.text(r3[i], vp + 0.12, f"{vp:.3f}s", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    # ── Clean Top Header ───────────────────────────────────────────────────────
    # 1. Main Title at y = 0.95
    fig.text(0.5, 0.95, "Aether Runtime vs Competitors — End-to-End Latency Comparison (Not TTFT)",
             fontsize=15, fontweight="bold", ha="center", va="top", color="#FFFFFF")
    # 2. Subtitle at y = 0.90
    fig.text(0.5, 0.90, "Measured Execution Time for 128 Generated Tokens · Kaggle Suite v2.0.0 (2× NVIDIA Tesla T4)",
             fontsize=11, ha="center", va="top", color=MUTED_TEXT)

    # 3. Legend at y = 0.83
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=True, fontsize=10.5, facecolor="#161B22", edgecolor=BORDER_COLOR)

    out_path = os.path.join(OUTPUT_DIR, "cross_engine_latency_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated latency comparison: {out_path}")
    return out_path


if __name__ == "__main__":
    p1 = generate_throughput_chart()
    p2 = generate_latency_chart()
    print("Successfully generated all README benchmark charts:")
    print(" ", p1)
    print(" ", p2)
