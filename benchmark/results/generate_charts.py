"""
Generate benchmark charts for Aether Runtime vs HuggingFace Transformers.
Produces 4 chart types per model:
  1. Bar chart — throughput comparison by (prompt, batch)
  2. Line chart — throughput vs prompt length (per batch size)
  3. Heatmap — speedup ratio across (prompt × batch) grid
  4. Latency bar chart — median latency comparison
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Palette ───────────────────────────────────────────────────────────────
AETHER_COLOR   = "#6C63FF"   # violet
TRANS_COLOR    = "#FF6B6B"   # coral red
BG_COLOR       = "#0D1117"   # GitHub dark
GRID_COLOR     = "#21262D"
TEXT_COLOR     = "#E6EDF3"
ACCENT         = "#58A6FF"
SPEEDUP_CMAP   = "RdYlGn"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ─── Raw benchmark data ─────────────────────────────────────────────────────
# Structure: {model_key: [ (prompt, batch, hf_toks, ae_toks, hf_lat, ae_lat), ... ]}

DATA = {
    "SmolLM2-135M-Instruct": [
        (32,   1,  23.95, 46.06, 5.3445, 2.7789),
        (32,   2,  49.37, 87.50, 5.1851, 2.9256),
        (32,   4,  95.03,172.41, 5.3880, 2.9697),
        (256,  1,  23.71, 45.80, 5.3995, 2.7945),
        (256,  2,  47.50, 85.56, 5.3900, 2.9921),
        (256,  4,  89.14,157.56, 5.7438, 3.2496),
        (1024, 1,  21.96, 41.50, 5.8285, 3.0841),
        (1024, 2,  43.36, 75.61, 5.9038, 3.3860),
        (1024, 4,  82.71,129.43, 6.1903, 3.9559),
    ],
    "Qwen3-0.6B": [
        (32,   1,  19.56, 41.96, 6.5429, 3.0508),
        (32,   2,  33.88, 42.78, 7.5559, 5.9842),
        (32,   4,  65.40, 81.40, 7.8282, 6.2902),
        (256,  1,  19.60, 40.24, 6.5308, 3.1811),
        (256,  2,  31.35, 37.73, 8.1665, 6.7859),
        (256,  4,  54.69, 66.04, 9.3616, 7.7532),
        (1024, 1,  18.44, 35.71, 6.9408, 3.5840),
        (1024, 2,  23.11, 27.38,11.0777, 9.3493),
        (1024, 4,  34.39, 39.81,14.8891,12.8603),
    ],
    "GPTNeo350M-Instruct-SFT": [
        (32,   1,  39.14, 71.67, 3.2707, 1.7860),
        (32,   2,  47.37, 64.57, 5.4047, 3.9647),
        (32,   4,  93.42,123.11, 5.4807, 4.1589),
        (256,  1,  39.34, 63.14, 3.2540, 2.0271),
        (256,  2,  45.53, 57.40, 5.6228, 4.4601),
        (256,  4,  85.60, 99.94, 5.9811, 5.1229),
        (1024, 1,  36.94, 54.41, 3.4647, 2.3526),
        (1024, 2,  39.08, 41.35, 6.5505, 6.1918),
        (1024, 4,  65.21, 60.38, 7.8522, 8.4795),
    ],
}

MODEL_SHORT = {
    "SmolLM2-135M-Instruct":   "SmolLM2-135M",
    "Qwen3-0.6B":              "Qwen3-0.6B",
    "GPTNeo350M-Instruct-SFT": "GPTNeo-350M",
}

PROMPTS = [32, 256, 1024]
BATCHES = [1, 2, 4]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  saved -> {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Grouped Bar: throughput (tok/s) for all (prompt, batch) combos
# ═══════════════════════════════════════════════════════════════════════════
def chart_throughput_bar(model, rows, out_dir):
    labels   = [f"P{p}\nB{b}" for p, b, *_ in rows]
    hf_vals  = [r[2] for r in rows]
    ae_vals  = [r[3] for r in rows]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)

    b1 = ax.bar(x - width/2, hf_vals, width, color=TRANS_COLOR,
                label="HuggingFace Transformers", zorder=3, alpha=0.9)
    b2 = ax.bar(x + width/2, ae_vals, width, color=AETHER_COLOR,
                label="Aether Runtime", zorder=3, alpha=0.9)

    # value labels on bars
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=8, color=TRANS_COLOR, fontweight="bold")
    for bar in b2:
        speedup = bar.get_height() / hf_vals[list(b2).index(bar)]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}\n({speedup:.2f}×)",
                ha="center", va="bottom", fontsize=7.5,
                color=AETHER_COLOR, fontweight="bold")

    ax.set_xlabel("Prompt tokens (P) × Batch size (B)", fontsize=11)
    ax.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax.set_title(f"{MODEL_SHORT[model]} — Throughput: Aether vs Transformers (BF16)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    slug = model.replace("/", "_").replace("-", "_").lower()
    _save(fig, os.path.join(out_dir, f"{slug}_throughput_bar.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Line: throughput vs prompt length, one line per batch
# ═══════════════════════════════════════════════════════════════════════════
def chart_throughput_line(model, rows, out_dir):
    by_batch = {b: [] for b in BATCHES}
    for p, b, hf, ae, *_ in rows:
        by_batch[b].append((p, hf, ae))

    batch_colors = ["#FFD700", "#00CED1", "#FF69B4"]   # gold, teal, pink

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"{MODEL_SHORT[model]} — Throughput vs Prompt Length (BF16)",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, (backend_name, col_idx) in zip(
        axes, [("HuggingFace Transformers", 1), ("Aether Runtime", 2)]
    ):
        for i, b in enumerate(BATCHES):
            pts   = sorted(by_batch[b], key=lambda x: x[0])
            xs    = [pt[0] for pt in pts]
            ys    = [pt[col_idx] for pt in pts]
            color = batch_colors[i]
            ax.plot(xs, ys, "o-", color=color, linewidth=2.5,
                    markersize=7, label=f"Batch {b}", zorder=3)
            for xi, yi in zip(xs, ys):
                ax.annotate(f"{yi:.0f}", (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color)

        ax.set_xscale("log")
        ax.set_xticks(PROMPTS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Prompt tokens", fontsize=11)
        ax.set_ylabel("Throughput (tok/s)", fontsize=11)
        ax.set_title(backend_name, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    plt.tight_layout()
    slug = model.replace("/", "_").replace("-", "_").lower()
    _save(fig, os.path.join(out_dir, f"{slug}_throughput_line.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3 — Heatmap: speedup ratio across prompt × batch grid
# ═══════════════════════════════════════════════════════════════════════════
def chart_speedup_heatmap(model, rows, out_dir):
    speedup = np.zeros((len(BATCHES), len(PROMPTS)))
    for p, b, hf, ae, *_ in rows:
        i = BATCHES.index(b)
        j = PROMPTS.index(p)
        speedup[i, j] = ae / hf

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR)

    im = ax.imshow(speedup, cmap=SPEEDUP_CMAP, aspect="auto",
                   vmin=0.8, vmax=2.3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup (Aether / Transformers)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    ax.set_xticks(range(len(PROMPTS)))
    ax.set_xticklabels([f"{p} tok" for p in PROMPTS], fontsize=10)
    ax.set_yticks(range(len(BATCHES)))
    ax.set_yticklabels([f"Batch {b}" for b in BATCHES], fontsize=10)

    for i in range(len(BATCHES)):
        for j in range(len(PROMPTS)):
            v   = speedup[i, j]
            txt = f"{v:.2f}×"
            col = "black" if 1.0 < v < 1.8 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=col)

    ax.set_title(f"{MODEL_SHORT[model]} — Speedup Heatmap (BF16)\n"
                 f"(Aether tok/s ÷ Transformers tok/s)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prompt length", fontsize=11)
    ax.set_ylabel("Batch size", fontsize=11)

    plt.tight_layout()
    slug = model.replace("/", "_").replace("-", "_").lower()
    _save(fig, os.path.join(out_dir, f"{slug}_speedup_heatmap.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4 — Latency bar: median generation latency (s) side-by-side
# ═══════════════════════════════════════════════════════════════════════════
def chart_latency_bar(model, rows, out_dir):
    labels   = [f"P{p}\nB{b}" for p, b, *_ in rows]
    hf_lat   = [r[4] for r in rows]
    ae_lat   = [r[5] for r in rows]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)

    b1 = ax.bar(x - width/2, hf_lat, width, color=TRANS_COLOR,
                label="HuggingFace Transformers", zorder=3, alpha=0.9)
    b2 = ax.bar(x + width/2, ae_lat, width, color=AETHER_COLOR,
                label="Aether Runtime", zorder=3, alpha=0.9)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}s", ha="center", va="bottom",
                fontsize=8, color=TRANS_COLOR, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}s", ha="center", va="bottom",
                fontsize=8, color=AETHER_COLOR, fontweight="bold")

    ax.set_xlabel("Prompt tokens (P) × Batch size (B)", fontsize=11)
    ax.set_ylabel("Median Latency (seconds)", fontsize=11)
    ax.set_title(f"{MODEL_SHORT[model]} — Latency: Aether vs Transformers (BF16)\n"
                 f"Lower is better",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    slug = model.replace("/", "_").replace("-", "_").lower()
    _save(fig, os.path.join(out_dir, f"{slug}_latency_bar.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Combined summary: one overview speedup bar per model  (for README header)
# ═══════════════════════════════════════════════════════════════════════════
def chart_overview_speedup(out_dir):
    model_labels = ["SmolLM2-135M", "Qwen3-0.6B", "GPTNeo-350M"]
    models_keys  = list(DATA.keys())

    # compute mean speedup per model
    mean_speedups = []
    for key in models_keys:
        rows = DATA[key]
        sp   = [ae / hf for _, _, hf, ae, *_ in rows]
        mean_speedups.append(np.mean(sp))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)

    colors = [AETHER_COLOR, "#00E5CC", "#FF9F43"]
    bars   = ax.bar(model_labels, mean_speedups, color=colors,
                    alpha=0.9, zorder=3, width=0.5)

    for bar, sp in zip(bars, mean_speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{sp:.2f}×", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=TEXT_COLOR)

    ax.axhline(1.0, color="white", linestyle="--", linewidth=1.5,
               alpha=0.5, label="1× baseline (equal speed)")
    ax.set_ylabel("Mean Speedup (Aether / Transformers)", fontsize=11)
    ax.set_title("Aether Runtime — Mean Speedup over HuggingFace Transformers\n"
                 "BF16 · 2× Tesla T4 · prompt 32–1024 · batch 1–4",
                 fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(mean_speedups) * 1.25)

    _save(fig, os.path.join(out_dir, "overview_speedup.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.ticker

    print("Generating benchmark charts…")
    for model, rows in DATA.items():
        print(f"\n  Model: {model}")
        chart_throughput_bar(model, rows, OUT)
        chart_throughput_line(model, rows, OUT)
        chart_speedup_heatmap(model, rows, OUT)
        chart_latency_bar(model, rows, OUT)

    print("\n  Overview chart…")
    chart_overview_speedup(OUT)

    print("\nAll charts generated successfully.")
