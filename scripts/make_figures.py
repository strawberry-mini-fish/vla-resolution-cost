#!/usr/bin/env python
"""画三张图。

    fig1_main.png      训练时间 vs 成功率，带 95% 置信区间
    fig2_persuite.png  分套件对比（每根柱子只有 50 集，用来看单套件的噪声有多大）
    fig3_cost.png      训练时间和评测时间的构成

数字都从 summary.jsonl 读，不手抄。Y 轴要留够范围，别把几点差异放大成满屏。
标签用英文，不然 WSL 里没中文字体会显示成方块。

用法:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FIGURES, RESULTS, read_summary  # noqa: E402

SUITES = ["spatial", "object", "goal", "libero_10"]


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def p_value(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if pool in (0.0, 1.0):
        return 1.0
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------


def load_runs() -> list[dict]:
    """从 summary.jsonl 组装每个配置的训练 + 评测数据。"""
    train, ev = {}, {}
    for r in read_summary():
        if r.get("stage") == "train":
            train[r["name"]] = r
        elif r.get("stage") == "eval":
            ev[r["name"]] = r  # 后来的覆盖前面的，即取最后一次

    out = []
    for name, e in ev.items():
        t = train.get(name, {})
        params = t.get("params") or {}
        raw = params.get("policy.resize_imgs_with_padding", "")
        res = 512
        try:
            res = int(raw.strip("[]").split(",")[0])
        except Exception:
            pass
        n = e.get("n_episodes") or 200
        avg = e.get("avg")
        if avg is None:
            continue
        out.append({
            "name": name,
            "res": res,
            "steps": t.get("steps") or 0,
            "hours": t.get("train_sec", 0) / 3600,
            "mem_mib": t.get("peak_mem_mib", 0),
            "avg": avg,
            "n": n,
            "k": round(avg * n),
            "suites": e.get("suites") or {},
        })
    # 先按分辨率降序，同一分辨率内按步数升序（20k 在 40k 前），
    # 这样 20k/40k 两个点不会在图上交错
    out.sort(key=lambda d: (-d["res"], d["steps"]))
    return out


# ----------------------------------------------------------------------
# 图 1：主图
# ----------------------------------------------------------------------


def fig_main(runs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for i, r in enumerate(runs):
        lo, hi = wilson(r["k"], r["n"])
        x = r["hours"]
        y = r["avg"] * 100
        yerr = [[ (r["avg"] - lo) * 100 ], [ (hi - r["avg"]) * 100 ]]
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=11,
                    capsize=6, capthick=1.8, elinewidth=1.8,
                    color="#2b6cb0", ecolor="#4a5568", zorder=3)
        steps_k = f"{r['steps'] // 1000}k" if r.get("steps") else "?"
        # 上下交错放标签：256@20k 与 384@20k 分数相同、横坐标又近，同侧必然重叠
        up = (i % 2 == 0)
        ax.annotate(f"{r['res']}×{r['res']} @ {steps_k}\n{r['hours']:.2f} h · {y:.1f}%",
                    (x, y), textcoords="offset points",
                    xytext=(10, 14) if up else (10, -14),
                    va="bottom" if up else "top",
                    fontsize=9.5, color="#1a202c")

    ax.set_xlabel("Training time (hours, batch 16)", fontsize=11)
    ax.set_ylabel("Average success rate (%)", fontsize=11)
    ax.set_title("SmolVLA on LIBERO: does high input resolution pay off?",
                 fontsize=12.5, pad=14)

    # 轴范围按数据自适应：加进 40k 的点后训练时间会到 8 h，
    # 不能像原来那样写死 1.1–3.8 h，否则新点会被裁到图外
    los = [wilson(r["k"], r["n"])[0] * 100 for r in runs]
    his = [wilson(r["k"], r["n"])[1] * 100 for r in runs]
    xs = [r["hours"] for r in runs]
    ax.set_ylim(max(0.0, min(los) - 8), min(100.0, max(his) + 8))
    ax.set_xlim(max(0.0, min(xs) * 0.75), max(xs) * 1.25)
    ax.grid(alpha=0.2, linestyle="--")

    # 标注"无显著差异"
    # 主线对比用同步数的 40k 那一对（同样 40000 步，只有分辨率不同）
    by_name = {r["name"]: r for r in runs}
    hi_res, lo_res = by_name.get("r512_40k"), by_name.get("r256_40k")
    if not (hi_res and lo_res):          # 40k 还没跑完时退回到 20k 那一对
        hi_res, lo_res = by_name.get("base_r512_bf16"), by_name.get("r256_bf16")
    if hi_res and lo_res:
        p = p_value(hi_res["avg"], hi_res["n"], lo_res["avg"], lo_res["n"])
        verdict = ("no significant accuracy change" if p >= 0.05
                   else "significant difference — see report")
        ax.text(0.02, 0.97,
                f"{hi_res['steps'] // 1000}k steps — 512 vs {lo_res['res']}: "
                f"Δ {(lo_res['avg'] - hi_res['avg']) * 100:+.1f} pt, "
                f"p = {p:.2f}  (n = {hi_res['n']}/config)\n"
                f"Training time reduced by "
                f"{(1 - lo_res['hours'] / hi_res['hours']) * 100:.0f}%, "
                f"{verdict}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7fafc",
                          edgecolor="#cbd5e0"))

    ax.text(0.02, 0.03, "Error bars: 95% Wilson interval (200 episodes per config)",
            transform=ax.transAxes, fontsize=8, color="#4a5568", style="italic")

    fig.tight_layout()
    out = FIGURES / "fig1_main.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ----------------------------------------------------------------------
# 图 2：分套件（展示噪声）
# ----------------------------------------------------------------------


def fig_persuite(runs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(SUITES))
    width = 0.26
    colors = ["#c53030", "#2b6cb0", "#2f855a", "#b7791f", "#6b46c1"]

    for i, r in enumerate(runs):
        vals, errs_lo, errs_hi = [], [], []
        for s in SUITES:
            v = (r["suites"].get(s) or 0)
            k = round(v * 50)
            lo, hi = wilson(k, 50)
            vals.append(v * 100)
            errs_lo.append((v - lo) * 100)
            errs_hi.append((hi - v) * 100)
        off = (i - (len(runs) - 1) / 2) * width
        bar_steps = f"{r['steps'] // 1000}k" if r.get("steps") else "?"
        ax.bar(x + off, vals, width, label=f"{r['res']}×{r['res']} @ {bar_steps}",
               color=colors[i % len(colors)], alpha=0.85,
               yerr=[errs_lo, errs_hi], capsize=3,
               error_kw={"elinewidth": 1.3, "ecolor": "#2d3748"})

    ax.set_xticks(x)
    ax.set_xticklabels(["Spatial", "Object", "Goal", "Long-horizon"], fontsize=10.5)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_title("Per-suite results — 50 episodes each, error bars overlap heavily",
                 fontsize=12, pad=12)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.text(0.99, 0.97,
            "n = 50 per bar → 95% CI ≈ ±14 pt\n"
            "Differences between bars are NOT reliable;\n"
            "only the 200-episode aggregate is.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#c53030",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffaf0",
                      edgecolor="#f6ad55"))

    fig.tight_layout()
    out = FIGURES / "fig2_persuite.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ----------------------------------------------------------------------
# 图 3：成本构成
# ----------------------------------------------------------------------


EVAL_HOURS = 2.07  # 实测：200 episode 约 2 h 04 min，与分辨率基本无关


def fig_cost(runs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    labels = [f"{r['res']}×{r['res']}\n{r['steps'] // 1000}k" if r.get("steps")
              else f"{r['res']}×{r['res']}" for r in runs]
    trainh = [r["hours"] for r in runs]
    evalh = [EVAL_HOURS] * len(runs)
    x = np.arange(len(runs))

    ax.bar(x, trainh, 0.55, label="Training", color="#2b6cb0")
    ax.bar(x, evalh, 0.55, bottom=trainh, label="Evaluation (200 episodes)",
           color="#a0aec0")

    for i, r in enumerate(runs):
        ax.text(i, trainh[i] / 2, f"{trainh[i]:.2f} h", ha="center",
                va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(i, trainh[i] + evalh[i] / 2, f"{evalh[i]:.2f} h", ha="center",
                va="center", color="#1a202c", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Input resolution", fontsize=11)
    ax.set_ylabel("Wall-clock time (hours)", fontsize=11)
    ax.set_title("Cost breakdown: resolution only affects training, not evaluation",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    fig.tight_layout()
    out = FIGURES / "fig3_cost.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ----------------------------------------------------------------------


def main() -> int:
    runs = load_runs()
    if not runs:
        print("summary.jsonl 里没有可用的评测数据")
        return 1

    print("=" * 60)
    print(" 数据源")
    print("=" * 60)
    for r in runs:
        lo, hi = wilson(r["k"], r["n"])
        print(f"  {r['res']:>3}×{r['res']:<3}  {r['hours']:.2f} h  "
              f"{r['avg'] * 100:5.1f}%  CI[{lo * 100:.1f}, {hi * 100:.1f}]  n={r['n']}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    print("\n生成图表：")
    fig_main(runs)
    if len(runs) >= 2:
        fig_persuite(runs)
        fig_cost(runs)
    print(f"\n全部输出到 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
