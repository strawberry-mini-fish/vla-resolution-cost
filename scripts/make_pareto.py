#!/usr/bin/env python
"""生成 results/table.md：显存、耗时、各套件成功率，带帕累托标记。

用法:
    python scripts/make_pareto.py
    python scripts/make_pareto.py --metric peak_mem_mib   # 换排序用的指标

产出:
    results/table.md    可以直接贴进 README
    stdout              表格 + 一段结论草稿

画图不用这个脚本了，看 scripts/make_figures.py。旧的帕累托图 Y 轴是截断的，
留着 --legacy-plot 只为了对照。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FIGURES, RESULTS, read_summary  # noqa: E402

SUITES = ["spatial", "object", "goal", "libero_10"]


def merge(records: list[dict]) -> dict[str, dict]:
    """把同一实验的 train / eval 两条记录合并成一条。"""
    merged: dict[str, dict] = {}
    for r in records:
        name = r.get("name")
        if not name:
            continue
        cur = merged.setdefault(name, {"name": name})
        if r.get("stage") == "train":
            for k in ("peak_mem_mib", "peak_mem_p95_mib", "train_sec", "steps", "batch_size"):
                if k in r:
                    cur[k] = r[k]
            cur.setdefault("params", {}).update(r.get("params") or {})
        elif r.get("stage") == "eval":
            cur["suites"] = r.get("suites")
            cur["avg"] = r.get("avg")
            cur["ci95"] = r.get("ci95")
            cur["n_episodes"] = r.get("n_episodes")
    return merged


def pareto(points: list[dict], mem_key: str) -> list[dict]:
    """返回帕累托前沿：显存更少且成功率更高，不被任何其他点支配。"""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_mem = q[mem_key] <= p[mem_key]
            better_acc = q["avg"] >= p["avg"]
            strict = q[mem_key] < p[mem_key] or q["avg"] > p["avg"]
            if better_mem and better_acc and strict:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda x: x[mem_key])


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="peak_mem_mib",
                    help="横轴指标：peak_mem_mib 或 train_sec")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--legacy-plot", action="store_true",
                    help="生成旧版画图（Y 轴截断、无误差棒，仅供对照，不建议使用）")
    args = ap.parse_args()

    records = read_summary()
    if not records:
        print("results/summary.jsonl 还是空的，先跑 run_sweep.py")
        return 1

    data = merge(records)
    usable = [
        d for d in data.values()
        if d.get("avg") is not None and d.get(args.metric) is not None
    ]
    if not usable:
        print(f"没有同时具备 {args.metric} 和成功率的实验。")
        print("检查：训练记录里有没有 peak_mem_mib？评测有没有跑过？")
        return 1

    usable.sort(key=lambda d: d[args.metric])
    front = pareto(usable, args.metric)
    front_names = {d["name"] for d in front}

    # ---------- markdown 表 ----------
    lines = []
    lines.append("| 实验 | 显存(MiB) | 训练(min) | spatial | object | goal | long | 平均 | 前沿 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for d in usable:
        s = d.get("suites") or {}
        mark = "●" if d["name"] in front_names else ""
        lines.append(
            f"| {d['name']} | {d.get(args.metric, 0):.0f} "
            f"| {d.get('train_sec', 0) / 60:.0f} "
            f"| {fmt_pct(s.get('spatial'))} | {fmt_pct(s.get('object'))} "
            f"| {fmt_pct(s.get('goal'))} | {fmt_pct(s.get('libero_10'))} "
            f"| **{fmt_pct(d.get('avg'))}** | {mark} |"
        )
    table = "\n".join(lines)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n已写入 {RESULTS / 'table.md'}")

    # ---------- 一句话结论草稿 ----------
    print("\n" + "=" * 70)
    print(" 结论草稿（数字来自上表，请人工核对后再用）")
    print("=" * 70)

    best_acc = max(usable, key=lambda d: d["avg"])
    cheapest = min(usable, key=lambda d: d[args.metric])
    print(f" 最高成功率 : {best_acc['name']}  {best_acc['avg'] * 100:.1f}%"
          f"  显存 {best_acc[args.metric]:.0f} MiB")
    print(f" 最省资源   : {cheapest['name']}  {cheapest['avg'] * 100:.1f}%"
          f"  显存 {cheapest[args.metric]:.0f} MiB")

    if best_acc["name"] != cheapest["name"]:
        d_acc = (best_acc["avg"] - cheapest["avg"]) * 100
        d_mem = best_acc[args.metric] - cheapest[args.metric]
        if d_mem > 0:
            print(f" 为多拿 {d_acc:.1f} 个点，需多花 {d_mem:.0f} MiB "
                  f"（{d_acc / (d_mem / 1024):.1f} 点/GB）")
        if abs(d_acc) < 4:
            print(" ⚠️ 差距 <4 点，在 400 episodes 的置信区间内，需要第二个 seed 才能下结论")

    print(f"\n 帕累托前沿: {', '.join(d['name'] for d in front)}")
    print("=" * 70)

    # ---------- 画图 ----------
    # ⚠️ 本函数的图已被 scripts/make_figures.py 取代，默认不再生成。
    #    原因：这里的 Y 轴会自动贴合数据范围（例如 44.4–49.6），
    #    把几个点的差异放大到整张图高，容易误导；而且没有误差棒。
    #    需要看老图请显式加 --legacy-plot。
    if not args.legacy_plot:
        print("\n（画图已迁移到 scripts/make_figures.py，本脚本默认只出表格）")
        return 0
    if args.no_plot:
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n没装 matplotlib，跳过画图：pip install matplotlib")
        return 0

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    fx = [d[args.metric] for d in front]
    fy = [d["avg"] * 100 for d in front]
    ax.plot(fx, fy, "--", color="#888", linewidth=1.2, zorder=1,
            label="Pareto frontier")

    for d in usable:
        is_front = d["name"] in front_names
        ax.scatter(d[args.metric], d["avg"] * 100,
                   s=90 if is_front else 60,
                   color="#d62728" if is_front else "#4c72b0",
                   zorder=3, alpha=0.9 if is_front else 0.6)
        ax.annotate(d["name"], (d[args.metric], d["avg"] * 100),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, alpha=0.9)

    xlabel = {"peak_mem_mib": "Peak GPU memory (MiB)",
              "train_sec": "Training time (s)"}.get(args.metric, args.metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average success rate (%)")
    ax.set_title("SmolVLA on LIBERO: memory–performance trade-off")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = FIGURES / "pareto.png"
    fig.savefig(out, dpi=160)
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
