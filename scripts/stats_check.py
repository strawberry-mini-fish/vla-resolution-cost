#!/usr/bin/env python
"""比较两个配置的成功率差异是不是真的。

用法:
    python scripts/stats_check.py                              # 读 summary.jsonl 两两比
    python scripts/stats_check.py --a 0.40 --b 0.34 --n 200    # 手算

200 个 episode、成功率 50% 时，95% 区间大概 ±7 个点。7 点以内的差异在这个样本量下
看不出来。看不出来不等于没有，报告里这两种说法得分清楚。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, read_summary  # noqa: E402

EVAL_LOGS = RESULTS / "eval_logs"
SUITES = ["spatial", "object", "goal", "libero_10"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（小样本比正态近似可靠）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_prop_p(a_p: float, a_n: int, b_p: float, b_n: int) -> float:
    """两比例 z 检验的 p 值（双侧）。"""
    if a_n == 0 or b_n == 0:
        return 1.0
    p_pool = (a_p * a_n + b_p * b_n) / (a_n + b_n)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / a_n + 1 / b_n))
    if se == 0:
        return 1.0
    z = abs(a_p - b_p) / se
    # 正态近似双侧 p 值
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def load_outcomes(name: str) -> dict[tuple[str, int, int], bool] | None:
    """从 eval_logs 读某个配置的逐集成败，用于配对检验。

    每个条件必须 4 个套件都在；缺任何一个就返回 None（不做配对分析）。
    键是 (套件, 任务号, 集号)——因为在 init_states=true 下它唯一对应一个初始状态。
    """
    out: dict[tuple[str, int, int], bool] = {}
    for s in SUITES:
        p = EVAL_LOGS / f"{name}_{s}" / "eval_info.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        for t in d.get("per_task", []):
            for i, ok in enumerate(t.get("metrics", {}).get("successes", [])):
                out[(s, t["task_id"], i)] = bool(ok)
    return out or None


def mcnemar_p(b: int, c: int) -> float:
    """配对 McNemar 检验（带连续性校正）的 p 值。b/c 是只有一方成功的集数。"""
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(max(chi2, 0.0) / 2))


def verdict(diff: float, p: float, n: int) -> str:
    if p < 0.05:
        return f"差异 {diff * 100:+.1f} 点，p={p:.3f} < 0.05 → **统计显著**"
    if abs(diff) < 0.10:
        return (f"差异 {diff * 100:+.1f} 点，p={p:.2f} ≥ 0.05 → **未检测到显著差异**\n"
                f"     ⚠️ 这不等于「没有差异」。当前 n={n} 只能检测 ≥10 点的差异；\n"
                f"        更小的真实差异需要更大样本才能分辨。")
    return f"差异 {diff * 100:+.1f} 点，p={p:.2f} ≥ 0.05 → 未达显著，但差异较大，建议加样本"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, help="配置 A 的成功率（0-1）")
    ap.add_argument("--b", type=float, help="配置 B 的成功率（0-1）")
    ap.add_argument("--n", type=int, default=200, help="每个配置的 episode 数")
    args = ap.parse_args()

    if args.a is not None and args.b is not None:
        pairs = [("<A>", args.a, args.n), ("<B>", args.b, args.n)]
    else:
        recs = {}
        for r in read_summary():
            if r.get("stage") == "eval" and r.get("avg") is not None:
                recs[r["name"]] = r
        if len(recs) < 2:
            print("summary.jsonl 里不足两个配置的评测结果")
            return 1
        pairs = [(n, r["avg"], r.get("n_episodes") or 200) for n, r in recs.items()]

    print("=" * 68)
    print(" 各配置的置信区间")
    print("=" * 68)
    for name, p, n in pairs:
        k = round(p * n)
        lo, hi = wilson(k, n)
        print(f"  {name:20} {p * 100:5.1f}%  n={n:<4}  CI95=[{lo * 100:.1f}%, {hi * 100:.1f}%]")

    if len(pairs) >= 2:
        print()
        print("=" * 68)
        print(" 两两比较")
        print("=" * 68)
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                na, pa, nna = pairs[i]
                nb, pb, nnb = pairs[j]
                p = two_prop_p(pa, nna, pb, nnb)
                print(f"\n  {na} vs {nb}")
                print(f"    {verdict(pa - pb, p, min(nna, nnb))}")

                # 配对检验：同一初始状态下的成对比较（比上面的非配对 z 检验更敏感）
                oa, ob = load_outcomes(na), load_outcomes(nb)
                if oa and ob:
                    keys = sorted(set(oa) & set(ob))
                    b = sum(1 for k in keys if oa[k] and not ob[k])
                    c = sum(1 for k in keys if ob[k] and not oa[k])
                    print(f"    配对 McNemar（n={len(keys)}）：仅前者成功 {b} 集，"
                          f"仅后者成功 {c} 集 → p = {mcnemar_p(b, c):.3f}")
                else:
                    print("    配对 McNemar：缺少 eval_logs 的逐集结果，跳过")

    print()
    print("=" * 68)
    print(" 怎么报")
    print("=" * 68)
    print("  ✅ 显著  → 「A 比 B 高 X 点，p<0.05」")
    print("  ✅ 不显著 → 「未检测到显著差异；在当前样本量下只能排除 ≥10 点的差异」")
    print("  ❌ 错误写法 → 「两者性能相同」「没有差异」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
