#!/usr/bin/env python
"""汇总机制验证结果：打印 2×2 矩阵，并给出解释判定。

用法:
    python scripts/mechanism_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS  # noqa: E402
from stats_check import two_prop_p  # noqa: E402

MECH = RESULTS / "mechanism.jsonl"
SUITES = ["spatial", "object", "goal", "libero_10"]

# 判定阈值不写死「几个点」：200 episodes 下 Wilson CI 半宽约 7 点，
# 硬编码 4 点会把 p=0.32 的差异误判成「超出噪声」（与 stats_check.py 矛盾）。
# 改为按各自样本量做两比例 z 检验，和主线统计保持同一套算法。


def load() -> dict[str, dict]:
    if not MECH.exists():
        return {}
    out = {}
    for line in MECH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["condition"]] = r
    return out


def pct(x: float | None) -> str:
    return "  —  " if x is None else f"{x * 100:5.1f}"


def judge(pa: float, na: int, pb: float, nb: int, label_a: str, label_b: str) -> str:
    """两比例 z 检验判定（与 stats_check.py 同一套算法）。"""
    diff = pa - pb
    p = two_prop_p(pa, na, pb, nb)
    if p < 0.05:
        better = label_a if diff > 0 else label_b
        return f"差异 {diff * 100:+.1f} 点，p={p:.3f} < 0.05 → {better} 更好（显著）"
    return (f"差异 {diff * 100:+.1f} 点，p={p:.2f} ≥ 0.05 → 未检测到显著差异"
            f"（n={min(na, nb)} 下只能排除约 ≥10 点的差异，不能说两者相同）")


def main() -> int:
    d = load()
    if not d:
        print("results/mechanism.jsonl 是空的，先跑 mechanism_test.py")
        return 1

    get = lambda k: d.get(k, {}).get("avg")  # noqa: E731
    c1, c2, c3, c4 = get("c1_512on512"), get("c2_512on256"), get("c3_256on256"), get("c4_256on512")

    # episode 数不一致时，格子之间不可直接比较
    ns = {k: d.get(k, {}).get("n_episodes") for k in d}
    if len({v for v in ns.values() if v}) > 1:
        print(f"⚠️  各条件 episode 数不一致：{ns}")
        print("    不同样本量的成功率不可直接比较，请用同一个 --n-episodes 重跑。")
        print()
    n_of = lambda k: int(d.get(k, {}).get("n_episodes") or 0)  # noqa: E731

    print("=" * 66)
    print(" 分辨率机制验证 · 结果矩阵")
    print("=" * 66)
    print()
    print(f"{'':<14}{'评测 512':>12}{'评测 256':>12}")
    print(f"{'模型训于 512':<14}{pct(c1):>12}{pct(c2):>12}")
    print(f"{'模型训于 256':<14}{pct(c4):>12}{pct(c3):>12}")
    print()
    print("  对角线 = 训练/评测一致；非对角线 = 分辨率失配")
    reused = [k for k in d if str(d[k].get("source", "")).startswith("reused")]
    if reused:
        print(f"  注：{', '.join(reused)} 复用主实验评测结果，非本次重跑")
    print()

    # ---------- 逐套件 ----------
    if c1 is not None:
        head = f"{'套件':<12}"
        for k in ("c1_512on512", "c2_512on256", "c3_256on256", "c4_256on512"):
            head += f"{k.split('_')[0].upper():>10}"
        print(head)
        print("-" * 52)
        for s in SUITES:
            row = f"{s:<12}"
            for k in ("c1_512on512", "c2_512on256", "c3_256on256", "c4_256on512"):
                v = d.get(k, {}).get("suites", {}).get(s)
                row += f"{pct(v):>10}"
            print(row)
        print()

    # ---------- 判定 ----------
    print("=" * 66)
    print(" 判定")
    print("=" * 66)
    print()

    ok = True

    # 完整性检查 + 改写是否生效
    if c1 is not None and c2 is not None and abs(c1 - c2) < 1e-6:
        print("⚠️  C1 与 C2 完全相同 —— 评测分辨率很可能没生效。")
        print("    1) 先看日志首行的命令里有没有 --policy.resize_imgs_with_padding=[256,256]。")
        print("       只改 config.json、不给这个显式参数时，--policy.type=smolvla 会用默认")
        print("       512 覆盖它（正是 2026-11-04 那轮 5 小时作废评测的根因）。")
        print("    2) 再检查 results/eval_variants/c2_512on256/config.json 的值。")
        print()
        ok = False

    # 核心比较：C2 vs C3
    if c2 is not None and c3 is not None:
        n2, n3 = n_of("c2_512on256"), n_of("c3_256on256")
        p_core = two_prop_p(c3, n3, c2, n2)
        print("【核心】C2 vs C3：同样喂 256，训练时适配过 256 的模型更好吗？")
        print(f"         C2(训512评256) = {c2 * 100:.1f}%   C3(训256评256) = {c3 * 100:.1f}%")
        print(f"         {judge(c3, n3, c2, n2, 'C3', 'C2')}")
        print()
        if p_core >= 0.05:
            print("  → 结论：训练分辨率适配的作用有限，掉点主要由【信息量】决定。")
            print("     即：8GB 上降分辨率确实只是「看少了东西」，不是「模型不适应」。")
            print("     实务含义：降分辨率是一条干净的路，不会引入额外的适配成本。")
        else:
            print("  → 结论：训练分辨率适配有实质作用，【信息量】不足以解释全部差异。")
            print("     即：模型需要在自己的分辨率上训练才能发挥，单纯降输入不够。")
            print("     实务含义：降分辨率必须重训，不能靠推理时改配置省事。")
        print()

    # 失配代价：C1 vs C2
    if c1 is not None and c2 is not None:
        print("【辅助】C1 vs C2：高分模型遇到低分输入，能扛住吗？")
        print(f"         C1 = {c1 * 100:.1f}%   C2 = {c2 * 100:.1f}%   "
              f"{judge(c2, n_of('c2_512on256'), c1, n_of('c1_512on512'), 'C2', 'C1')}")
        print()

    # 反向：C3 vs C4
    if c3 is not None and c4 is not None:
        print("【辅助】C3 vs C4：低分模型遇到高分输入，会变好吗？")
        print(f"         C3 = {c3 * 100:.1f}%   C4 = {c4 * 100:.1f}%   "
              f"{judge(c4, n_of('c4_256on512'), c3, n_of('c3_256on256'), 'C4', 'C3')}")
        print()

    # ---------- 结果摘要（数字自动填入，可直接粘贴）----------
    print("=" * 66)
    print(" 结果摘要")
    print("=" * 66)
    print()
    if c1 is not None and c2 is not None and c3 is not None:
        n2, n3 = n_of("c2_512on256"), n_of("c3_256on256")
        p_phrase = two_prop_p(c3, n3, c2, n2)
        print(f"  为区分「信息瓶颈」与「分辨率失配」两种解释，我们构造了 2×2")
        print(f"  训测分辨率矩阵（纯推理，无额外训练）：把模型训于 512 评于 256 时")
        print(f"  成功率 {c2 * 100:.1f}%，而训于 256 评于 256 时为 {c3 * 100:.1f}%，"
              f"相差 {abs(c3 - c2) * 100:.1f} 点（p={p_phrase:.2f}，n={min(n2, n3)}/条件）。")
        if p_phrase >= 0.05:
            print(f"  该差异未达统计显著，说明训练侧的分辨率适配并非性能差异的主要来源；")
            print(f"  但只能表述为「未检测到显著差异」，不能声称「两者相同」。")
        else:
            print(f"  该差异达到统计显著，说明分辨率适配本身有独立贡献，")
            print(f"  低分辨率的影响不能仅用「信息变少」解释。")
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
