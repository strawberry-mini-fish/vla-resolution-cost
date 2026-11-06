#!/usr/bin/env python
"""机制验证：区分"信息量损失"和"训练-评测分辨率失配"两种解释。

背景
----
前面测出 r256 比 r512 掉点。但这个现象有两种完全不同的解释：

  解释 A（信息瓶颈）：256 的图信息少，模型看不到物体边界细节，所以做不对。
  解释 B（表示失配） ：SmolVLA 的 VLM 在 512 预处理上预训练，喂 256 属于
                        分布外输入，特征本身就坏了，跟信息量无关。

两者的预测不同，可以用下面的 2×2 矩阵区分。全部是纯推理，不需要重训。

  ┌────────────┬──────────┬──────────┐
  │            │ 评测 512 │ 评测 256 │
  ├────────────┼──────────┼──────────┤
  │ 模型训于512 │   C1     │   C2     │
  │ 模型训于256 │   C4     │   C3     │
  └────────────┴──────────┴──────────┘

关键比较是 C2 vs C3：同样是 256 的输入，训练时见过 256 的模型是否更好？

  * C2 ≈ C3  → 训练分辨率适配几乎无用，掉点纯由信息量决定（支持 A）
  * C2 << C3  → 适配有用，256 模型学到了不同的特征提取方式（A 不完整）

用法
----
    python scripts/mechanism_test.py --list
    python scripts/mechanism_test.py --all
    python scripts/mechanism_test.py --only c2
    python scripts/mechanism_test.py --all --n-episodes 10

前置条件
--------
需要先有 base_r512_bf16 和 r256_bf16 两个训练完成的 checkpoint。

注意
----
本脚本通过"复制 checkpoint 并改写 config.json 里的 resize_imgs_with_padding"
来改变评测分辨率，不改 LeRobot 源码。如果 C2 的结果与 C1 完全相同，
说明改写没有生效，脚本会给出警告。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LOGS, RESULTS, RUNS, ensure_dirs, lerobot_commit, load_config,
    match_names, read_summary, run_streaming, which,
)
from eval_all import find_checkpoint, parse_success, SUITES, SUITE_SHORT  # noqa: E402

MECH = RESULTS / "mechanism.jsonl"
VARIANTS = RESULTS / "eval_variants"

# (条件名, 说明, 训练时分辨率, 评测时分辨率)
CONDITIONS = [
    ("c1_512on512", "基线：训512 评512", 512, 512),
    ("c2_512on256", "失配：训512 评256", 512, 256),
    ("c3_256on256", "自洽：训256 评256", 256, 256),
    ("c4_256on512", "反向失配：训256 评512", 256, 512),
]

MODEL_CFG = {
    512: "base_r512_bf16",
    256: "r256_bf16",
}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("\ufeff")
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def append_jsonl(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_variant(src_ckpt: Path, out_dir: Path, resize: int) -> tuple[Path, int]:
    """复制 checkpoint，改写 resize_imgs_with_padding。返回 (新路径, 原值)。

    大文件用硬链接避免重复占盘；失败则退回复制。
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for f in src_ckpt.iterdir():
        dst = out_dir / f.name
        if f.is_file() and f.stat().st_size > 10 * 1024 * 1024:
            try:
                os.link(f, dst)
                continue
            except OSError:
                pass  # 跨文件系统等情况，退回复制
        if f.is_file():
            shutil.copy2(f, dst)
        elif f.is_dir():
            shutil.copytree(f, dst)

    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"variant 里没有 config.json: {out_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    old = cfg.get("resize_imgs_with_padding")
    cfg["resize_imgs_with_padding"] = [resize, resize]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir, old if old else -1


def eval_cmd(variant: Path, eval_res: int, suite: str, n_episodes: int,
             seed: int, tag: str) -> list[str]:
    """构造评测命令。

    ⚠️ 必须显式传 --policy.resize_imgs_with_padding。
    lerobot-eval 在只给 --policy.type=smolvla 时，会用 SmolVLAConfig 的默认值
    (512, 512) 重新构造 policy，从而【覆盖】config.json 里改过的值——
    这正是 2026-11-04 那轮作废 5 小时评测的根因。只改 config.json 不够。
    """
    return [
        "lerobot-eval",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={variant}",
        f"--policy.resize_imgs_with_padding=[{eval_res},{eval_res}]",
        "--env.type=libero",
        f"--env.task={suite}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--env.init_states=true",
        f"--seed={seed}",
        f"--output_dir={RESULTS / 'eval_logs' / tag}",
    ]


# c1 / c3 与主实验完全等价（同模型、同训练分辨率、同评测分辨率、同协议），
# 可以直接复用 summary.jsonl 的结果，省掉两次约 2 小时的评测。
REUSE_CONDITIONS = {"c1_512on512": 512, "c3_256on256": 256}


def find_reusable(model_name: str, want_n: int) -> dict | None:
    """从 summary.jsonl 取该模型最近一次有效评测；episode 数必须一致才可复用。"""
    hits = [
        r for r in read_summary()
        if r.get("stage") == "eval"
        and r.get("name") == model_name
        and r.get("suites")
        and r.get("avg") is not None
    ]
    if not hits:
        return None
    rec = hits[-1]
    if rec.get("n_episodes") != want_n:
        return None
    return rec


def reuse_suite(log: Path, cmd: list[str]) -> float | None:
    """日志首行命令与本次完全一致、且已出成功率时，直接复用该套件结果。

    用于「跑到一半进程被杀」的续跑。命令里包含分辨率 flag、episode 数、seed
    和输出目录，所以只有同一次实验的日志才会命中。
    """
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="replace")
    if not txt.startswith("$ " + " ".join(cmd)):
        return None
    # 只有当次命令真的跑完（run_streaming 会写 "[exit 0]"）才复用；
    # 被中途杀死的日志里只有进度条，解析会得到假的 0%
    if "[exit 0]" not in txt:
        return None
    return parse_success(txt)


def main() -> int:
    ap = argparse.ArgumentParser(description="分辨率机制验证（纯推理）")
    ap.add_argument("--only", action="append", help="只跑指定条件，前缀匹配")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-main", action="store_true",
                    help="c1/c3 直接复用 summary.jsonl 的结果，省约 4 小时（要求 episode 数一致）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要执行的命令，不真跑（先用它核对参数）")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    ensure_dirs()

    if args.list:
        done = {r["condition"] for r in read_jsonl(MECH)}
        print(f"{'条件':<16}{'训练':<6}{'评测':<6}{'状态':<10}说明")
        print("-" * 78)
        for name, desc, tr, ev in CONDITIONS:
            st = "已完成" if name in done else "待跑"
            print(f"{name:<16}{tr:<6}{ev:<6}{st:<10}{desc}")
        return 0

    if args.all:
        todo = CONDITIONS
    elif args.only:
        names, missing = match_names([c[0] for c in CONDITIONS], args.only)
        if missing:
            raise SystemExit(f"未知条件: {', '.join(missing)}（用 --list 查看）")
        todo = [c for c in CONDITIONS if c[0] in names]
    else:
        raise SystemExit("请指定 --all 或 --only <条件名>")

    if not which("lerobot-eval"):
        raise SystemExit("找不到 lerobot-eval")

    cfg_all = load_config()
    by_name = {e["name"]: e for e in cfg_all["experiments"]}

    # 先检查两个基础模型是否都在
    missing_ckpt = []
    for res, model_name in MODEL_CFG.items():
        if model_name not in by_name:
            missing_ckpt.append(f"{model_name}(不在 experiments.yaml)")
            continue
        if not find_checkpoint(model_name):
            missing_ckpt.append(f"{model_name}(无 checkpoint)")
    if missing_ckpt:
        print("✗ 缺少基础模型：")
        for m in missing_ckpt:
            print(f"    {m}")
        print("\n先跑训练：")
        print("    python scripts/run_sweep.py --only base_r512_bf16,r256_bf16")
        return 1

    commit = lerobot_commit()
    done = {r["condition"] for r in read_jsonl(MECH)}
    ok = 0
    want_n = args.n_episodes * 10 * 4

    if args.reuse_main:
        for cond, model_res in REUSE_CONDITIONS.items():
            if cond in done:
                continue
            model_name = MODEL_CFG[model_res]
            rec = find_reusable(model_name, want_n)
            if rec is None:
                print(f"  ! {cond} 无法复用（summary.jsonl 里没有 {model_name} "
                      f"的 {want_n}-episode 有效评测），将重新跑")
                continue
            append_jsonl(MECH, {
                "condition": cond,
                "desc": f"复用主实验：训{model_res}评{model_res}",
                "train_res": model_res, "eval_res": model_res,
                "base_model": model_name,
                "suites": rec["suites"], "avg": rec["avg"],
                "n_episodes": rec["n_episodes"],
                "seed": rec.get("seed", args.seed),
                "source": "reused:summary.jsonl",
                "lerobot_commit": commit,
            })
            done.add(cond)
            print(f"  ✓ {cond} 复用 {model_name}：{rec['avg'] * 100:.1f}%"
                  f"（{rec['n_episodes']} episodes）")

    for i, (name, desc, train_res, eval_res) in enumerate(todo, 1):
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(todo)}] {name} — {desc}")
        print("=" * 78)

        if name in done and not args.force:
            print("  已完成，跳过（--force 重跑）")
            continue

        base_model = MODEL_CFG[train_res]
        src = find_checkpoint(base_model)
        if not src:
            print(f"  ✗ 找不到 {base_model} 的 checkpoint")
            continue

        out_dir = VARIANTS / name
        if args.dry_run:
            print(f"  源模型     : {base_model}  ({src})")
            print(f"  variant    : {out_dir}（config.json → {eval_res}）")
            print("  待执行命令（每个 suite 一条）：")
            for suite, short in zip(SUITES, SUITE_SHORT):
                print("    $ " + " ".join(eval_cmd(
                    out_dir, eval_res, suite, args.n_episodes, args.seed,
                    f"mech_{name}_{short}")))
            continue

        try:
            variant, old_res = make_variant(src, out_dir, eval_res)
        except Exception as e:
            print(f"  ✗ 生成 variant 失败: {e}")
            continue

        print(f"  源模型     : {base_model}")
        print(f"  评测分辨率 : {eval_res}（config.json 原值 {old_res} → {eval_res}）")

        suites: dict[str, float] = {}
        failed = False

        for suite, short in zip(SUITES, SUITE_SHORT):
            log = LOGS / f"mech.{name}.{short}.log"
            cmd = eval_cmd(variant, eval_res, suite, args.n_episodes, args.seed,
                           f"mech_{name}_{short}")

            # 断点续跑：同一条命令的日志已经出了结果，就不重跑
            reused = None if args.force else reuse_suite(log, cmd)
            if reused is not None:
                suites[short] = round(reused, 4)
                print(f"\n  [{short}] 复用已有结果 → {reused:.4f}")
                continue

            print(f"\n  [{short}]")
            run_streaming(cmd, log)
            val = parse_success(log.read_text(encoding="utf-8", errors="replace"))
            if val is None:
                print(f"  ✗ 解析失败，日志: {log}")
                failed = True
                break
            suites[short] = round(val, 4)
            print(f"      → {val:.4f}")

        if failed or len(suites) != 4:
            continue

        avg = round(sum(suites.values()) / 4, 4)
        rec = {
            "condition": name, "desc": desc,
            "train_res": train_res, "eval_res": eval_res,
            "base_model": base_model,
            "suites": suites, "avg": avg,
            "n_episodes": want_n,
            "seed": args.seed, "source": "measured", "lerobot_commit": commit,
        }
        append_jsonl(MECH, rec)
        ok += 1
        print(f"\n  ✓ {name}  平均 {avg * 100:.1f}%")

    print(f"\n{'=' * 78}\n 完成 {ok} 个条件\n{'=' * 78}")
    if ok:
        print("\n下一步: python scripts/mechanism_test.py --list && python scripts/mechanism_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
