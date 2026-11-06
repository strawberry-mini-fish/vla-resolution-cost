#!/usr/bin/env python
"""批量评测：对每个训练完成的实验跑 lerobot-eval，汇总 4 个 suite 的成功率。

用法:
    python scripts/eval_all.py --list
    python scripts/eval_all.py --only r256
    python scripts/eval_all.py --all
    python scripts/eval_all.py --only r256 --n-episodes 10

    # 如果自动解析失败，可手工录入（顺序固定为 spatial,object,goal,libero_10）
    python scripts/eval_all.py --only r256 --manual 0.82,0.88,0.79,0.61

说明:
    lerobot-eval 的输出格式可能随版本变化，本脚本用多种模式尝试解析。
    解析失败时会保留原始日志并提示你手工录入，不会静默写入错误数据。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    FIGURES, LOGS, RUNS, append_summary, ensure_dirs, lerobot_commit,
    load_config, match_names, read_summary, run_streaming, which,
)

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_SHORT = ["spatial", "object", "goal", "libero_10"]


def read_ckpt_resize(ckpt: Path) -> tuple[int, int] | None:
    """读取 checkpoint 训练时用的输入分辨率。

    ⚠️ 必须显式传给 lerobot-eval，否则 `--policy.type=smolvla` 会用
    SmolVLAConfig 的默认值 512 覆盖掉 checkpoint 里存的值，导致
    训练/评测分辨率失配，成功率会掉到接近 0（而 config 打印出来还会
    显示 512，很容易误判成"模型坏了"）。
    """
    try:
        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    v = cfg.get("resize_imgs_with_padding")
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return int(v[0]), int(v[1])
    return None


def find_checkpoint(name: str) -> Path | None:
    """找到该实验最新的 checkpoint 目录。"""
    base = RUNS / name / "checkpoints"
    if not base.exists():
        return None

    for cand in (base / "last" / "pretrained_model", base / "last"):
        if cand.exists() and (cand / "config.json").exists():
            return cand

    steps = []
    for d in base.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue
        pm = d / "pretrained_model"
        if (pm / "config.json").exists():
            steps.append((int(d.name), pm))
    if steps:
        return max(steps)[1]
    return None


NUM = r"([01]?\.\d+|[01])"


_NUM = r"[0-9]+(?:\.[0-9]+)?"


def parse_success(text: str) -> float | None:
    """从评测输出里抠出成功率。

    LeRobot 0.6.x 的输出长这样：

        {'libero_object': {'avg_sum_reward': 0.55, 'pc_success': 55.00000000000001,
                           'n_episodes': 20, 'n_success': 11, ...}}

    几个坑：键名是单引号不是双引号；pc_success 是百分数，得除以 100；
    同一段输出里 per-suite 和 aggregated 各出现一次，取后面那个。
    解析不出来就返回 None，宁缺一条也不要写错数字。
    """
    patterns = [
        r"pc_success['\"]?\s*[:=]\s*(" + _NUM + r")",
        # 必须排除进度条里的 running_success_rate=0.0% —— 那是当前批次的瞬时值，
        # 不是最终结果。日志被中途杀死时只剩它，会把「没跑完」误读成「0%」。
        r"(?<!running_)success_rate['\"]?\s*[:=]\s*(" + _NUM + r")",
        r"avg_success['\"]?\s*[:=]\s*(" + _NUM + r")",
    ]
    for p in patterns:
        vals = re.findall(p, text, re.IGNORECASE)
        if vals:
            v = float(vals[-1])
            return v / 100.0 if v > 1.0 else v
    return None


def find_result_json(out_dir: Path) -> float | None:
    if not out_dir.exists():
        return None
    for f in sorted(out_dir.rglob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("pc_success", "success_rate", "avg_success"):
            if isinstance(data, dict) and key in data:
                try:
                    v = float(data[key])
                    return v / 100 if v > 1.0 else v
                except (TypeError, ValueError):
                    pass
    return None


def select(cfg: dict, args) -> list[dict]:
    exps = cfg["experiments"]
    if args.all:
        return exps
    if args.only:
        wanted = [n for c in args.only for n in c.split(",")]
        names, missing = match_names([e["name"] for e in exps], wanted)
        if missing:
            raise SystemExit(
                f"找不到匹配的实验: {', '.join(missing)}\n用 --list 查看全部实验名"
            )
        return [e for e in exps if e["name"] in names]
    raise SystemExit("请指定 --only <name> 或 --all")


def main() -> int:
    ap = argparse.ArgumentParser(description="批量评测")
    ap.add_argument("--only", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--manual", type=str, default=None,
                    help="手工录入成功率，逗号分隔 4 个 suite")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cfg = load_config()
    ensure_dirs()

    if args.list:
        have = {r["name"] for r in read_summary() if r.get("stage") == "eval"}
        for e in cfg["experiments"]:
            ck = find_checkpoint(e["name"])
            status = "已评测" if e["name"] in have else ("有 ckpt" if ck else "无 ckpt")
            print(f"{e['name']:<26}{status}")
        return 0

    exps = select(cfg, args)
    if not exps:
        return 0

    if not which("lerobot-eval"):
        raise SystemExit("找不到 lerobot-eval")

    commit = lerobot_commit()
    ok_count = 0

    for i, exp in enumerate(exps, 1):
        name = exp["name"]
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(exps)}] 评测 {name}")
        print("=" * 78)

        if args.manual:
            vals = [float(x) for x in args.manual.split(",")]
            if len(vals) != 4:
                raise SystemExit("--manual 需要 4 个值")
            record = dict(zip(SUITE_SHORT, vals))
            append_summary({
                "stage": "eval", "name": name, "source": "manual",
                "suites": record, "avg": round(sum(vals) / 4, 4),
                "ci95": None, "n_episodes": None,
                "lerobot_commit": commit,
            })
            print(f"  已手工录入: {record}")
            ok_count += 1
            continue

        ckpt = find_checkpoint(name)
        if not ckpt:
            print("  ✗ 找不到 checkpoint，先跑 run_sweep.py")
            continue

        print(f"  checkpoint: {ckpt}")
        resize = read_ckpt_resize(ckpt)
        if resize:
            print(f"  训练分辨率: {resize[0]}×{resize[1]}（评测时显式指定）")
        else:
            print("  ⚠️ 读不到 checkpoint 分辨率，将用默认值（可能失配）")
        suites: dict[str, float] = {}
        failed = False

        for suite, short in zip(SUITES, SUITE_SHORT):
            out_dir = FIGURES.parent / "eval_logs" / f"{name}_{short}"
            log = LOGS / f"{name}.eval.{short}.log"

            # 断点复用：已有完整结果【且分辨率一致】才跳过
            if not args.force and log.exists():
                txt = log.read_text(encoding="utf-8", errors="replace")
                if "End of eval" in txt:
                    cached = parse_success(txt)
                    m = re.search(r"resize_imgs_with_padding': \((\d+), (\d+)\)", txt)
                    log_res = (int(m.group(1)), int(m.group(2))) if m else None
                    if cached is not None and (resize is None or log_res == resize):
                        suites[short] = round(cached, 4)
                        print(f"  [{short}] 复用已有结果 {cached:.4f}（--force 可重跑）")
                        continue
                    if log_res and resize and log_res != resize:
                        print(f"  [{short}] 旧结果分辨率 {log_res} ≠ checkpoint 的 {resize}"
                              f" → 作废重跑")

            cmd = [
                "lerobot-eval",
                "--policy.type=smolvla",
                f"--policy.pretrained_path={ckpt}",
                "--env.type=libero",
                f"--env.task={suite}",
                "--eval.batch_size=1",
                f"--eval.n_episodes={args.n_episodes}",
                "--env.init_states=true",
                f"--seed={args.seed}",
                f"--output_dir={out_dir}",
            ]
            # 必须显式传训练时的分辨率。只靠 checkpoint 里的 config.json 不够：
            # 一旦给了 --policy.type=smolvla，就会用 SmolVLAConfig 的默认值 (512,512)
            # 覆盖 config.json，变成训练 256、评测 512，成功率直接归零。
            if resize:
                cmd.insert(3, f"--policy.resize_imgs_with_padding=[{resize[0]},{resize[1]}]")
            print(f"\n  [{short}] $ {' '.join(cmd)}")
            rc = run_streaming(cmd, log)

            val = parse_success(log.read_text(encoding="utf-8", errors="replace"))
            if val is None:
                val = find_result_json(out_dir)

            if val is None:
                print(f"  ✗ 无法解析 {short} 的成功率（rc={rc}）")
                print(f"     日志: {log}")
                print(f"     可手工录入: python scripts/eval_all.py --only {name} "
                      f"--manual 0.82,0.88,0.79,0.61")
                failed = True
                break
            suites[short] = round(val, 4)
            print(f"  → {short} = {val:.4f}")

        if failed or len(suites) != 4:
            continue

        avg = round(sum(suites.values()) / 4, 4)
        append_summary({
            "stage": "eval", "name": name, "source": "lerobot-eval",
            "suites": suites, "avg": avg,
            "n_episodes": args.n_episodes * 10 * 4,
            "seed": args.seed, "lerobot_commit": commit,
        })
        ok_count += 1
        print(f"\n  ✓ {name}  平均成功率 {avg:.4f}")

    print(f"\n{'=' * 78}\n 评测完成 {ok_count} 个\n{'=' * 78}")
    if ok_count:
        print("\n下一步: python scripts/make_pareto.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
