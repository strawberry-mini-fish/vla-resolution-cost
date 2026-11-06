#!/usr/bin/env python
"""批量训练：按 configs/experiments.yaml 逐个跑 lerobot-train，记录显存峰值与耗时。

用法:
    python scripts/run_sweep.py --list
    python scripts/run_sweep.py --print-cmd --only r256
    python scripts/run_sweep.py --only r256 --force
    python scripts/run_sweep.py --all
    python scripts/run_sweep.py --only base_r512_bf16 --steps 200   # 冒烟测试

几个实现上的选择：
  跑完的实验会在 results/runs/<name>/ 下留一个 .done，重跑自动跳过。
  显存峰值用 nvidia-smi 轮询，torch.cuda.max_memory_allocated 会少算
  （漏掉 CUDA context 和 cuDNN workspace）。
  想确认生成的命令对不对，用 --print-cmd 打出来看一眼。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LOGS, RUNS, GpuPoller, append_summary, build_train_cmd, ensure_dirs,
    is_done, lerobot_commit, load_config, looks_like_oom, mark_done,
    match_names, merged_params, run_streaming, which,
)


def select(cfg: dict, args) -> list[dict]:
    exps = cfg["experiments"]
    if args.list:
        return []
    if args.all:
        return exps
    if args.only:
        wanted = [n for chunk in args.only for n in chunk.split(",")]
        names, missing = match_names([e["name"] for e in exps], wanted)
        if missing:
            raise SystemExit(
                f"找不到匹配的实验: {', '.join(missing)}\n用 --list 查看全部实验名"
            )
        return [e for e in exps if e["name"] in names]
    raise SystemExit("请指定 --only <name> 或 --all（用 --list 查看全部）")


def main() -> int:
    ap = argparse.ArgumentParser(description="批量训练 VLA 配置")
    ap.add_argument("--only", action="append", help="只跑指定实验，可重复或逗号分隔")
    ap.add_argument("--all", action="store_true", help="跑全部实验")
    ap.add_argument("--list", action="store_true", help="列出实验，不执行")
    ap.add_argument("--print-cmd", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--force", action="store_true", help="忽略已完成标记，重跑")
    ap.add_argument("--steps", type=int, default=None, help="覆盖所有实验的 steps（冒烟测试用）")
    ap.add_argument("--timeout", type=int, default=None, help="单个实验超时秒数")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()

    if args.list:
        print(f"{'名称':<26}{'batch':<8}{'说明'}")
        print("-" * 78)
        for e in cfg["experiments"]:
            p = merged_params(cfg["common"], e)
            print(f"{e['name']:<26}{str(p.get('batch_size', '?')):<8}{e.get('desc', '')}")
        print(f"\n共 {len(cfg['experiments'])} 个实验")
        return 0

    exps = select(cfg, args)
    if not exps:
        return 0

    if not args.print_cmd and not which("lerobot-train"):
        raise SystemExit("找不到 lerobot-train，先在环境里 pip install -e \".[libero,peft]\"")

    print("=" * 78)
    print(f" 准备执行 {len(exps)} 个实验")
    print("=" * 78)

    commit = lerobot_commit()
    if commit:
        print(f" lerobot commit: {commit}")
    print()

    done_count = 0
    fail_count = 0
    skip_count = 0

    for i, exp in enumerate(exps, 1):
        name = exp["name"]
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(exps)}] {name} — {exp.get('desc', '')}")
        print("=" * 78)

        cmd = build_train_cmd(exp, cfg["common"])
        if args.steps is not None:
            cmd = [c if not c.startswith("--steps=") else f"--steps={args.steps}" for c in cmd]
        if args.print_cmd:
            print("$ " + " ".join(cmd))
            continue

        if is_done(name) and not args.force:
            print("  已完成，跳过（--force 可重跑）")
            skip_count += 1
            continue

        print("$ " + " ".join(cmd) + "\n")

        poller = GpuPoller(interval=1.0)
        poller.start()
        t0 = time.time()
        rc = run_streaming(cmd, LOGS / f"{name}.train.log", timeout=args.timeout)
        elapsed = time.time() - t0
        peak = poller.stop()

        params = merged_params(cfg["common"], exp)
        if args.steps is not None:
            params["steps"] = args.steps

        if rc == 0:
            done_count += 1
            mark_done(name)
            append_summary({
                "stage": "train", "name": name, "rc": rc,
                "peak_mem_mib": peak,
                "peak_mem_p95_mib": poller.p95(),
                "train_sec": round(elapsed, 1),
                "steps": params.get("steps"),
                "batch_size": params.get("batch_size"),
                "params": {k: str(v) for k, v in params.items()},
                "lerobot_commit": commit,
            })
            print(f"\n  ✓ 完成  {elapsed / 60:.1f} 分钟  显存峰值 {peak} MiB")
        else:
            fail_count += 1
            kind = "OOM" if looks_like_oom(LOGS / f"{name}.train.log") else "FAIL"
            append_summary({
                "stage": "train", "name": name, "rc": rc, "error": kind,
                "peak_mem_mib": peak, "train_sec": round(elapsed, 1),
                "lerobot_commit": commit,
            })
            print(f"\n  ✗ {kind}  (rc={rc})  日志: results/logs/{name}.train.log")
            if kind == "OOM":
                print("      → 降低 batch_size，或用 --accelerator.gradient_accumulation.steps")
                print("      → 也可以先降分辨率：policy.resize_imgs_with_padding=[256,256]")

    if args.print_cmd:
        return 0

    print(f"\n{'=' * 78}")
    print(f" 完成 {done_count}  跳过 {skip_count}  失败 {fail_count}")
    print("=" * 78)
    if done_count:
        print("\n下一步: python scripts/eval_all.py --all")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
