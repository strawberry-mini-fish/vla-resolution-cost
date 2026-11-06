"""共享工具：路径、实验配置、结果记录、显存轮询、命令构建。

所有脚本都从这里 import，保证路径和格式一致。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "experiments.yaml"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
LOGS = RESULTS / "logs"
FIGURES = RESULTS / "figures"
SUMMARY = RESULTS / "summary.jsonl"


# ----------------------------------------------------------------------
# 实验清单
# ----------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("缺少 PyYAML：pip install pyyaml")

    if not CONFIG.exists():
        raise SystemExit(f"找不到实验清单：{CONFIG}")

    with CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    common = cfg.get("common") or {}
    exps = cfg.get("experiments") or []
    if not exps:
        raise SystemExit("configs/experiments.yaml 里没有 experiments")

    seen = set()
    for e in exps:
        name = e.get("name")
        if not name:
            raise SystemExit(f"实验缺少 name：{e}")
        if name in seen:
            raise SystemExit(f"实验名重复：{name}")
        seen.add(name)
        e.setdefault("overrides", {})

    return {"common": common, "experiments": exps}


def match_names(all_names: list[str], wanted: list[str]) -> tuple[list[str], list[str]]:
    """把用户输入的实验名映射到真实名字，支持前缀/子串匹配。

    r256      → r256_bf16, r256_fp32, ...（多个都匹配，全部返回）
    r256_bf16 → r256_bf16（精确）
    返回 (matched, unmatched_inputs)
    """
    matched: list[str] = []
    missing: list[str] = []
    for w in wanted:
        w = w.strip()
        if not w:
            continue
        if w in all_names:
            matched.append(w)
            continue
        prefix = [n for n in all_names if n.startswith(w)]
        if prefix:
            matched.extend(prefix)
            continue
        sub = [n for n in all_names if w in n]
        if sub:
            matched.extend(sub)
            continue
        missing.append(w)
    # 去重保序
    seen: set[str] = set()
    out = []
    for n in matched:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, missing


def merged_params(common: dict, exp: dict) -> dict[str, Any]:
    """合并 common 与 override，override 优先。"""
    out = dict(common)
    out.update(exp.get("overrides") or {})
    return out


def fmt_value(v: Any) -> str:
    """把值格式化成 draccus 能接受的命令行字面量。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def build_train_cmd(exp: dict, common: dict) -> list[str]:
    params = merged_params(common, exp)
    cmd = ["lerobot-train"]
    for k, v in params.items():
        cmd.append(f"--{k}={fmt_value(v)}")
    cmd.append(f"--output_dir={RUNS / exp['name']}")
    return cmd


# ----------------------------------------------------------------------
# 结果记录
# ----------------------------------------------------------------------


def ensure_dirs() -> None:
    for d in (RESULTS, RUNS, LOGS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def append_summary(record: dict) -> None:
    ensure_dirs()
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_summary() -> list[dict]:
    if not SUMMARY.exists():
        return []
    out = []
    # utf-8-sig：兼容 Windows 编辑器/Set-Content 写入的 BOM，否则首行会被静默丢弃
    for line in SUMMARY.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def latest_by_name(records: list[dict]) -> dict[str, dict]:
    """同名实验取最后一条（支持重跑）。"""
    out: dict[str, dict] = {}
    for r in records:
        out[r.get("name", "")] = r
    return out


def is_done(name: str) -> bool:
    return (RUNS / name / ".done").exists()


def mark_done(name: str) -> None:
    p = RUNS / name
    p.mkdir(parents=True, exist_ok=True)
    (p / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def git_commit(path: Path) -> str | None:
    """记录依赖库版本，便于复现。"""
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def lerobot_commit() -> str | None:
    try:
        import lerobot  # noqa: F401
        return git_commit(Path(lerobot.__file__).resolve().parent.parent)
    except Exception:
        return None


# ----------------------------------------------------------------------
# 显存轮询
# ----------------------------------------------------------------------


class GpuPoller:
    """后台轮询 nvidia-smi，记录显存峰值。

    用 nvidia-smi 而不是 torch.cuda.max_memory_allocated，因为后者只统计
    PyTorch 分配器视角，会漏掉 CUDA context、cuDNN workspace 和渲染缓冲。
    用户真正感受到的是 nvidia-smi 的数字。
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.peak = 0
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> int | None:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            first = r.stdout.strip().splitlines()[0]
            return int(first.strip())
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._query()
            if v is not None:
                self.samples.append(v)
                self.peak = max(self.peak, v)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.peak

    def p95(self) -> int:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        return s[min(len(s) - 1, int(len(s) * 0.95))]


# ----------------------------------------------------------------------
# 杂项
# ----------------------------------------------------------------------


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_streaming(cmd: list[str], log_path: Path, timeout: int | None = None) -> int:
    """执行命令，输出同时写日志和终端。返回退出码。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
            )
        except FileNotFoundError:
            print(f"  ✗ 找不到命令：{cmd[0]}")
            log.write(f"[命令不存在: {cmd[0]}]\n")
            return 127

        assert proc.stdout is not None
        rc = None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="")
            rc = proc.wait(timeout=timeout) if timeout else proc.wait()
        except subprocess.TimeoutExpired:
            print(f"\n  ✗ 超时（{timeout}s），正在终止进程")
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            log.write(f"\n[超时 {timeout}s，已终止]\n")
            rc = 124
        except KeyboardInterrupt:
            print("\n  中断，正在终止子进程…")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.write("\n[用户中断]\n")
            raise

        log.write(f"\n\n[exit {rc}] elapsed {time.time() - t0:.1f}s\n")

    return rc if rc is not None else 1


def looks_like_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    txt = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return any(k in txt for k in (
        "out of memory", "cuda error: out of memory",
        "cublas_status_alloc_failed", "cuda_error_out_of_memory",
    ))
