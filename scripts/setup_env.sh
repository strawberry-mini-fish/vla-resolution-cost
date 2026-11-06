#!/usr/bin/env bash
# 环境自检：8 项检查，任何一项不过都会说明原因和修法。
# 用法: bash scripts/setup_env.sh

set -uo pipefail

PASS=0
FAIL=0
WARN=0

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; echo "      → $2"; FAIL=$((FAIL+1)); }
warn() { echo "  ! $1"; echo "      → $2"; WARN=$((WARN+1)); }

echo "=============================================="
echo " VLA 实验框架 · 环境自检"
echo "=============================================="
echo

# ---------- 1. WSL 内存 ----------
echo "[1/8] WSL 内存分配"
TOTAL_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$TOTAL_MB" -ge 9000 ]; then
  ok "可用内存 ${TOTAL_MB} MiB"
else
  warn "可用内存仅 ${TOTAL_MB} MiB（建议 ≥ 9.5GB）" \
       "在 C:\\Users\\<你>\\.wslconfig 里设置 memory=10GB，然后 wsl --shutdown"
fi
echo

# ---------- 2. Python ----------
echo "[2/8] Python 版本"
if command -v python >/dev/null 2>&1; then
  PYV=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    ok "Python $PYV"
  else
    bad "Python $PYV 过低" "conda create -n vla python=3.11 -y && conda activate vla"
  fi
else
  bad "找不到 python" "conda create -n vla python=3.11 -y && conda activate vla"
fi
echo

# ---------- 3. PyTorch + CUDA ----------
echo "[3/8] PyTorch 与 GPU"
python - <<'PY' 2>/dev/null
import sys
try:
    import torch
except ImportError:
    print("  ✗ 没装 torch")
    print("      → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

print(f"  · torch {torch.__version__} / cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    print("  ✗ CUDA 不可用")
    print("      → 确认 WSL 里能跑 nvidia-smi；torch 必须是 cu128 或更新")
    sys.exit(1)

cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  · {name}  {vram:.1f} GiB  capability={cap}")

if cap[0] < 12:
    print(f"  ✓ GPU 可用（capability {cap}）")
else:
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        _ = a @ a
        print(f"  ✓ GPU 可用，Blackwell (sm_{cap[0]}{cap[1]}) 内核正常")
    except Exception as e:
        print(f"  ✗ Blackwell 内核不可用: {e}")
        print("      → torch 版本过旧，重装 cu128 版本")
        sys.exit(1)
PY
if [ $? -ne 0 ]; then FAIL=$((FAIL+1)); else PASS=$((PASS+1)); fi
echo

# ---------- 4. HuggingFace 镜像 ----------
echo "[4/8] HuggingFace 连通性"
EP="${HF_ENDPOINT:-https://huggingface.co}"
echo "  · 当前 HF_ENDPOINT = $EP"
if [ "$EP" != "https://huggingface.co" ] && [ "${HF_HUB_DISABLE_XET:-0}" != "1" ]; then
  warn "用了镜像但没关 Xet" \
       "export HF_HUB_DISABLE_XET=1 —— 否则下载会报 CAS Client Error (401 Unauthorized)"
fi
if curl -sS -m 12 -o /dev/null -w "%{http_code}" "$EP/api/models/lerobot/smolvla_base" 2>/dev/null | grep -q '^200$'; then
  ok "$EP 可达"
elif curl -sS -m 12 -o /dev/null -w "%{http_code}" "https://hf-mirror.com/api/models/lerobot/smolvla_base" 2>/dev/null | grep -q '^200$'; then
  bad "$EP 不可达，但镜像可用" \
      "export HF_ENDPOINT=https://hf-mirror.com  (建议写进 ~/.bashrc)"
else
  bad "镜像也不可达" "检查网络；或在 Windows 侧配置代理"
fi
echo

# ---------- 5. LeRobot ----------
echo "[5/8] LeRobot"
if python -c "import lerobot" 2>/dev/null; then
  LV=$(python -c "import lerobot; print(getattr(lerobot,'__version__','unknown'))")
  ok "lerobot $LV"
  for b in lerobot-train lerobot-eval; do
    command -v "$b" >/dev/null 2>&1 && ok "$b 可用" || bad "$b 不在 PATH" "pip install -e \".[libero,peft]\""
  done
else
  bad "没装 lerobot" "git clone https://github.com/huggingface/lerobot.git && cd lerobot && pip install -e \".[libero,peft]\""
fi
echo

# ---------- 6. MuJoCo 渲染 ----------
echo "[6/8] MuJoCo 渲染"
MUJOCO_GL="${MUJOCO_GL:-egl}"
echo "  · MUJOCO_GL=$MUJOCO_GL"
if MUJOCO_GL=$MUJOCO_GL python - <<'PY' 2>/dev/null
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
PY
then
  ok "渲染后端可用"
else
  bad "渲染失败（MUJOCO_GL=$MUJOCO_GL）" \
      "试 export MUJOCO_GL=osmesa（软件渲染）或 glfw；LIBERO 评测依赖渲染"
fi
echo

# ---------- 7. 数据集与模型 ----------
echo "[7/8] 数据与模型缓存"
python - <<'PY' 2>/dev/null || true
import os
from pathlib import Path
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
ds = list((hub / "lerobot").glob("libero*")) if (hub / "lerobot").exists() else []
md = list((hub / "hub").glob("models--lerobot--smolvla_base")) if (hub / "hub").exists() else []
print(f"  · 数据集: {'已缓存' if ds else '未下载（首次训练会自动下载 1.9GB）'}")
print(f"  · 模型:   {'已缓存' if md else '未下载（首次训练会自动下载约 1GB）'}")
PY
PASS=$((PASS+1))
echo

# ---------- 8. 磁盘 ----------
echo "[8/8] 磁盘空间"
AVAIL_G=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL_G=${AVAIL_G:-0}
if [ "$AVAIL_G" -ge 50 ]; then
  ok "家目录可用 ${AVAIL_G} GB"
else
  warn "家目录可用仅 ${AVAIL_G} GB" "每个实验 checkpoint 约 1–2 GB，建议 ≥ 50 GB"
fi
echo

# ---------- 汇总 ----------
echo "=============================================="
echo " 通过 $PASS  警告 $WARN  失败 $FAIL"
echo "=============================================="
if [ "$FAIL" -gt 0 ]; then
  echo
  echo "有 $FAIL 项未通过，先修完再往下走。"
  exit 1
fi
echo
echo "下一步：bash scripts/probe_batch.sh 4 6 8"
