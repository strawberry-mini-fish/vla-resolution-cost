# 安装步骤（Ubuntu 26.04 / WSL2）

> 你的环境实测结果：
> - 用户 `xsh`，家目录 `/home/xsh`
> - Ubuntu 26.04 LTS，默认 Python **3.14**（PyTorch 不支持，必须另建 3.11 环境）
> - 无 pip、无 conda
> - **GPU 已可用**：WSL 里 `nvidia-smi` 能识别 RTX 5060
> - 网络：huggingface.co 超时；pypi / github / conda-forge / download.pytorch.org 均可达

---

## 第 0 步：打开 WSL

Windows 里按 `Win` 键 → 输入 `Ubuntu` → 回车。

也可以从 PowerShell 输入 `wsl`。

打开后提示符应该类似：

```
xsh@LAPTOP-35D2KK9F:~$
```

**后面所有命令都在这个 Ubuntu 窗口里敲，不是 PowerShell。**

---

## 第 1 步：给 WSL 分配更多内存

你现在 WSL 只有 **7.3 GB** 内存，LIBERO 渲染 + 训练会不够。

### 1a. 在 Windows 侧改配置

按 `Win + R` → 输入 `notepad` → 回车。

在记事本里 `Ctrl + O`，地址栏输入：

```
C:\Users\<你的用户名>\.wslconfig
```

如果提示文件不存在，点"是"新建。

粘贴以下内容：

```ini
[wsl2]
memory=10GB
swap=8GB
processors=8
```

保存（`Ctrl + S`），关闭记事本。

### 1b. 重启 WSL

在 **PowerShell**（不是 Ubuntu 窗口）里执行：

```powershell
wsl --shutdown
```

等 10 秒，重新打开 Ubuntu。

### 1c. 验证

```bash
free -h
```

期望第一行 `Mem:` 的 total 约为 `10Gi`。

> 如果 Windows 变卡，把 `memory=10GB` 改成 `9GB` 再来一次。

---

## 第 2 步：设置 HuggingFace 镜像

你的网络直连 huggingface.co 会超时。执行：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
echo $HF_ENDPOINT
```

期望输出：

```
https://hf-mirror.com
```

---

## 第 3 步：安装 miniforge（conda）

因为系统 Python 是 3.14，PyTorch 还不支持，必须建一个 3.11 的环境。

```bash
cd ~
curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec bash
```

验证：

```bash
conda --version
```

期望类似 `conda 25.x.x`。

> 提示符前面会多出 `(base)`。

---

## 第 4 步：创建 Python 3.12 环境

**必须是 3.12 或更高。** LeRobot 的 `pyproject.toml` 里写死了 `requires-python = ">=3.12"`，3.11 装不上。

```bash
conda create -n vla python=3.12 -y
conda activate vla
```

验证：

```bash
python --version
```

期望 `Python 3.12.x`

提示符变成 `(vla) xsh@...`。

> **以后每次新开终端，都要先 `conda activate vla`。**

---

## 第 5 步：安装 LeRobot（含 PyTorch，一步到位）

**不要单独装 PyTorch。** LeRobot 约束了 `torch>=2.7,<2.12`，先装 2.15 会被强制降级，等于白下 4 GB。

让它自己解算：

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[libero,peft]"
```

会下载 5–8 GB（torch + torchvision + nvidia 运行时 + mujoco + robosuite + hf-libero），约 15–25 分钟。

### 验证（关键）

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability())"
```

期望：

```
2.11.x  （或 2.10.x / 2.9.x，只要在 2.7–2.12 之间）
13.0    （或 12.8）
True
(12, 0)
```

- `(12, 0)` 是 Blackwell 的能力值，**这是你要看到的**
- 如果是 `False`，或 `(12,0)` 但训练报 "no kernel image available"，说明 torch 版本不对

### 再验证 LeRobot 本身

```bash
python -c "import lerobot; print('lerobot ok')"
lerobot-train --help | head -20
lerobot-eval --help | head -5
```

能打印出帮助信息就算成功。

---

## 关于 PyTorch 源的说明（踩过的坑）

| 源 | 结果 |
|---|---|
| `download.pytorch.org/whl/cu128` | ❌ **HTTP 403**，文件拿不到 |
| 清华 pytorch-wheels 镜像 | ❌ 404，不存在 |
| 清华 PyPI | ✅ 5.9 MB/s |

**任何教程让你加 `--index-url https://download.pytorch.org/whl/cu128` 的，在这台机器上都会失败。**

用 PyPI 即可（已在第 4 步之前配好镜像）。

---

## 第 6 步：把项目代码拷进 WSL

**注意：不要放在 `/mnt/c` 下跑训练**，跨文件系统读写很慢。

```bash
cp -r /mnt/c/Users/<你的用户名>/path/to/vla-resolution-cost ~/vla-resolution-cost
cd ~/vla-resolution-cost
ls
```

期望看到 `README.md  configs  requirements.txt  results  scripts`。

```bash
chmod +x scripts/*.sh
```

---

## 第 7 步：环境自检

```bash
cd ~/vla-resolution-cost
bash scripts/setup_env.sh
```

它会检查 8 项，每项告诉你通过还是失败、失败了怎么修。

**这一步跑完，把完整输出贴出来。**

预期会有 1–3 项红，正常，我们逐个修。

---

## 第 8 步：显存探针（第 7 步全绿之后再做）

```bash
bash scripts/probe_batch.sh 4 6 8
```

它会用 30 步快速试探，约 5–10 分钟，输出：

```
batch    状态      显存峰值(MiB)
4        OK        6xxx
6        OK        7xxx
8        OOM       78xx
```

**这张表决定后面所有实验的 batch 上限。**

---

## 第 9 步：第一次真实训练（冒烟测试）

```bash
python scripts/run_sweep.py --print-cmd --only base
```

先看生成的命令长什么样，确认参数名没问题。

然后：

```bash
python scripts/run_sweep.py --only base --steps 200
```

只跑 200 步，验证整条链路（下载数据 → 训练 → 存 checkpoint）。

---

## 常见问题

| 现象 | 原因 | 修法 |
|---|---|---|
| `python --version` 是 3.11 或 3.14 | 环境不对 | LeRobot 要 3.12+，`conda create -n vla python=3.12 -y` |
| `torch.cuda.is_available()` 是 False | torch 装成了 CPU 版 | `pip install --force-reinstall torch` 重来 |
| 训练报 `no kernel image` | torch 版本过旧 | 确认 torch 在 2.7–2.12 之间 |
| 装 LeRobot 时 torch 被降级 | 正常，它在解算 `torch<2.12` | 别中断，等它装完 |
| 下载卡住 | 没设镜像 | `echo $HF_ENDPOINT`、`pip config list` 检查 |
| `command not found: lerobot-train` | 环境没激活或没装 | `conda activate vla` |
| MuJoCo 渲染报错 | EGL 不可用 | `export MUJOCO_GL=osmesa` |
| `Permission denied` 跑 .sh | 没执行权限 | `chmod +x scripts/*.sh` |
| `bad interpreter` | CRLF 换行 | `sed -i 's/\r$//' scripts/*.sh` |

---

## 速查：每次新开终端要做的两件事

```bash
conda activate vla
export HF_ENDPOINT=https://hf-mirror.com   # 已写进 .bashrc 的话不需要
```
