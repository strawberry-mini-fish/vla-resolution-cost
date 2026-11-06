# 低资源下的 VLA 微调：输入分辨率的成本与收益

在单卡 8GB 上微调 SmolVLA(450M)，把输入分辨率从 512² 降到 256²：训练时间降 57%，显存峰值降 34%，平均成功率没有统计显著变化（52.5% → 56.0%，p = 0.48，n = 200/配置）。但分辨率不能在推理时改——训练和评测用的分辨率不一致，成功率会掉到 0–1%。

> *Abstract*: On a single 8 GB GPU, fine-tuning SmolVLA (450M) on LIBERO, dropping the input
> resolution from 512² to 256² cuts training time by 57% and peak memory by 34%, with no
> statistically significant change in success rate (52.5% → 56.0%, p = 0.48, n = 200/config).
> However, the policy is brittle to any train/eval resolution mismatch: accuracy collapses to
> 0–1% when the evaluation resolution differs from what was trained.

## 1. 设定

| 项 | 值 |
|---|---|
| 模型 | SmolVLA 450M（VLM 主干 16 层 + 动作专家，flow matching，chunk 50） |
| 基准 | LIBERO，4 套件 × 10 任务，每配置 200 episodes |
| 训练 | batch 16，bf16，冻结视觉编码器，只训动作专家，seed 1000 |
| 硬件 | RTX 5060 Laptop 8 GB / WSL2 |
| 变量 | 输入分辨率 512 / 384 / 256；512 和 256 另外各跑一遍 40k 步 |

选这个题是因为 LeRobot 默认把 LIBERO 原始的 256×256 观测放大到 512×512 再喂给 VLM。视觉编码器的 patch 是 16，放大之后视觉 token 从 256 个变成 1024 个，而编码器是冻结的——这部分算力按理说换不来任何适配收益。但这只是假设，得测。

## 2. 主结果

| 配置 | spatial | object | goal | long | 平均 | 训练 | 显存峰值 |
|---|---|---|---|---|---|---|---|
| 512 @ 20k | 40.0 | 54.0 | 50.0 | 34.0 | 44.5% | 3.31 h | 4948 MiB |
| 384 @ 20k | 50.0 | 44.0 | 72.0 | 32.0 | 49.5% | 2.23 h | 3948 MiB |
| 256 @ 20k | 46.0 | 56.0 | 64.0 | 32.0 | 49.5% | 1.49 h | 3332 MiB |
| 512 @ 40k | 50.0 | 60.0 | 68.0 | 32.0 | 52.5% | 6.93 h | 5223 MiB |
| 256 @ 40k | 56.0 | 62.0 | 74.0 | 32.0 | 56.0% | 2.97 h | 3436 MiB |

统计（n = 200/配置，两比例 z 检验）：

- 512@20k vs 256@20k：+5.0 点，p = 0.32
- 512@40k vs 256@40k：+3.5 点，p = 0.48
- 512@20k vs 512@40k：+8.0 点，p = 0.11
- 256@20k vs 256@40k：+6.5 点，p = 0.19

n = 200 的能力只够发现 10 点以上的差异。所以正确写法是"未检测到显著差异"，不是"两者相同"。另外注意连训练翻倍（+8.0 点）都没到显著——这个样本量的灵敏度就到这里了。

评测用固定初始状态（`init_states=true`），各配置跑的是同一批 200 个初始条件，属于配对设计，所以补了一个 McNemar 检验，只看两组结果不一致的集：

| 对比 | 只有前者成功 | 只有后者成功 | 非配对 z | 配对 McNemar |
|---|---|---|---|---|
| 512@40k vs 256@40k | 40 | 47 | p = 0.48 | p = 0.52 |
| 512@20k vs 512@40k | 30 | 46 | p = 0.11 | p = 0.085 |

两种检验结论一致：主对比不显著；训练翻倍在配对下更接近显著（0.11 → 0.085），但也没过线。

## 3. 机制：2×2 训测分辨率矩阵

不重训。复制 checkpoint，改写 `config.json` 里的 `resize_imgs_with_padding`，同时显式传 `--policy.resize_imgs_with_padding`（只改 config 不够，会被 `--policy.type` 的默认值 512 覆盖），然后分别评测。

| | 评测 512 | 评测 256 |
|---|---|---|
| 训 512 | 44.5% | 1.0% |
| 训 256 | 0.0% | 49.5% |

对角线正常，非对角线两个方向都塌到 0–1%（各 200 episodes，p < 0.001）。

这个结果说明掉点不是"256 的信息量不够"——256 训出来的模型能到 49.5%。问题出在训练和评测对不上。模型对分辨率一致性的依赖是崩溃级的，比分辨率本身的选择重要得多。

## 4. 怎么用

| 场景 | 做法 | 别这么干 |
|---|---|---|
| 小显存 / 慢卡 | 从训练开始就选 256 | 训 512 再降级部署（会崩到 0–1%） |
| 已经有 512 模型 | 保持 512 推理 | 为了省推理时间改成 256 |
| 想提高绝对分数 | 加训练步数（方向为正，但 n=200 测不出显著） | 靠提高分辨率 |

另外 LIBERO-10 这个长程套件在 5 个配置、3 种分辨率、2 种训练长度下都稳定在 32–34%，同一时期其他套件在 40k 时涨了 6–14 点。看着像瓶颈在多步误差累积，跟视觉信息量和训练量关系都不大。

## 5. 局限

1. 单 seed（1000），训练随机性带来的方差没测。
2. n = 200 只够发现 10 点以上的差异。
3. 绝对性能 52–56% 远没饱和（20k 相当于 1.17 个 epoch，40k 是 2.34），结论限定在低资源区间。
4. 纯仿真，没验证真机。
5. 机制那部分只适用于冻结视觉编码器的情况。解冻之后模型可能能适应新分辨率，没测。
6. 没做的轴：fp32、LoRA、梯度累积、第二 seed、24GB 对照、解冻视觉编码器。

## 6. 复现

```bash
cd vla-resolution-cost

# 训练（有断点续跑，已经完成的会跳过）
python scripts/run_sweep.py --only r512_40k,r256_40k

# 评测。--n-episodes 5 = 每任务 5 集 = 每配置 200 集
python scripts/eval_all.py --only r512_40k,r256_40k --n-episodes 5

# 机制矩阵。--reuse-main 让 C1/C3 直接复用主实验，只跑 C2/C4
python scripts/mechanism_test.py --all --reuse-main --n-episodes 5 --dry-run   # 先看命令
python scripts/mechanism_test.py --all --reuse-main --n-episodes 5
python scripts/mechanism_report.py

# 统计与出图
python scripts/stats_check.py
python scripts/make_figures.py
```

写自己的评测命令时要注意：`lerobot-eval` 必须显式给 `--policy.resize_imgs_with_padding=[H,W]`，否则 `--policy.type=smolvla` 会用默认的 512 覆盖 checkpoint 里存的值。这个坑我踩过一次，一整轮 5 小时的评测全部作废。

---

数字来自 `results/summary.jsonl` 和 `results/mechanism.jsonl`，没有手抄。
