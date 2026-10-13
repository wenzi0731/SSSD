# SSSD-baseline8：天气/时间条件下的联合源荷场景生成

本实现选择 **SSSD-S4**，基于仓库原始提交
`4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed`。新增代码位于 `baseline8/`，
原始 `src/` 和 MIT 许可证保持不变。论文中建议命名为 **conditional SSSD-S4 (adapted)**。
这是适配 HEEW 条件场景生成的 baseline，不是对原论文数据集实验结果的复现声明。

原论文：Juan Lopez Alcaraz and Nils Strodthoff,
[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://openreview.net/forum?id=hHiIbk7ApW)，TMLR 2022。

## 保留与修改

保留原始 SSSDS4Imputer 的输入投影、扩散步嵌入、每个残差块中的两个 S4 层、
条件投影、门控、残差/skip 连接和零初始化输出层；不是用 Transformer 替代 S4。
测试检查去除 import 后的模型代码 AST 与原始文件一致。

| 项目 | baseline8 |
| --- | --- |
| 生成目标 | 一个模型联合生成 Electricity、Heat、Cooling、PV，形状 `[B,S,4,24]` |
| 条件 | 当天 10 个天气特征 + 8 个周期时间特征 + 已知年份 `(year-2014)/6` |
| 掩码 | 四个能源通道整天未知；19 个外生条件通道整天已知 |
| 扩散与损失 | 只对四个能源通道加噪、计算噪声 MSE、反向采样和评分 |
| 网络通道 | 原始 imputer 读取 23 通道及掩码，输出投影保留 23 通道，仅取前四个 |
| 历史能源 | 不使用；测试真实能源仅供评分，不进入条件 |
| 扩散 | 原仓库 SSSDS4 配置的线性 beta 0.0001→0.02，T=200；测试完整 200 步 DDPM |
| 模型规模 | 固定 4 个残差块、S4 状态维度 64、skip=64、双向 S4；宽度参加六组搜索 |

容量适配需要明确披露：上游示例为 36 个残差块、res/skip=256，本任务采用较小容量，
不是保持原示例参数量不变；上游序列长度 100 改为一天 24。核心层类型、连接方式不变。
S4 的 HiPPO 参数可训练性保持上游默认设置：C 为可训练参数，默认 A/B/dt 等为缓冲区。
双向 S4 处理整段带噪目标日，不是读取测试真实未来能源。

原训练工具对已知通道的变换与采样时固定已知通道不完全一致。本适配统一为训练、采样
都使用干净已知条件，能源通道走标准 DDPM。年份作为共享条件，并非只接到 PV 输出头。
天气来自你提供的 CSV；代码没有把观测天气转换成前一天发布的天气预报。

### 运行兼容与加速

- 使用保留 NPLR/HiPPO、双线性离散化和 FFT 卷积的原始 S4 实现。
- 用原生 PyTorch 计算含共轭对的 Cauchy 和；修正原 slow fallback 缺失共轭半边的问题，
  与上游加速分支的 `symmetric=True` 数学口径一致。无需编译 CUDA 扩展或安装 PyKeOps。
- 每次场景生成调用内缓存长度 24 的 S4 核，返回/异常时清空，不跨训练更新保留缓存。
  固定随机流下缓存/未缓存采样逐元素一致性有测试覆盖。
- 将展开视图缓冲区实体化，修复新版 einops 下 checkpoint 加载时的重叠内存问题。
- 去掉未使用的 wandb/Lightning 导入；保留原 weight_norm，可能看到弃用警告。
- 不使用 EMA，`ema_decay=0`。不支持将其改为非零。
- 日志记录模型实际设备、参数量、训练/验证时间；`SSSD_CPU_THREADS` 默认 1。

上述兼容改动与来源哈希见 [vendor/README](baseline8/vendor/README.md)。

## 安装与数据

在仓库根目录运行，使用 Python >=3.10 和已有 CUDA 兼容的 PyTorch >=2.0：

```bash
python -m pip install -r requirements-baseline8.txt
```

不用安装原始实验的全部旧依赖。将与 baseline5 完全相同的清洗文件放到：

```text
Data/CN03_energy_cleaned.csv
Data/weather_cleaned.csv
```

只保留能量/天气时间戳对齐、每天 00–23 点完整的日期。训练 2014–2020，验证 2021，
测试 2022；仅用完整训练日拟合 target/weather z-score，标准差加 1e-6。
NaN/Inf、重复时间戳直接报错，不自动填补。修复数据时务必同步所有 baseline 的数据版本。
缺小时的日期被排除，可检查 `split_dates.json`；不要将不同有效日期集的分数混入主表。
checkpoint 保存 CSV SHA256 和训练统计量；评估允许移动相同文件，不允许悄悄更换内容。
此提交不包含你的私有 CSV、权重或实验结果。

## 1. 六组验证集调参

```bash
python -m baseline8.sweep --seed 42 --set run.device=cuda:0
```

`res_channels ∈ {32,64,128}` × `learning_rate ∈ {0.0002,0.001}`，共六组，其他配置固定。
Adam，batch=16，上限 300 epochs，每 5 epochs 验证，12 次验证无提升早停；每个验证日
生成 20 个场景，仅依据 **2021 val_macro_nCRPS** 选最优配置和 checkpoint。
不依据测试分数选步数或超参数。六组配置数量相同不意味着与其他模型训练 FLOPs 相同。

```text
experiments/baseline8/tuning_budget_6/
  protocol.yaml
  partial_results.json
  tuning_results.csv
  best_config.yaml
  best_config.json
```

`--skip-completed` 只跳过配置/数据一致且完整结束的运行，不恢复中途 epoch。
不覆盖非空运行目录。改变协议需同时换结果根目录和调参汇总目录，例如：

```bash
python -m baseline8.sweep --seed 42 --set run.device=cuda:0 \
  --set run.output_root=experiments/baseline8_v2 \
  --output-dir experiments/baseline8_v2/tuning_budget_6
```

## 2. 最优配置的五种子训练

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline8.train \
    --config experiments/baseline8/tuning_budget_6/best_config.yaml \
    --run-name sssd_final \
    --set run.device=cuda:0 \
    --set run.seed=$seed || break
done
```

模型分别保存至 `experiments/baseline8/sssd_final_seed42/best.pt` 等目录。
`best.pt` 按各自验证集最优 epoch 保存。训练不会自动测试。

## 3. 测试

```bash
python -m baseline8.evaluate \
  --checkpoint experiments/baseline8/sssd_final_seed42/best.pt \
  --outdir evaluation_results/baseline8_seed42 \
  --device cuda:0
```

默认全部 2022 完整日、每天 100 个场景、batch=8、sample_chunk_size=32。
`--seed` 默认沿用 checkpoint 的训练 seed；实际独立采样 RNG 为 **seed+90000**，
例如训练 seed=42 对应 sample_seed=90042。验证使用 seed+50000。
固定 batch/chunk、软硬件环境才适合比较逐元素复现；更改 batch/chunk 不改变目标分布，
但可能改变具体随机样本。只对反归一化后的生成 PV 截断到非负，其余通道不截断。

支持 `--energy-path`、`--weather-path`、`--seed`、`--scenarios`、`--batch-size`、
`--no-plots`、`--max-days`。`--max-days 2` 可快速验收输出，但属于部分测试，不可作为
完整测试集论文结果。输出目录必须为空。

## 与 baseline5 一致的输出和指标

`baseline8/metrics.py` 与 CSDI baseline5 的提交
`17c217bca33a8e769cbb094c177cfe13c2f298bf` 中 `baseline5/metrics.py` **逐字节一致**，
Git blob=`ea17ab63806f45539fc4dce630115b1717e1ad55`，由测试校验。

| 文件 | 内容 |
| --- | --- |
| global_metrics.json / .csv | 全局与逐通道分数、种子、步数、耗时；CSV 为 metric,value |
| channel_metrics.csv | 每通道 MAE、RMSE、MAE_Z、RMSE_Z、R2、CRPS、nCRPS、Precision_Z、Recall_Z、CR、IW、IS、IS_Z |
| daily_scores.csv | 每天 ES_Z、VS_Z 及各通道 CRPS/IS/IS_Z |
| metric_definitions.json | 指标公式口径、区间和 VS 配对范围 |
| global_pearson.png | 真实测试数据的全局 Pearson 矩阵 |
| pearson/ | 真实/生成 Pearson 矩阵及绝对差 |
| random_timeseries_50/ | 固定种子随机选取 50 天的四通道图，不足则全部绘制 |
| baseline8_scenarios.npz | 物理量场景 `[D,S,4,24]`、真值、条件、日期、训练统计量和元数据 |
| global_metrics_wide.csv / metrics_summary.json | 便于合并多种子结果的额外汇总 |

点预测使用场景均值，在全部测试日/小时上计算每个通道的 MAE、RMSE、R2=1−SSE/SST。
`_Z` 使用训练目标标准差，不能解释为测试集 min-max 归一化。
nCRPS=sum(CRPS)/(sum(abs(target))+1e-8)，macro_nCRPS 为四通道 nCRPS 算术平均，
mean_nCRPS 则合并分子/分母；论文请明确使用哪一个。
CR/IW/IS 使用 95% 区间，CR 是 0–1 比例；兼容字段保留另一些 90% 区间指标。
联合 ES/VS 将最终物理场景按训练统计量重新标准化后展开成 96 维向量。
ES 使用经验分布 S² 分母。VS 的 p=0.5，仅跨通道、含同时刻与不同时刻全部 3456 对，
单位权重，**对配对求和再对日期求均值**，不对配对数求平均。ES=ES_Z、VS=VS_Z 为别名。
Precision_Z/Recall_Z 沿用 baseline5 最近中心半径的 kNN 近似，k=5、生成池上限10000；
不是完整 union-of-balls 实现。未定义 R2/小样本指标输出 JSON null、CSV 空值。
未新增 FID，因为它不在所对齐 baseline5 的指标集合内。

仅重新计算指标/画图而不重新训练、采样：

```bash
python -m baseline8.score_npz \
  --npz evaluation_results/baseline8_seed42/baseline8_scenarios.npz \
  --outdir evaluation_results/baseline8_seed42_rescored
```

## 测试与验收范围

```bash
MPLBACKEND=Agg python -m pytest -c pytest_baseline8.ini -q
```

包括原始 imputer 架构一致性、共轭 Cauchy 计算/梯度、S4 FFT 核与稠密状态递推的一致性、
已知条件固定与目标掩码、DDPM 更新公式、缓存等价、随机数隔离、四通道/条件梯度、
T=200 有限性、训练日统计量、错误数据、六组 sweep、checkpoint 加载、所有图表输出、
NPZ 重算、CLI 及 baseline5 公式/源码校验。
本地 CPU 合成数据测试通过不代表真实数据收敛或 GPU 性能已验证。
