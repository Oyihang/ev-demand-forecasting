# ST-TFT：基于时空融合 Transformer 的电动汽车充电需求预测研究报告

> **状态：实验模板。** 本报告的结果章节必须在实际运行训练与测试后填入。不得用虚构的指标替代实验结果。

## 摘要

本文研究城市区域级电动汽车充电需求的多步预测问题，提出一种结合图卷积网络（GNN）与 Temporal Fusion Transformer（TFT）的时空预测框架 ST-TFT。空间编码器利用区域邻接关系聚合历史充电负荷及占用、时长信号，TFT 建模历史时序依赖，并输出未来多个时间步的分位数预测。实验使用 UrbanEV 小时级区域数据，采用按时间顺序划分的训练、验证与测试集，并与朴素预测、LSTM、TFT 及其他时空基线进行比较。**摘要中的最终结果应在实验完成后补充。**

关键词：电动汽车；充电需求预测；Temporal Fusion Transformer；图神经网络；时空建模

## 1. 研究背景与问题定义

充电需求具有明显的日内周期、周内周期及区域异质性。仅依赖单一区域历史序列的模型可能忽略邻近区域之间的需求联动。本研究以区域为节点，预测每个区域未来 24 小时的充电电量。

给定历史窗口长度 $L$、区域图 $G=(V,E)$ 和未来预测长度 $H$，目标为：

\[
\hat{Y}_{t+1:t+H}^{(i)}=f_\theta(X_{t-L+1:t}^{(i)},\,G,\,S_i,\,K_{t-L+1:t+H})
\]

其中 $i$ 表示区域，$S_i$ 是静态区域属性，$K$ 是预测时已知的日历特征。

## 2. 数据集与预处理

- 数据来源：UrbanEV 官方数据仓库（原始数据不随本项目仓库分发；下载与许可说明见 [data/README.md](../data/README.md)）。
- 时间粒度：1 小时。
- 目标变量：`volume.csv` 中的区域充电电量。
- 动态历史特征：历史目标、占用率、充电时长。
- 静态特征：经纬度、充电桩数量、区域面积和周长，以及价格历史均值。
- 空间数据：区域邻接矩阵；距离矩阵用于后续替代图实验。
- 时间特征：小时、星期、周末、年内日期的周期编码。

### 2.1 数据清理与切分

数据按时间戳与区域 ID 对齐；缺失历史信号以前向填充并对剩余缺失置零。目标使用 `log1p` 变换，随后以训练期每个区域的均值和标准差标准化。时间切分比例为 70%/15%/15%，不随机打乱时间轴。所有标准化参数仅使用训练期计算。

运行 `python scripts/audit_data.py` 后，将 `results/metrics/data_audit.json` 的实际内容整理到此处，包括样本数、区域数、时间范围、缺失情况及图的稀疏度。

## 3. 方法

### 3.1 空间图编码器

在每个历史时刻，图卷积按归一化邻接矩阵聚合邻域特征：

\[
H_t^{(l+1)}=\sigma(\tilde{A}H_t^{(l)}W^{(l)}),\quad
\tilde{A}=D^{-1/2}(A+I)D^{-1/2}
\]

得到的空间表示沿历史时间维进行池化，并与区域静态特征一起作为 TFT 的静态上下文。

### 3.2 Temporal Fusion Transformer

TFT 使用变量选择网络、门控残差网络、LSTM 编码器/解码器、静态变量编码和可解释时间注意力，对每个区域的历史目标与未来已知日历特征建模。

### 3.3 预测头与损失函数

模型预测 0.1、0.5、0.9 三个分位数，使用 Pinball Loss：

\[
L_q(y,\hat y)=\max(q(y-\hat y),(q-1)(y-\hat y))
\]

点预测指标使用中位数分位数（0.5）。

## 4. 实验设计

### 4.1 对比模型

1. Last-value / seasonal-naive 基线。
2. LSTM。
3. TFT（不使用跨区域图信息）。
4. GCN-LSTM 或公开时空基线。
5. ST-TFT（本文模型）。

所有模型必须使用相同的时间切分、目标变换、预测步长和评价指标。

### 4.2 评价指标

- MAE：平均绝对误差。
- RMSE：均方根误差。
- WAPE：总绝对误差除以实际需求总量。
- MAPE（非零目标）：只在非零目标上计算，并注明其对低需求值敏感。

### 4.3 消融实验

- 去掉跨区域边（只保留自环）。
- 使用原始邻接矩阵。
- 使用距离加权邻接矩阵。
- 移除 occupancy / duration 特征。
- 不使用静态区域属性。

## 5. 实验结果

**待运行后填写。** 指标文件默认保存在 `results/metrics/*_test_metrics.json`。

| 模型 | MAE | RMSE | WAPE (%) | 训练时间 | 参数量 |
|---|---:|---:|---:|---:|---:|
| Last-value | 待测 | 待测 | 待测 | — | — |
| LSTM | 待测 | 待测 | 待测 | 待测 | 待测 |
| TFT | 待测 | 待测 | 待测 | 待测 | 待测 |
| ST-TFT | 待测 | 待测 | 待测 | 待测 | 待测 |

## 6. 误差与空间分析

运行绘图脚本后，分析 `results/figures/` 下生成的预测曲线与训练曲线。建议额外生成：

- 每个预测步长的 MAE/RMSE 曲线；
- 各区域误差分布地图；
- 高需求区与低需求区的分组误差；
- 工作日与周末的预测误差；
- 空间图消融结果表。

在此处解释哪些区域更难预测，以及空间模块是否对邻接关系更强的区域产生更明显收益。

## 7. 可解释性与业务含义

TFT 的变量选择权重和时间注意力可用于分析模型依赖的输入信号与时间位置，但它们不等同于因果效应。若用于能源调度，应进一步评估预测误差对峰值负荷、容量配置和调度成本的影响。

## 8. 局限性

1. 数据覆盖约六个月，不能完整评估全年季节性。
2. 本版空间上下文通过历史窗口池化，可能丢失空间影响的短期动态变化。
3. 未来天气与价格只有在预测时真实可得时才能加入 decoder；未来实际值会造成数据泄漏。
4. 区域级预测结果不等同于单个充电站或单个充电桩级预测。
5. 结果只对当前数据集、时间切分和实验配置负责，不能直接推断到其他城市。

## 9. 结论

待所有基线与消融实验完成后，基于实际测试集指标总结 ST-TFT 是否优于时间模型，以及空间图结构在哪些区域和预测步长上最有帮助。不要仅凭单次训练或单一指标得出结论。

## 复现说明

```bash
pip install -r requirements.txt
python scripts/audit_data.py
python -m src.training.train --smoke-test --max-zones 8
python -m src.training.train
python scripts/plot_results.py
pytest -q
```

## 参考资料

- Lim, B. et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 2021.
- IntelligentSystemsLab, UrbanEV dataset and repository: https://github.com/IntelligentSystemsLab/UrbanEV
- IntelligentSystemsLab, ST-EVCDP repository: https://github.com/IntelligentSystemsLab/ST-EVCDP
