# 零售行业客户细分与购买模式分析

> 用 WEKA 走完一遍完整的数据挖掘流程：从数据预处理、特征选择，到分类算法对比、K-Means 客户分群，最后落成可执行的精准营销策略。
>
> 课程项目 · 数据挖掘（Data Mining），全英文授课与报告

![tool](https://img.shields.io/badge/tool-WEKA%203.8.6-1f6feb)
![task](https://img.shields.io/badge/task-classification%20%7C%20clustering%20%7C%20association-2ea043)
![lang](https://img.shields.io/badge/report-English-informational)

---

## 一、项目概览

零售行业的核心矛盾是：客户需求高度分化，而传统营销靠经验拍板、ROI 越来越低。这个项目想回答三个问题：

1. **能不能用行为数据预测客户价值？**——能，KNN 在 10 折交叉验证下准确率 **94.8%**。
2. **客户能不能自然分成几群？**——能，K-Means 在 K=3 时轮廓系数 **0.68**，三群特征清晰可分。
3. **分完之后业务怎么用？**——按客户生命周期与价值分层配置营销资源，见[第六节](#六业务落地从分群到策略)。

| 关键结果 | 数值 |
|---|---|
| 数据集规模 | 500 条客户记录 / 10 个行为特征（5 数值 + 5 类别） |
| 最优分类器 | KNN（K=10）准确率 **94.8%** |
| 最佳可解释模型 | J48 决策树，4 层、8 个叶节点 |
| 聚类方案 | K-Means，K=3（肘部法则确定），轮廓系数 **0.68** |
| 强关联规则 | 高活跃 → 高消费（支持度 15%、置信度 85%、提升度 2.1） |

---

## 二、数据集

原始数据为课程提供的模拟零售交易数据（`Online_Retail.csv`），字段设计贴近真实 CRM 场景：

| 字段 | 类型 | 含义 |
|---|---|---|
| `TotalAmount` | 数值（比率） | 累计消费金额 |
| `OrderCount` | 数值（比率） | 订单数 |
| `ActiveDays` | 数值（比率） | 活跃天数 |
| `ProductVariety` | 数值（比率） | 购买商品品类数 |
| `AvgOrderValue` | 数值（比率） | 客单价 |
| `Country` | 类别（名义） | 客户所在国家 |
| `Channel` | 类别（名义） | 主要获客渠道 |
| `PaymentMethod` | 类别（名义） | 主要支付方式 |
| `ProductCategory` | 类别（名义） | 主要购买品类 |
| `CustomerValue` | 类别（有序） | 客户价值标签 High / Medium / Low（**分类目标**） |

数据质量评估：无缺失值，取值逻辑自洽，各属性分布相对均匀。

> **关于本仓库的数据**
> `data/online_retail_demo.csv` 是按同一套字段结构、用固定随机种子生成的**演示数据**（500 条，可直接跑通全部流程）。原始课程数据不在本仓库分发。
> 生成脚本见 `scripts/generate_data.py`，客户价值标签由「归一化客单价 × 活跃天数」的加权得分按分位数切三档得到，因此三类大致均衡（High 160 / Medium 190 / Low 150），适合做分类对比实验。
> 下文所有指标均来自**原始数据集**上的实验结果，演示数据用于复现流程，具体数值会有差异。

---

## 三、分析流程

```
原始数据 (500 × 10)
      │
      ├─ ① 数据清洗     缺失值检测 → 无；异常值保留合理业务值
      │
      ├─ ② 数据变换     Min-Max 归一化，消除量纲差异
      │
      ├─ ③ 特征选择     InfoGainAttributeEval + Ranker
      │                 → 关键特征：ActiveDays、TotalAmount
      │
      ├─ ④ 分类建模     J48 / NaiveBayes / IBk(KNN)，10 折交叉验证
      │                 → 目标变量 CustomerValue
      │
      ├─ ⑤ 聚类分群     SimpleKMeans，K=3，欧氏距离
      │                 → 忽略 CustomerValue，保持无监督性
      │
      └─ ⑥ 关联分析     Apriori，minSupport=0.1，minConfidence=0.7
                        → 挖掘购买行为组合规则
```

---

## 四、三个核心实验

### 实验一：数据预处理与特征选择

| 步骤 | 操作 | 结果 |
|---|---|---|
| 缺失值处理 | WEKA Preprocess 面板摘要统计 | 无缺失值，跳过插补 |
| 异常值处理 | 业务视角审查 | 保留合理高值（大客户不应被剔除） |
| 归一化 | `unsupervised.attribute.Normalize`（Min-Max） | 所有数值属性压缩到 [0,1]，避免金额量纲主导距离计算 |
| 特征选择 | `InfoGainAttributeEval` + `Ranker`，Use full training set | 信息增益排序靠前：**ActiveDays、TotalAmount** |

**为什么必须归一化**：KNN 用欧氏距离度量相似度，`TotalAmount` 取值范围上千、`ProductVariety` 只有 1–5，不归一化的话距离几乎完全由金额决定，聚类和 KNN 都会失真。

**为什么聚类时要忽略 `CustomerValue`**：这是标签泄漏。带着标签聚类等于让模型偷看答案，失去"无监督发现"的意义——这一步是实验里最容易做错、也最容易被面试官追问的地方。

### 实验二：分类算法对比

统一采用 **10 折交叉验证**，目标变量 `CustomerValue`（三分类），对比三个算法：

| 算法 | WEKA 类 | 关键参数 | 优势 | 结果 |
|---|---|---|---|---|
| J48 决策树（C4.5） | `trees.J48` | confidenceFactor = 0.25，minNumObj = 2 | 可解释性最强，能直接读出业务规则 | 结构：4 层 / 8 叶节点 |
| 朴素贝叶斯 | `bayes.NaiveBayes` | 默认参数，KernelEstimator = false | 训练最快，适合高维 | 速度最优 |
| KNN | `lazy.IBk` | K = 10，欧氏距离 | 无需训练，多分类友好 | **准确率最高：94.8%** |

**结论与取舍**：

- 追求**准确率** → 选 KNN（94.8%）
- 追求**可解释性与落地** → 选 J48，因为业务方需要的是"什么样的客户算高价值"，而不是一个黑盒分类结果
- 追求**训练效率** → 选朴素贝叶斯

这也说明一件事：模型选型不是只看准确率，还要看**谁用、怎么用**。

### 实验三：K-Means 客户分群

| 配置项 | 取值 |
|---|---|
| 算法 | `SimpleKMeans` |
| 簇数 K | 3（肘部法则确定，SSE 在 K=3 后下降明显放缓） |
| 距离度量 | 欧氏距离 |
| 最大迭代 | 500 |
| 初始化 | 随机初始化 |
| 忽略属性 | `CustomerValue` |

**质量评估**：轮廓系数 **0.68**（>0.5 即认为簇结构合理），簇间距离清晰分离，SSE 随 K 增长显著下降。

**三群画像**：

| 簇 | 命名 | 特征 | 规模判断 |
|---|---|---|---|
| Cluster 3 | 核心客户 | 高消费 + 高活跃，集中在散点图右上区域 | 少数但贡献主要营收 |
| Cluster 2 | 潜力客户 | 处于中间过渡区，有向上迁移空间 | 中等 |
| Cluster 1 | 一般客户 | 低价值区密集分布，客单与频次均低 | 多数 |

**可视化发现**：`TotalAmount` 与 `ActiveDays` 两个维度的分离度最明显；`OrderCount` 与 `ProductVariety` 呈较强正相关（买得多的人往往买得杂）。

### 附加：关联规则挖掘（Apriori）

参数：最小支持度 0.1、最小置信度 0.7、最多输出 10 条规则。

| 规则 | 支持度 | 置信度 | 提升度 |
|---|---|---|---|
| 高活跃 → 高消费 | 15% | 85% | 2.1 |
| 多品类购买 → 高客户价值 | 12% | 78% | 1.9 |

提升度均 > 1，说明这两条规则具备正向关联而非偶然共现。业务含义：**活跃度是消费力的前置信号**，把运营资源投在"提升活跃"上，比直接打折促销更可能带来客单价提升。

---

## 五、从决策树读出的业务规则

J48 的根节点是 `AvgOrderValue > 0.568`（归一化后），往下 4 层、8 个叶节点，可以翻译成两句人话：

> **规则 1**：客单价高于阈值 **且** 活跃天数 > 45 天 → 高价值客户
> **规则 2**：客单价不达阈值 **但** 下单频率 > 0.1 → 中价值客户

这两条规则的价值在于**可执行**：销售团队不需要理解决策树，只需要一张"客单价 + 活跃天数"的判定表，就能在 CRM 里给客户打标。

---

## 六、业务落地：从分群到策略

### 按客户生命周期配置动作

| 阶段 | 对应群体 | 策略 |
|---|---|---|
| 引入期 | Cluster 2（潜力） | 识别并培育潜力客户，引导完成第二次购买 |
| 成长期 | Cluster 2 → 3 | 交叉销售、升级激励 |
| 成熟期 | Cluster 3（核心） | 深度维护忠诚度，防流失 |
| 衰退期 | 全部 | 流失预警 + 召回策略 |

### 按分层配置营销资源

| 群体 | 服务策略 | 成本取向 |
|---|---|---|
| 核心客户（Cluster 3） | 专属客服、个性化推荐、VIP 权益与优先服务 | 高投入高回报 |
| 潜力客户（Cluster 2） | 定向促销活动、会员升级计划、成长路径设计 | 中等投入，看转化 |
| 一般客户（Cluster 1） | 标准化服务、自动化营销流程 | 严格控制成本 |

---

## 七、局限与改进方向

**数据层面**

- 使用模拟数据，与真实业务数据存在分布差异
- 未纳入时间序列特征，无法刻画客户行为的动态变化

**算法层面**

- 未尝试深度学习等更复杂的模型
- 缺乏实时能力与在线学习机制

**后续可以做的**

1. **数据层**：接入实时交易流；融合多源数据（社交媒体、客服记录）
2. **算法层**：尝试 LSTM / Transformer 建模行为序列；实现在线学习与实时更新
3. **系统层**：搭建 CDP（客户数据平台），开发实时推荐引擎

---

## 八、如何复现

### 方式一：用演示数据快速跑通（推荐）

```bash
git clone https://github.com/dyc94588/weka-customer-segmentation.git
cd weka-customer-segmentation

# 重新生成演示数据（可选，仓库里已包含）
python scripts/generate_data.py
```

然后用 WEKA 3.8.6 打开 `data/online_retail_demo.csv`（Explorer → Preprocess → Open file）。

> WEKA 对中文路径敏感，建议把项目放在纯英文路径下。

### 方式二：按步骤手工复现

1. **Preprocess**：Open file → 加载 CSV → Choose `unsupervised.attribute.Normalize` → Apply
2. **特征选择**：Select attributes 标签 → Evaluator 选 `InfoGainAttributeEval` → Search 选 `Ranker` → Start
3. **分类**：Classify 标签 → 依次选 `trees.J48` / `bayes.NaiveBayes` / `lazy.IBk`（K=10）→ Test options 选 Cross-validation、Folds=10 → Target 选 `CustomerValue` → Start
4. **聚类**：Cluster 标签 → `SimpleKMeans`，numClusters=3 → Ignore attributes 里把 `CustomerValue` 加入忽略列表 → Start → 右键结果 → Visualize cluster assignments

更细的界面操作（含每个按钮的位置）见 [`docs/WEKA实验报告原文.txt`](docs/WEKA实验报告原文.txt)，项目完整汇报逻辑见 [`docs/项目PPT原文.txt`](docs/项目PPT原文.txt)。

---

## 九、目录结构

```
weka-customer-segmentation/
├── README.md
├── data/
│   └── online_retail_demo.csv     # 500 条演示数据（结构同原始数据集）
├── scripts/
│   └── generate_data.py           # 演示数据生成脚本（固定种子，可复现）
└── docs/
    ├── 项目PPT原文.txt             # 17 页汇报内容（已脱敏）
    └── WEKA实验报告原文.txt        # 三次实验的完整操作步骤
```

---

## 十、技术栈与收获

**工具**：WEKA 3.8.6（Explorer / J48 / NaiveBayes / IBk / SimpleKMeans / Apriori / InfoGainAttributeEval）

**方法**：CRISP-DM 思路的完整走通——业务理解 → 数据预处理 → 建模 → 评估 → 部署建议

**三条最实在的收获**：

1. **归一化不是可选步骤**。做 KNN 和聚类时，量纲差异会直接毁掉距离度量，这是踩过才知道的。
2. **聚类前必须屏蔽标签**。忽略 `CustomerValue` 这一步做错，整个"无监督发现"就成了自欺欺人。
3. **准确率不是唯一指标**。KNN 准确率最高，但真正被业务采纳的是 J48——因为它能输出一句人话规则。

---

## 参考资料

1. Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
2. Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques*. Morgan Kaufmann.
3. WEKA Machine Learning Software, Version 3.8.6 — https://www.cs.waikato.ac.nz/ml/weka/

---

> 本项目为课程学习实践，数据集与实验设计来自 Data Mining 课程要求，分析过程、结论与文档整理为个人完成。
