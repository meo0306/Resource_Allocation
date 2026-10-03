# 教学方法分配方案与阶段实施状态

> 状态截止：2026 年 9 月 8 日 22:45（北京时间）。本文件记录五拓扑历史版本的模型、实现和阶段结果；不纳入截止时点以后的四拓扑重构与新目标实验。整理于 2026 年 9 月 27 日。

核心问题是“为已选择知识点分配教学方法”，采用 PA-MOAP、Effect-only greedy 初解与共享 Masked PPO 局部改进。已完成算例处理、数据集划分、参数筛选、三种子训练、基线对比及 Gurobi/混合精炼验证 pilot。当前结论为阶段性计算实验结果，尚不意味着教学效果验证和方法优势论证全部完成。

# 1. 教学属性空间与模型参数构造

## 1.1. 教学交互属性空间

表1 教学交互属性空间

| **维度**       | **知识点侧含义** $\mathbf{u}_{i}$ | **方法侧含义** $r_{m}$ | **学生侧含义** $s$     | **教师侧含义** $t$                              |
|----------------|----------------------------------|-----------------------|-----------------------|------------------------------------------------|
| guidance       | 是否需要强引导                   | 方法提供的引导程度    | 学生对引导的需求      | 教师讲授、示范与过程引导倾向                   |
| interaction    | 是否适合互动讨论                 | 方法的互动强度        | 学生互动偏好          | 教师组织讨论、问答和课堂互动的倾向/能力        |
| practice       | 是否需要实践/练习                | 方法的实践性          | 学生实践偏好          | 教师开展实验、练习、项目实践的经验与能力       |
| autonomy       | 是否适合自主探究                 | 方法的开放探究程度    | 学生自主学习能力/偏好 | 教师组织开放探究、自主学习活动的倾向           |
| collaboration  | 是否适合协作                     | 方法的协作程度        | 学生协作偏好          | 教师组织小组协作、同伴互评的倾向/能力          |
| cognitive_load | 该知识点认知负荷                 | 方法引入的认知负荷    | 学生认知负荷承受能力  | 教师对高认知负荷、高组织复杂度教学活动的承受度 |

## 1.2. 知识点需求向量

定义类别原型矩阵：

$$\mathbf{D}^{C} \in \lbrack 0,1\rbrack^{K_{c} \times 5}$$

其中 $D_{ck}^{C}$表示类别 $c$的知识点在第 $k$个教学活动维度上的典型需求强度，$k\mathcal{\in L}$。

取 $K_{c} = 6$，行对应知识点类别，列对应前5个教学交互维度。可设：

| **知识点类别** | **guidance** | **interaction** | **practice** | **autonomy** | **collaboration** |
|----------------|--------------|-----------------|--------------|--------------|-------------------|
| 基础           | 0.90         | 0.20            | 0.15         | 0.20         | 0.20              |
| 理论           | 0.70         | 0.60            | 0.25         | 0.45         | 0.35              |
| 算法           | 0.55         | 0.45            | 0.80         | 0.50         | 0.45              |
| 实验           | 0.45         | 0.55            | 0.95         | 0.55         | 0.60              |
| 案例           | 0.35         | 0.70            | 0.70         | 0.60         | 0.80              |
| 扩展           | 0.30         | 0.65            | 0.50         | 0.90         | 0.60              |

对知识点 $i$，定义其教学需求向量：

$$\mathbf{u}_{i} = (u_{i1},u_{i2},\ldots,u_{i6}) \in \lbrack 0,1\rbrack^{6}$$

其中前五维由知识点类别决定：

$$u_{ik} = D_{c_{i}k}^{C},k\mathcal{\in L}$$

第六维复用知识点认知负荷：

$$u_{i6} = q_{i}$$

因此，内容二阶段不再额外引入新的知识点原始属性，而是基于知识点类别和认知负荷构造派生需求向量。

## 1.3. 教学方法属性向量

对任意教学方法 $m\mathcal{\in M}$，定义其教学属性向量：

$$r_{m} = (r_{m1},r_{m2},\ldots,r_{m6}) \in \lbrack 0,1\rbrack^{6}$$

其中 $r_{mk}$表示教学方法 $m$在第 $k$个教学交互属性维度上的强度。

所有教学方法的属性构成方法属性矩阵：

$$R \in \lbrack 0,1\rbrack^{M \times 6}$$

其中第 $m$行为 $r_{m}$。列对应：guidance, interaction, practice, autonomy, collaboration, cognitive_load，然后使用下列专家规则赋值：

| **方法**       | **guidance** | **interaction** | **practice** | **autonomy** | **collaboration** | **cognitive_load** |
|----------------|--------------|-----------------|--------------|--------------|-------------------|--------------------|
| 讲授法         | 0.90         | 0.20            | 0.10         | 0.10         | 0.10              | 0.30               |
| 讨论法         | 0.40         | 0.90            | 0.25         | 0.55         | 0.75              | 0.55               |
| 案例分析法     | 0.45         | 0.65            | 0.60         | 0.55         | 0.55              | 0.60               |
| 实验法         | 0.35         | 0.45            | 0.95         | 0.55         | 0.45              | 0.75               |
| 项目式学习     | 0.30         | 0.70            | 0.85         | 0.75         | 0.90              | 0.85               |
| 翻转课堂       | 0.55         | 0.65            | 0.45         | 0.70         | 0.45              | 0.60               |
| 探究式学习     | 0.30         | 0.65            | 0.70         | 0.95         | 0.60              | 0.85               |
| 协作学习       | 0.35         | 0.85            | 0.45         | 0.60         | 0.95              | 0.65               |
| 游戏化教学     | 0.40         | 0.75            | 0.65         | 0.70         | 0.65              | 0.70               |
| 辅导法         | 0.95         | 0.45            | 0.35         | 0.30         | 0.25              | 0.40               |
| 在线自适应学习 | 0.75         | 0.35            | 0.55         | 0.80         | 0.20              | 0.55               |

## 1.4. 教学效果匹配度 $\mathbf{E}_{\mathbf{im}}$

定义方法—类别教学效果矩阵：

$$\mathbf{A}^{E} \in \lbrack 0,1\rbrack^{M \times K}$$

其中 $A_{mc}^{E}$表示教学方法 $m$与知识点类别 $c$的理论教学效果匹配度。

若知识点 $i$的类别为 $c_{i}$，则知识点 $i$分配教学方法 $m$时的教学效果匹配度定义为：

$$E_{im} = A_{mc_{i}}^{E},i\mathcal{\in I,}\ m\mathcal{\in M}$$

其中 $E_{im} \in \lbrack 0,1\rbrack$。

| **教学方法 \ 知识点** | **基础** | **理论** | **算法** | **实验** | **案例** | **扩展** |
|------------------------|----------|----------|----------|----------|----------|----------|
| 讲授法                 | 0.90     | 0.70     | 0.40     | 0.20     | 0.30     | 0.40     |
| 讨论法                 | 0.30     | 0.70     | 0.50     | 0.40     | 0.60     | 0.80     |
| 案例分析法             | 0.20     | 0.50     | 0.70     | 0.40     | 0.90     | 0.70     |
| 实验法                 | 0.10     | 0.30     | 0.60     | 0.95     | 0.50     | 0.40     |
| 项目式学习             | 0.10     | 0.40     | 0.60     | 0.50     | 0.85     | 0.80     |
| 翻转课堂               | 0.50     | 0.70     | 0.65     | 0.40     | 0.55     | 0.60     |
| 探究式学习             | 0.20     | 0.50     | 0.50     | 0.80     | 0.60     | 0.85     |
| 协作学习               | 0.30     | 0.50     | 0.50     | 0.40     | 0.75     | 0.70     |
| 游戏化教学             | 0.20     | 0.40     | 0.60     | 0.50     | 0.70     | 0.90     |
| 辅导法                 | 0.60     | 0.60     | 0.60     | 0.60     | 0.60     | 0.60     |
| 在线自适应学习         | 0.80     | 0.70     | 0.60     | 0.30     | 0.40     | 0.75     |

# 2. 教学方法分配问题优化模型

## 2.1. 问题描述

教学方法分配任务以前置资源组合阶段输出的已选知识点集合为输入。本阶段接收前置阶段的知识点筛选结果，并以前置结果满足其先修约束和资源约束为输入前提；本阶段转换程序不重新证明前置优化结果的可行性。因此，本阶段不再重复考虑教学时长、认知负荷和外部资源供给等资源容量约束，而是在已选知识点集合上进一步决定每个知识点应采用的教学方法。

该问题的核心决策是：对于每个已选知识点 $i$，从候选教学方法集合中选择一种方法 $m$，使整体分配方案同时具有较高的理论教学效果、学生偏好满足度、教师偏好满足度以及全局方法分布合理性，并避免方法使用过度单一或过度集中。

## 2.2. 集合、索引与基本参数

令：

$$\mathcal{I} = \{ 1,2,\ldots,n\}$$

表示前置资源组合阶段输出的已选知识点集合，其中 $n = \mid \mathcal{I} \mid$为待分配教学方法的知识点数量。

$$\mathcal{M} = \{ 1,2,\ldots,M\}$$

表示候选教学方法集合，其中 $M = \mid \mathcal{M} \mid$。当前方案中 $M = 11$。

$$\mathcal{C} = \{ 1,2,\ldots,K_{c}\}$$

表示知识点类别集合，其中 $K_{c} = 6$，对应基础、理论、算法、实验、案例、扩展六类知识点。

$$\mathcal{D} = \{ 1,2,\ldots,6\}$$

表示教学交互属性维度集合，六个维度依次为 guidance、interaction、practice、autonomy、collaboration 和 cognitive_load。

$$\mathcal{L} = \{ 1,2,\ldots,5\}$$

表示前五个教学活动属性维度，不包含 cognitive_load。

对任意知识点 $i\mathcal{\in I}$，其类别记为：

$$c_{i}\mathcal{\in C}$$

其认知负荷记为：

$$q_{i} \in \lbrack 0,1\rbrack$$

其中 $q_{i}$直接复用前置算例中的 CognitiveLoad 属性。

## 2.3. 教学属性与效果匹配参数

见1.2、1.3、1.4节

## 2.4. 偏好评分参数

学生偏好：评价**学生学习特征**与当前**知识点-方法匹配对**特征是否符合

教师偏好：评价**教师风格**与当前**知识点-方法匹配对**特征是否符合

偏好评分用于刻画学生或教师对某一“知识点—方法”匹配对的满意程度。学生偏好和教师偏好采用统一的计算形式，仅主体画像参数不同。

令偏好主体集合为：

$$\mathcal{B} = \{stu,tea\}$$

其中 $stu$表示学生，$tea$ 表示教师。

对任一主体 $b\mathcal{\in B}$，定义其画像向量：

$$\mathbf{\theta}^{b} = (\theta_{1}^{b},\theta_{2}^{b},\ldots,\theta_{6}^{b}) \in \lbrack 0,1\rbrack^{6}$$

其中前五维表示主体对教学活动形式的偏好或能力倾向，第六维表示主体对认知负荷或组织复杂度的承受能力。

### 2.4.1. 知识点—方法匹配对交互特征

对任意知识点 $i\mathcal{\in I}$和教学方法 $m\mathcal{\in M}$，定义其匹配对交互特征向量：

$$\mathbf{z}_{im} = (z_{im1},z_{im2},\ldots,z_{im6}) \in \lbrack 0,1\rbrack^{6}$$

其中前五维表示教学方法 $m$用于知识点 $i$时形成的教学活动特征：

$$z_{imk} = u_{ik}r_{mk},k\mathcal{\in L}$$

第六维表示综合认知负荷：

$$z_{im6} = \lambda_{q}q_{i} + \lambda_{r}r_{m6}$$

其中：

$$\lambda_{q} + \lambda_{r} = 1,\lambda_{q},\lambda_{r} \geq 0$$

当前取：

$$\lambda_{q} = 0.6,\lambda_{r} = 0.4$$

该定义表示综合认知负荷主要由知识点本身认知负荷决定，同时考虑教学方法自身引入的认知或组织复杂度。

### 2.4.2. 局部偏好得分

（1）对任一主体 $b\mathcal{\in B}$，定义知识点 $i$与教学方法 $m$的前五维活动匹配得分：

$$\Phi_{im}^{b} = \sum_{k\mathcal{\in L}}^{}\omega_{k}\left( 1- \mid z_{imk} - \theta_{k}^{b} \mid \right)$$

其中：

$$\omega_{k} \geq 0,\sum_{k\mathcal{\in L}}^{}\omega_{k} = 1$$

取等权重：$\omega_{k} = 0.2,k\mathcal{\in L}$。

（2）对任一主体 $b\mathcal{\in B}$，定义负荷兼容度：

$$\Psi_{im}^{b} = 1 - \max(0,z_{im6} - \theta_{6}^{b})$$

该项表示当匹配对的综合认知负荷不超过主体承受能力时不产生扣分；当综合认知负荷超过主体承受能力时，按照超出部分降低兼容度。

（3）对任一主体 $b\mathcal{\in B}$，定义局部偏好得分：

$$P_{im}^{b} = clip\left( \alpha_{b}\Phi_{im}^{b} + (1 - \alpha_{b})\Psi_{im}^{b},0,1 \right)$$

其中：$\alpha_{b} \in \lbrack 0,1\rbrack$，表示活动形式匹配得分与负荷兼容度之间的相对权重。当前取：$\alpha_{stu} = \alpha_{tea} = 0.75$

因此，学生偏好得分为：$P_{im}^{S} = clip\left( 0.75\Phi_{im}^{stu} + 0.25\Psi_{im}^{stu},0,1 \right)$，教师偏好得分为：$P_{im}^{T} = clip\left( 0.75\Phi_{im}^{tea} + 0.25\Psi_{im}^{tea},0,1 \right)$，其中 $P_{im}^{S},P_{im}^{T} \in \lbrack 0,1\rbrack$。

## 2.5. 决策变量

定义二元决策变量：

$$x_{im} = \left\{ \begin{matrix}
1, & \text{若知识点}\text{ }v_{i}\text{ }\text{采用方法}\text{ }m, \\
0, & \text{否则},
\end{matrix} \right.$$

其中：

$$i\mathcal{\in I,}m\mathcal{\in M}$$

在算法实现中，可使用与 $y_{im}$等价的整数 assignment 表示：

$$a_{i} = m$$

表示知识点 $i$当前被分配教学方法 $m$。

## 2.6. 方法分布与全局评价

评价**教学场景**与**整个分配方案**的结构是否符合

### 2.6.1. 当前方法分布

给定分配方案 $x$，定义教学方法 $m$在当前方案中的使用比例：

$$\pi_{m}(x) = \frac{1}{n}\sum_{i\mathcal{\in I}}^{}x_{im},m\mathcal{\in M}$$

对应的当前方法分布向量为：

$$\mathbf{\pi}(x) = (\pi_{1}(x),\pi_{2}(x),\ldots,\pi_{M}(x))$$

显然有：

$$\sum_{m\mathcal{\in M}}^{}\pi_{m}(x) = 1$$

### 2.6.2. 目标方法分布

定义目标方法分布：

$$\mathbf{\rho} = (\rho_{1},\rho_{2},\ldots,\rho_{M})$$

其中：

$$\rho_{m} \geq 0,\sum_{m\mathcal{\in M}}^{}\rho_{m} = 1$$

$\rho_{m}$表示教学设计者或教学场景期望教学方法 $m$在整体方案中的使用比例。

## 2.7. 目标函数

### 2.7.1. 局部教学效果目标

$$F_{E}(x) = \frac{1}{n}\sum_{i\mathcal{\in I}}^{}{\sum_{m\mathcal{\in M}}^{}x_{im}}E_{im}$$

该项表示整体分配方案的平均理论教学效果匹配度。

### 2.7.2. 局部偏好目标

①学生偏好目标：表示整体方案对学生偏好的平均满足程度。

$$F_{S}(x) = \frac{1}{n}\sum_{i\mathcal{\in I}}^{}{\sum_{m\mathcal{\in M}}^{}x_{im}}P_{im}^{S}$$

②教师偏好目标：表示整体方案对教师偏好的平均满足程度。

$$F_{T}(x) = \frac{1}{n}\sum_{i\mathcal{\in I}}^{}{\sum_{m\mathcal{\in M}}^{}x_{im}}P_{im}^{T}$$

### 2.7.3. 全局方法分布目标

定义全局方法分布得分：

$$F_{G}(x) = 1 - \frac{1}{2}\sum_{m\mathcal{\in M}}^{}{\mid \pi_{m}(x) - \rho_{m} \mid}$$

该项用于衡量当前方法分布 $\mathbf{\pi}(x)$与目标方法分布 $\mathbf{\rho}$的接近程度。由于二者均为概率分布，$\frac{1}{2}\sum_{m}^{} \mid \pi_{m}(x) - \rho_{m} \mid \in \lbrack 0,1\rbrack$，因此：

$$F_{G}(y) \in \lbrack 0,1\rbrack$$

数值越大表示整体方法分布越接近目标结构,后续可改用$\mathbf{F}_{\mathbf{G}}(x) = - D_{KL}(\rho \parallel \mathbf{\pi}(x) + \epsilon)$

### 2.7.4. 软约束违反程度 $\mathbf{V}_{\text{soft}}(\mathbf{x})$——做罚函数

定义总软约束违反程度：

$$V_{\text{soft}}(y) = \lambda_{\text{div}}V_{\text{div}}(y) + \lambda_{\text{cap}}V_{\text{cap}}(y)$$

其中：$\lambda_{div} \geq 0,\lambda_{cap} \geq 0$，分别表示多样性软约束与占比上限软约束的惩罚权重，取：$\lambda_{\text{div}} = \lambda_{\text{cap}} = 1.0$。

**①方法多样性软约束**

定义当前方法分布的归一化熵：

$$H(y) = - \frac{1}{\log M}\sum_{m\mathcal{\in M}}^{}\pi_{m}(y)\log(\pi_{m}(y) + \varepsilon)$$

其中 $\varepsilon > 0$是避免数值错误的极小常数，例如 $10^{- 8}$。

归一化熵 $H(y) \in \lbrack 0,1\rbrack$。其值越大，表示教学方法使用越多样；若所有知识点均使用同一种教学方法，则 $H(y)$接近 0。

设最低多样性要求为：

$$H_{\min} \in \lbrack 0,1\rbrack$$

则多样性违反程度定义为：

$$V_{div}(y) = \left\lbrack \max(0,H_{\min} - H(y)) \right\rbrack^{2}$$

当前取：$H_{\min} = 0.55$

**②方法占比上限软约束**

为避免某一教学方法在整体方案中过度集中，设方法 $m$的最大允许使用比例为：

$$\left. \ {\overset{ˉ}{\pi}}_{m} \in (0,1 \right\rbrack$$

则方法占比上限违反程度定义为：

$$V_{cap}(y) = \sum_{m\mathcal{\in M}}^{}\left\lbrack \max(0,\pi_{m}(y) - {\overset{ˉ}{\pi}}_{m}) \right\rbrack^{2}$$

第一版可统一取：${\overset{ˉ}{\pi}}_{m} = 0.35,m\mathcal{\in M}$，即任一教学方法在整体方案中的使用比例不应过度超过 35%。

## 2.8. 约束

### 2.8.1. 二元变量约束

$$x_{im} \in \{ 0,1\},\forall i\mathcal{\in I,}\ m\mathcal{\in M}$$

### 2.8.2. 唯一分配约束

每个已选知识点必须且只能分配一种教学方法：

$$\sum_{m}^{}x_{im} = 1,\forall i\mathcal{\in I}$$

### 2.8.3. 方法可行性约束

定义方法可行性掩码：

$$\Gamma_{im} \in \{ 0,1\}$$

其中：$\Gamma_{im} = 1$表示教学方法 $m$对知识点 $i$可用；$\Gamma_{im} = 0$表示教学方法 $m$对知识点 $i$被场景规则或人工规则禁用。

方法可行性约束为：

$$x_{im} \leq \Gamma_{im},\forall i\mathcal{\in I,}\ m\mathcal{\in M}$$

该约束仅处理硬性禁用规则，例如教学场景不支持实验法、教学平台不支持在线自适应学习，或人工明确禁用某类方法等。

为保证模型可行，要求每个知识点至少存在一种可用方法：

$$\sum_{m\mathcal{\in M}}^{}\Gamma_{im} \geq 1,\forall i\mathcal{\in I}$$

该条件属于实例有效性条件，而非优化决策约束。

## 2.9. 多目标表达与实际标量目标

用分配矩阵 $x$ 统一记号。模型同时考虑教学效果、学生偏好、教师偏好、全局方法分布和软违反：

$$\max_x\left(F_E(x),F_S(x),F_T(x),F_G(x),-V_{\mathrm{soft}}(x)\right).$$

约束为 $\sum_m x_{im}=1$、$x_{im}\le\Gamma_{im}$、$x_{im}\in\{0,1\}$。局部目标可逐点计算，全局分布项与软惩罚依赖整体方法使用比例，因此逐点局部最优不等于完整目标最优。

截至本版实际训练与比较均使用固定加权标量化，未求取完整 Pareto 前沿：

$$J(x)=0.30F_E(x)+0.25F_S(x)+0.25F_T(x)+0.10F_G(x)-0.10V_{\mathrm{soft}}(x).$$

其中 $V_{\mathrm{soft}}=V_{\mathrm{div}}+V_{\mathrm{cap}}$，$H_{\min}=0.55$、方法占比阈值均为 0.35。正向项权重合计 0.90，不能将 J 直接解释为百分制成绩或相对全局最优的比例。

参考方法分布固定为：

$$\rho=(0.18,0.12,0.10,0.10,0.10,0.08,0.08,0.08,0.05,0.05,0.06).$$

即时奖励为 $r_t=J(x_{t+1})-J(x_t)$。Effect-only 初解不构成 $F_E$ 下限约束，后续优化可以用效果分项的下降换取其他分项收益。当前方案不属于词典序优化，也未验证任意新权重或新目标分布下无需重训的适应能力。评分依据为规则参数与合成画像，不等同于实测教学效果。

# 3. RL 设计

## 3.1. 状态

### 3.1.1. 单 instance

```text
assignment: 当前每个知识点分配的方法，shape = [n]
category_id: 知识点类别编号，shape = [n]
cognitive_load: 知识点认知负荷，shape = [n]
concept_need: 知识点派生需求向量，shape = [n, 6]
method_attr: 教学方法属性矩阵，shape = [M, 6]
effect_matrix: 教学效果匹配矩阵，shape = [n, M]
student_pref_matrix: 学生偏好矩阵，shape = [n, M]
teacher_pref_matrix: 教师偏好矩阵，shape = [n, M]
feasible_mask: 方法可行性掩码，shape = [n, M]
method_distribution: 当前方法使用比例 π(a)，shape = [M]
target_distribution: 目标方法分布 ρ，shape = [M]
current_scores: 当前解各项得分，shape = [6]，current_scores = [F_E, F_S, F_T, F_G, V_soft, J]
```

### 3.1.2. Batch 训练

```text
category_id: [B, N_max]
cognitive_load: [B, N_max]
concept_need: [B, N_max, 6]
node_mask: [B, N_max]
assignment: [B, N_max]
method_attr: [M, 6]
effect_matrix: [B, N_max, M]
student_pref: [B, N_max, M]
teacher_pref: [B, N_max, M]
feasible_mask: [B, N_max, M]
method_dist: [B, M]
target_dist: [B, M]
current_scores: [B, 6]
```

注：node_mask 只用于 **batch padding**。

因为不同 instance 的知识点数量 $n$不同。为了组成 batch，需要把所有 instance padding 到同一个长度 $N_{\max}$。

定义：

$$\mathrm{node\_mask}_{bi} = \left\{ \begin{matrix}
1, & i < n_{b} \\
0, & i \geq n_{b}
\end{matrix} \right.$$

其中 $n_{b}$是 batch 中第 $b$个 instance 的真实知识点数量。

作用有三个：

- masked mean pooling 时不把 padding 节点算进去；

- actor 输出动作时不允许选择 padding 节点；

- 计算 loss 或统计指标时忽略 padding 节点。

单个 instance 运行时：node_mask = 全 1 向量，shape = [n]

Batch 运行时：node_mask: [B, N_max]

动态动作掩码应写成：

$$\Omega_{bim}(a) = \mathrm{node\_mask}_{bi} \land \Gamma_{bim} \land \mathbb{I}\lbrack m \neq a_{bi}\rbrack$$

## 3.2. 动作

单点替换 $a = (i,m')$，表示把知识点 $i$的方法替换为 $m'$。

动作空间大小 $N \times M$，通过 mask 禁止：不可行方法；当前已经使用的方法；违反硬约束的方法。

## 3.3. Reward

即时增量奖励 $r_{t} = J(x_{t + 1}) - J(x_{t})$

工程细节：
- **保留 best-so-far 解**：即使某一步奖励为负，也可以允许执行，用来跳出局部最优，但最终返回本 episode 中最好的解。
- **奖励归一化**：所有目标用平均值，不用总和，避免 $N$变化导致 reward scale 不稳定。

## 3.4. 终止条件

动作步数达到上限 $T_{\max}$，或连续若干步未超过历史最好值时，当前回合结束。正式推理使用 max_steps=128、patience=32，最终返回 best-so-far 解，即：

```python
if new_score > best_score + eps:
    best_score = new_score
    best_assignment = assignment.copy()
    no_improve_steps = 0
else:
    no_improve_steps += 1

done = (step_count >= max_steps) or (no_improve_steps >= patience)
```

## 3.5. 网络架构



### 3.5.1. 网络中的特征类型区分

| **类型**           | **名称**         | **符号**                         | **是否原始输入** | **是否随 assignment 改变** | **说明**                       |
|--------------------|------------------|----------------------------------|------------------|----------------------------|--------------------------------|
| 节点原始特征       | 类别             | $c_i$                            | 是               | 否                         | 来自前置算例                   |
| 节点原始特征       | 认知负荷         | $q_i$                            | 是               | 否                         | 只保留原算例 CognitiveLoad     |
| 节点派生特征       | 知识点需求向量   | $\mathbf{u}_i$                   | 否               | 否                         | 由 $(c_i,q_i)$ 派生            |
| 方法原始/配置特征  | 方法属性向量     | $\mathbf{r}_m$                   | 是               | 否                         | 人工定义或配置文件给出         |
| 静态评分矩阵       | 教学效果         | $E_{im}$                         | 派生             | 否                         | 由方法—类别矩阵得到            |
| 静态评分矩阵       | 学生偏好         | $P^S_{im}$                       | 派生             | 否                         | 由偏好函数得到                 |
| 静态评分矩阵       | 教师偏好         | $P^T_{im}$                       | 派生             | 否                         | 由偏好函数得到                 |
| 动态状态           | 当前分配         | $a_i$                            | 是               | 是                         | RL 状态核心                    |
| 动态状态           | 当前方法分布     | $\boldsymbol{\pi}(a)$            | 派生             | 是                         | 由 assignment 统计得到         |
| embedding          | 类别 embedding   | $\operatorname{Emb}_C(c_i)$      | 否               | 否                         | 网络学习得到                   |
| embedding          | 方法 ID embedding | $\operatorname{Emb}_M(m)$        | 否               | 否                         | 网络学习得到                   |
| 局部状态表示       | 知识点状态表示   | $h_i$                            | 否               | 是                         | Node Encoder 输出              |
| 方法表示           | 方法语义表示     | $\mu_m$                          | 否               | 否                         | Method Encoder 输出            |
| 全局状态表示       | 整体方案表示     | $g$                              | 否               | 是                         | Global Encoder 输出            |
| 候选动作特征       | pair feature     | $\xi_{im}$                       | 否               | 是                         | 候选动作 $((i,m))$ 的特征      |

### 3.5.2. 数据流

节点线：(c_i, q_i) → concept_need u_i → **Node Encoder** → h_i

方法线：(method_id, r_m) → **Method Encoder** → μ_m

全局线：{h_i} + 当前方法分布 π(a) + 目标分布 ρ + 当前得分 → **Global Encoder** → g

动作线：(h_i, μ_m, g, E_im, P_im^S, P_im^T, 各类 delta) → **Pair Feature** → **Actor** → logit_{im}

### 3.5.3. 原始输入

对 batch 训练，设：

- $B$：batch size；

- $N_{\max}$：batch 内最大知识点数量；

- $m = 11$：方法数；

- $K = 6$：类别数。

节点相关：

```text
category_id: [B, N_max]
cognitive_load: [B, N_max]
concept_need: [B, N_max, 6]
node_mask: [B, N_max]
```

方法相关：

```text
method_id: [M]
method_attr: [M, 6]
```

状态相关：

```text
assignment: [B, N_max]
effect_matrix: [B, N_max, M]
student_pref: [B, N_max, M]
teacher_pref: [B, N_max, M]
feasible_mask: [B, N_max, M]
method_dist: [B, M]
target_dist: [B, M]
current_scores: [B, 6] # F_E, F_S, F_T, F_G, V_soft, J
```

### 3.5.4. 关键模块

#### 3.5.4.1. Node Encoder-知识点状态编码器

作用/功能：确定在当前 assignment 下，知识点 $i$处于什么局部状态。

需要知道：

1. 这个知识点是什么类别；

2. 这个知识点的教学需求是什么；

3. 当前给它分配了什么方法；

4. 当前这个分配在效果、学生偏好、教师偏好上表现如何。

**（1）输入**

对知识点 $i$，构造：

$$\mathbf{x}_{i}^{N} = \lbrack{Emb}_{C}(c_{i}),\mathbf{u}_{i},{Emb}_{M}(a_{i}),E_{i,a_{i}},P_{i,a_{i}}^{S},P_{i,a_{i}}^{T}\rbrack$$

其中：

$$
\begin{aligned}
{Emb}_{C}(c_{i}) &\in \mathbb{R}^{d_{c}} \\
\mathbf{u}_{i} &\in \mathbb{R}^{6} \\
{Emb}_{M}(a_{i}) &\in \mathbb{R}^{d_{m}} \\
E_{i,a_{i}},P_{i,a_{i}}^{S},P_{i,a_{i}}^{T} &\in \mathbb{R}
\end{aligned}
$$

若取：$d_{c} = 8,d_{m} = 8$，则：$\dim\left( \mathbf{x}_{i}^{N} \right) = 8 + 6 + 8 + 3 = 25$

**（2）输出**

$$h_{i} = \phi_{N}(\mathbf{x}_{i}^{N}) \in \mathbb{R}^{H}$$

取 $H = 64$，则 batch shape：node_repr h: [B, N_max, H]

#### 3.5.4.2. Method Encoder-方法编码器

作用/功能：确定教学方法 $m$本身具有什么教学活动特征，与当前 assignment 无关，是静态方法表示。

**（1）输入**

对每个方法 $m$，构造：

$$\mathbf{x}_{m}^{M} = \lbrack{Emb}_{M}(m),\mathbf{r}_{m}\rbrack$$

其中：

$$
\begin{aligned}
{Emb}_{M}(m) &\in \mathbb{R}^{d_{m}} \\
\mathbf{r}_{m} &\in \mathbb{R}^{6}
\end{aligned}
$$

若：$d_{m} = 8$，则：$\dim(\mathbf{x}_{m}^{M}) = 8 + 6 = 14$

**（2）输出**

$$\mu_{m} = \phi_{M}(\mathbf{x}_{m}^{M}) \in \mathbb{R}^{H}$$

batch 中方法表示可以共享：method_repr μ: [M, H]

扩展到 batch：method_repr μ: [B, M, H]

#### 3.5.4.3. Global Encoder-全局状态编码器

作用/功能：确定当前整个分配方案处于什么状态，需要表达全局耦合信息，尤其是方法分布、多样性、均衡性和当前总分。

**（1）输入**

**①节点池化表示**

先对所有知识点状态表示做 masked mean pooling：

$$\bar{h} = \frac{\sum_{i = 1}^{N_{\max}}\mathrm{node\_mask}_{i}h_{i}}{\sum_{i = 1}^{N_{\max}}\mathrm{node\_mask}_{i} + \varepsilon}$$

其中：$\bar{h} \in \mathbb{R}^{H}$

②全局分布和得分

当前方法分布：$\mathbf{\pi}(a) \in \mathbb{R}^{M}$，目标方法分布：$\mathbf{\rho} \in \mathbb{R}^{M}$，

当前得分：$\mathbf{s}(a) = \lbrack F_{E},F_{S},F_{T},F_{G},V_{\text{soft}},J\rbrack \in \mathbb{R}^{6}$

③合并得到全局输入向量：

$$\mathbf{x}^{G} = \lbrack\bar{h},\mathbf{\pi}(a),\mathbf{\rho},\mathbf{\pi}(a) - \mathbf{\rho},\mathbf{s}(a)\rbrack$$

维度：

$$\dim(\mathbf{x}^{G}) = H + 3M + 6$$

当：$H = 64,M = 11$，则：$\dim(\mathbf{x}^{G}) = 64 + 33 + 6 = 103$

**（2）输出：**

$$g = \phi_{G}(\mathbf{x}^{G}) \in \mathbb{R}^{H}$$

shape：global_repr g: [B, H]

#### 3.5.4.4. Pair Feature Builder候选动作特征构造器

作用/功能：判断如果把知识点 $i$的当前方法 $a_{i}$替换成方法 $m$，这个动作从局部和全局看是否值得执行。

**（1）输入**

必须同时包含：

1. 当前知识点状态 $h_{i}$；

2. 候选方法表示 $\mu_{m}$；

3. 当前全局状态 $g$；

4. 该候选方法的局部得分；

5. 从当前方法换到候选方法的增量；

6. 对全局分布和软约束的影响。

候选动作：

$$\left( i,m \right)$$

表示将知识点 $i$的当前方法从 $a_{i}$替换为 $m$。

对每个 $\left( i,m \right)$，构造局部增量：

$$
\begin{aligned}
\Delta E_{im} &= E_{im} - E_{i,a_{i}} \\
\Delta P_{im}^{S} &= P_{im}^{S} - P_{i,a_{i}}^{S} \\
\Delta P_{im}^{T} &= P_{im}^{T} - P_{i,a_{i}}^{T}
\end{aligned}
$$

设执行动作 $\left( i,m \right)$后的新方法分布为：

$$\mathbf{\pi}'(a,i,m) = \mathbf{\pi}(a) + \frac{1}{n}(\mathbf{e}_{m} - \mathbf{e}_{a_{i}})$$

则：

$$\Delta F_{G}(i,m) = F_{G}\left( \mathbf{\pi}'(a,i,m) \right) - F_{G}\left( \mathbf{\pi}(a) \right)$$

$$\Delta V_{\text{soft}}(i,m) = V_{\text{soft}}(\mathbf{\pi}'(a,i,m)) - V_{\text{soft}}(\mathbf{\pi}(a))$$

**（2）输出**

候选动作特征pair feature 定义为：

$$\xi_{im} = \lbrack h_{i},\mu_{m},g,E_{im},P_{im}^{S},P_{im}^{T},\Delta E_{im},\Delta P_{im}^{S},\Delta P_{im}^{T},\Delta F_{G}(i,m),\Delta V_{\text{soft}}(i,m),\mathbb{I}\lbrack m = a_{i}\rbrack\rbrack$$

这里的 $\mathbb{I}\lbrack m = a_{i}\rbrack$只是告诉网络“这是不是当前方法”，但实际中该动作会被 action mask 屏蔽

维度：

$$\dim(\xi_{im}) = 3H + 9$$

当 $H = 64$时：

$$\dim\left( \xi_{im} \right) = 201$$

shape：pair_feature: [B, N_max, M, 3H + 9]

#### 3.5.4.5. Actor Head

作用/功能：给每个候选替换动作 $\left( i,m \right)$打分

$$\mathcal{l}_{im} = \phi_{A}(\xi_{im})$$

输出：action_logits: [B, N_max, M]

然后应用 action mask：

$${\widetilde{\mathcal{l}}}_{im} = \left\{ \begin{matrix}
\mathcal{l}_{im}, & \Omega_{im}(a) = 1 \\
 - \infty, & \Omega_{im}(a) = 0
\end{matrix} \right.$$

代码：masked_logits = action_logits.masked_fill(~action_mask, -1e9)

其中action mask动态动作掩码，由 node_mask、feasible_mask 和当前 assignment 共同生成

公式：

$$\Omega_{im}(a) = \Gamma_{im} \cdot \mathbb{I}\lbrack m \neq a_{i}\rbrack$$

batch 下：

$$\Omega_{bim}(a) = \mathrm{node\_mask}_{bi} \land \Gamma_{bim} \land \mathbb{I}\lbrack m \neq a_{bi}\rbrack$$

#### 3.5.4.6. Critic Head

作用/功能：只需要估计当前整体状态价值

Critic 只使用全局状态表示：

$$V(s) = \phi_{V}(g)$$

输出：state_value: [B, 1]

### 3.5.5. 训练框架：Actor-Critic

### 3.5.6. 策略更新：PPO

采用 GAE 与 clipped PPO。正式训练使用多环境同步采样与联合更新：每次选择同规模桶中 4 个不同实例，尽量覆盖不同拓扑；每环境采样 128 步，分别计算并标准化 GAE，合并 512 条 transition 后更新。不同节点数通过 padding 和 node mask 处理。

| 参数 | 五拓扑正式训练值 |
|---|---:|
| rollout mode / instances per update | joint / 4 |
| rollout steps / minibatch | 128 / 64 |
| PPO epochs / learning rate | 4 / 3e-4 |
| entropy coefficient / hidden dimension | 0.01 / 64 |
| gamma / GAE lambda | 0.99 / 0.95 |
| clip epsilon / value coefficient | 0.20 / 0.50 |
| max grad norm / target KL | 0.50 / 关闭 |
| 训练种子 / 每种子更新数 | 0、1、2 / 1500 |
| 验证频率 / 固定样本数 | 每 25 updates / 90 |

训练随机采样动作，评估采用确定性动作选择。rollout 可跨环境回合，update 不能直接等同于 episode。best.pt 按固定验证分数保存，用于部署和比较；latest.pt 保存完整恢复状态。TensorBoard、CSV、checkpoint 索引和运行元数据共同记录训练过程。

网络为节点/方法/全局 MLP 编码与候选动作打分，当前不直接编码原始网络邻接关系；不同拓扑实验反映内容结构与属性分布差异，不等同于对未见图拓扑类型的泛化。

# 4. 算例构造

## 4.1. assignment instance 字段

```python
assignment_instance = {
    "instance_name": str,
    "source_sm": str,
    "selected_ids": list[int],
    "N": int,
    "M": 11,
    "K": 6,
    "category_names": ["基础", "理论", "算法", "实验", "案例", "扩展"],
    "method_names": [...],
    "category_id": np.ndarray,        # [N]
    "cognitive_load": np.ndarray,     # [N]
    "concept_need": np.ndarray,       # [N, 6]，派生，不是新增原始属性
    "method_attr": np.ndarray,        # [M, 6]
    "effect_matrix": np.ndarray,      # [N, M]
    "student_profile": np.ndarray,    # [6]
    "teacher_profile": np.ndarray,    # [6]
    "student_pref": np.ndarray,       # [N, M]
    "teacher_pref": np.ndarray,       # [N, M]
    "target_distribution": np.ndarray,# [M]
    "feasible_mask": np.ndarray,      # [N, M]
    "weights": dict
}
```

## 4.2. adapter 逻辑

1. 解析原算例文件获取 category_id 和 cognitive_load

2. 读取原算例上一步求解结果，获取选中节点 id 并重新从 0 开始编码

3. 构造节点原始属性和方法原始属性

4. 构造学生/教师场景：训练从六类学生模板与六类教师模板的 30 个允许组合中在线采样并连续扰动，评估使用固定场景库；另有 6 个组合仅用于 held-out 测试。

5. 构造 $E_{im}$

6. 构造 $P^{S},P^{T}$

```python
z_im = build_pair_feature_vector(concept_need[i], method_attr[m], q[i, 1])
student_pref[i, m] = preference_score(z_im, student_profile)
teacher_pref[i, m] = preference_score(z_im, teacher_profile)
```

7. 构造 feasible_mask

默认为 `np.ones((N, M), dtype=bool)`，如果禁用某类则对应位置为 `false`。

8. 构造 target_distribution

## 4.3. 五拓扑数据集与泄漏控制

原始数据含 uniform、random、hourglass、bottleneck、staircase 五类拓扑，每类九档规模、每档 100 个实例，总计 4500。原始 V=60–1000；经内容选择后的实际 N=37–775。manual_bottleneck 只作为输入目录名，元数据规范为 bottleneck。以 topology/group/instance 分层输出，防止同名覆盖。

按 group 与原始随机种子形成底层身份，使用 seed20260906 作 70/15/15 划分，五拓扑变体同属一组：训练3150、验证675、测试675，对应底层身份630/135/135。每拓扑—规模单元为70/15/15。

学生/教师模板各6类；留出组合为 S1–T4、S2–T5、S3–T6、S4–T1、S5–T2、S6–T3。扰动标准差0.05，最大绝对扰动0.10，画像裁剪到[0.05,0.95]。固定验证插值库30个场景，测试插值库60个，测试留出库30个。按实例索引循环配对；每套测试含675个配对样本，而非内容与全部场景的笛卡尔积。

实例 JSON 的类别、需求和偏好矩阵保留来源与原始ID。当前4500个实例的静态可用性掩码均为全真，动态掩码仍排除原方法和padding；复杂禁用关系下的鲁棒性尚未在本批主实验中检验。

# 5. 已实现的程序与流程

| 文件或目录 | 截止时点的作用 |
|---|---|
| data/parse_sm.py、selection_loader.py | 原始属性解析与内容选择结果接入 |
| data/build_assignment_instance.py、convert_pre_data.py | assignment实例与manifest构造 |
| data/build_dataset.py、scenarios.py | 内容划分、偏好场景采样与固定场景库 |
| utils/scoring.py、masks.py | 统一目标分项、软惩罚和动作掩码 |
| envs/method_assignment_env.py | 单点替换、ΔJ奖励、终止与best-so-far |
| models/encoders.py、actor_critic.py | 共享编码、Masked Actor–Critic |
| solvers/ppo_solver.py | rollout、GAE、PPO更新与策略求解 |
| experiments/formal_training.py | 联合训练、固定验证、日志和checkpoint |
| experiments/select_ppo_candidates.py、freeze_ppo_parameters.py | 受控参数比较和多种子冻结门槛 |
| experiments/evaluate_checkpoint.py、summarize_multiseed_evaluation.py | 冻结后测试及跨种子统计 |
| solvers/greedy_solver.py、local_search_solver.py | 构造式基线与增量单点局部搜索 |
| solvers/one_point_local_search.py | 现有单点最优改进搜索的规范别名，不是新增算法 |
| solvers/gurobi_exact_solver.py、hybrid_solver.py | 精确/有界参照与PPO后短程精炼 |
| experiments/compare_exact_hybrid.py、remeasure_gurobi_runtime.py | validation pilot与Gurobi计时补正 |
| experiments/prepare_publication_results.py | 既有图件、结果表与源数据整理 |

表内代码入口相对 pa_moap_rl。当前工作区代码已有截止日期以后的变更，本表仅记录当时已存在的模块职责；历史参数以run_metadata和当时实验文件为准，不直接采用今天CLI默认值重新解释旧实验。

# 6. 已执行的比较与评价协议

## 6.1. 五算法测试

主表包含随机可行、Effect-only greedy、标量独立贪婪、单点最优改进局部搜索与Masked PPO。标量贪婪仅最大化局部0.30E+0.25P_S+0.25P_T；局部搜索从该初解出发优化完整J，默认最多N×M次接受迭代，无正收益单点替换即停止。PPO从Effect-only初解出发，采用验证选出的seed1/update1450。

参数冻结后，interpolation和held-out各评价675个相同实例—场景样本；三种子PPO也严格配对。随机重启和first-improvement不是主表中的独立算法。评价包括J、各分项、软惩罚、硬可行率、非法动作和运行时间。不同初解和不同计算预算构成完整求解流程比较，尚不是公平控制初解的算法消融。

## 6.2. 精确求解和混合精炼

固定抽取5拓扑×9规模中的中位N验证实例，共45个，N=40–736。Gurobi首轮120秒，仅未证最优且gap>1%时追加600秒。混合方法在PPO解上追加4/8/16/32次接受的单点改进。Gurobi用整数方法计数的PWL表达熵，并由项目评分器重算结果；未证最优时应报告上下界，不能把incumbent称作全局最优。

混合预算仅用validation选择：恢复至少50%的PPO—完整局部搜索差距、总时间不超过局部搜索20%，选择满足条件的最小预算。本次无候选同时达标。

## 6.3. 统计与时间口径

训练结果报告三种子，部署模型只按validation选。跨种子SD使用ddof=0；历史bootstrap区间按实例重采样，未校正底层身份和重复场景的依赖，不等同于教学效果证据。

Gurobi初版只记录optimize时间，晚间已补测构建加求解调用时间；使用fair_runtime修正版。该计时仍不含共享实例读取/场景构造，不是完整服务时延。五算法675例测试与45例pilot的时间必须分别报告。PPO推理不包含训练和模型加载成本，不宣称对所有算法有速度优势。

## 6.4. 未完成的验证

偏好输入、全局分布项和初解机制的正式公平消融；权重及目标分布变化的泛化；动态偏好变化后的最小调整；批量服务吞吐和训练摊销；真实教学效果验证。已有脚本入口不等于相关实验完成。

# 7. 阶段进展与训练结果

已完成数据工程、联合采样实现与受控比较、参数搜索、多种子长程训练、冻结后测试和精确求解pilot。取消早期Day1–Day14排期的“待执行”表述，用已有证据记录实际进度。

joint对比sequential的历史100-update受控试验中，吞吐约为2.04倍，update500配对验证提高约0.00667；该结果是工程采用依据，不能替代从零多种子训练。

| 候选 | 质量分 S | 最佳验证 J | KL P90 | 最弱拓扑差 | 资格 |
|---|---|---|---|---|---|
| B | 0.654349 | 0.662251 | 0.04420 | 0.000000 | 通过 |
| C1 | 0.657075 | 0.658264 | 0.04085 | -0.006885 | 未通过 |
| C2 | 0.655388 | 0.657917 | 0.03650 | -0.006810 | 未通过 |
| C3 | 0.650257 | 0.658985 | 0.03601 | -0.006830 | 未通过 |
| C4 | 0.652867 | 0.657824 | 0.04529 | -0.007834 | 未通过 |

B为3e-4/4epochs，C1为3epochs，C2为2epochs，C3为2e-4，C4为target-KL0.03；候选比较300updates，未通过质量/拓扑门槛后保留B。三种子通过1000-update门槛，满足延长条件后统一训练到1500。

| 种子 | 最佳 update | 最佳验证 J | 最终验证 J | 更新计算/分钟 | 验证推理/分钟 |
|---|---|---|---|---|---|
| 0 | 1150 | 0.669582 | 0.663991 | 48.46 | 36.77 |
| 1 | 1450 | 0.671094 | 0.664512 | 32.67 | 23.79 |
| 2 | 875 | 0.668755 | 0.661607 | 31.65 | 24.98 |

每种子768000条transition，合计2304000条。有效日志更新计时与验证实例推理计时合计约3.31小时，不含参数搜索、启动、保存和恢复等完整墙钟成本。正式日志硬可行率100%、非法动作为0；训练存在回撤，部署使用best.pt。seed1曾发生中断及错误默认参数恢复，相关无效分支已隔离，不纳入上述统计。

# 8. 初步结果与局限

## 8.1. 冻结后的675例测试

| 算法 | 插值 J | 留出组合 J | 插值平均时间/s | 留出平均时间/s |
|---|---|---|---|---|
| 随机可行 | 0.603897 | 0.604979 | 0.002533 | 0.002376 |
| Effect-only | 0.619974 | 0.620868 | 0.000024 | 0.000023 |
| 标量贪婪 | 0.655647 | 0.656164 | 0.000029 | 0.000029 |
| PPO seed1 | 0.671166 | 0.671739 | 0.331693 | 0.326528 |
| 单点局部搜索 | 0.687444 | 0.688414 | 22.144741 | 22.224597 |

PPO对标量贪婪平均提高0.015519/0.015575，对单点局部搜索低0.016278/0.016675。相对当前Python局部搜索的平均求解时间比约为66.8/68.1；它只支持此版本实现下的初步质量—时间折中。

## 8.2. 三种子与画像组合

| 场景 | 三种子 J 均值 ± SD | staircase J 均值 ± SD | 硬可行率 / 非法动作 |
|---|---|---|---|
| interpolation | 0.669342 ± 0.001292 | 0.613729 ± 0.002150 | 100% / 0 |
| heldout | 0.670409 ± 0.000944 | 0.616328 ± 0.002549 | 100% / 0 |

held-out指已知学生/教师模板的未见组合，五种拓扑均在训练中出现过。没有验证任意新画像分布、新权重、新目标分布和全新拓扑类型。

## 8.3. 精确与混合对照

| 方法 | 平均 J | 绝对最优差 | 相对最优差/% | 平均时间/s | 中位时间/s |
|---|---|---|---|---|---|
| PPO | 0.673091 | 0.018331 | 2.659 | 1.091 | 1.165 |
| PPO+4 | 0.674827 | 0.016595 | 2.406 | 1.500 | 1.473 |
| PPO+8 | 0.676073 | 0.015349 | 2.225 | 1.919 | 1.719 |
| PPO+16 | 0.677772 | 0.013650 | 1.976 | 2.655 | 2.174 |
| PPO+32 | 0.680054 | 0.011368 | 1.642 | 4.098 | 2.744 |
| 单点局部搜索 | 0.690365 | 0.001057 | 0.150 | 21.604 | 4.124 |
| Gurobi | 0.691422 | 0.000000 | 0.000 | 1.869 | 0.327 |

45个Gurobi算例全部首轮证得最优，无600秒追加，重算误差最大3.79e-14。修正计时后，Gurobi在30/45个实例快于PPO、45/45个快于当前Python单点局部搜索。PPO+32恢复40.31%的局部搜索增益、耗时占18.97%，未达50%质量门槛。不能宣称PPO全面优于精确求解。

## 8.4. 后续改进方向

由当前数据出发，优先分解质量差距与规模/拓扑/目标分项的关系，诊断推理步数、动作重复与停止条件；用控制变量实验检验偏好和全局分布项的作用；进一步统一初始化、加载、构建和求解的计时边界。更大邻域、不同混合预算与摊销部署价值均属于待验证方向，不能预写正面结论。本版保留staircase，不纳入截止以后对训练范围和目标权重的更改。

# 9. 研究动机与可用表述

研究问题在于协调理论教学匹配、双主体偏好与整体方法组合。规则目标包含非加性全局项，构造式局部贪婪未必充分；从效果匹配初解出发学习局部替换是一条已实现、可验证的求解路径。采用PPO的价值需要由质量、成本、适用场景与消融证据具体支持，不能仅从解空间大或偏好存在就推导其优越性。

建议表述：“本研究实现了以教学效果贪婪初解为起点的共享Masked PPO方法，完成五拓扑、多规模与合成偏好场景下的训练和初步比较。PPO改善了简单基线的综合目标，在当前实现下较完整单点局部搜索节省推理时间；精确求解对照揭示了剩余最优差距及方法适用性问题，后续将据此完善算法和实验。”

不将该方法称为“先保证效果再优化偏好”的词典序或约束优化，不将单点局部最优当作全局最优，不将规则评分提高当作实测教学收益，不将后续计划记为已完成实验。

## 9.1. 可追溯依据

历史依据：data_processed/splits、三种子results/formal_joint_seed*_run01、results/final_evaluation、results/publication_ready_20260908、results/exact_hybrid_validation_pilot/fair_runtime和summary_fair_runtime，以及docs/ppo_parameter_search_and_final_evaluation_20260908.md。

9月8日晚间pilot没有进一步使用测试集，但此前675×2的冻结后测试已经完成，不能说“所有测试仍封存”。本次整理不重跑训练，不回滚当前代码，也不将9月22日状态当成9月8日代码快照。
