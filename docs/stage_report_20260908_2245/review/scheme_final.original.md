核心问题是“为已选择知识点分配教学方法”，并采用 PA-MOAP、启发式初解、Actor-Critic/PPO 的技术路线

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

教学方法分配任务以前置资源组合阶段输出的已选知识点集合为输入。前置阶段已经完成知识点筛选，并保证所选知识点集合满足先修约束和资源约束。因此，本阶段不再重复考虑教学时长、认知负荷和外部资源供给等资源容量约束，而是在已选知识点集合上进一步决定每个知识点应采用的教学方法。

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

（3）对任一主体 $b\mathcal{\in B}$，定义局部偏好偏好得分：

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

## 2.9. 多目标优化模型

综合上述目标与软约束，教学方法分配问题可表述为如下偏好感知的多目标分配优化模型：

$$\underset{y}{\max}\mathbf{F}(x) = \left( F_{E}(x),F_{S}(x),F_{T}(x),F_{G}(y),F_{H}(x),{- V}_{C}(y) \right)$$

s.t.

$$\sum_{m\mathcal{\in M}}^{}x_{im} = 1,\forall i\mathcal{\in I}$$

$$x_{im} \leq \Gamma_{im},\forall i\mathcal{\in I,}\ m\mathcal{\in M}$$

$$x_{im} \in \{ 0,1\},\forall i\mathcal{\in I,}\ m\mathcal{\in M}$$

由于上述目标之间可能存在冲突，本问题通常不存在一个同时使所有目标达到最优的单一解。

首先，教学效果目标 $F_{E}(x)$主要反映基于知识点类别和教学方法理论适配关系的规范性评价，而学生偏好目标 $F_{S}(x)$与教师偏好目标 $F_{T}(x)$则反映不同主体的个性化需求，三者可能在局部知识点—方法匹配上产生冲突。其次，$F_{E}(x)$、$F_{S}(x)$ 和 $F_{T}(x)$均为局部加和目标，而 $F_{G}(x)$、$F_{H}(x)$ 和 $F_{C}(x)$依赖整体方法分布，体现跨知识点的全局组合结构要求。因此，逐点选择局部得分最高的方法并不一定能得到全局上合理的教学方法分配方案。

虽然理论模型采用多目标形式，但在启发式搜索或强化学习求解过程中，仍需要将多目标向量转化为可比较的标量评价值。因此，工程实现阶段采用加权标量化形式：

$$J(x;\mathbf{w}) = w_{E}F_{E}(x) + w_{S}F_{S}(x) + w_{T}F_{T}(x) + w_{G}F_{G}(x) + w_{H}F_{H}(x) + w_{C}F_{C}(x)$$

其中：

$$w_{E},w_{S},w_{T},w_{G},w_{H},w_{C} \geq 0$$

并可设置：

$$w_{E} + w_{S} + w_{T} + w_{G} + w_{H} + w_{C} = 1$$

强化学习环境中的即时奖励可定义为标量化目标的增量：

$$r_{t} = J(y_{t + 1};\mathbf{w}) - J(y_{t};\mathbf{w})$$

这样处理后，模型层面保留多目标优化问题的本质，算法层面则通过不同权重设置获得不同偏好下的折中解。

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

动作步数达到上限 $T_{\max}$ 或连续若干步有超过历史最好值时当前回合结束，即：

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

4. 构造学生/教师画像（固定、单一配置，后续实验考虑设置多组画像）

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

# 5. 代码实现

## 5.1. 项目结构

pa_moap_rl/

configs/

default.yaml

method_config.yaml

profile_config.yaml

docs/

model_definition.md

data/

parse_sm.py

selection_loader.py

from_content_selection.py

build_assignment_instance.py

instance_schema.py

envs/

method_assignment_env.py

models/

encoders.py

actor_critic.py

solvers/

random_solver.py

greedy_solver.py

local_search_solver.py

ppo_solver.py

utils/

scoring.py

masks.py

metrics.py

seed.py

logger.py

io.py

experiments/

train_ppo.py

run_batch.py

run_ablation.py

analyze_results.py

tests/

test_parse_sm.py

test_instance_builder.py

test_scoring.py

test_masks.py

test_env.py

test_encoders.py

test_actor_critic.py

results/

figures/

tables/

## 5.2. 具体程序文件实现

各脚本基本功能/职责如下：

| **文件**                       | **职责**                                                                   |
|--------------------------------|----------------------------------------------------------------------------|
| instance_schema.py             | 定义 instance 字段、shape、合法范围                                        |
| method_assignment_generator.py | 生成内容二专用算例                                                         |
| from_content_selection.py      | 从内容一输出的 selected concepts 构造内容二输入                            |
| loader.py                      | 读写 JSON/NPZ/YAML 格式 instance                                           |
| method_assignment_env.py       | 封装 reset/step/action_mask/reward/done/best-so-far成类似 Gymnasium 的环境 |
| encoders.py                    | 节点、方法、pair feature 编码模块                                          |
| actor_critic.py                | PPO 用 Actor-Critic 网络                                                   |
| scoring.py                     | 计算 (F_E,F_S,F_T,F_G,J)                                                   |
| masks.py                       | 生成 feasible mask 和 action mask                                          |
| metrics.py                     | 计算实验指标                                                               |
| random_solver.py               | 随机合法 baseline，做对比算法                                              |
| greedy_solver.py               | 见后文，做对比算法                                                         |
| local_search_solver.py         | 见后文，做对比算法                                                         |
| ppo_solver.py                  | masked PPO，本研究核心贡献算法                                             |
| run_batch.py                   | 批量生成实例、运行算法、保存结果                                           |
| run_ablation.py                | 权重、全局项、初解方式等消融                                               |
| analyze_results.py             | 汇总 csv，画图和表格                                                       |

### 5.2.1. greedy_solver.py——做对比算法

实现：

effect-only greedy：仅考虑理论匹配度 $E$

scalarized independent greedy：不考虑全局偏好

scalarized one-point greedy improvement：考虑所有偏好，从一个初解出发，每次选择使总目标增量最大的单点替换：$(i^{*},m^{*}) = \arg\max_{i,m}\Delta J(i,m)$。如果 $\Delta J(i^{*},m^{*}) > 0$，执行替换；否则停止。

### 5.2.2. env.py

将分配等交互过程封装成

obs = env.reset(instance)

obs, reward, done, info = env.step(action)

### 5.2.3. local_search_solver.py

best-improvement one-point local search

first-improvement one-point local search

random restart local search

### 5.2.4. ppo_solver.py

实现 masked PPO

# 6. 实验设计

## 6.1. 指标

| **类别** | **指标**                 | **含义**                                 |
|----------|--------------------------|------------------------------------------|
| 可行性   | hard_feasible_rate       | 输出方案是否满足硬约束                   |
|          | mask_violation_count     | 是否选择了被 mask 禁止的方法             |
|          | valid_action_ratio       | 每个状态下可行动作比例，辅助判断问题难度 |
| 目标函数 | total_score              | 总目标 (J(x))                            |
|          | effect_score             | 教学效果匹配 (F_E(x))                    |
|          | student_pref_score       | 学生偏好满足 (F_S(x))                    |
|          | teacher_pref_score       | 教师偏好满足 (F_T(x))                    |
|          | global_score             | 全局分布偏好满足 (F_G(x))                |
|          | soft_penalty             | 软约束违反惩罚                           |
| 优化过程 | improvement_over_initial | 相比启发式初解提升多少                   |
|          | improvement_over_greedy  | 相比 greedy 提升多少                     |
|          | steps_to_best            | 多少步找到 best-so-far                   |
|          | episode_return           | PPO episode 累计 reward                  |
|          | runtime                  | 运行时间                                 |

## 6.2. 对比 baseline

Random legal

Effect-only greedy

Scalarized greedy

Local search

PPO local improvement

## 6.3. 消融

建议三组：

**无偏好 vs 有偏好：**证明偏好项确实改变了分配结果。

**无全局分布项 vs 有全局分布项：**证明加入全局偏好后，问题不再是简单逐点贪婪。

**启发式初解 vs 随机初解：**支撑你的“启发式引导”设计。

## 6.4. 需要的结果图

1. PPO training reward curve

2. 不同方法的 total score 对比柱状图

3. effect score 与 preference score 的 trade-off 散点图

4. 方法使用比例与目标分布的对比图

# 7. 时间计划

| **时间** | **任务**                       | **具体 list**                                                                                                                                                                                                                                                        | **输出文件/脚本**                                                                  | **备注 / 验收要点**                                                                                                                                             |
|----------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Day 1    | 冻结优化模型与符号体系         | 明确已选知识点集合 (\mathcal I)、候选方法集合 (\mathcal M)、类别集合 (\mathcal C)；统一使用 (y\_{im}) 作为分配变量；区分 feasible mask (\Gamma\_{im}) 与 action mask (\Omega\_{im})；整理目标函数 (J(y))、软约束 (V\_{\mathrm{soft}}(y))                             | docs/model_definition.md                                                           | 这一版作为后续代码实现的唯一数学依据；避免后续脚本中出现 (x_i)、(S)、(A\_{im}) 等易混符号                                                                       |
| Day 2    | 固定配置文件                   | 写入 6 类知识点类别；写入 11 类教学方法；写入类别原型矩阵 (\mathbf D^C)；写入方法属性矩阵 (\mathbf R)；写入方法—类别教学效果矩阵 (\mathbf A^E)；写入默认学生画像 (\boldsymbol\theta^{stu})、教师画像 (\boldsymbol\theta^{tea})、目标分布 (\boldsymbol\rho)、权重参数 | configs/default.yaml；configs/method_config.yaml；configs/profile_config.yaml      | 配置文件中必须保证所有矩阵 shape 一致：(\mathbf D^C:[K_c,5])，(\mathbf R:[M,6])，(\mathbf A^E:[M,K_c])                                                    |
| Day 3    | 实现 .sm 算例解析器            | 解析原始算例文件；读取节点编号、类别、CognitiveLoad；必要时保留原节点 ID 与重新编码 ID 的映射；暂不使用 Importance、Timeliness、Measurability、TeachingTime、ExternalResourceDemand                                                                                  | data/parse_sm.py；tests/test_parse_sm.py                                           | 验收：输入 .sm 后能输出 category_id: [V]、cognitive_load: [V]、id_mapping                                                                                   |
| Day 4    | 对接前置资源组合结果           | 读取内容一算法输出的 selected_ids 或 x_star；从完整 .sm 中抽取已选知识点；将原节点编号重新映射为 (0,\dots,n-1)；保存 source instance 信息                                                                                                                            | data/from_content_selection.py；data/selection_loader.py                           | 若暂时没有标准化内容一输出文件，可先支持 JSON 格式 {"selected_ids":[...]}；后续再接 CAMGA 输出                                                                |
| Day 5    | 构造内容二 assignment instance | 根据 (c_i,q_i) 构造知识点需求向量 (\mathbf u_i)；根据 (\mathbf A^E) 构造 effect_matrix；根据偏好函数构造 student_pref、teacher_pref；构造 feasible_mask 与 target_distribution                                                                                       | data/build_assignment_instance.py；data/instance_schema.py                         | 输出字段应与当前方案一致：category_id、cognitive_load、concept_need、method_attr、effect_matrix、student_pref、teacher_pref、feasible_mask、target_distribution |
| Day 6    | 实现评分函数                   | 实现 (F_E(y))、(F_S(y))、(F_T(y))、(F_G(y))、(H(y))、(V\_{\mathrm{div}}(y))、(V\_{\mathrm{cap}}(y))、(V\_{\mathrm{soft}}(y))、(J(y))；实现单点替换后的 delta score 计算                                                                                              | utils/scoring.py；tests/test_scoring.py                                            | 验收：所有得分在合理范围；同一 assignment 重复计算一致；单点替换的 delta 与重算结果一致                                                                         |
| Day 7    | 实现 mask 与合法动作逻辑       | 实现 feasible mask (\Gamma\_{im})；实现动态 action mask (\Omega\_{im})；支持单 instance 和 batch 两种形式；处理当前方法不可重复选择，即 (m\ne a_i)                                                                                                                   | utils/masks.py；tests/test_masks.py                                                | 明确：feasible_mask 是静态方法可行性掩码；action_mask 是动态动作掩码；二者不能混用                                                                              |
| Day 8    | 封装 RL 环境                   | 实现 reset()、step(action)、get_action_mask()、compute_reward()、done 判断、best-so-far 记录；reward 定义为 (J(y\_{t+1})-J(y_t))                                                                                                                                     | envs/method_assignment_env.py；tests/test_env.py                                   | 验收：非法动作不能执行；执行合法动作后 assignment、method distribution、current scores 同步更新                                                                 |
| Day 9    | 实现 baseline                  | 实现 Random legal；Effect-only greedy；Scalarized independent greedy；One-point greedy improvement；Local search                                                                                                                                                     | solvers/random_solver.py；solvers/greedy_solver.py；solvers/local_search_solver.py | baseline 是后续证明问题非平凡性和 PPO 改进效果的基础，优先级高于复杂网络调参                                                                                    |
| Day 10   | 实现网络编码模块               | 实现 Category Embedding、Method ID Embedding、Node Encoder、Method Encoder、Global Encoder、Pair Feature Builder；统一输入输出 shape                                                                                                                                 | models/encoders.py；tests/test_encoders.py                                         | 验收：输入 batch 后输出 h:[B,N_max,H]、\mu:[B,M,H]、g:[B,H]、\xi:[B,N_max,M,3H+9]                                                               |
| Day 11   | 实现 Actor-Critic              | 实现 Actor Head 输出 action_logits:[B,N_max,M]；应用 action mask；实现 Critic Head 输出 state_value:[B,1]；检查 flatten 后动作索引与 ((i,m)) 的映射                                                                                                              | models/actor_critic.py；tests/test_actor_critic.py                                 | 先保证前向传播、mask、采样、log_prob、entropy 全部正确，再接 PPO                                                                                                |
| Day 12   | 跑通 masked PPO 最小闭环       | 实现 rollout buffer、GAE、clipped policy loss、value loss、entropy bonus、参数更新；先在小规模固定 instance 上训练                                                                                                                                                   | solvers/ppo_solver.py；experiments/train_ppo.py                                    | 验收目标不是立刻超过所有 baseline，而是训练过程稳定、reward 不爆炸、非法动作率为 0                                                                              |
| Day 13   | 批量实验与消融                 | 批量生成 assignment instances；运行 baseline 和 PPO；做无偏好/有偏好、无全局项/有全局项、启发式初解/随机初解消融；保存所有结果                                                                                                                                       | experiments/run_batch.py；experiments/run_ablation.py；results/\*.csv              | 建议规模先取 (n=20,50,100)，每组若时间允许跑多 seed；优先保证结果表完整                                                                                         |
| Day 14   | 结果分析与汇报材料整理         | 汇总 total score、effect score、student/teacher preference score、global score、soft penalty、runtime、improvement；绘制训练曲线、baseline 对比图、方法分布图、case table；整理当前问题与下一步                                                                      | experiments/analyze_results.py；figures/\*.png；tables/\*.csv；slides_or_report.md | 汇报主线：内容一输出如何进入内容二 → 偏好函数如何构造 → 非加性全局项为何必要 → baseline 与 PPO 初步结果                                                         |

# 8. 结果

## 8.1. 预期成果

第一版（当前）：效用函数偏好 + 全局分布偏好 + 启发式初始化 + masked PPO

把教学方法分配问题从概念建模推进到可运行的优化环境，完成了偏好评分函数、全局分布偏好、启发式初解、baseline 和 PPO 局部改进框架的初步实现，并通过合成算例验证了偏好项会改变分配结果，RL 框架可以在非加性全局偏好下进行方案改进。

## 8.2. 后续可能做的在第一版基础上的改进

### 8.2.1. 动作空间

①两点交换swap

在不改变方法使用总量的情况下调整分配结构，尤其是引入了全局分布/多样性要求的情况下

作用：改善 credit assignment，提供略强的跳出局部最优能力，改变搜索的邻域几何结构

②top-K重采样：选一小组知识点重新用 greedy/prob 选方法

③方法整体迁移（method reassignment）：把某一类知识点全部换方法，对应“教学策略调整”

### 8.2.2. 偏好

用 PbRL / 主动学习 / 贝叶斯偏好模型替换手工偏好函数

# 9. Motivation 相关

## 9.1. 优化目标冲突

### 9.1.1. 两个目标之间是否真的存在结构性冲突

两类目标在决策空间中呈现“局部冲突、全局相关”的特征，使得简单加权难以稳定获得高质量解。

| **结构性冲突来源** | **解释** |
|--------------------|----------|
| 认知收益 vs 使用成本 | 高匹配度方法：可能认知负荷高（如探究式学习）；偏好方法：可能更轻松（如讲授） |
| 全局 vs 局部 | 匹配度：逐点最优；偏好：可能有“组合结构”（例如希望方法多样性） |
| 规范性目标 vs 主观性目标 | 匹配度：基于理论/经验规则（normative）；偏好：个体差异（subjective） |

### 9.1.2. 为什么采用“先匹配度 → 再偏好”的分阶段策略【Two-stage preference-aware optimization】是合理的

（1）可行性与搜索空间角度

- 当前组合优化问题解空间巨大，如果直接优化偏好，preference 是 weak signal（偏好函数不稳定、甚至是噪声；RL early stage exploration 很差；容易学到低质量但高偏好解。）

- Feasible-region restriction + preference optimization：匹配度 $F_{1}$：是**强结构信息**，提供**搜索空间的“骨架”，**先用 $F_{1}$把搜索空间收缩到“合理区域”，再在这个子空间内优化 $F_{2}$。


（2）学习稳定性角度

初始解足够好，RL 做改进：将问题从“全局构造”转化为“局部改进”，显著降低 RL 训练难度；如果交换顺序：reward signal 噪声大、policy gradient 方差高、容易 collapse。

（3）认知与实际应用角度

在实际教学设计中，教学方法首先需满足基本的教学适配原则，其次才在可行范围内调整以满足个性化偏好。

（4）易扩展，可以插入学习偏好模型

### 9.1.3. 这种顺序 vs 其他处理方式（加权、同时优化、先偏好等）在方法论上的差异与优劣

加权单目标：权重确定，鲁棒性（偏好变化权重失效）

多目标优化：得到 Pareto 前沿还需要决策

交换顺序/纯RL/混合：容易产生认知不合理方案、偏好信号不可靠（特别是冷启动）、RL exploration 质量差

### 9.1.4. 理论支撑（硬蹭）

Lexicographic optimization 词典序优化~近似实现

Constrained reinforcement learning

### 9.1.5. 论文可用文字描述

本研究将偏好优化问题从全局搜索转化为受约束的局部改进问题，从而在保证教学合理性的前提下实现偏好驱动的个性化优化。

| **角度/核心逻辑** | **描述** |
|------------------|----------|
| 问题本质 | 教学方法分配需要同时满足：基于知识结构的理论适配性；面向个体差异的偏好满足 |
| 挑战 | 两类目标在局部决策层面存在冲突，同时偏好信息具有不完备性与不稳定性，使得直接联合优化困难。 |
| 现有方法不足 | 加权方法：依赖权重，鲁棒性差；多目标方法：难以决策；纯RL：搜索空间大、训练不稳定 |
| 我的核心思想 | 将问题分解为：“先保证合理性，再优化个性化” |
| 方法本质 | 在高匹配度子空间中进行偏好优化 |
