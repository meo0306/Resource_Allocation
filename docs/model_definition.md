# PA-MOAP 工程数学定义

本文档是 `docs/scheme_final.md` 的工程实现版摘要。后续代码以本文件和 YAML 配置为直接依据。

## 输入与派生特征

内容二阶段只为内容一已选知识点分配教学方法，不重复考虑先修约束、`TeachingTime`、`ExternalResourceDemand` 等容量约束。

- `.sm` 节点属性字段：`nodnr. <type> <p_1> <p_2> <p_3> <q_1> <q_2> <q_3>`。
- `type -> category_id`。
- `q_2 -> cognitive_load`。
- `concept_need` 是由 `category_id` 和 `cognitive_load` 派生的 `[N, 6]` 特征，不是新的原始节点属性。

对知识点 `i`：

```text
concept_need[i, :5] = D_C[category_id[i]]
concept_need[i, 5] = cognitive_load[i]
```

## 固定配置

第一版固定：

- 知识点类别数 `K = 6`。
- 教学方法数 `M = 11`。
- 类别原型矩阵 `D_C: [K, 5]`。
- 方法属性矩阵 `method_attr/R: [M, 6]`。
- 方法-类别教学效果矩阵 `A_E: [M, K]`。
- 学生画像、教师画像、目标方法分布、权重均从 `pa_moap_rl/configs/*.yaml` 读取。

## 偏好评分

对知识点 `i` 和教学方法 `m`，匹配对交互特征为：

```text
z_im[k] = concept_need[i, k] * method_attr[m, k], k = 0..4
z_im[5] = lambda_q * cognitive_load[i] + lambda_r * method_attr[m, 5]
```

默认：

```text
lambda_q = 0.6
lambda_r = 0.4
alpha_student = 0.75
alpha_teacher = 0.75
activity_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
```

主体画像 `theta` 的局部偏好为：

```text
Phi_im = sum_k activity_weights[k] * (1 - abs(z_im[k] - theta[k]))
Psi_im = 1 - max(0, z_im[5] - theta[5])
P_im = clip(alpha * Phi_im + (1 - alpha) * Psi_im, 0, 1)
```

## 目标函数

代码中使用 `assignment: np.ndarray[int], shape = [N]` 表示数学变量 `x_im` 的等价整数形式。

局部目标均使用平均值：

```text
F_E = mean_i effect_matrix[i, assignment[i]]
F_S = mean_i student_pref[i, assignment[i]]
F_T = mean_i teacher_pref[i, assignment[i]]
```

方法分布：

```text
pi[m] = count(assignment == m) / N
```

全局分布目标：

```text
F_G = 1 - 0.5 * sum_m abs(pi[m] - target_distribution[m])
```

软约束：

```text
H = -sum_m pi[m] * log(pi[m] + epsilon) / log(M)
V_div = max(0, entropy_min - H) ** 2
V_cap = sum_m max(0, pi[m] - method_cap) ** 2
V_soft = lambda_div * V_div + lambda_cap * V_cap
```

标量目标统一为：

```text
J = w_E * F_E + w_S * F_S + w_T * F_T + w_G * F_G - w_soft * V_soft
```

默认权重为可修改的中性配置：

```text
w_E = w_S = w_T = w_G = w_soft = 0.2
```

## Mask 语义

- `feasible_mask` 是静态方法可行性掩码，shape 为 `[N, M]` 或 batch 下 `[B, N_max, M]`。
- `action_mask` 是动态可执行替换动作掩码。
- 当前已经使用的方法 `m == assignment[i]` 不能作为替换动作。
- `node_mask` 只用于 batch padding；单 instance 下全为 `True`。

单实例：

```text
action_mask[i, m] = feasible_mask[i, m] and m != assignment[i]
```

batch：

```text
action_mask[b, i, m] =
    node_mask[b, i] and feasible_mask[b, i, m] and m != assignment[b, i]
```

## 环境与奖励

`MethodAssignmentEnv` 的动作是单点替换 `(i, m)`。

```text
reward = J(next_assignment) - J(current_assignment)
done = step_count >= max_steps or no_improve_steps >= patience
```

环境必须保留 episode 内的 best-so-far assignment。
