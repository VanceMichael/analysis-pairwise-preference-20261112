# 含平局的成对偏好模型（rank 命令）

`rank` 在现有记录校验与数据指纹之上，拟合 Davidson（1970）对 Bradley–Terry 的
平局扩展，输出相对能力、比较不足诊断、前 K 排名稳定性与补比较建议。
模型档案落盘后可用 `--model` 重载直接解释，不重新拟合。

## 概率模型

对商品 i、j，能力为 beta_i、beta_j，平局参数 theta = exp(eta) >= 0，令
d = beta_i - beta_j：

```
P(i 胜 j) = exp(d) / Z
P(j 胜 i) = exp(-d) / Z
P(平局)   = 2*theta / Z
Z = exp(d) + exp(-d) + 2*theta
```

- theta → 0 时退化为普通 Bradley–Terry；theta 越大平局概率越高。
- 平局是独立的第三类结果，**不拆成两条半胜**，经验胜率与模型概率都保留平局列。
- 能力只在同一连通分量内可识别。每个分量约束 `sum(beta) = 0` 固定尺度，
  断开分量之间不存在共同尺度，**不生成全局总榜、不硬排先后**。

## 正则与完全分离

无正则的 MLE 在完全分离（某商品在所有比较中全胜/全负）时参数发散。
模型对 beta 加独立高斯先验 `N(0, 1/lambda)`、对 eta 加 `N(0, 1/lambdaEta)`，
即负对数后验增加 `0.5*lambda*sum(beta^2)` 与 `0.5*lambdaEta*eta^2`，
把参数收缩到有限值。`policy.regularization` 给出 lambda 网格，主拟合使用
`primaryLambda`，报告 `sensitivity` 给出每个 lambda 下的 theta、目标值、
分量内顺序与相对主拟合的 Spearman 相关；分量级 `orderingStableAcrossGrid`
标明名次对正则强度是否敏感。先验收缩越强，分离商品的能力绝对值越小，但
通常不改变已连通商品的先后顺序。

## 拟合与数值实现

- 参数化：每分量 m 个商品用 m-1 个自由 beta，锚点由和为零约束导出；
  eta 自由取值，theta 恒正。
- 牛顿迭代 + 回溯线搜索，解析梯度与 Hessian（softmax 评分协方差形式）：
  - `I_dd  = p_win + p_loss - (p_win-p_loss)^2`
  - `I_dη  = -(p_win-p_loss)*p_tie`
  - `I_ηη  = p_tie*(1-p_tie)`
  解析导数有有限差分校验测试（任意内点，含锚点投影与评委权重）。
- 收敛后取观测信息的逆作为后验协方差；矩阵求解用带部分主元的高斯-若尔当消元。

## 评委处理

同一评委的重复比较不是独立评委。两层处理：

1. **影响界限 judgeCap**：评委有效权重
   `min(1, judgeCap / 该评委原始比较数)`，单评委的有效比较数封顶在 judgeCap；
   `judgeCap: null` 表示不设界限。每个评委的原始数、有效权重与有效比较数
   写入档案。
2. **评委整组 cluster bootstrap**：重采样单位是评委，抽中评委的全部比较
   一起进入，重复比较始终同组。单评委数据下 bootstrap 退化为零宽度，
   报告显式警告“不反映评委间变异”，而不是给出虚假的精确区间。

## 输出结构要点

- `components[]`：每个连通分量的点估计名次、beta、渐近标准误、bootstrap
  95% beta 区间、名次 95% 区间、各名次概率、前 K 概率、`credibleRank`
  （名次区间宽度不超过 1）、点估计并列组 `tiedGroups`。
- `pairwise[]`：同分量对给经验胜/负/平计数与胜率、模型三分类概率及
  bootstrap 区间、能力对比 d 的区间是否跨零；跨分量对标记“不可比较”，
  `model` 与 `contrast` 为 null。
- `coverage`：每个商品的比较数、不同对手数、不同评委数与覆盖缺口原因。
- `suggestions`：补比较建议，方法与每条建议的理由随报告给出。
- `sensitivity`：正则强度网格与顺序稳定性。

## 补比较建议的口径

建议**只依据已拟合模型的期望 Fisher 信息与覆盖约束**，不假设新比较的任何
结果，也不预测谁会赢：

- 同分量对的期望信息
  `I = p_win + p_loss - (p_win-p_loss)^2`，只依赖当前拟合概率；
  结合能力对比的后验方差与覆盖缺口打分。
- 跨分量对属于“桥接比较”，做一次即可首次连通两个断开分量，优先级最高，
  不输出信息数值（当前没有共同尺度）。
- 已达到 `pairCap = minComparisons` 且双方覆盖充分的对不再推荐。

## 数据校验

`rank` 在拟合前统一拒绝：相同商品比较（left === right）、outcome 不在
left/right/tie 内、空商品或评委标签。记录按商品字典序规范化，左右交换且
outcome 同步翻转得到相同的 canonical 比较，整条推断对记录朝向不变，并有
回归测试覆盖。
