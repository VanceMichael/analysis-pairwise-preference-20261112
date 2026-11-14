# 数据与现有口径

outcome 为 left/right/tie；round 是场次，不是重复记录权重。

`data/records.json` 是版本为 1 的演练数据，所有人员、机构、商品和交易均为本地合成记录。`id` 是导入记录键，不自动代表企业、访问或其他统计单位；记录键不得重复。数值字段必须有限，整数字段不得超过安全整数范围。空字符串可用于字段约定允许的未公开标签，不用零冒充缺测。

| 字段 | 类型 |
| --- | --- |
| `id` | `string` |
| `judge` | `string` |
| `left` | `string` |
| `right` | `string` |
| `outcome` | `string` |
| `round` | `integer` |

现有 `summary` 按 `outcome` 清点记录，`sum` 为 `round` 的逐行核对值，不代表统计推断或跨期业务指标。分组按键排序，行按记录键稳定处理。`data/expected-summary.json` 是独立生成的现有核算对照；`data/policy.json` 保存业务配置的演练值，当前清点命令不消费分析配置。

`rank` 命令消费 `data/policy.json` 的 `analysis` 段（缺省字段由代码默认值补齐，生效配置写入模型档案）：

| 字段 | 含义 |
| --- | --- |
| `topK` | 前 K 排名稳定性的 K |
| `regularization` | 高斯先验强度 lambda 网格，用于敏感性分析 |
| `primaryLambda` | 主拟合使用的 lambda，必须出现在网格中 |
| `lambdaEta` | 平局参数 eta=log(theta) 的先验强度 |
| `judgeCap` | 单评委有效比较数上限，`null` 表示不设界限 |
| `bootstrapReplicates` | 评委整组 bootstrap 次数（不小于 50） |
| `minComparisons` | 商品与商品对的最小比较覆盖数 |
| `suggestionCount` | 补比较建议返回条数 |
| `seed` | bootstrap 与确定性平局处理的随机种子 |

`rank` 额外拒绝相同商品比较（`left === right`）与不在 left/right/tie 内的 outcome；平局在模型中保留为独立第三类结果，不拆成两条半胜。模型数学与输出口径见 `docs/model.md`。

`inspect` 返回输入行数、字段列表与 SHA-256 指纹；指纹不依赖字段或记录排列。数据版本不支持时应直接拒绝，不做静默迁移。错误发生时命令返回非零退出码。模型档案另存 `canonicalFingerprint`（报告规范序列化的哈希）与 `dataFingerprint`（输入记录哈希），`--model` 重载时重新计算输入指纹并比对，不一致即拒绝。
