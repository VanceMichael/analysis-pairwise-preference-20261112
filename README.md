# pairwise-preference

outcome 为 left/right/tie；round 是场次，不是重复记录权重。

项目提供本地记录校验、输入指纹、按 `outcome` 的原始核算，以及含平局的成对
偏好模型（`rank`）：相对能力、比较不足诊断、前 K 排名稳定性与补比较建议。
分析人员可以保留完整输入并复查每次导出的 JSON；原始数据不被命令改写。
业务配置位于 `data/policy.json`，字段约定在 `docs/data.md`，模型口径在
`docs/model.md`。

## 运行

使用 Node.js 22 LTS（>=22.22.1 且 <23）和 npm，无第三方运行依赖。
`.nvmrc` 选择 22 主版本，清单与锁文件记录相同的兼容范围；Dockerfile 仍固定基础镜像摘要，容器构建可精确复现。

```sh
npm ci --ignore-scripts --no-audit --no-fund --offline
npm run build
npm test
npm run summary
node src/cli.js inspect --input data/records.json
node src/cli.js summary --input data/records.json --out var/summary.json
node src/cli.js rank --input data/records.json --out var/model-archive.json
node src/cli.js rank --input data/records.json --model var/model-archive.json
npm run demo
```

`rank` 的模型采用 Davidson（1970）的 Bradley–Terry 平局扩展，平局保留为
独立结果、不拆半胜；完全分离由高斯先验收敛处理并报告正则强度敏感性。
比较图按连通分量分别排名，没有比较路径的商品不会为了出总榜被硬排先后。
不确定性按评委整组 bootstrap 计算，同一评委的重复比较不冒充独立评委。
补比较建议只基于已拟合的期望 Fisher 信息与覆盖约束，不假设新比较结果。

`--out` 在 `rank` 下落盘的是完整模型档案：商品字典、参数约束、评委处理、
生效配置与数据指纹。`--model` 重载档案时会重新计算输入指纹，指纹不一致
直接拒绝，不会静默重算。

`npm run demo` 在本地打印输入、现有核算、rank 报告及配置并生成
`var/summary.json` 与 `var/model-archive.json`，不访问网络、不等待真实业务时间，通常几秒内完成。输出目录可删除后重新运行，测试只使用临时目录。

## 容器

```sh
docker build -t pairwise-preference:local .
docker run --rm --network none pairwise-preference:local
```

Dockerfile 固定 Node.js 基础镜像摘要，构建时执行语法检查和测试。镜像构建完成后演示无需联网；可覆盖命令运行 `npm test`。构建不需要真实账号、邮件服务、数据库或线上数据。

## 目录

- `src/dataset.js`：记录校验、规范序列化和数据指纹。
- `src/domain.js`：当前业务分组核算。
- `src/model.js`：Davidson 平局扩展、连通分量、评委界限与整组 bootstrap。
- `src/ranking.js`：排名报告、覆盖诊断、补比较建议与模型档案。
- `src/cli.js`：本地读取、检查、rank 拟合与档案重载入口。
- `src/output.js`：以临时文件写入后替换完整 JSON。
- `test/`：输入约束、模型数值（含有限差分）、领域核对、命令与输出回归。
- `data/`：合成记录、独立核对值及本地业务配置。
- `docs/`：数据口径（`data.md`）与模型口径（`model.md`）。
