// 含平局的成对偏好模型：Davidson (1970) 对 Bradley–Terry 的平局扩展。
//
// 对商品 i、j，能力参数 beta_i、beta_j（同一连通分量内约束 sum(beta)=0），
// 平局参数 theta = exp(eta) >= 0，令 d = beta_i - beta_j：
//   P(i 胜 j) = exp(d)          / (exp(d) + exp(-d) + theta*exp((d+0)/2)*2 归一)
// 等价写法（程序内实际使用，数值稳定）：
//   a = d, b = -d, t = 0.5*d + 0.5*b + log(2*theta) = log(2*theta)
//   P_i   = exp(a) / Z,  P_j = exp(b) / Z,  P_tie = exp(t) / Z,
//   Z = exp(a) + exp(b) + exp(t)。
// theta -> 0 时退化为普通 Bradley–Terry；theta 越大平局越多。
// 平局是独立的第三类结果，不拆成两条半胜。
//
// 正则：每个 beta 加独立高斯先验 N(0, 1/lambda)，即目标函数增加
// 0.5*lambda*sum(beta^2)；eta 加 N(0, 1/lambdaEta) 先验。完全分离时
// 无正则的 MLE 会发散，先验把参数拉回有限值，强度敏感性由 rank 报告给出。

export const OUTCOMES = Object.freeze({ LEFT: 'left', RIGHT: 'right', TIE: 'tie' });

// 并查集：根据“曾经比较过”的边求无向连通分量，图上没有比较路径的商品
// 之间不存在可识别的相对能力，后续不得跨分量硬排名次。
class UnionFind {
  constructor(nodes) {
    this.parent = new Map(nodes.map(node => [node, node]));
    this.size = new Map(nodes.map(node => [node, 1]));
  }

  find(node) {
    let root = node;
    while (this.parent.get(root) !== root) root = this.parent.get(root);
    let current = node;
    while (this.parent.get(current) !== current) {
      const next = this.parent.get(current);
      this.parent.set(current, root);
      current = next;
    }
    return root;
  }

  union(a, b) {
    const rootA = this.find(a);
    const rootB = this.find(b);
    if (rootA === rootB) return;
    const [small, large] = this.size.get(rootA) <= this.size.get(rootB)
      ? [rootA, rootB] : [rootB, rootA];
    this.parent.set(small, large);
    this.size.set(large, this.size.get(large) + this.size.get(small));
  }
}

function logSumExp(terms) {
  const max = Math.max(...terms);
  if (!Number.isFinite(max)) return max;
  return max + Math.log(terms.reduce((sum, term) => sum + Math.exp(term - max), 0));
}

// 三分类概率，输入 d = beta_i - beta_j，i 记为“左侧/胜者方向”。
export function outcomeProbabilities(d, theta) {
  const logTwoTheta = Math.log(2 * theta); // theta 由 eta=log(theta) 保证为正
  const logZ = logSumExp([d, -d, logTwoTheta]);
  return {
    win: Math.exp(d - logZ),          // i 胜 j
    loss: Math.exp(-d - logZ),        // j 胜 i
    tie: Math.exp(logTwoTheta - logZ),
  };
}

// 规范化一条记录：统一按商品字典序放置 a/b，记录结果相对于 a 的方向，
// 同时保留原始左右朝向以便按原始 outcome 口径核对。
// 左右交换（left/right 互换且 outcome 同步翻转）经过这里得到完全相同的
// canonical 比较，因此整条推断管线对记录朝向不变。
export function canonicalComparison(record) {
  const swap = record.left > record.right;
  const a = swap ? record.right : record.left;
  const b = swap ? record.left : record.right;
  let label;
  if (record.outcome === OUTCOMES.TIE) {
    label = OUTCOMES.TIE;
  } else {
    const winnerIsLeft = record.outcome === OUTCOMES.LEFT;
    const winner = winnerIsLeft ? record.left : record.right;
    label = winner === a ? 'win' : 'loss';
  }
  return {
    id: record.id, judge: record.judge, a, b, label, round: record.round,
    originalLeft: record.left, originalRight: record.right, originalOutcome: record.outcome,
  };
}

// 记录级校验：outcome 编码、相同商品比较、空标签统一在此拦截。
export function validateComparisons(records) {
  for (const [index, record] of records.entries()) {
    const where = `记录 ${record.id ?? `第 ${index + 1} 行`}`;
    if (![OUTCOMES.LEFT, OUTCOMES.RIGHT, OUTCOMES.TIE].includes(record.outcome)) {
      throw new Error(`${where} 的 outcome 必须是 left/right/tie`);
    }
    if (typeof record.left !== 'string' || typeof record.right !== 'string'
      || record.left.length === 0 || record.right.length === 0) {
      throw new Error(`${where} 的商品标签必须是非空字符串`);
    }
    if (record.left === record.right) {
      throw new Error(`${where} 是相同商品比较（left === right），不进入偏好模型`);
    }
    if (typeof record.judge !== 'string' || record.judge.length === 0) {
      throw new Error(`${where} 的 judge 必须是非空字符串`);
    }
  }
}

export function buildGraph(records) {
  validateComparisons(records);
  const items = [...new Set(records.flatMap(record => [record.left, record.right]))].sort();
  const judges = [...new Set(records.map(record => record.judge))].sort();
  const uf = new UnionFind(items);
  const comparisons = records.map(canonicalComparison);
  for (const comparison of comparisons) uf.union(comparison.a, comparison.b);

  const componentByRoot = new Map();
  for (const item of items) {
    const root = uf.find(item);
    if (!componentByRoot.has(root)) componentByRoot.set(root, []);
    componentByRoot.get(root).push(item);
  }
  const components = [...componentByRoot.values()]
    .map(members => members.sort())
    .sort((left, right) => right.length - left.length || left[0].localeCompare(right[0], 'en'));
  const componentId = new Map();
  components.forEach((members, index) => {
    for (const item of members) componentId.set(item, index);
  });
  return { items, judges, comparisons, components, componentId };
}

// 评委影响设界限：同一评委的重复比较不能冒充独立评委。
// 每个评委的有效权重 = min(1, judgeCap / 该评委原始比较数)，
// 即单评委超过 judgeCap 条之后的比较按比例降权，总有效比较数封顶在
// judgeCap；judgeCap 为 null 时全部权重为 1。权重写进模型档案可复查。
export function judgeWeights(comparisons, judges, judgeCap) {
  const counts = new Map(judges.map(judge => [judge, 0]));
  for (const comparison of comparisons) {
    counts.set(comparison.judge, counts.get(comparison.judge) + 1);
  }
  const weights = new Map();
  for (const judge of judges) {
    const count = counts.get(judge);
    weights.set(judge, judgeCap === null || judgeCap === undefined
      ? 1 : Math.min(1, judgeCap / count));
  }
  return { counts, weights };
}

// 高斯-若尔当消元解线性方程组 A x = b（A 为稠密方阵），带部分主元。
function solveLinear(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (!Number.isFinite(a[pivot][col]) || Math.abs(a[pivot][col]) < 1e-12) {
      throw new Error('牛顿迭代遇到奇异矩阵，参数在当前约束下不可识别');
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const pivotValue = a[col][col];
    for (let k = col; k <= n; k += 1) a[col][k] /= pivotValue;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      if (factor === 0) continue;
      for (let k = col; k <= n; k += 1) a[row][k] -= factor * a[col][k];
    }
  }
  return a.map(row => row[n]);
}

function invertMatrix(matrix) {
  const n = matrix.length;
  const columns = [];
  for (let col = 0; col < n; col += 1) {
    const unit = Array.from({ length: n }, (_, i) => i === col ? 1 : 0);
    columns.push(solveLinear(matrix.map(row => [...row]), unit));
  }
  // columns[col] 是逆矩阵的第 col 列，转置回按行存储
  return Array.from({ length: n }, (_, row) =>
    Array.from({ length: n }, (_, col) => columns[col][row]));
}

// 负对数后验（未加权数据似然 + 先验项），用于收敛判断与档案留痕。
export function negativeLogPosterior(comparisons, betaByName, eta, lambda, lambdaEta, weightOf) {
  const theta = Math.exp(eta);
  let nll = 0;
  for (const comparison of comparisons) {
    const d = betaByName.get(comparison.a) - betaByName.get(comparison.b);
    const { win, loss, tie } = outcomeProbabilities(d, theta);
    const p = comparison.label === 'win' ? win : comparison.label === 'loss' ? loss : tie;
    nll -= (weightOf ? weightOf(comparison) : 1) * Math.log(p);
  }
  let penalty = 0;
  for (const beta of betaByName.values()) penalty += 0.5 * lambda * beta * beta;
  penalty += 0.5 * lambdaEta * eta * eta;
  return { nll, penalty, objective: nll + penalty };
}

// 拟合 Davidson 模型。
// 参数布局：先按分量顺序放各分量的自由 beta（分量大小 m 个商品放 m-1 个
// 自由值，最后一个商品由 sum(beta)=0 约束解出），末尾放 eta。
export function fitModel(graph, options = {}) {
  const { comparisons, components, items, componentId } = graph;
  const lambda = options.lambda ?? 1;
  const lambdaEta = options.lambdaEta ?? 1;
  const weightMap = options.weightMap ?? null;
  const maxIterations = options.maxIterations ?? 100;
  const tolerance = options.tolerance ?? 1e-10;
  const weightOf = comparison => weightMap ? weightMap.get(comparison.judge) ?? 1 : 1;

  // 自由参数索引：freeIndex[item] 为 null 表示该商品是分量的锚点（最后一个）。
  const freeIndex = new Map();
  const anchorOf = new Map();
  let cursor = 0;
  for (const members of components) {
    for (let k = 0; k < members.length - 1; k += 1) {
      freeIndex.set(members[k], cursor);
      cursor += 1;
    }
    const anchor = members[members.length - 1];
    freeIndex.set(anchor, null);
    anchorOf.set(componentId.get(anchor), anchor);
  }
  const freeCount = cursor;
  const dimension = freeCount + 1;
  const etaIndex = freeCount;

  const expand = free => {
    const beta = new Map();
    for (const members of components) {
      let sum = 0;
      for (let k = 0; k < members.length - 1; k += 1) {
        const value = free[freeIndex.get(members[k])];
        beta.set(members[k], value);
        sum += value;
      }
      // 归一化 -0 为 0，保证档案与名次比较不出现符号零歧义
      beta.set(members[members.length - 1], sum === 0 ? 0 : -sum);
    }
    return beta;
  };

  let free = new Array(freeCount).fill(0);
  let eta = 0; // theta 初始为 1
  let objective = Infinity;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const beta = expand(free);
    const theta = Math.exp(eta);
    const gradient = new Array(dimension).fill(0);
    const hessian = Array.from({ length: dimension }, () => new Array(dimension).fill(0));

    for (const comparison of comparisons) {
      const weight = weightOf(comparison);
      const d = beta.get(comparison.a) - beta.get(comparison.b);
      const { win, loss, tie } = outcomeProbabilities(d, theta);
      // 指示量：结果相对 a 的方向
      const yWin = comparison.label === 'win' ? 1 : 0;
      const yLoss = comparison.label === 'loss' ? 1 : 0;
      const yTie = comparison.label === 'tie' ? 1 : 0;

      // 全 beta 空间的负对数似然一阶导：
      //   g_a = w*(p_win-p_loss-(y_win-y_loss)),  g_b = -g_a
      // 二阶导由 softmax 对数似然的链式法则得到（评分协方差，必半正定）：
      //   I_dd  = p_win + p_loss - (p_win-p_loss)^2
      //   I_dη  = -(p_win-p_loss)*p_tie
      //   I_ηη  = p_tie*(1-p_tie)
      // 这些量只依赖当前拟合概率，与未知的比较结果无关。
      const ga = weight * (win - loss - (yWin - yLoss));
      const info = weight * (win + loss - (win - loss) ** 2);
      const hCross = -weight * (win - loss) * tie;
      const hEtaEta = weight * tie * (1 - tie);
      const gEta = weight * (tie - yTie);

      // 和为零约束下，锚点 beta_anchor = -sum_k beta_k。
      // 对每个自由变量记录其在 d = beta_a-beta_b 中的系数（+1/-1/0），
      // 锚点参与的比较通过该系数把导数投影回自由参数。
      const factors = new Map();
      const collect = (item, sign) => {
        const free = freeIndex.get(item);
        if (free !== null && free !== undefined) {
          factors.set(free, (factors.get(free) ?? 0) + sign);
        } else {
          const members = components[componentId.get(item)];
          for (let k = 0; k < members.length - 1; k += 1) {
            const index = freeIndex.get(members[k]);
            factors.set(index, (factors.get(index) ?? 0) - sign);
          }
        }
      };
      collect(comparison.a, 1);
      collect(comparison.b, -1);
      const touched = [...factors.entries()];

      for (const [index, factor] of touched) gradient[index] += factor * ga;
      for (const [indexI, factorI] of touched) {
        for (const [indexJ, factorJ] of touched) {
          hessian[indexI][indexJ] += factorI * factorJ * info;
        }
        // 自由 beta 与 eta 的交叉项
        hessian[indexI][etaIndex] += factorI * hCross;
        hessian[etaIndex][indexI] += factorI * hCross;
      }
      gradient[etaIndex] += gEta;
      hessian[etaIndex][etaIndex] += hEtaEta;
    }

    // 高斯先验（二次惩罚）；锚点 beta 由和为零线性约束导出，其先验贡献在
    // 自由参数上的 Hessian 不是对角的，这里按展开后的 beta 重新累加。
    let penaltyGradient = new Array(freeCount).fill(0);
    for (const members of components) {
      // beta_anchor = -sum_k beta_k；d penalty/d beta_k = lambda*(beta_k - beta_anchor)
      const anchor = members[members.length - 1];
      const anchorBeta = beta.get(anchor);
      for (let k = 0; k < members.length - 1; k += 1) {
        const item = members[k];
        const index = freeIndex.get(item);
        penaltyGradient[index] += lambda * (beta.get(item) - anchorBeta);
        for (let l = 0; l < members.length - 1; l += 1) {
          const jIndex = freeIndex.get(members[l]);
          hessian[index][jIndex] += lambda * (k === l ? 2 : 1);
        }
      }
    }
    for (let index = 0; index < freeCount; index += 1) gradient[index] += penaltyGradient[index];
    gradient[etaIndex] += lambdaEta * eta;
    hessian[etaIndex][etaIndex] += lambdaEta;

    const current = negativeLogPosterior(comparisons, beta, eta, lambda, lambdaEta, weightOf).objective;
    let step;
    try {
      step = solveLinear(hessian, gradient.map(value => -value));
    } catch (error) {
      return { converged: false, reason: error.message, iterations: iteration + 1,
        beta, eta, theta, lambda, lambdaEta, components, items, componentId, freeIndex };
    }

    // 回溯线搜索保证目标值单调下降，避免步长越过数值域。
    let accepted = null;
    let stepScale = 1;
    for (let backtrack = 0; backtrack < 40; backtrack += 1) {
      const trialFree = free.map((value, index) => value + stepScale * step[index]);
      const trialEta = eta + stepScale * step[etaIndex];
      if (trialFree.some(value => !Number.isFinite(value)) || !Number.isFinite(trialEta)) {
        stepScale *= 0.5;
        continue;
      }
      const trialBeta = expand(trialFree);
      const trialObjective = negativeLogPosterior(
        comparisons, trialBeta, trialEta, lambda, lambdaEta, weightOf).objective;
      if (Number.isFinite(trialObjective) && trialObjective <= current + 1e-12) {
        accepted = { trialFree, trialEta, trialObjective };
        break;
      }
      stepScale *= 0.5;
    }
    if (!accepted) {
      return { converged: false, reason: '线搜索无法找到下降步', iterations: iteration + 1,
        beta, eta, theta, lambda, lambdaEta, components, items, componentId, freeIndex };
    }
    free = accepted.trialFree;
    eta = accepted.trialEta;
    const improvement = objective === Infinity ? Infinity : objective - accepted.trialObjective;
    objective = accepted.trialObjective;
    if (improvement !== Infinity && improvement < tolerance) {
      const finalBeta = expand(free);
      const finalTheta = Math.exp(eta);
      // 收敛后重算 Hessian 取后验协方差（观测信息的逆）
      const covariance = invertMatrix(hessian);
      const losses = negativeLogPosterior(comparisons, finalBeta, eta, lambda, lambdaEta, weightOf);
      return {
        converged: true, iterations: iteration + 1, beta: finalBeta, eta,
        theta: finalTheta, lambda, lambdaEta, objective,
        nll: losses.nll, penalty: losses.penalty,
        covariance, freeIndex, etaIndex, components, items, componentId,
      };
    }
  }
  const beta = expand(free);
  return { converged: false, reason: '达到最大迭代次数仍未收敛', iterations: maxIterations,
    beta, eta, theta: Math.exp(eta), lambda, lambdaEta, components, items, componentId, freeIndex };
}

// 可复现的整数 PRNG（mulberry32），cluster bootstrap 的抽样种子写进档案。
export function seededRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 按评委整组重采样：同一评委的全部比较始终一起抽出，重复比较不会被
// 当作独立评委。抽中的评委保留其全部原始记录（可能重复多次），未抽中
// 则缺席；重采样输出恢复成原始记录形状，保证后续校验与规范化一致。
export function clusteredResample(records, judges, random) {
  const byJudge = new Map(judges.map(judge => [judge, []]));
  for (const record of records) {
    const judge = record.judge;
    const restored = record.originalOutcome !== undefined
      ? { id: record.id, judge: record.judge, left: record.originalLeft,
        right: record.originalRight, outcome: record.originalOutcome, round: record.round }
      : record;
    if (!byJudge.has(judge)) byJudge.set(judge, []);
    byJudge.get(judge).push(restored);
  }
  const resampled = [];
  for (let draw = 0; draw < judges.length; draw += 1) {
    const judge = judges[Math.floor(random() * judges.length)];
    resampled.push(...byJudge.get(judge));
  }
  return resampled;
}

// 评委整组 bootstrap：每次重采样后重新构图、重新拟合，逐次保存 beta 与
// theta，名次分布和成对区间都由这份对齐的重采样结果派生。重采样产生的
// 临时新分量按主图分量对齐，缺席商品在该次重采样中记为缺测。
export function bootstrapRanks(graph, policy, weightMap, replicates, seed) {
  const random = seededRandom(seed);
  const { comparisons, judges, items, components } = graph;
  const draws = { replicates: [], failed: 0, seed, requested: replicates };
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    const resampled = clusteredResample(comparisons, judges, random);
    if (resampled.length === 0) {
      draws.failed += 1;
      continue;
    }
    let resampledGraph;
    try {
      resampledGraph = buildGraph(resampled);
    } catch {
      draws.failed += 1;
      continue;
    }
    let fit;
    try {
      fit = fitModel(resampledGraph, {
        lambda: policy.lambda, lambdaEta: policy.lambdaEta,
        weightMap: reweight(resampledGraph, weightMap, judges, policy.judgeCap),
      });
    } catch {
      draws.failed += 1;
      continue;
    }
    if (!fit.converged) {
      draws.failed += 1;
      continue;
    }
    // 只保留主图商品，保证各次重采样结构对齐
    const beta = new Map();
    for (const item of items) {
      if (fit.beta.has(item)) beta.set(item, fit.beta.get(item));
    }
    const componentOf = new Map();
    for (const item of items) {
      componentOf.set(item, resampledGraph.componentId.has(item)
        ? resampledGraph.componentId.get(item) : null);
    }
    draws.replicates.push({ beta, theta: fit.theta, componentOf });
  }
  return draws;
}

// bootstrap 重采样后按“原始评委身份”重新应用评委界限权重；
// 原始权重表只在原评委集合上有定义，缺席/新标签回退到 1 再由 cap 限制。
function reweight(resampledGraph, originalWeightMap, originalJudges, judgeCap) {
  const { counts, weights } = judgeWeights(resampledGraph.comparisons, resampledGraph.judges, judgeCap);
  const map = new Map();
  for (const judge of resampledGraph.judges) {
    const capped = weights.get(judge);
    const original = originalJudges.includes(judge) ? originalWeightMap.get(judge) : 1;
    map.set(judge, Math.min(capped, original));
  }
  return map;
}

// 分量内排名：beta 高者在前，严格相等时用传入随机数做稳定平局打破，
// 保证 bootstrap 的名次分布不被字典序系统性偏向。
export function rankItems(items, betaOf, random = Math.random) {
  return [...items].map(item => ({ item, key: random() }))
    .sort((left, right) => {
      const difference = betaOf(right.item) - betaOf(left.item);
      if (difference !== 0) return difference;
      return left.key - right.key;
    })
    .map(entry => entry.item);
}

export function quantile(sortedValues, probability) {
  if (sortedValues.length === 0) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const position = (sortedValues.length - 1) * probability;
  const base = Math.floor(position);
  const fraction = position - base;
  if (base + 1 >= sortedValues.length) return sortedValues[base];
  return sortedValues[base] * (1 - fraction) + sortedValues[base + 1] * fraction;
}

export function summarizeDistribution(values) {
  if (values.length === 0) {
    return { replicates: 0, mean: null, sd: null, q025: null, q50: null, q975: null };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0)
    / Math.max(1, values.length - 1);
  return {
    replicates: values.length,
    mean,
    sd: Math.sqrt(variance),
    q025: quantile(sorted, 0.025),
    q50: quantile(sorted, 0.5),
    q975: quantile(sorted, 0.975),
  };
}

export { UnionFind, invertMatrix, solveLinear };
