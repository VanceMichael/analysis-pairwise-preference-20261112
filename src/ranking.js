// 排名报告、不确定性汇总、比较不足诊断与补比较建议。
// 所有建议只使用已拟合模型的期望 Fisher 信息与覆盖约束，
// 不假设任何“新比较会得到什么结果”。

import { readFile } from 'node:fs/promises';
import {
  buildGraph, judgeWeights, fitModel, bootstrapRanks, rankItems,
  outcomeProbabilities, seededRandom, summarizeDistribution,
} from './model.js';
import { validateRecords, fingerprint, canonical } from './dataset.js';

export const MODEL_ARCHIVE_VERSION = 1;

// policy.json 的分析字段（清点命令不消费这些字段）。缺省时给出可复查的
// 默认值，实际生效的完整配置写入模型档案。
export const DEFAULT_POLICY = Object.freeze({
  topK: 3,
  regularization: [0.1, 1, 10],
  primaryLambda: 1,
  lambdaEta: 1,
  judgeCap: 5,
  bootstrapReplicates: 1000,
  minComparisons: 3,
  suggestionCount: 5,
  seed: 31,
});

export function resolvePolicy(input = {}) {
  const policy = { ...DEFAULT_POLICY, ...input };
  if (!Number.isSafeInteger(policy.topK) || policy.topK < 1) {
    throw new Error('policy.topK 必须是正整数');
  }
  if (!Array.isArray(policy.regularization) || policy.regularization.length < 2
    || !policy.regularization.every(value => typeof value === 'number' && value > 0)) {
    throw new Error('policy.regularization 必须是至少两个正数值的数组');
  }
  if (!policy.regularization.includes(policy.primaryLambda)) {
    throw new Error('policy.primaryLambda 必须出现在 regularization 中');
  }
  for (const [name, value] of Object.entries({
    lambdaEta: policy.lambdaEta, primaryLambda: policy.primaryLambda,
  })) {
    if (typeof value !== 'number' || value <= 0) throw new Error(`policy.${name} 必须为正数`);
  }
  if (policy.judgeCap !== null && (!Number.isSafeInteger(policy.judgeCap) || policy.judgeCap < 1)) {
    throw new Error('policy.judgeCap 必须是正整数或 null');
  }
  if (!Number.isSafeInteger(policy.bootstrapReplicates) || policy.bootstrapReplicates < 50) {
    throw new Error('policy.bootstrapReplicates 必须是不小于 50 的整数');
  }
  if (!Number.isSafeInteger(policy.minComparisons) || policy.minComparisons < 1) {
    throw new Error('policy.minComparisons 必须是正整数');
  }
  if (!Number.isSafeInteger(policy.suggestionCount) || policy.suggestionCount < 1) {
    throw new Error('policy.suggestionCount 必须是正整数');
  }
  if (!Number.isSafeInteger(policy.seed)) throw new Error('policy.seed 必须是安全整数');
  return policy;
}

// 自由参数协方差 -> 按商品索引的完整 beta 协方差（锚点由和为零约束导出）。
function betaCovariance(fit) {
  const { covariance, freeIndex, etaIndex, items, components } = fit;
  const indexOf = new Map(items.map((item, index) => [item, index]));
  const n = items.length;
  const result = Array.from({ length: n }, () => new Array(n).fill(0));
  const freeVars = item => {
    const direct = freeIndex.get(item);
    if (direct !== null && direct !== undefined) return [[direct, 1]];
    const component = fit.componentId.get(item);
    const members = components[component];
    return members.slice(0, -1).map(member => [freeIndex.get(member), -1]);
  };
  for (const itemA of items) {
    for (const itemB of items) {
      let value = 0;
      for (const [indexA, signA] of freeVars(itemA)) {
        for (const [indexB, signB] of freeVars(itemB)) {
          value += signA * signB * covariance[indexA][indexB];
        }
      }
      result[indexOf.get(itemA)][indexOf.get(itemB)] = value;
    }
  }
  return { matrix: result, indexOf, etaVariance: covariance[etaIndex][etaIndex] };
}

function empiricalPairStats(comparisons) {
  const stats = new Map();
  for (const comparison of comparisons) {
    const key = `${comparison.a}|${comparison.b}`;
    if (!stats.has(key)) stats.set(key, { a: comparison.a, b: comparison.b, wins: 0, losses: 0, ties: 0, judges: new Set(), rounds: new Set() });
    const entry = stats.get(key);
    if (comparison.label === 'win') entry.wins += 1;
    else if (comparison.label === 'loss') entry.losses += 1;
    else entry.ties += 1;
    entry.judges.add(comparison.judge);
    entry.rounds.add(comparison.round);
  }
  return stats;
}

// 点估计的模型概率（含平局，平局不拆半胜）。
function modelPairProbabilities(fit, itemA, itemB) {
  const d = fit.beta.get(itemA) - fit.beta.get(itemB);
  const { win, loss, tie } = outcomeProbabilities(d, fit.theta);
  return { aWins: win, bWins: loss, tie };
}

function spearmanCorrelation(leftOrder, rightOrder) {
  const rankOf = order => {
    const map = new Map();
    order.forEach((item, index) => map.set(item, index + 1));
    return map;
  };
  const leftRank = rankOf(leftOrder);
  const rightRank = rankOf(rightOrder);
  const items = leftOrder.filter(item => rightRank.has(item));
  const n = items.length;
  if (n < 2) return null;
  const mean = (n + 1) / 2;
  let numerator = 0;
  let denominatorLeft = 0;
  let denominatorRight = 0;
  for (const item of items) {
    const x = leftRank.get(item) - mean;
    const y = rightRank.get(item) - mean;
    numerator += x * y;
    denominatorLeft += x * x;
    denominatorRight += y * y;
  }
  if (denominatorLeft === 0 || denominatorRight === 0) return null;
  return numerator / Math.sqrt(denominatorLeft * denominatorRight);
}

// 主入口：从原始记录构建完整 rank 报告与模型档案内容。
export function buildRankReport(records, policyInput, options = {}) {
  validateRecords(records);
  const policy = resolvePolicy(policyInput);
  const graph = buildGraph(records);
  const { items, judges, comparisons, components, componentId } = graph;
  const { counts: judgeCounts, weights: judgeWeightMap } = judgeWeights(comparisons, judges, policy.judgeCap);

  const weightMap = new Map(judges.map(judge => [judge, judgeWeightMap.get(judge)]));
  const primary = fitModel(graph, {
    lambda: policy.primaryLambda, lambdaEta: policy.lambdaEta, weightMap,
  });
  if (!primary.converged) throw new Error(`主模型未收敛：${primary.reason}`);

  const covariance = betaCovariance(primary);
  const pairStats = empiricalPairStats(comparisons);

  // 正则强度敏感性：在 policy.regularization 的每个 lambda 下重新拟合，
  // 比较分量内名次顺序与 Spearman 相关。
  const sensitivity = policy.regularization.map(lambda => {
    const fit = fitModel(graph, { lambda, lambdaEta: policy.lambdaEta, weightMap });
    if (!fit.converged) throw new Error(`lambda=${lambda} 的敏感性拟合未收敛：${fit.reason}`);
    const componentOrders = components.map(members =>
      rankItems(members, item => fit.beta.get(item), seededRandom(policy.seed)));
    return {
      lambda,
      theta: fit.theta,
      objective: fit.objective,
      nll: fit.nll,
      componentOrders,
      primary: lambda === policy.primaryLambda,
    };
  });
  const primarySensitivity = sensitivity.find(entry => entry.primary);
  const componentSensitivity = components.map((members, componentIndex) => {
    const baseOrder = primarySensitivity.componentOrders[componentIndex];
    const grid = sensitivity.map(entry => {
      const order = entry.componentOrders[componentIndex];
      const sameOrder = order.every((item, index) => item === baseOrder[index]);
      return {
        lambda: entry.lambda,
        order,
        orderIdentical: sameOrder,
        spearmanVsPrimary: spearmanCorrelation(baseOrder, order),
      };
    });
    return {
      component: componentIndex,
      orderingStableAcrossGrid: grid.every(entry => entry.orderIdentical),
      grid,
    };
  });

  // 评委整组 bootstrap
  const bootstrap = bootstrapRanks(graph, policy, weightMap,
    options.bootstrapReplicates ?? policy.bootstrapReplicates, policy.seed);
  const bootstrapRandom = seededRandom(policy.seed ^ 0x9e3779b9);

  // 每个商品的 beta 分布与分量内名次分布
  const itemResults = items.map(item => {
    const involvingItem = comparisons.filter(comparison =>
      comparison.a === item || comparison.b === item);
    const itemJudgeCount = new Set(involvingItem.map(comparison => comparison.judge)).size;
    const values = bootstrap.replicates
      .map(replicate => replicate.beta.get(item))
      .filter(value => value !== undefined);
    const distribution = summarizeDistribution(values);
    const component = componentId.get(item);
    const members = components[component];
    const rankValues = [];
    let eligibleReplicates = 0;
    for (const replicate of bootstrap.replicates) {
      // 该次重采样中主图分量全体成员都仍被观测到，才参与名次分布；
      // 分量是否仍然连通由“成员全部存在”隐含（重采样只删边不增边，
      // 全体成员存在且原图连通时，新图中它们仍在同一分量）。
      const present = members.every(member => replicate.beta.has(member));
      if (!present) continue;
      eligibleReplicates += 1;
      const order = rankItems(members, member => replicate.beta.get(member), bootstrapRandom);
      rankValues.push(order.indexOf(item) + 1);
    }
    const rankSummary = summarizeDistribution(rankValues);
    const rankProbability = {};
    for (let rank = 1; rank <= members.length; rank += 1) {
      rankProbability[rank] = rankValues.length
        ? rankValues.filter(value => value === rank).length / rankValues.length : null;
    }
    const topK = Math.min(policy.topK, members.length);
    const topKProbability = rankValues.length
      ? rankValues.filter(value => value <= topK).length / rankValues.length : null;
    const seAsymptotic = Math.sqrt(covariance.matrix[covariance.indexOf.get(item)][covariance.indexOf.get(item)]);
    return {
      item,
      component,
      beta: primary.beta.get(item),
      seAsymptotic,
      betaBootstrap: distribution,
      rankBootstrap: { eligibleReplicates, ...rankSummary, probabilityByRank: rankProbability },
      topKProbability,
      warnings: [
        ...(itemJudgeCount < 2 ? ['只有单一评委覆盖该商品：cluster bootstrap 退化为零宽度，置信区间不反映评委间变异，请按评委界限后的点估计谨慎使用'] : []),
      ],
    };
  });
  const itemResultByName = new Map(itemResults.map(entry => [entry.item, entry]));

  // 分量内点估计名次（报告主名次）：beta 高者在前，严格相同时按商品 id
  // 字典序确定展示顺序并显式标记并列，不用随机数决定展示给选品的名次。
  const componentReports = components.map((members, componentIndex) => {
    const pointOrder = [...members].sort((left, right) => {
      const difference = primary.beta.get(right) - primary.beta.get(left);
      if (difference !== 0) return difference;
      return left.localeCompare(right, 'en');
    });
    const tiedGroups = [];
    for (let index = 0; index < pointOrder.length; index += 1) {
      const group = [pointOrder[index]];
      while (index + 1 < pointOrder.length
        && primary.beta.get(pointOrder[index + 1]) === primary.beta.get(pointOrder[index])) {
        index += 1;
        group.push(pointOrder[index]);
      }
      if (group.length > 1) tiedGroups.push(group);
    }
    const topK = Math.min(policy.topK, members.length);
    const tiedItems = new Set(tiedGroups.flat());
    const rankedItems = pointOrder.map((item, index) => {
      const result = itemResultByName.get(item);
      const credible = (result.rankBootstrap.q025 !== null
        && result.rankBootstrap.q975 !== null
        && result.rankBootstrap.q975 - result.rankBootstrap.q025 <= 1);
      return {
        rank: index + 1,
        item,
        beta: result.beta,
        seAsymptotic: result.seAsymptotic,
        betaCi95: [result.betaBootstrap.q025, result.betaBootstrap.q975],
        rankCI95: [result.rankBootstrap.q025, result.rankBootstrap.q975],
        rankProbability: result.rankBootstrap.probabilityByRank,
        topKProbability: result.topKProbability,
        credibleRank: credible,
        pointEstimateTie: tiedItems.has(item),
        warnings: result.warnings,
      };
    });
    return {
      component: componentIndex,
      connected: members.length > 1,
      members,
      order: pointOrder,
      tiedGroups,
      topK,
      rankedItems,
      sensitivity: componentSensitivity[componentIndex],
      note: members.length === 1
        ? '该商品没有任何比较路径，能力与名次不可识别' : undefined,
    };
  });

  // 成对结果：直接比较给经验胜率，模型概率对同分量所有商品对给出；
  // 跨分量对明确标记不可比较，不输出谁先谁后的概率。
  const pairwise = buildPairwiseReport(graph, primary, pairStats, covariance, bootstrap);

  // 比较不足诊断
  const coverage = buildCoverage(graph, pairStats, judgeCounts, policy, weightMap);

  // 补比较建议（只依据已拟合信息与覆盖约束）
  const suggestions = buildSuggestions(graph, primary, covariance, pairStats, coverage, policy);

  const report = {
    model: {
      family: 'Davidson-BradleyTerry-ties',
      reference: 'Davidson (1970), The Analysis of Frequency Data in Which Some Observations are Paired',
      outcomeEncoding: { left: 'left 商品获胜', right: 'right 商品获胜', tie: '平局（独立第三类结果，不拆为两条半胜）' },
      tieParameter: { name: 'theta', value: primary.theta, eta: primary.eta,
        prior: { family: 'log-normal via N(eta; 0, 1/lambdaEta)', lambdaEta: policy.lambdaEta },
        bootstrap: summarizeDistribution(bootstrap.replicates.map(r => r.theta)) },
      abilityParameter: {
        name: 'beta', constraint: '每个连通分量内 sum(beta)=0',
        prior: { family: 'N(beta; 0, 1/lambda)', lambda: policy.primaryLambda },
        separationHandling: '完全分离时无正则 MLE 发散；高斯先验把参数收缩到有限值，强度敏感性见 sensitivity',
      },
      fit: { converged: primary.converged, iterations: primary.iterations,
        objective: primary.objective, nll: primary.nll, penalty: primary.penalty },
      judgeTreatment: {
        rule: policy.judgeCap === null ? '不设界限，每条比较权重为 1'
          : `同一评委有效权重 = min(1, judgeCap / 该评委比较数)，单评委有效比较数封顶 ${policy.judgeCap}`,
        judgeCap: policy.judgeCap,
        judges: judges.map(judge => ({
          judge, rawComparisons: judgeCounts.get(judge),
          effectiveWeight: weightMap.get(judge),
          effectiveComparisons: judgeCounts.get(judge) * weightMap.get(judge),
        })),
        clusteredBootstrap: {
          resamplingUnit: '评委整组（同一评委的全部比较一起抽，重复比较不冒充独立评委）',
          replicates: bootstrap.replicates.length,
          requested: bootstrap.requested,
          failed: bootstrap.failed,
          seed: bootstrap.seed,
        },
      },
    },
    components: componentReports,
    pairwise,
    coverage,
    suggestions,
    sensitivity: {
      primaryLambda: policy.primaryLambda,
      grid: sensitivity.map(entry => ({
        lambda: entry.lambda, theta: entry.theta, objective: entry.objective,
        nll: entry.nll, primary: entry.primary,
        spearmanVsPrimary: entry.componentOrders.map((order, componentIndex) =>
          spearmanCorrelation(primarySensitivity.componentOrders[componentIndex], order)),
      })),
      orderingStableAcrossGrid: componentSensitivity.every(entry => entry.orderingStableAcrossGrid),
    },
    dataInfo: {
      recordCount: records.length,
      itemCount: items.length,
      judgeCount: judges.length,
      componentCount: components.length,
      outcomeCounts: countOutcomes(comparisons),
    },
  };

  const archive = buildArchive(records, policy, graph, report);
  return { report, archive };
}

function countOutcomes(comparisons) {
  const counts = { left: 0, right: 0, tie: 0 };
  for (const comparison of comparisons) counts[comparison.originalOutcome] += 1;
  return counts;
}

function buildPairwiseReport(graph, fit, pairStats, covariance, bootstrap) {
  const { items, components, componentId } = graph;
  const pairs = [];
  for (let indexA = 0; indexA < items.length; indexA += 1) {
    for (let indexB = indexA + 1; indexB < items.length; indexB += 1) {
      const itemA = items[indexA];
      const itemB = items[indexB];
      const sameComponent = componentId.get(itemA) === componentId.get(itemB);
      const key = itemA < itemB ? `${itemA}|${itemB}` : `${itemB}|${itemA}`;
      const observed = pairStats.get(key);
      const model = sameComponent ? modelPairProbabilities(fit, itemA, itemB) : null;

      let modelCi95 = null;
      if (sameComponent) {
        const winSamples = [];
        const lossSamples = [];
        const tieSamples = [];
        for (const replicate of bootstrap.replicates) {
          if (replicate.beta.get(itemA) === undefined || replicate.beta.get(itemB) === undefined) continue;
          const d = replicate.beta.get(itemA) - replicate.beta.get(itemB);
          const probabilities = outcomeProbabilities(d, replicate.theta);
          winSamples.push(probabilities.win);
          lossSamples.push(probabilities.loss);
          tieSamples.push(probabilities.tie);
        }
        // 三类概率的区间分别从各自的 bootstrap 样本计算，不做相减以免越界
        const winDistribution = summarizeDistribution(winSamples);
        const lossDistribution = summarizeDistribution(lossSamples);
        const tieDistribution = summarizeDistribution(tieSamples);
        modelCi95 = {
          aWins: [winDistribution.q025, winDistribution.q975],
          bWins: [lossDistribution.q025, lossDistribution.q975],
          tie: [tieDistribution.q025, tieDistribution.q975],
        };
      }

      // 解析近似：对比 d = beta_a - beta_b 的方差与 95% 区间
      let contrast = null;
      if (sameComponent) {
        const i = covariance.indexOf.get(itemA);
        const j = covariance.indexOf.get(itemB);
        const variance = covariance.matrix[i][i] + covariance.matrix[j][j]
          - 2 * covariance.matrix[i][j];
        const d = fit.beta.get(itemA) - fit.beta.get(itemB);
        contrast = {
          d,
          se: Math.sqrt(variance),
          ci95: [d - 1.96 * Math.sqrt(variance), d + 1.96 * Math.sqrt(variance)],
          excludesZero: Math.abs(d) > 1.96 * Math.sqrt(variance),
        };
      }

      pairs.push({
        items: [itemA, itemB],
        sameComponent,
        observed: observed ? {
          comparisons: observed.wins + observed.losses + observed.ties,
          aWins: observed.wins, bWins: observed.losses, ties: observed.ties,
          aWinRate: observed.wins / (observed.wins + observed.losses + observed.ties),
          tieRate: observed.ties / (observed.wins + observed.losses + observed.ties),
          distinctJudges: observed.judges.size,
          distinctRounds: observed.rounds.size,
        } : null,
        model: model ? {
          aWins: model.aWins, bWins: model.bWins, tie: model.tie,
          bootstrapCi95: modelCi95,
        } : null,
        contrast,
        ordering: sameComponent
          ? (fit.beta.get(itemA) > fit.beta.get(itemB) ? `${itemA} > ${itemB}`
            : fit.beta.get(itemA) < fit.beta.get(itemB) ? `${itemB} > ${itemA}` : `${itemA} = ${itemB}`)
          : '不可比较：图上无比较路径，不硬排先后',
      });
    }
  }
  return pairs;
}

function buildCoverage(graph, pairStats, judgeCounts, policy, weightMap) {
  const { items, comparisons, componentId } = graph;
  const perItem = items.map(item => {
    const involving = comparisons.filter(comparison =>
      comparison.a === item || comparison.b === item);
    const opponents = new Set();
    const judgesSeen = new Set();
    let effectiveComparisons = 0;
    for (const comparison of involving) {
      opponents.add(comparison.a === item ? comparison.b : comparison.a);
      judgesSeen.add(comparison.judge);
      effectiveComparisons += weightMap.get(comparison.judge) ?? 1;
    }
    return {
      item,
      component: componentId.get(item),
      rawComparisons: involving.length,
      effectiveComparisons,
      distinctOpponents: opponents.size,
      distinctJudges: judgesSeen.size,
      undercovered: involving.length < policy.minComparisons
        || judgesSeen.size < Math.min(2, involving.length),
      gaps: [
        ...(involving.length < policy.minComparisons
          ? [`比较数 ${involving.length} < 最小覆盖 ${policy.minComparisons}`] : []),
        ...(judgesSeen.size < Math.min(2, involving.length)
          ? ['只有单一评委覆盖，评委整组不确定性高'] : []),
      ],
    };
  });
  return {
    minComparisons: policy.minComparisons,
    items: perItem,
    undercoveredItems: perItem.filter(entry => entry.undercovered).map(entry => entry.item),
  };
}

// 补比较建议。打分依据：
//  1. 同分量对：再做一次比较对 beta 对比带来的期望 Fisher 信息
//     w = p_win*p_loss + 0.25*p_tie*(1-p_tie)（与未知结果无关，只依赖当前拟合），
//     信息越大、当前对比方差越大，越值得补；
//  2. 跨分量对：比较会首次建立连接路径（桥），价值最高，按分量大小与
//     覆盖缺口排序；
//  3. 覆盖约束：覆盖不足商品的对优先；已有比较达到 pairCap 的对降级。
// 建议只指定“补哪两个商品、为什么、期望信息是多少”，不预设结果。
function buildSuggestions(graph, fit, covariance, pairStats, coverage, policy) {
  const { items, components, componentId } = graph;
  const coverageByName = new Map(coverage.items.map(entry => [entry.item, entry]));
  const pairCap = policy.minComparisons;
  const candidates = [];

  for (let indexA = 0; indexA < items.length; indexA += 1) {
    for (let indexB = indexA + 1; indexB < items.length; indexB += 1) {
      const itemA = items[indexA];
      const itemB = items[indexB];
      const key = `${itemA}|${itemB}`;
      const observed = pairStats.get(key);
      const observedCount = observed ? observed.wins + observed.losses + observed.ties : 0;
      const sameComponent = componentId.get(itemA) === componentId.get(itemB);
      const undercovered = coverageByName.get(itemA).undercovered
        || coverageByName.get(itemB).undercovered;

      if (sameComponent) {
        const d = fit.beta.get(itemA) - fit.beta.get(itemB);
        const { win, loss, tie } = outcomeProbabilities(d, fit.theta);
        // 期望 Fisher 信息（与拟合 Hessian 同一公式），只依赖当前拟合概率，
        // 与“下一次比较会出现什么结果”无关。
        const expectedInfo = win + loss - (win - loss) ** 2;
        const i = covariance.indexOf.get(itemA);
        const j = covariance.indexOf.get(itemB);
        const contrastVariance = covariance.matrix[i][i] + covariance.matrix[j][j]
          - 2 * covariance.matrix[i][j];
        // 已充分覆盖且无覆盖缺口的对不再推荐
        if (observedCount >= pairCap && !undercovered) continue;
        const reasons = [];
        if (observedCount < pairCap) reasons.push(`该对仅比较 ${observedCount} 次（覆盖上限 ${pairCap}）`);
        if (undercovered) reasons.push('至少一方比较不足或评委单一');
        if (contrastVariance > 1) reasons.push('当前能力对比的后验方差较大');
        candidates.push({
          items: [itemA, itemB],
          kind: 'within-component',
          connected: true,
          observedComparisons: observedCount,
          expectedFisherInfo: expectedInfo,
          contrastVariance,
          score: expectedInfo * contrastVariance * (undercovered ? 2 : 1)
            * (observedCount === 0 ? 1.5 : 1),
          reasons,
        });
      } else {
        const sizeA = components[componentId.get(itemA)].length;
        const sizeB = components[componentId.get(itemB)].length;
        const reasons = ['比较后将首次连通两个断开分量，消除不可比较状态'];
        if (undercovered) reasons.push('至少一方比较不足或评委单一');
        candidates.push({
          items: [itemA, itemB],
          kind: 'bridge',
          connected: false,
          observedComparisons: observedCount,
          expectedFisherInfo: null,
          contrastVariance: null,
          score: 1000 * (sizeA + sizeB) * (undercovered ? 2 : 1) + (observedCount === 0 ? 1 : 0),
          reasons,
        });
      }
    }
  }

  candidates.sort((left, right) => {
    if (left.kind !== right.kind) return left.kind === 'bridge' ? -1 : 1;
    if (right.score !== left.score) return right.score - left.score;
    return left.items[0].localeCompare(right.items[0], 'en')
      || left.items[1].localeCompare(right.items[1], 'en');
  });

  return {
    method: '期望 Fisher 信息（Davidson 模型，与未知比较结果无关）+ 覆盖约束；不假设新比较的任何结果',
    pairCap,
    candidates: candidates.slice(0, policy.suggestionCount),
    totalEligiblePairs: candidates.length,
  };
}

function buildArchive(records, policy, graph, report) {
  const sortedRecords = [...records].sort((left, right) => left.id.localeCompare(right.id, 'en'));
  const dataFingerprint = fingerprint(sortedRecords);
  const itemDictionary = graph.items.map(item => ({
    item,
    component: graph.componentId.get(item),
    componentMembers: graph.components[graph.componentId.get(item)],
  }));
  return {
    archiveVersion: MODEL_ARCHIVE_VERSION,
    createdAt: new Date().toISOString(),
    dataFingerprint,
    recordIds: sortedRecords.map(record => record.id),
    itemDictionary,
    parameterConstraints: {
      family: report.model.family,
      ability: report.model.abilityParameter,
      tie: report.model.tieParameter,
      componentConstraint: '每个连通分量内 beta 之和为零；跨分量不定义共同尺度',
      identifiability: '断开分量之间不存在可识别的相对能力，档案不提供全局总榜',
    },
    judgeTreatment: report.model.judgeTreatment,
    policy,
    report,
    canonicalFingerprint: fingerprint(canonical(report)),
  };
}

// 重载档案：校验数据指纹一致后直接用档案中的报告解释，不重新拟合。
export async function loadArchive(path) {
  const archive = JSON.parse(await readFile(path, 'utf8'));
  if (archive.archiveVersion !== MODEL_ARCHIVE_VERSION) {
    throw new Error(`不支持的模型档案版本：${archive.archiveVersion}`);
  }
  return archive;
}

export async function reloadArchive(archivePath, inputPath) {
  const archive = await loadArchive(archivePath);
  const { records } = await (async () => {
    const { loadDataset } = await import('./dataset.js');
    return loadDataset(inputPath);
  })();
  const currentFingerprint = fingerprint(
    [...records].sort((left, right) => left.id.localeCompare(right.id, 'en')));
  if (currentFingerprint !== archive.dataFingerprint) {
    throw new Error('输入数据指纹与模型档案不一致，档案不能用于解释当前数据；请重新运行 rank');
  }
  return {
    reloaded: true,
    dataFingerprint: archive.dataFingerprint,
    policy: archive.policy,
    itemDictionary: archive.itemDictionary,
    parameterConstraints: archive.parameterConstraints,
    judgeTreatment: archive.judgeTreatment,
    report: archive.report,
  };
}
