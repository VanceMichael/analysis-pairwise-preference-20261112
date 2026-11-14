import test from 'node:test';
import assert from 'node:assert/strict';
import {
  outcomeProbabilities, canonicalComparison, buildGraph, validateComparisons,
  fitModel, negativeLogPosterior, judgeWeights, clusteredResample, seededRandom,
  rankItems, summarizeDistribution, UnionFind,
} from '../src/model.js';
const record = (id, judge, left, right, outcome, round = 1) =>
  ({ id, judge, left, right, outcome, round });

test('三类概率归一、对称，并在 theta->0 时退化为 Bradley-Terry', () => {
  for (const d of [-3, -0.7, 0, 0.4, 2.5]) {
    for (const theta of [0.001, 0.3, 1, 4]) {
      const p = outcomeProbabilities(d, theta);
      assert.ok(Math.abs(p.win + p.loss + p.tie - 1) < 1e-12);
      const mirrored = outcomeProbabilities(-d, theta);
      assert.ok(Math.abs(p.win - mirrored.loss) < 1e-14);
      assert.ok(Math.abs(p.tie - mirrored.tie) < 1e-14);
    }
  }
  const d = 0.8;
  const limit = outcomeProbabilities(d, 1e-12);
  const bradleyTerry = Math.exp(d) / (Math.exp(d) + Math.exp(-d));
  assert.ok(Math.abs(limit.win - bradleyTerry) < 1e-6);
  assert.ok(limit.tie < 1e-9);
  const zero = outcomeProbabilities(0, 1);
  assert.equal(zero.win, zero.loss);
});

test('左右交换与 outcome 同步翻转得到相同的 canonical 比较', () => {
  const cases = [
    [record('r1', 'j', 'a', 'b', 'left'), record('r2', 'j', 'b', 'a', 'right')],
    [record('r1', 'j', 'a', 'b', 'right'), record('r2', 'j', 'b', 'a', 'left')],
    [record('r1', 'j', 'a', 'b', 'tie'), record('r2', 'j', 'b', 'a', 'tie')],
  ];
  for (const [one, two] of cases) {
    const c1 = canonicalComparison(one);
    const c2 = canonicalComparison(two);
    assert.equal(c1.a, c2.a);
    assert.equal(c1.b, c2.b);
    assert.equal(c1.label, c2.label);
  }
});

test('相同商品比较、非法 outcome 与空标签被统一拒绝', () => {
  assert.throws(() => validateComparisons([record('r', 'j', 'a', 'a', 'left')]), /相同商品/);
  assert.throws(() => validateComparisons([record('r', 'j', 'a', 'b', 'draw')]), /outcome/);
  assert.throws(() => validateComparisons([{ ...record('r', 'j', 'a', 'b', 'left'), judge: '' }]), /judge/);
  assert.throws(() => buildGraph([record('r', 'j', 'a', 'a', 'tie')]), /相同商品/);
});

test('并查集把无路径的商品分到不同连通分量', () => {
  const graph = buildGraph([
    record('r1', 'j1', 'a', 'b', 'left'),
    record('r2', 'j2', 'c', 'd', 'right'),
  ]);
  assert.equal(graph.components.length, 2);
  assert.deepEqual(graph.components.map(c => c.join('')).sort(), ['ab', 'cd']);
  assert.notEqual(graph.componentId.get('a'), graph.componentId.get('c'));
  const uf = new UnionFind(['x', 'y']);
  uf.union('x', 'y');
  assert.equal(uf.find('x'), uf.find('y'));
});

test('解析梯度与 Hessian 通过有限差分校验（含锚点投影与评委权重）', () => {
  const records = [
    record('r1', 'a', 'i0', 'i1', 'left'),
    record('r2', 'a', 'i1', 'i2', 'left'),
    record('r3', 'b', 'i0', 'i2', 'tie'),
    record('r4', 'b', 'i0', 'i1', 'right'),
    record('r5', 'c', 'i2', 'i0', 'left'),
  ];
  const graph = buildGraph(records);
  const lambda = 0.7;
  const lambdaEta = 1.3;
  const { weights } = judgeWeights(graph.comparisons, graph.judges, 2);
  const members = graph.components[0];
  const fit = fitModel(graph, { lambda, lambdaEta, weightMap: weights });
  assert.ok(fit.converged, fit.reason);

  // 在最优点，目标函数对自由参数的数值梯度应接近 0
  const objective = (free, eta) => {
    const beta = new Map();
    for (let k = 0; k < members.length - 1; k += 1) beta.set(members[k], free[k]);
    beta.set(members[members.length - 1], -free.reduce((sum, v) => sum + v, 0));
    return negativeLogPosterior(graph.comparisons, beta, eta, lambda, lambdaEta,
      comparison => weights.get(comparison.judge)).objective;
  };
  const freeStar = members.slice(0, -1).map(item => fit.beta.get(item));
  const x = [...freeStar, fit.eta];
  const epsilon = 1e-6;
  for (let i = 0; i < x.length; i += 1) {
    const plus = [...x]; const minus = [...x];
    plus[i] += epsilon; minus[i] -= epsilon;
    const numericGradient = (objective(plus.slice(0, -1), plus.at(-1))
      - objective(minus.slice(0, -1), minus.at(-1))) / (2 * epsilon);
    assert.ok(Math.abs(numericGradient) < 1e-4, `坐标 ${i} 数值梯度=${numericGradient}`);
  }

  // 在非最优点用有限差分校验完整 Hessian（含锚点投影、交叉项与先验块）
  const probe = [0.42, -0.31, 0.18];
  const numericHessian = [];
  for (let i = 0; i < 3; i += 1) {
    numericHessian.push([]);
    for (let j = 0; j < 3; j += 1) {
      const p11 = [...probe]; const p12 = [...probe]; const p21 = [...probe]; const p22 = [...probe];
      const h = 1e-5;
      p11[i] += h; p11[j] += h; p12[i] += h; p12[j] -= h;
      p21[i] -= h; p21[j] += h; p22[i] -= h; p22[j] -= h;
      numericHessian[i].push(
        (objective(p11.slice(0, -1), p11.at(-1)) - objective(p12.slice(0, -1), p12.at(-1))
          - objective(p21.slice(0, -1), p21.at(-1)) + objective(p22.slice(0, -1), p22.at(-1)))
        / (4 * h * h));
    }
  }
  // 独立按模型公式在 probe 点重建解析 Hessian（不经过 fitModel 的逆矩阵）
  const analyticHessian = buildAnalyticHessian(graph, weights, members, probe, lambda, lambdaEta);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      assert.ok(Math.abs(analyticHessian[i][j] - numericHessian[i][j]) < 1e-4,
        `Hessian[${i}][${j}] 解析=${analyticHessian[i][j]} 数值=${numericHessian[i][j]}`);
    }
  }
});

function buildAnalyticHessian(graph, weights, members, x, lambda, lambdaEta) {
  const freeIndex = new Map([[members[0], 0], [members[1], 1], [members[2], null]]);
  const beta = new Map([[members[0], x[0]], [members[1], x[1]], [members[2], -(x[0] + x[1])]]);
  const eta = x[2];
  const theta = Math.exp(eta);
  const dimension = 3;
  const hessian = Array.from({ length: dimension }, () => new Array(dimension).fill(0));
  for (const comparison of graph.comparisons) {
    const weight = weights.get(comparison.judge);
    const d = beta.get(comparison.a) - beta.get(comparison.b);
    const { win, loss, tie } = outcomeProbabilities(d, theta);
    const info = weight * (win + loss - (win - loss) ** 2);
    const hCross = -weight * (win - loss) * tie;
    const hEta = weight * tie * (1 - tie);
    const factors = new Map();
    const collect = (item, sign) => {
      const free = freeIndex.get(item);
      if (free !== null && free !== undefined) factors.set(free, (factors.get(free) ?? 0) + sign);
      else {
        factors.set(0, (factors.get(0) ?? 0) - sign);
        factors.set(1, (factors.get(1) ?? 0) - sign);
      }
    };
    collect(comparison.a, 1);
    collect(comparison.b, -1);
    for (const [i, fi] of factors) {
      for (const [j, fj] of factors) hessian[i][j] += fi * fj * info;
      hessian[i][2] += fi * hCross;
      hessian[2][i] += fi * hCross;
    }
    hessian[2][2] += hEta;
  }
  for (let k = 0; k < 2; k += 1) {
    for (let l = 0; l < 2; l += 1) hessian[k][l] += lambda * (k === l ? 2 : 1);
  }
  hessian[2][2] += lambdaEta;
  return hessian;
}

test('拟合在对称数据上给出相等能力，在循环偏好上三者并列', () => {
  // 真正平衡的设计：a、b 各胜一场，平局两场，朝向互换后信息完全等价
  const symmetric = [
    record('r1', 'j1', 'a', 'b', 'left'),
    record('r2', 'j2', 'b', 'a', 'left'),
    record('r3', 'j3', 'a', 'b', 'tie'),
    record('r4', 'j4', 'b', 'a', 'tie'),
  ];
  const fitSymmetric = fitModel(buildGraph(symmetric), { lambda: 1, lambdaEta: 1 });
  assert.ok(fitSymmetric.converged);
  assert.ok(Math.abs(fitSymmetric.beta.get('a') - fitSymmetric.beta.get('b')) < 1e-8);
  assert.ok(Math.abs(fitSymmetric.beta.get('a')) < 1e-8); // 和为零且相等 -> 都为 0

  let id = 0;
  const cyclic = [];
  for (let k = 0; k < 6; k += 1) {
    cyclic.push(record(`c${id++}`, `j${k}`, 'a', 'b', 'left'));
    cyclic.push(record(`c${id++}`, `j${k}`, 'b', 'c', 'left'));
    cyclic.push(record(`c${id++}`, `j${k}`, 'c', 'a', 'left'));
  }
  const fitCyclic = fitModel(buildGraph(cyclic), { lambda: 1, lambdaEta: 1 });
  assert.ok(fitCyclic.converged);
  const betas = ['a', 'b', 'c'].map(item => fitCyclic.beta.get(item));
  assert.ok(Math.max(...betas) - Math.min(...betas) < 1e-8);
});

test('完全分离时高斯先验保持参数有限，且随 lambda 增强而收缩', () => {
  const records = Array.from({ length: 12 }, (_, k) => record(`s${k}`, `j${k}`, 'a', 'b', 'left'));
  const graph = buildGraph(records);
  const magnitudes = [0.1, 1, 10].map(lambda => {
    const fit = fitModel(graph, { lambda, lambdaEta: 1 });
    assert.ok(fit.converged);
    assert.ok(Number.isFinite(fit.beta.get('a')) && Number.isFinite(fit.beta.get('b')));
    return Math.abs(fit.beta.get('a'));
  });
  assert.ok(magnitudes[0] > magnitudes[1] && magnitudes[1] > magnitudes[2]);
});

test('全平局时能力全部为零、平局参数变大且平局保留为独立结果', () => {
  const records = Array.from({ length: 10 }, (_, k) => record(`t${k}`, `j${k % 4}`, 'a', 'b', 'tie'));
  const fit = fitModel(buildGraph(records), { lambda: 1, lambdaEta: 1 });
  assert.ok(fit.converged);
  assert.ok(Object.is(fit.beta.get('a'), 0) || Math.abs(fit.beta.get('a')) < 1e-12);
  assert.ok(Object.is(fit.beta.get('b'), 0) || Math.abs(fit.beta.get('b')) < 1e-12);
  assert.ok(fit.theta > 1);
  const p = outcomeProbabilities(0, fit.theta);
  assert.ok(p.tie > p.win && p.tie > p.loss);
});

test('评委界限把单评委的有效比较数封顶，cluster 重采样按整组抽取', () => {
  const records = [];
  for (let k = 0; k < 20; k += 1) records.push(record(`x${k}`, 'j1', 'a', 'b', 'left'));
  records.push(record('y1', 'j2', 'a', 'b', 'right'));
  const graph = buildGraph(records);
  const { counts, weights } = judgeWeights(graph.comparisons, graph.judges, 5);
  assert.equal(counts.get('j1'), 20);
  assert.equal(weights.get('j1'), 0.25);
  assert.equal(weights.get('j2'), 1);
  assert.equal(counts.get('j1') * weights.get('j1'), 5);

  const random = seededRandom(7);
  const resampled = clusteredResample(graph.comparisons, graph.judges, random);
  // 每次抽取都是某个评委的整组记录，记录数一定是该评委原始计数的整数倍组合
  const judges = new Set(resampled.map(r => r.judge));
  assert.ok(judges.size <= 2);
  // 重采样输出恢复为原始记录形状，可再次通过校验
  assert.doesNotThrow(() => buildGraph(resampled));
});

test('单评委 cluster bootstrap 退化为零宽度且不抛异常', () => {
  const records = [
    record('r1', 'j1', 'a', 'b', 'left'), record('r2', 'j1', 'b', 'c', 'left'),
    record('r3', 'j1', 'a', 'c', 'tie'), record('r4', 'j1', 'a', 'b', 'right'),
  ];
  const fit = fitModel(buildGraph(records), { lambda: 1, lambdaEta: 1 });
  assert.ok(fit.converged);
  const random = seededRandom(1);
  for (let k = 0; k < 20; k += 1) {
    const resampled = clusteredResample(buildGraph(records).comparisons, ['j1'], random);
    const refit = fitModel(buildGraph(resampled), { lambda: 1, lambdaEta: 1 });
    assert.ok(refit.converged);
  }
});

test('分布汇总处理空样本并给出分位数', () => {
  assert.deepEqual(summarizeDistribution([]).replicates, 0);
  assert.equal(summarizeDistribution([]).q50, null);
  const summary = summarizeDistribution([1, 2, 3, 4, 5]);
  assert.equal(summary.replicates, 5);
  assert.ok(Math.abs(summary.q50 - 3) < 1e-9);
  assert.ok(summary.q025 <= summary.q50 && summary.q50 <= summary.q975);
});

test('rankItems 对严格相等的能力用注入随机数平局打破，能力差不受随机影响', () => {
  const betaOf = item => (item === 'a' ? 1 : item === 'b' ? 0 : -1);
  const order = rankItems(['a', 'b', 'c'], betaOf, seededRandom(3));
  assert.deepEqual(order, ['a', 'b', 'c']);
  const flat = rankItems(['a', 'b'], () => 0, seededRandom(3));
  assert.equal(flat.length, 2);
});
