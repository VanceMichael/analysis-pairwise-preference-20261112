import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildRankReport, resolvePolicy, DEFAULT_POLICY, loadArchive, reloadArchive, MODEL_ARCHIVE_VERSION } from '../src/ranking.js';
import { fingerprint } from '../src/dataset.js';

const policy = {
  topK: 2, regularization: [0.1, 1, 10], primaryLambda: 1, lambdaEta: 1,
  judgeCap: 5, bootstrapReplicates: 200, minComparisons: 3, suggestionCount: 5, seed: 31,
};
let counter = 0;
const record = (judge, left, right, outcome, round = 1) =>
  ({ id: `t${counter++}`, judge, left, right, outcome, round });

function reportOf(records, overrides = {}) {
  counter = 0;
  return buildRankReport(records, { ...policy, ...overrides }).report;
}

test('policy 校验拒绝非法配置并提供默认值', () => {
  assert.equal(resolvePolicy({}).primaryLambda, DEFAULT_POLICY.primaryLambda);
  assert.throws(() => resolvePolicy({ primaryLambda: 5, regularization: [1, 2] }), /primaryLambda/);
  assert.throws(() => resolvePolicy({ regularization: [1] }), /regularization/);
  assert.throws(() => resolvePolicy({ judgeCap: 0 }), /judgeCap/);
  assert.throws(() => resolvePolicy({ bootstrapReplicates: 10 }), /bootstrapReplicates/);
  assert.doesNotThrow(() => resolvePolicy({ judgeCap: null }));
});

test('断开分量不生成全局总榜，跨分量对标记不可比较并优先推荐桥接比较', () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j2', 'a', 'b', 'right'), record('j3', 'a', 'b', 'tie'),
    record('j4', 'c', 'd', 'left'), record('j5', 'c', 'd', 'right'), record('j6', 'c', 'd', 'tie'),
  ];
  const report = reportOf(records);
  assert.equal(report.components.length, 2);
  assert.ok(!('globalRanking' in report));
  const crossPairs = report.pairwise.filter(pair => !pair.sameComponent);
  assert.equal(crossPairs.length, 4);
  for (const pair of crossPairs) {
    assert.equal(pair.model, null);
    assert.equal(pair.contrast, null);
    assert.match(pair.ordering, /不可比较/);
  }
  const bridges = report.suggestions.candidates.filter(s => s.kind === 'bridge');
  assert.ok(bridges.length >= 1);
  assert.ok(bridges.every(bridge => {
    const [one, two] = bridge.items;
    return report.components.some(c => c.members.includes(one))
      && report.components.some(c => c.members.includes(two))
      && report.components.find(c => c.members.includes(one))
        !== report.components.find(c => c.members.includes(two));
  }));
});

test('同分量商品对给出经验胜率、模型概率与 bootstrap 区间，区间不越界', () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j2', 'a', 'b', 'left'),
    record('j3', 'a', 'b', 'right'), record('j4', 'a', 'b', 'tie'),
    record('j5', 'b', 'c', 'left'), record('j6', 'b', 'c', 'right'),
  ];
  const report = reportOf(records);
  const ab = report.pairwise.find(pair => pair.items.join('') === 'ab');
  assert.equal(ab.observed.comparisons, 4);
  assert.equal(ab.observed.aWins + ab.observed.bWins + ab.observed.ties, 4);
  assert.ok(Math.abs(ab.observed.aWinRate - 0.5) < 1e-12);
  assert.equal(ab.model.aWins + ab.model.bWins + ab.model.tie - 1 < 1e-12, true);
  for (const bound of ab.model.bootstrapCi95.aWins) {
    assert.ok(bound >= 0 && bound <= 1);
  }
  assert.ok(ab.contrast.se > 0);
});

test('补比较建议只基于已拟合信息与覆盖约束，不含任何假设结果', () => {
  const records = [
    record('j1', 'a', 'b', 'left'),
    record('j2', 'c', 'd', 'left'),
  ];
  const report = reportOf(records, { minComparisons: 3, suggestionCount: 8 });
  assert.ok(report.suggestions.candidates.length > 0);
  for (const candidate of report.suggestions.candidates) {
    assert.ok(!('assumedOutcome' in candidate) && !('predictedWinner' in candidate));
    assert.ok(Array.isArray(candidate.reasons) && candidate.reasons.length > 0);
    if (candidate.kind === 'within-component') {
      assert.ok(candidate.expectedFisherInfo > 0 && candidate.expectedFisherInfo <= 0.5);
      assert.ok(candidate.contrastVariance > 0);
    } else {
      assert.equal(candidate.expectedFisherInfo, null);
    }
  }
  assert.match(report.suggestions.method, /Fisher/);
});

test('覆盖不足与单评委商品被明确诊断', () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j1', 'a', 'b', 'right'),
    record('j2', 'b', 'c', 'left'), record('j3', 'b', 'c', 'right'),
    record('j4', 'b', 'c', 'tie'),
  ];
  const report = reportOf(records, { minComparisons: 3 });
  const coverageA = report.coverage.items.find(item => item.item === 'a');
  assert.equal(coverageA.distinctJudges, 1);
  assert.ok(coverageA.undercovered);
  assert.ok(coverageA.gaps.some(gap => /单一评委/.test(gap)));
  assert.ok(report.coverage.undercoveredItems.includes('a'));
  const rankedA = report.components[0].rankedItems.find(item => item.item === 'a');
  assert.ok(rankedA.warnings.some(warning => /单一评委/.test(warning)));
});

test('正则敏感性网格覆盖全部 lambda 并报告顺序稳定性', () => {
  const records = [];
  for (let k = 0; k < 8; k += 1) {
    records.push(record(`j${k}`, 'a', 'b', k % 3 === 0 ? 'tie' : 'left'));
    records.push(record(`k${k}`, 'b', 'c', k % 2 === 0 ? 'right' : 'left'));
  }
  const report = reportOf(records);
  assert.deepEqual(report.sensitivity.grid.map(entry => entry.lambda), [0.1, 1, 10]);
  assert.equal(report.sensitivity.primaryLambda, 1);
  for (const entry of report.sensitivity.grid) {
    assert.ok(Number.isFinite(entry.theta) && Number.isFinite(entry.objective));
    assert.ok(entry.spearmanVsPrimary.every(value => value === null || Math.abs(value) <= 1));
  }
  for (const component of report.components) {
    assert.equal(component.sensitivity.grid.length, 3);
    assert.equal(typeof component.sensitivity.orderingStableAcrossGrid, 'boolean');
  }
});

test('前 K 稳定性以 bootstrap 概率与名次区间呈现，而非单点数字', () => {
  const records = [];
  for (let k = 0; k < 12; k += 1) {
    records.push(record(`j${k}`, 'a', 'b', 'left'));
    records.push(record(`x${k}`, 'b', 'c', k % 4 === 0 ? 'right' : 'left'));
  }
  const report = reportOf(records, { topK: 1 });
  const ranked = report.components[0].rankedItems;
  for (const entry of ranked) {
    assert.ok(entry.topKProbability >= 0 && entry.topKProbability <= 1);
    assert.ok(entry.rankCI95[0] >= 1 && entry.rankCI95[1] <= ranked.length);
    assert.ok(entry.rankCI95[0] <= entry.rankCI95[1]);
    const probabilitySum = Object.values(entry.rankProbability)
      .reduce((sum, value) => sum + (value ?? 0), 0);
    assert.ok(Math.abs(probabilitySum - 1) < 1e-9);
  }
  const topProbabilities = ranked.map(entry => entry.topKProbability);
  assert.ok(Math.abs(topProbabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
});

test('单评委刷票被界限降权，经验胜率与收缩后的模型概率同时呈现', () => {
  const records = [];
  for (let k = 0; k < 20; k += 1) records.push(record('j1', 'a', 'b', 'left'));
  records.push(record('j2', 'a', 'b', 'right'), record('j3', 'a', 'b', 'tie'));
  const report = reportOf(records, { judgeCap: 5 });
  const judge = report.model.judgeTreatment.judges.find(entry => entry.judge === 'j1');
  assert.equal(judge.rawComparisons, 20);
  assert.equal(judge.effectiveComparisons, 5);
  const ab = report.pairwise.find(pair => pair.items.join('') === 'ab');
  assert.ok(ab.observed.aWinRate > 0.85); // 原始记录里的压倒性“胜场榜”
  assert.ok(ab.model.aWins < ab.observed.aWinRate); // 界限后模型明显收缩
});

test('全平局报告并列名次、高平局参数，且不把平局拆成半胜', () => {
  const records = Array.from({ length: 10 }, (_, k) => record(`j${k % 4}`, 'a', 'b', 'tie'));
  const report = reportOf(records);
  assert.deepEqual(report.components[0].tiedGroups, [['a', 'b']]);
  assert.ok(report.components[0].rankedItems.every(entry => entry.pointEstimateTie));
  assert.ok(report.model.tieParameter.value > 1);
  const ab = report.pairwise.find(pair => pair.items.join('') === 'ab');
  assert.equal(ab.observed.ties, 10);
  assert.equal(ab.observed.aWins, 0);
  assert.equal(ab.observed.bWins, 0);
  assert.ok(ab.model.tie > ab.model.aWins);
});

test('循环偏好呈现三者并列而不是硬排出先后', () => {
  const records = [];
  for (let k = 0; k < 6; k += 1) {
    records.push(record(`j${k}`, 'a', 'b', 'left'));
    records.push(record(`j${k}`, 'b', 'c', 'left'));
    records.push(record(`j${k}`, 'c', 'a', 'left'));
  }
  const report = reportOf(records);
  const betas = report.components[0].rankedItems.map(entry => entry.beta);
  assert.ok(Math.max(...betas) - Math.min(...betas) < 1e-8);
  assert.deepEqual(report.components[0].tiedGroups.map(group => group.sort()).sort(),
    [['a', 'b', 'c']]);
});

test('左右交换整条记录集合不改变拟合结果', () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j2', 'a', 'b', 'tie'),
    record('j3', 'b', 'c', 'right'), record('j4', 'a', 'c', 'left'),
  ];
  const swapped = records.map((entry, index) => {
    if (entry.outcome === 'tie') return { ...entry, id: `s${index}`, left: entry.right, right: entry.left };
    return {
      ...entry, id: `s${index}`, left: entry.right, right: entry.left,
      outcome: entry.outcome === 'left' ? 'right' : 'left',
    };
  });
  const first = reportOf(records);
  const second = reportOf(swapped);
  for (const item of ['a', 'b', 'c']) {
    const beta1 = first.components[0].rankedItems.find(r => r.item === item).beta;
    const beta2 = second.components[0].rankedItems.find(r => r.item === item).beta;
    assert.ok(Math.abs(beta1 - beta2) < 1e-10);
  }
  assert.ok(Math.abs(first.model.tieParameter.value - second.model.tieParameter.value) < 1e-10);
});

test('模型档案保存字典、约束、评委处理与数据指纹，重载后可直接解释', async () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j2', 'a', 'b', 'right'),
    record('j3', 'c', 'd', 'tie'),
  ];
  counter = 0;
  const { archive } = buildRankReport(records, policy);
  const folder = await mkdtemp(join(tmpdir(), 'archive-'));
  try {
    const path = join(folder, 'model.json');
    await writeFile(path, JSON.stringify(archive));
    const reloaded = await loadArchive(path);
    assert.equal(reloaded.archiveVersion, MODEL_ARCHIVE_VERSION);
    assert.equal(reloaded.dataFingerprint, archive.dataFingerprint);
    assert.deepEqual(reloaded.itemDictionary.map(entry => entry.item).sort(), ['a', 'b', 'c', 'd']);
    assert.match(reloaded.parameterConstraints.family, /Davidson/);
    assert.match(reloaded.parameterConstraints.identifiability, /分量/);
    assert.ok(reloaded.judgeTreatment.clusteredBootstrap.seed === policy.seed);
    assert.equal(reloaded.policy.judgeCap, policy.judgeCap);

    const expectedFingerprint = fingerprint([...records].sort((l, r) => l.id.localeCompare(r.id, 'en')));
    assert.equal(reloaded.dataFingerprint, expectedFingerprint);

    const explained = await reloadArchive(path, await writeInput(folder, records));
    assert.equal(explained.reloaded, true);
    assert.equal(explained.report.components.length, archive.report.components.length);

    const tampered = records.map((entry, index) =>
      index === 0 ? { ...entry, outcome: 'tie' } : entry);
    await writeInput(folder, tampered, 'tampered.json');
    await assert.rejects(
      reloadArchive(path, join(folder, 'tampered.json')), /指纹/);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

async function writeInput(folder, records, name = 'records.json') {
  const path = join(folder, name);
  await writeFile(path, JSON.stringify({ version: 1, records }));
  return path;
}

test('相同商品比较与非法 outcome 在报告入口被拒绝', () => {
  assert.throws(() => reportOf([record('j', 'a', 'a', 'left')]), /相同商品/);
  assert.throws(() => reportOf([record('j', 'a', 'b', 'half-win')]), /outcome/);
});

test('结果对输入行序不变且可复现（固定种子）', () => {
  const records = [];
  for (let k = 0; k < 10; k += 1) {
    records.push(record(`j${k}`, 'a', 'b', k % 3 === 0 ? 'tie' : 'left'));
    records.push(record(`x${k}`, 'b', 'c', k % 2 ? 'right' : 'left'));
  }
  const first = reportOf(records);
  const second = reportOf([...records].reverse());
  assert.deepEqual(
    first.components.map(c => c.order),
    second.components.map(c => c.order));
  assert.deepEqual(
    first.components[0].rankedItems.map(r => r.rankProbability),
    second.components[0].rankedItems.map(r => r.rankProbability));
});

test('固定种子下重复构建报告完全一致，换种子只改变 bootstrap 抽样细节', () => {
  const records = [];
  for (let k = 0; k < 10; k += 1) {
    records.push(record(`j${k}`, 'a', 'b', k % 3 === 0 ? 'tie' : 'left'));
    records.push(record(`x${k}`, 'b', 'c', k % 2 ? 'right' : 'left'));
  }
  const snapshot = report => JSON.stringify(
    report.components[0].rankedItems.map(entry =>
      [entry.item, entry.beta, entry.rankProbability, entry.topKProbability]));
  const first = reportOf(records);
  const second = reportOf(records);
  assert.equal(snapshot(first), snapshot(second));
  const otherSeed = reportOf(records, { seed: 777 });
  // 点估计与种子无关，bootstrap 概率允许小幅差异
  assert.deepEqual(first.components[0].order, otherSeed.components[0].order);
  const firstProb = first.components[0].rankedItems.map(entry => entry.topKProbability);
  const otherProb = otherSeed.components[0].rankedItems.map(entry => entry.topKProbability);
  for (let index = 0; index < firstProb.length; index += 1) {
    assert.ok(Math.abs(firstProb[index] - otherProb[index]) < 0.2);
  }
});

test('对称比较（互换朝向的等价记录集）给出相同的三分类概率与能力', () => {
  const records = [
    record('j1', 'a', 'b', 'left'), record('j2', 'a', 'b', 'right'),
    record('j3', 'a', 'b', 'tie'), record('j4', 'b', 'a', 'tie'),
  ];
  const report = reportOf(records);
  const ab = report.pairwise.find(pair => pair.items.join('') === 'ab');
  assert.ok(Math.abs(ab.model.aWins - ab.model.bWins) < 1e-12);
  const betas = report.components[0].rankedItems.map(entry => entry.beta);
  assert.ok(Math.abs(betas[0] - betas[1]) < 1e-10);
  assert.deepEqual(report.components[0].tiedGroups, [['a', 'b']]);
});
