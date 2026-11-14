import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { buildRankReport } from '../src/ranking.js';
import { fingerprint } from '../src/dataset.js';

// 独立核对的小数据集与期望值（data/expected-rank.json 由一次性脚本生成，
// 这里重新拟合并逐项对照，防止模型公式或正则被悄悄改动）。
const records = [
  { id: 'e1', judge: 'j1', left: 'x', right: 'y', outcome: 'left', round: 1 },
  { id: 'e2', judge: 'j2', left: 'x', right: 'y', outcome: 'left', round: 1 },
  { id: 'e3', judge: 'j3', left: 'x', right: 'y', outcome: 'right', round: 1 },
  { id: 'e4', judge: 'j4', left: 'x', right: 'y', outcome: 'tie', round: 1 },
  { id: 'e5', judge: 'j1', left: 'y', right: 'z', outcome: 'left', round: 1 },
  { id: 'e6', judge: 'j2', left: 'y', right: 'z', outcome: 'right', round: 1 },
  { id: 'e7', judge: 'j3', left: 'x', right: 'z', outcome: 'tie', round: 1 },
];
const policy = {
  topK: 2, regularization: [0.5, 1, 2], primaryLambda: 1, lambdaEta: 1,
  judgeCap: null, bootstrapReplicates: 400, minComparisons: 3, suggestionCount: 3, seed: 31,
};

test('rank 结果与独立核对档案一致（参数、theta、指纹、经验计数）', async () => {
  const expected = JSON.parse(await readFile('data/expected-rank.json', 'utf8'));
  const { report, archive } = buildRankReport(records, policy);

  assert.equal(archive.dataFingerprint, expected.dataFingerprint);
  assert.equal(fingerprint([...records].sort((l, r) => l.id.localeCompare(r.id, 'en'))),
    expected.dataFingerprint);

  assert.deepEqual(report.components[0].order, expected.order);
  for (const [item, value] of Object.entries(expected.betas)) {
    const actual = report.components[0].rankedItems.find(entry => entry.item === item).beta;
    assert.ok(Math.abs(actual - value) < 1e-9, `${item} beta=${actual} 期望 ${value}`);
  }
  assert.ok(Math.abs(report.model.tieParameter.value - expected.theta) < 1e-9);
  assert.deepEqual(report.dataInfo.outcomeCounts, expected.outcomeCounts);

  const xy = report.pairwise.find(pair => pair.items.join('') === 'xy');
  assert.equal(xy.observed.comparisons, expected.observedXY.comparisons);
  assert.equal(xy.observed.aWins, expected.observedXY.aWins);
  assert.equal(xy.observed.ties, expected.observedXY.ties);
  assert.equal(xy.observed.aWinRate, expected.observedXY.aWinRate);

  report.sensitivity.grid.forEach((entry, index) => {
    assert.equal(entry.lambda, expected.sensitivity[index].lambda);
    assert.ok(Math.abs(entry.theta - expected.sensitivity[index].theta) < 1e-9);
  });

  // beta 约束：分量内和为零
  const sum = report.components[0].rankedItems.reduce((total, entry) => total + entry.beta, 0);
  assert.ok(Math.abs(sum) < 1e-12);
});
