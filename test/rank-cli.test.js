import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { parseArguments, main } from '../src/cli.js';

const run = args => spawnSync(process.execPath, ['src/cli.js', ...args], { encoding: 'utf8' });

test('parseArguments 接受 rank 及其新参数并拒绝冲突组合', () => {
  assert.equal(parseArguments(['rank']).command, 'rank');
  assert.equal(parseArguments(['rank', '--model', 'm.json']).model, 'm.json');
  assert.equal(parseArguments(['rank', '--policy', 'p.json', '--input', 'x.json']).policy, 'p.json');
  assert.throws(() => parseArguments(['rank', '--model', 'm.json', '--policy', 'p.json']), /不能同时/);
  assert.throws(() => parseArguments(['rank', '--policy']), /参数/);
  assert.throws(() => parseArguments(['frobnicate']), /命令/);
});

test('rank 命令输出报告并把完整模型档案落盘，stdout 与档案指纹一致', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'rank-cli-'));
  try {
    const archivePath = join(folder, 'model.json');
    const result = run(['rank', '--input', 'data/records.json', '--out', archivePath]);
    assert.equal(result.status, 0, result.stderr);
    const printed = JSON.parse(result.stdout);
    const archive = JSON.parse(await readFile(archivePath, 'utf8'));
    assert.equal(printed.inputFingerprint, archive.dataFingerprint);
    assert.equal(printed.modelArchive.saved, true);
    assert.ok(Array.isArray(archive.itemDictionary));
    assert.ok(archive.parameterConstraints.constraint || archive.parameterConstraints.componentConstraint);
    assert.match(archive.parameterConstraints.family, /Davidson/);
    assert.equal(archive.judgeTreatment.judges.length, printed.dataInfo.judgeCount);
    assert.ok(archive.report.components.length >= 1);
    // 档案是规范 JSON，可被独立复查
    assert.ok(archive.canonicalFingerprint.length === 64);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('rank --model 重载档案，指纹不一致时非零退出', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'rank-reload-'));
  try {
    const archivePath = join(folder, 'model.json');
    const first = run(['rank', '--input', 'data/records.json', '--out', archivePath]);
    assert.equal(first.status, 0, first.stderr);

    const reloaded = run(['rank', '--input', 'data/records.json', '--model', archivePath]);
    assert.equal(reloaded.status, 0, reloaded.stderr);
    const body = JSON.parse(reloaded.stdout);
    assert.equal(body.reloaded, true);
    assert.ok(body.itemDictionary.length > 0);
    assert.match(body.parameterConstraints.identifiability, /分量/);

    const original = JSON.parse(await readFile('data/records.json', 'utf8'));
    const tampered = {
      ...original,
      records: original.records.map((row, index) =>
        index === 0 ? { ...row, outcome: row.outcome === 'tie' ? 'left' : 'tie' } : row),
    };
    const tamperedPath = join(folder, 'tampered.json');
    await writeFile(tamperedPath, JSON.stringify(tampered));
    const rejected = run(['rank', '--input', tamperedPath, '--model', archivePath]);
    assert.equal(rejected.status, 1);
    assert.match(rejected.stderr, /指纹/);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('rank 通过 main 直接调用时返回结构化报告', async () => {
  const document = await main(['rank', '--input', 'data/records.json']);
  assert.equal(document.dataInfo.recordCount, 96);
  assert.ok(document.components.every(component => Array.isArray(component.rankedItems)));
  assert.ok(Array.isArray(document.suggestions.candidates));
  assert.equal(document.suggestions.pairCap, document.coverage.minComparisons);
  assert.equal(document.inputFingerprint.length, 64);
});

test('rank 拒绝含相同商品比较的输入并返回非零退出码', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'rank-invalid-'));
  try {
    const original = JSON.parse(await readFile('data/records.json', 'utf8'));
    const invalid = {
      ...original,
      records: [{ ...original.records[0], right: original.records[0].left, id: 'bad-self' }],
    };
    const path = join(folder, 'invalid.json');
    await writeFile(path, JSON.stringify(invalid));
    const result = run(['rank', '--input', path]);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /相同商品/);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
