import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { summarize } from '../src/domain.js';

const seed = JSON.parse(await readFile('data/records.json', 'utf8')).records[0];
const probe = patches => summarize(patches.map((patch, index) => ({...seed, id: `p${index}`, ...patch}))).groups[0];

test('评委人数不因多轮比较重复增加', () => {

  assert.equal(probe([{judge: 'a', round: 1}, {judge: 'a', round: 2}, {judge: 'b', round: 1}]).judges, 2);
  assert.equal(probe([{judge: 'a'}]).judges, 1);
});
