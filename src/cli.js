import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { loadDataset } from './dataset.js';
import { summarize } from './domain.js';
import { writeResult } from './output.js';
import { buildRankReport, reloadArchive, resolvePolicy } from './ranking.js';

const COMMANDS = ['summary', 'inspect', 'rank'];

export function parseArguments(args) {
  const [command = 'summary', ...options] = args;
  if (!COMMANDS.includes(command)) throw new Error('命令应为 summary、inspect 或 rank');
  const result = { command, input: 'data/records.json' };
  const allowed = ['--input', '--out', '--policy', '--model'];
  for (let index = 0; index < options.length; index += 2) {
    const name = options[index];
    const value = options[index + 1];
    if (!allowed.includes(name) || !value || value.startsWith('--')) {
      throw new Error(`参数应为 ${allowed.join(' / ')} 路径`);
    }
    if (result[name.slice(2)] && name !== '--input') throw new Error('参数不可重复');
    result[name.slice(2)] = value;
  }
  if (result.model && result.policy) {
    throw new Error('--model 重载档案与 --policy 重新拟合不能同时使用');
  }
  return result;
}

export async function main(args) {
  const options = parseArguments(args);
  const dataset = await loadDataset(options.input);

  let document;
  if (options.command === 'summary') {
    document = { ...summarize(dataset.records), inputFingerprint: dataset.fingerprint };
  } else if (options.command === 'inspect') {
    document = {
      records: dataset.records.length,
      fingerprint: dataset.fingerprint,
      fields: dataset.records.length ? Object.keys(dataset.records[0]).sort() : [],
      inputFingerprint: dataset.fingerprint,
    };
  } else if (options.model) {
    // 重载模型档案：指纹不一致直接拒绝，不静默重算
    document = await reloadArchive(options.model, options.input);
  } else {
    const policyInput = options.policy
      ? JSON.parse(await readFile(resolve(options.policy), 'utf8'))
      : JSON.parse(await readFile(resolve('data/policy.json'), 'utf8')).analysis ?? {};
    const policy = resolvePolicy(policyInput);
    const { report, archive } = buildRankReport(dataset.records, policy);
    document = {
      ...report,
      inputFingerprint: dataset.fingerprint,
      modelArchive: {
        archiveVersion: archive.archiveVersion,
        dataFingerprint: archive.dataFingerprint,
        canonicalFingerprint: archive.canonicalFingerprint,
        saved: Boolean(options.out),
      },
    };
    // --out 落盘的是完整模型档案（商品字典、参数约束、评委处理、数据指纹）
    if (options.out) await writeResult(options.out, archive);
    return document;
  }

  if (options.out) await writeResult(options.out, document);
  return document;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(result => {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  }).catch(error => {
    process.stderr.write(`处理失败：${error.message}\n`);
    process.exitCode = 1;
  });
}
