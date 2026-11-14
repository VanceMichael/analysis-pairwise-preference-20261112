import { main } from '../src/cli.js';
import { readFile } from 'node:fs/promises';

const started = performance.now();
console.log('读取本地演练记录并核对输入指纹');
console.log(JSON.stringify(await main(['inspect']), null, 2));
console.log('现有领域核算');
console.log(JSON.stringify(await main(['summary', '--out', 'var/summary.json']), null, 2));
console.log('含平局成对偏好模型与补比较建议（rank，模型档案写入 var/model-archive.json）');
console.log(JSON.stringify(
  await main(['rank', '--out', 'var/model-archive.json']), null, 2));
console.log('业务配置');
console.log(JSON.stringify(JSON.parse(await readFile('data/policy.json', 'utf8')), null, 2));
console.log(`完成，用时 ${Math.ceil(performance.now() - started)} 毫秒`);
