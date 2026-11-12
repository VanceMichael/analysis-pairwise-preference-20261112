import { validateRecords } from './dataset.js';

export function summarize(records) {
  validateRecords(records);
  const grouped = new Map();
  for (const row of [...records].sort((left, right) => left.id.localeCompare(right.id, 'en'))) {
    const key = String(row["outcome"]);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  const groups = [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right, 'en'))
    .map(([key, members]) => {
      const totals = Object.fromEntries(["round"].map(field => [field,
        members.reduce((sum, row) => sum + row[field], 0)]));
      return { key, count: members.length, sum: totals["round"], judges: new Set(members.map(row => row.judge)).size };
    });
  return { groupBy: "outcome", valueField: "round", count: records.length, groups };
}
