import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const rootDir = new URL('..', import.meta.url).pathname;
const docsRoots = [
  'src/content/docs',
  'sites/cv/src/content/docs',
  'sites/llm/src/content/docs',
  'sites/multimodal/src/content/docs',
  'sites/interview/src/content/docs',
];
const requiredFields = ['title', 'description', 'category', 'tags', 'status', 'order'];
const allowedStatuses = new Set(['draft', 'review', 'stable']);
const allowedCategories = new Set([
  'guide',
  'cv',
  'llm',
  'multimodal',
  'interview',
  'basics',
  'image-processing',
  'detection',
  'segmentation',
  'deployment',
]);

const errors = [];

for (const docsRoot of docsRoots) {
  for (const file of walk(join(rootDir, docsRoot))) {
    if (!/\.(md|mdx)$/.test(file)) continue;
    validateDoc(file);
  }
}

if (errors.length > 0) {
  console.error('内容元信息校验失败：');
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log('内容元信息校验通过。');

function walk(dir) {
  const files = [];

  for (const item of readdirSync(dir)) {
    const path = join(dir, item);
    const stat = statSync(path);

    if (stat.isDirectory()) {
      files.push(...walk(path));
    } else {
      files.push(path);
    }
  }

  return files;
}

function validateDoc(file) {
  const rel = relative(rootDir, file);
  const source = readFileSync(file, 'utf8');
  const frontmatter = parseFrontmatter(source);

  if (!frontmatter) {
    errors.push(`${rel}: 缺少 frontmatter`);
    return;
  }

  for (const field of requiredFields) {
    if (!(field in frontmatter)) errors.push(`${rel}: 缺少 ${field}`);
  }

  if (typeof frontmatter.title !== 'string' || frontmatter.title.trim() === '') {
    errors.push(`${rel}: title 不能为空`);
  }

  if (typeof frontmatter.description !== 'string' || frontmatter.description.trim() === '') {
    errors.push(`${rel}: description 不能为空`);
  }

  if (!allowedCategories.has(frontmatter.category)) {
    errors.push(`${rel}: category 必须是 ${[...allowedCategories].join(', ')} 之一`);
  }

  if (!Array.isArray(frontmatter.tags) || frontmatter.tags.length === 0) {
    errors.push(`${rel}: tags 必须是非空数组`);
  }

  if (!allowedStatuses.has(frontmatter.status)) {
    errors.push(`${rel}: status 必须是 draft、review 或 stable`);
  }

  if (!Number.isInteger(frontmatter.order) || frontmatter.order < 1) {
    errors.push(`${rel}: order 必须是正整数`);
  }
}

function parseFrontmatter(source) {
  const normalized = source.replace(/\r\n/g, '\n');
  if (!normalized.startsWith('---\n')) return null;

  const end = normalized.indexOf('\n---', 4);
  if (end === -1) return null;

  const block = normalized.slice(4, end).trim();
  const data = {};

  const lines = block.split('\n');
  let currentKey = '';
  let currentArray = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    // 多行数组项：  - value
    const arrayMatch = line.match(/^-\s+(.+)$/);
    if (arrayMatch && currentKey) {
      if (currentArray === null) currentArray = [];
      currentArray.push(stripQuotes(arrayMatch[1].trim()));
      data[currentKey] = currentArray;
      continue;
    }

    // 键值对
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;

    const [, key, rawValue] = match;
    currentKey = key;
    currentArray = null;

    if (rawValue.trim() === '') {
      // 值为空，可能是多行数组的开始
      data[key] = undefined;
    } else {
      data[key] = parseValue(rawValue);
    }
  }

  return data;
}

function parseValue(rawValue) {
  const value = rawValue.trim();

  if (/^\[.*\]$/.test(value)) {
    const inner = value.slice(1, -1).trim();
    if (!inner) return [];
    return inner.split(',').map((item) => stripQuotes(item.trim())).filter(Boolean);
  }

  if (/^\d+$/.test(value)) return Number(value);
  if (value === 'true') return true;
  if (value === 'false') return false;

  return stripQuotes(value);
}

function stripQuotes(value) {
  return value.replace(/^['"]|['"]$/g, '');
}
