import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';

const rootDir = new URL('..', import.meta.url).pathname;
const sites = [
  { name: 'root', docsRoot: 'src/content/docs', base: '/ai-wiki' },
  { name: 'cv', docsRoot: 'sites/cv/src/content/docs', base: '/ai-wiki/cv' },
  { name: 'llm', docsRoot: 'sites/llm/src/content/docs', base: '/ai-wiki/llm' },
  { name: 'multimodal', docsRoot: 'sites/multimodal/src/content/docs', base: '/ai-wiki/multimodal' },
  { name: 'interview', docsRoot: 'sites/interview/src/content/docs', base: '/ai-wiki/interview' },
];
const externalSchemes = /^(https?:|mailto:|tel:|ftp:|data:|javascript:)/i;
const routes = new Map();
const errors = [];

for (const site of sites) {
  for (const file of walk(join(rootDir, site.docsRoot))) {
    if (!/\.(md|mdx)$/.test(file)) continue;

    const page = buildPage(site, file);
    routes.set(page.url, page);
  }
}

for (const page of routes.values()) validatePageLinks(page);

if (errors.length > 0) {
  console.error('内部链接校验失败：');
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log('内部链接校验通过。');

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

function buildPage(site, file) {
  const rel = relative(join(rootDir, site.docsRoot), file).replace(/\\/g, '/');
  const slug = rel.replace(/\.(md|mdx)$/, '');
  const routePath = slug === 'index' ? '' : slug.endsWith('/index') ? slug.slice(0, -6) : slug;
  const url = normalizePath(`${site.base}/${routePath}`);
  const source = readFileSync(file, 'utf8');

  return {
    site,
    file,
    source,
    url,
    anchors: collectAnchors(source),
  };
}

function collectAnchors(source) {
  const anchors = new Set(['_top']);
  const used = new Map();

  for (const line of stripCodeBlocks(source).split('\n')) {
    const match = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (!match) continue;

    const text = match[2]
      .replace(/\{#.+?\}\s*$/, '')
      .replace(/<[^>]+>/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/[`*_~]/g, '')
      .trim();
    const baseSlug = slugify(text);
    const count = used.get(baseSlug) ?? 0;

    used.set(baseSlug, count + 1);
    anchors.add(count === 0 ? baseSlug : `${baseSlug}-${count}`);
  }

  return anchors;
}

function validatePageLinks(page) {
  const links = extractLinks(stripCodeBlocks(page.source));

  for (const link of links) {
    const target = link.target.trim();
    if (!target || externalSchemes.test(target) || target.startsWith('<')) continue;

    const resolved = resolveInternalTarget(page, target);
    if (!resolved) continue;

    if (resolved.assetPath) {
      if (!existsSync(resolved.assetPath)) {
        errors.push(`${formatFile(page.file)}:${link.line}: 静态资源不存在 ${target}`);
      }
      continue;
    }

    const targetPage = routes.get(resolved.path);
    if (!targetPage) {
      errors.push(`${formatFile(page.file)}:${link.line}: 页面不存在 ${target}`);
      continue;
    }

    if (resolved.hash && !targetPage.anchors.has(resolved.hash)) {
      errors.push(`${formatFile(page.file)}:${link.line}: 锚点不存在 ${target}`);
    }
  }
}

function extractLinks(source) {
  const links = [];
  const lines = source.split('\n');
  const markdownLinkPattern = /!?\[[^\]]*?\]\(([^)\s]+)(?:\s+['"][^'"]*['"])?\)/g;
  const htmlAttrPattern = /\b(?:href|src)=["']([^"']+)["']/g;

  for (const [index, line] of lines.entries()) {
    for (const match of line.matchAll(markdownLinkPattern)) {
      links.push({ line: index + 1, target: match[1] });
    }

    for (const match of line.matchAll(htmlAttrPattern)) {
      links.push({ line: index + 1, target: match[1] });
    }
  }

  return links;
}

function resolveInternalTarget(page, target) {
  const cleaned = target.replace(/^<|>$/g, '');
  const [withoutHash, rawHash = ''] = cleaned.split('#');
  const pathWithoutQuery = withoutHash.split('?')[0];
  const hash = decodeURIComponent(rawHash);

  if (!pathWithoutQuery) return { path: page.url, hash };

  if (/\.(avif|css|gif|ico|jpeg|jpg|js|json|pdf|png|svg|webp|xml)$/i.test(pathWithoutQuery)) {
    return resolveAsset(page, pathWithoutQuery);
  }

  if (pathWithoutQuery.startsWith('/')) {
    if (pathWithoutQuery === page.site.base || pathWithoutQuery.startsWith(`${page.site.base}/`)) {
      return { path: normalizePath(pathWithoutQuery), hash };
    }

    if (pathWithoutQuery === '/ai-wiki' || pathWithoutQuery.startsWith('/ai-wiki/')) {
      return { path: normalizePath(pathWithoutQuery), hash };
    }

    errors.push(
      `${formatFile(page.file)}: 内部绝对链接缺少站点 base ${target}，请使用 ${page.site.base} 前缀或相对链接`
    );
    return null;
  }

  const resolved = new URL(pathWithoutQuery, `https://local.invalid${page.url}`).pathname;
  return { path: normalizePath(resolved), hash };
}

function resolveAsset(page, pathWithoutQuery) {
  if (pathWithoutQuery.startsWith('/')) {
    // Remove base path prefix for root site assets
    let publicPath = pathWithoutQuery;
    if (page.site.name === 'root' && pathWithoutQuery.startsWith('/ai-wiki/')) {
      publicPath = pathWithoutQuery.slice('/ai-wiki'.length);
    }

    // For multi-site projects, check the site's public directory first
    if (page.site.name !== 'root') {
      // For sub-sites, the asset path after base should be relative to site's public dir
      const siteBase = page.site.base;
      if (pathWithoutQuery.startsWith(siteBase + '/')) {
        const relativePath = pathWithoutQuery.slice(siteBase.length);
        const sitePublicPath = join(rootDir, 'sites', page.site.name, 'public', relativePath);
        if (existsSync(sitePublicPath)) {
          return { assetPath: sitePublicPath };
        }
      }

      // Also check site's public dir for absolute paths without site base prefix
      const sitePublicPath = join(rootDir, 'sites', page.site.name, 'public', publicPath);
      if (existsSync(sitePublicPath)) {
        return { assetPath: sitePublicPath };
      }
    }

    return { assetPath: join(rootDir, 'public', publicPath) };
  }

  return { assetPath: join(dirname(page.file), pathWithoutQuery) };
}

function normalizePath(path) {
  const normalized = path.replace(/\/+/g, '/').replace(/\/index$/, '');
  return normalized.endsWith('/') ? normalized : `${normalized}/`;
}

function slugify(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, '')
    .replace(/\s+/g, '-');
}

function stripCodeBlocks(source) {
  return source.replace(/```[\s\S]*?```/g, '').replace(/~~~[\s\S]*?~~~/g, '');
}

function formatFile(file) {
  return relative(rootDir, file);
}
