// 各站共享的 Starlight 配置。各站 astro.config.mjs 导入后用 spread 覆盖差异化字段。

/** 根站 base 路径，用于子站返回首页等场景 */
export const rootBase = '/ai-wiki/';

export const sharedLocales = {
  root: {
    label: '简体中文',
    lang: 'zh-CN',
  },
};

export const sharedComponents = {
  Head: './src/components/Head.astro',
  SiteTitle: './src/components/SiteTitle.astro',
  MarkdownContent: './src/components/MarkdownContent.astro',
};

export const sharedSocial = [
  {
    icon: 'github',
    label: 'GitHub',
    href: 'https://github.com/Yeezy7/ai-wiki',
  },
];

/** 生成子站的 editLink.baseUrl */
export function editBaseUrl(siteDir) {
  return `https://github.com/Yeezy7/ai-wiki/edit/main/sites/${siteDir}`;
}

/**
 * 返回子站通用的 Starlight 选项（不含 sidebar）。
 * 调用方用 spread 覆盖 title / description / sidebar 等字段。
 */
export function sharedStarlightOpts(siteDir) {
  return {
    locales: sharedLocales,
    components: sharedComponents,
    customCss: ['./src/styles/custom.css'],
    editLink: { baseUrl: editBaseUrl(siteDir) },
    lastUpdated: true,
    pagination: false,
    tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
    social: sharedSocial,
  };
}
