import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/interview',
  integrations: [
    starlight({
      title: '面试题库',
      description: 'AI 面试题库',
      locales: {
        root: {
          label: '简体中文',
          lang: 'zh-CN',
        },
      },
      components: {
        Head: './src/components/Head.astro',
        SiteTitle: './src/components/SiteTitle.astro',
        MarkdownContent: './src/components/MarkdownContent.astro',
      },
      customCss: ['./src/styles/custom.css'],
      editLink: {
        baseUrl: 'https://github.com/Yeezy7/ai-wiki/edit/main/sites/interview',
      },
      lastUpdated: true,
      pagination: true,
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Yeezy7/ai-wiki',
        },
      ],
      sidebar: [
        {
          label: '面试题',
          items: [
            { label: 'CV 面试题', slug: 'cv' },
          ],
        },
      ],
    }),
  ],
});
