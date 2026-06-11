import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts, sharedMarkdown } from './src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki',
  markdown: sharedMarkdown,
  integrations: [
    starlight({
      ...sharedStarlightOpts(''),
      title: 'AI Wiki',
      description: '面向 AI 学习、面试和工程实践的开源知识库',
      editLink: { baseUrl: 'https://github.com/Yeezy7/ai-wiki/edit/main' },
      components: {
        Head: './src/components/Head.astro',
        SiteTitle: './src/components/SiteTitle.astro',
        Sidebar: './src/components/Sidebar.astro',
        MarkdownContent: './src/components/MarkdownContent.astro',
      },
      sidebar: [
        {
          label: '学习指南',
          items: [
            { label: '项目介绍', slug: 'index' },
            { label: '学习路线', slug: 'guide/roadmap' },
            { label: '如何使用', slug: 'guide/how-to-use' },
            { label: '标签索引', slug: 'tags' },
          ],
        },
      ],
    }),
  ],
});
