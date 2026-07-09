import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';
import { sharedStarlightOpts, sharedMarkdown } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/interview',
  markdown: sharedMarkdown,
  integrations: [
    sitemap(),
    starlight({
      ...sharedStarlightOpts('interview'),
      title: '面试题库',
      description: 'AI 面试题库',
      sidebar: [
        {
          label: '开始',
          items: [
            { label: '面试准备路线', slug: 'index' },
          ],
        },
        {
          label: '面试题',
          items: [
            { label: 'CV 面试题', slug: 'cv' },
            { label: 'LLM 面试题', slug: 'llm' },
          ],
        },
      ],
    }),
  ],
});
