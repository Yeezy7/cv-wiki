import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts, sharedMarkdown } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/interview',
  markdown: sharedMarkdown,
  integrations: [
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
          ],
        },
      ],
    }),
  ],
});
