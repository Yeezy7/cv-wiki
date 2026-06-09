import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/multimodal',
  integrations: [
    starlight({
      ...sharedStarlightOpts('multimodal'),
      title: 'Multimodal Wiki',
      description: '多模态知识库',
      sidebar: [
        {
          label: '开始',
          items: [
            { label: '领域概述与学习路线', slug: 'index' },
          ],
        },
        {
          label: '核心模型',
          items: [
            { label: 'CLIP', slug: 'clip' },
          ],
        },
      ],
    }),
  ],
});
