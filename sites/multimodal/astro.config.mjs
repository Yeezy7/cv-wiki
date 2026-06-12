import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts, sharedMarkdown } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/multimodal',
  markdown: sharedMarkdown,
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
            { label: 'ViT 视觉 Transformer', slug: 'vit' },
          ],
        },
      ],
    }),
  ],
});
