import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';
import { sharedStarlightOpts, sharedMarkdown } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/multimodal',
  markdown: sharedMarkdown,
  integrations: [
    sitemap(),
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
            { label: 'BLIP', slug: 'blip' },
            { label: 'BLIP-2', slug: 'blip2' },
            { label: 'SigLIP', slug: 'siglip' },
            { label: 'LLaVA', slug: 'llava' },
          ],
        },
      ],
    }),
  ],
});
