import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts, sharedMarkdown } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/llm',
  markdown: sharedMarkdown,
  integrations: [
    starlight({
      ...sharedStarlightOpts('llm'),
      title: 'LLM Wiki',
      description: '大语言模型知识库',
      sidebar: [
        {
          label: '开始',
          items: [
            { label: '领域概述与学习路线', slug: 'index' },
          ],
        },
        {
          label: '基础架构',
          items: [
            { label: 'Transformer', slug: 'transformer' },
          ],
        },
        {
          label: '训练与应用',
          items: [
            { label: 'Fine-tuning 与 LoRA', slug: 'fine-tuning' },
            { label: 'RAG 检索增强生成', slug: 'rag' },
          ],
        },
      ],
    }),
  ],
});
