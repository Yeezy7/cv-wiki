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
          ],
        },
        {
          label: '计算机视觉',
          items: [
            { label: '概述', slug: 'cv' },
            {
              label: '深度学习基础',
              collapsed: false,
              items: [
                { label: 'CNN 基础', slug: 'cv/basics/cnn' },
                { label: 'BatchNorm', slug: 'cv/basics/batchnorm' },
                { label: '激活函数', slug: 'cv/basics/activation' },
                { label: '损失函数', slug: 'cv/basics/loss-functions' },
              ],
            },
            {
              label: '目标检测',
              collapsed: false,
              items: [
                { label: '目标检测综述', slug: 'cv/detection/overview' },
                { label: 'YOLO 系列', slug: 'cv/detection/yolo' },
                { label: 'NMS', slug: 'cv/detection/nms' },
                { label: 'mAP', slug: 'cv/detection/map' },
              ],
            },
            {
              label: '图像分割',
              collapsed: false,
              items: [
                { label: '图像分割综述', slug: 'cv/segmentation/overview' },
              ],
            },
          ],
        },
        {
          label: '大语言模型',
          items: [
            { label: '概述', slug: 'llm' },
            { label: 'Transformer', slug: 'llm/transformer' },
          ],
        },
        {
          label: '多模态',
          items: [
            { label: '概述', slug: 'multimodal' },
            { label: 'CLIP', slug: 'multimodal/clip' },
          ],
        },
        {
          label: '面试题库',
          items: [
            { label: '概述', slug: 'interview' },
            { label: 'CV 面试题', slug: 'interview/cv' },
          ],
        },
      ],
    }),
  ],
});
