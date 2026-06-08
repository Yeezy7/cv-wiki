import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/cv',
  integrations: [
    starlight({
      title: 'CV Wiki',
      description: '计算机视觉知识库',
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
        baseUrl: 'https://github.com/Yeezy7/ai-wiki/edit/main/sites/cv',
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
          label: '深度学习基础',
          items: [
            { label: 'CNN 基础', slug: 'basics/cnn' },
            { label: 'BatchNorm', slug: 'basics/batchnorm' },
            { label: '激活函数', slug: 'basics/activation' },
            { label: '损失函数', slug: 'basics/loss-functions' },
          ],
        },
        {
          label: '目标检测',
          items: [
            { label: '目标检测综述', slug: 'detection/overview' },
            { label: 'YOLO 系列', slug: 'detection/yolo' },
            { label: 'NMS', slug: 'detection/nms' },
            { label: 'mAP', slug: 'detection/map' },
          ],
        },
        {
          label: '图像分割',
          items: [
            { label: '图像分割综述', slug: 'segmentation/overview' },
          ],
        },
      ],
    }),
  ],
});
