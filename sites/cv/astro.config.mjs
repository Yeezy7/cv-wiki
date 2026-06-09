import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { sharedStarlightOpts } from '../../src/starlight-shared.mjs';

export default defineConfig({
  site: 'https://Yeezy7.github.io',
  base: '/ai-wiki/cv',
  integrations: [
    starlight({
      ...sharedStarlightOpts('cv'),
      title: 'CV Wiki',
      description: '计算机视觉知识库',
      sidebar: [
        {
          label: '开始',
          items: [
            { label: '领域概述与学习路线', slug: 'index' },
          ],
        },
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
