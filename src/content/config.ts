import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { z } from 'astro/zod';

const allowedCategories = [
  'guide',
  'cv',
  'llm',
  'multimodal',
  'interview',
  'basics',
  'image-processing',
  'detection',
  'segmentation',
  'deployment',
] as const;

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        category: z.enum(allowedCategories),
        tags: z.array(z.string()).min(1),
        status: z.enum(['draft', 'review', 'stable']),
        order: z.number().int().positive(),
      }),
    }),
  }),
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
