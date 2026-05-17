// src/content/config.ts
// Strict schemas for writing posts and projects. .strict() makes typos in
// frontmatter (e.g. `puDate`) fail the build instead of silently ignoring.

import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const writing = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/writing' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    math: z.boolean().default(false),
    draft: z.boolean().default(false),
  }).strict(),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()),
    link: z.string().url().optional(),
    repo: z.string().url().optional(),
  }).strict(),
});

const publications = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    authors: z.array(z.string()),
    venue: z.string(),
    arxiv: z.string().url().optional(),
    pdf: z.string().url().optional(),
    code: z.string().url().optional(),
    tags: z.array(z.string()).optional(),
  }).strict(),
});

export const collections = { writing, projects, publications };
