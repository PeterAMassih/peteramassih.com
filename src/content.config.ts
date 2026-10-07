// src/content/config.ts
// Strict schemas for writing posts and projects. .strict() makes typos in
// frontmatter (e.g. `puDate`) fail the build instead of silently ignoring.

import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

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

// Top-level files only: entries moved into projects/archive/ stay in the repo
// but are not built, listed or put in the sitemap.
const projects = defineCollection({
  loader: glob({ pattern: '*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    // The result line: shown under the title and used as the meta description.
    description: z.string(),
    // End month. It orders the list; a startDate turns it into a range.
    pubDate: z.coerce.date(),
    startDate: z.coerce.date().optional(),
    context: z.string(),
    links: z.array(z.object({ label: z.string(), href: z.url() })),
  }).strict(),
});

const publications = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    authors: z.array(z.string()),
    // The byline after the authors: status, advisor, host.
    venue: z.string(),
    // href can be a local asset path (e.g. /master-thesis.pdf), so it is a
    // plain string rather than z.url().
    links: z.array(z.object({ label: z.string(), href: z.string() })),
  }).strict(),
});

export const collections = { writing, projects, publications };
