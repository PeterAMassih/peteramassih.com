// src/content.config.ts
// Strict schemas for writing posts, projects and publications. .strict() makes typos in
// frontmatter (e.g. `puDate`) fail the build instead of silently ignoring.

import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const writing = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/writing' }),
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
  loader: glob({ pattern: '*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    // The result line: shown under the title and used as the meta description.
    description: z.string(),
    // End month. It orders the list; a startDate turns it into a range.
    pubDate: z.coerce.date(),
    startDate: z.coerce.date().optional(),
    // Course or setting, shown on the project's own page.
    context: z.string().optional(),
    links: z.array(z.object({ label: z.string(), href: z.url() })).default([]),
    tags: z.array(z.string()).default([]),
  }).strict(),
});

const publications = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    // Orders the list and dates the paper's own page.
    pubDate: z.coerce.date(),
    authors: z.array(z.string()),
    // The byline after the authors: status, advisor, host.
    venue: z.string(),
    // href can be a local asset path (e.g. /master-thesis.pdf), so it is a
    // plain string rather than z.url().
    links: z.array(z.object({ label: z.string(), href: z.string() })),
  }).strict(),
});

export const collections = { writing, projects, publications };
