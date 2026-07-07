// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  site: 'https://peteramassih.com',
  integrations: [mdx(), sitemap()],

  // Allow any *.trycloudflare.com hostname so phone testing through a quick tunnel works.
  // The leading dot tells Vite to treat it as a subdomain wildcard; only applies in dev.
  vite: { server: { allowedHosts: ['.trycloudflare.com'] } },

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      // Long lines scroll horizontally instead of wrapping, so code
      // indentation stays intact (wrapping folds lines to the left margin).
      wrap: false,
    },
  },

  adapter: cloudflare(),
});