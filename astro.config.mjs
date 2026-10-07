// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSectionLinks from './src/plugins/rehype-section-links.mjs';

import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  site: 'https://peteramassih.com',
  // /play/ is noindex, so the sitemap leaves it out.
  integrations: [mdx(), sitemap({ filter: (page) => !page.endsWith('/play/') })],

  // Allow any *.trycloudflare.com hostname so phone testing through a quick tunnel works.
  // The leading dot tells Vite to treat it as a subdomain wildcard; only applies in dev.
  vite: { server: { allowedHosts: ['.trycloudflare.com'] } },

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeSectionLinks, rehypeKatex],
    shikiConfig: {
      themes: {
        // The high-contrast variants keep every token at 4.5:1 or more.
        light: 'github-light-high-contrast',
        dark: 'github-dark-high-contrast',
      },
      // Long lines scroll horizontally instead of wrapping, so code
      // indentation stays intact (wrapping folds lines to the left margin).
      wrap: false,
    },
  },

  adapter: cloudflare(),
});