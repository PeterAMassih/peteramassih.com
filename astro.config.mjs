// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSectionLinks from './src/plugins/rehype-section-links.mjs';

import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  site: 'https://peteramassih.com',
  // /play/ is noindex, so the sitemap leaves it out.
  integrations: [sitemap({ filter: (page) => !page.endsWith('/play/') })],
  // Astro 7's default 'jsx' whitespace mode drops the spaces between inline
  // elements, such as the separators in "CV · Email"; keep the old behavior.
  compressHTML: true,

  // Allow any *.trycloudflare.com hostname so phone testing through a quick tunnel works.
  // The leading dot tells Vite to treat it as a subdomain wildcard; only applies in dev.
  vite: { server: { allowedHosts: ['.trycloudflare.com'] } },

  markdown: {
    // The unified pipeline runs the remark/rehype plugins; Astro 7's default
    // Satteri processor does not.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeSectionLinks, rehypeKatex],
    }),
    shikiConfig: {
      themes: {
        // The high-contrast variants keep every token at 4.5:1 or more.
        light: 'github-light-high-contrast',
        dark: 'github-dark-high-contrast',
      },
    },
  },

  // Images are served as-is from public/; passthrough keeps the adapter from
  // shipping an unused on-the-fly resizing endpoint in the Worker.
  adapter: cloudflare({ imageService: 'passthrough' }),
});