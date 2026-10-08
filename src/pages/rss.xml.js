// src/pages/rss.xml.js
// RSS feed for the writing section. Astro emits this as /rss.xml at build time.
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = await getCollection('writing', ({ data }) => !data.draft);
  return rss({
    title: 'Writing — Peter Massih',
    description: 'Writing by Peter Massih, including a post on Mask2Former.',
    site: context.site,
    customData: '<language>en-us</language>',
    items: posts
      .sort((a, b) => b.data.pubDate.getTime() - a.data.pubDate.getTime())
      .map((post) => ({
        title: post.data.title,
        pubDate: post.data.pubDate,
        description: post.data.description,
        link: `/writing/${post.id}/`,
      })),
  });
}
