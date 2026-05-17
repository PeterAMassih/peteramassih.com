// src/pages/rss.xml.js
// RSS feed for Peter's Patches. Astro emits this as /rss.xml at build time.
// Feed readers (Feedly, NetNewsWire, etc.) poll this URL for new posts.
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = await getCollection('writing', ({ data }) => !data.draft);
  return rss({
    title: "Peter's Patches",
    description: 'Essays and notes by Peter Massih.',
    site: context.site,
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
