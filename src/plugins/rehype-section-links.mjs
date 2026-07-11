// Turns inline section references like "§8.1" into links to that section's
// heading. Heading ids are computed with the same slugger Astro uses, so the
// links match its anchors; the map is derived from the headings themselves and
// never goes stale. Runs before rehype-katex, while headings are still plain
// text, so a heading id computes the same way Astro computes it.
import GithubSlugger from 'github-slugger';
import { visit } from 'unist-util-visit';

const textOf = (node) =>
  node.type === 'text'
    ? node.value
    : node.children
      ? node.children.map(textOf).join('')
      : '';

export default function rehypeSectionLinks() {
  return (tree) => {
    // Map each leading section number (e.g. "8" or "8.1") to its heading id.
    const slugger = new GithubSlugger();
    const idFor = {};
    visit(tree, (node) => {
      if (!node.tagName || !/^h[1-6]$/.test(node.tagName)) return;
      const text = textOf(node);
      const id = slugger.slug(text);
      const m = text.match(/^\s*(\d+(?:\.\d+)?)/);
      if (m) idFor[m[1]] = id;
    });

    // Replace §N and §N.M in prose text with links to that heading.
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || parent.tagName === 'a' || index == null
          || !node.value.includes('§')) return;
      const out = [];
      let last = 0;
      const re = /§(\d+(?:\.\d+)?)/g;
      let m;
      while ((m = re.exec(node.value))) {
        const id = idFor[m[1]];
        if (!id) continue;
        if (m.index > last)
          out.push({ type: 'text', value: node.value.slice(last, m.index) });
        out.push({
          type: 'element',
          tagName: 'a',
          properties: { href: '#' + id, className: ['section-ref'] },
          children: [{ type: 'text', value: '§' + m[1] }],
        });
        last = m.index + m[0].length;
      }
      if (!out.length) return;
      if (last < node.value.length)
        out.push({ type: 'text', value: node.value.slice(last) });
      parent.children.splice(index, 1, ...out);
      return index + out.length;
    });
  };
}
