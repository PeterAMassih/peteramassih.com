# peteramassih.com

Source of Peter Massih's site: publications, projects, writing and a few toys.

Astro with MDX content, deployed as a Cloudflare Worker that serves the static
build. Pushing `main` triggers Cloudflare Workers Builds; `npm run deploy` is
the manual path.

```sh
npm install
npm run dev       # local dev server
npm run build     # static build into dist/
npm run preview   # build, then serve it locally with wrangler dev
```

Content lives in `src/content/` (projects, publications, writing). Projects
moved to `src/content/projects/archive/` stay in the repo but are not built.
The multiplayer world at `/play/` talks to a separate Worker in `game/`.
