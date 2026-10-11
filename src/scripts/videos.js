// src/scripts/videos.js
// Figure animations. Each one is rendered twice, for the light and the dark
// theme, so it sits on the page like the inline SVG figures. A video loads as
// it nears the viewport and shows its first frame, which poses the question.
// It plays once when it reaches the middle of the screen and stops on its last
// frame, which holds the whole result. It does not loop: motion beside text
// pulls the eye off the sentence being read. Click, Enter or Space pauses,
// resumes, or replays from the start. Under prefers-reduced-motion nothing
// autoplays or preloads: the poster is the last frame, and native controls
// can play it.

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const isDark = () => {
  const theme = document.documentElement.dataset.theme;
  return theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
};
const file = (v) => `${v.dataset.base}-${isDark() ? 'dark' : 'light'}`;
const videos = document.querySelectorAll('video[data-base]');

// Posters are set here rather than in the markup, where every figure's poster
// would download with the page.
const load = (v) => {
  v.poster = `${file(v)}-${reduceMotion ? 'end' : 'start'}.webp`;
  v.preload = reduceMotion ? 'none' : 'auto';
  v.src = `${file(v)}.mp4`;
};

// Autoplay can be refused (battery saver), and the reader then gets controls.
// A play() cut short by pause() also rejects, which needs no handling.
const play = (v) => v.play().catch((e) => { if (e.name === 'NotAllowedError') v.controls = true; });

const applyTheme = (v) => {
  if (!v.src || v.src.endsWith(`${file(v)}.mp4`)) return;
  const { currentTime, paused } = v;
  load(v);
  v.currentTime = currentTime;
  if (!paused) play(v);
};

// Figures, not videos, are observed: the post skips layout for offscreen
// blocks, so only the figure's own box is reliable before it is rendered.
// Both observers can fire in the same frame after a jump, in either order, so
// whichever runs first loads the video. Loading twice would restart it.
const loadOnce = (v) => { if (!v.src) load(v); };
const near = new IntersectionObserver((entries) => {
  for (const { isIntersecting, target } of entries) {
    if (!isIntersecting) continue;
    loadOnce(target.querySelector('video'));
    near.unobserve(target);
  }
}, { rootMargin: '1200px 0px' });

// A band across the middle of the screen rather than a visible fraction: on a
// short screen, such as a phone held sideways, a figure is never mostly in view.
const inView = new IntersectionObserver((entries) => {
  for (const { isIntersecting, target } of entries) {
    const v = target.querySelector('video');
    if (!isIntersecting) {
      v.pause();
    } else if (!v.dataset.held && !v.ended) {
      loadOnce(v);
      play(v);
    }
  }
}, { rootMargin: '-30% 0px' });

for (const v of videos) {
  near.observe(v.parentElement);
  if (reduceMotion) {
    v.controls = true;
    continue;
  }
  // A reader's pause sticks: scrolling away and back does not resume it.
  // Once native controls appear, they handle clicks and keys themselves.
  const toggle = () => {
    v.dataset.held = v.paused ? '' : 'true';
    if (v.paused) play(v);
    else v.pause();
  };
  v.title = 'Click, Enter or Space to pause or replay';
  v.tabIndex = 0;
  v.addEventListener('click', () => { if (!v.controls) toggle(); });
  v.addEventListener('keydown', (e) => {
    if (v.controls || (e.key !== 'Enter' && e.key !== ' ')) return;
    e.preventDefault();
    toggle();
  });
  inView.observe(v.parentElement);
}

const retheme = () => videos.forEach(applyTheme);
new MutationObserver(retheme).observe(document.documentElement, {
  attributes: true, attributeFilter: ['data-theme'],
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', retheme);
