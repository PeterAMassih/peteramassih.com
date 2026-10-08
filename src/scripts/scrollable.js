// src/scripts/scrollable.js
// Wide math, tables and figures scroll sideways on narrow screens; a tab stop
// lets keyboard users scroll them too. Off-screen blocks are not laid out yet
// (content-visibility in the post layout), so each box is measured when it
// gets a size: first render, rotation, and once more after the KaTeX fonts load.

const markScroller = (el) => {
  if (el.scrollWidth > el.clientWidth) el.tabIndex = 0;
  else el.removeAttribute('tabindex');
};
const scrollers = document.querySelectorAll('.katex-display, figure.viz, .body table');
const sized = new ResizeObserver((entries) => entries.forEach((e) => markScroller(e.target)));
scrollers.forEach((el) => sized.observe(el));
document.fonts.ready.then(() => scrollers.forEach(markScroller));
