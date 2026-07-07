// src/scripts/copy-code.js
// Wraps every <pre> in a .code-block and adds a "Copy" button pinned to that
// wrapper, so the button stays in the corner while the code scrolls
// horizontally inside the <pre>. Hidden until the block is hovered or the
// button is focused. Idempotent — safe to re-run.

function attach() {
  document.querySelectorAll('pre').forEach((pre) => {
    if (pre.parentElement?.classList.contains('code-block')) return;
    const source = pre.querySelector('code') ?? pre;

    // The button is pinned to the wrapper, not the <pre>: an absolute child of
    // the scrolling <pre> would slide away with the code.
    const wrap = document.createElement('div');
    wrap.className = 'code-block';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    // Clipboard access is permission-gated; a denied write shows "Failed"
    // instead of breaking the button.
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(source.textContent ?? '');
        btn.textContent = 'Copied';
      } catch {
        btn.textContent = 'Failed';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
    wrap.appendChild(btn);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', attach);
} else {
  attach();
}
