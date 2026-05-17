// src/scripts/copy-code.js
// Adds a "Copy" button to every <pre>. Hidden until the pre is hovered or
// the button gets keyboard focus. Idempotent — safe to re-run.

function attach() {
  document.querySelectorAll('pre').forEach((pre) => {
    if (pre.querySelector('.copy-btn')) return;
    const source = pre.querySelector('code') ?? pre;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(source.textContent ?? '');
        btn.textContent = 'Copied';
      } catch {
        btn.textContent = 'Failed';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
    pre.appendChild(btn);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', attach);
} else {
  attach();
}
