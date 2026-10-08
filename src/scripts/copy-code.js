// src/scripts/copy-code.js
// Wraps every <pre> in a .code-block and adds a "Copy" button pinned to that
// wrapper, so the button stays in the corner while the code scrolls
// horizontally inside the <pre> (an absolute child of the scrolling <pre>
// would slide away with the code). Hidden until the block is hovered or the
// button is focused. Module scripts run after parsing, so every <pre> exists.

for (const pre of document.querySelectorAll('pre')) {
  const wrap = document.createElement('div');
  wrap.className = 'code-block';
  pre.replaceWith(wrap);
  wrap.append(pre);

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'copy-btn';
  btn.textContent = 'Copy';
  btn.setAttribute('aria-label', 'Copy code to clipboard');
  // Clipboard access is permission-gated; a denied write shows "Failed"
  // instead of breaking the button.
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      btn.textContent = 'Copied';
    } catch {
      btn.textContent = 'Failed';
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
  wrap.append(btn);
}
