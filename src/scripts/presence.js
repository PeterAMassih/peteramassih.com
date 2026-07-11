// src/scripts/presence.js
// The homepage invitation comes alive when the world is inhabited: swap the
// static line for a live headcount. When the count is zero or the fetch
// fails (adblock, offline), the static line already says everything needed.
const PRESENCE_URL = import.meta.env.DEV
  ? 'http://localhost:8787/presence'
  : 'https://peteramassih-play.peteramassih.workers.dev/presence';

const invite = document.getElementById('play-invite');
if (invite) {
  fetch(PRESENCE_URL)
    .then((r) => r.json())
    .then(({ count }) => {
      if (count === 1) invite.textContent = '1 person is in the tiny world on this site right now →';
      else if (count > 1) invite.textContent = `${count} people are in the tiny world on this site right now →`;
    })
    .catch(() => {});
}
