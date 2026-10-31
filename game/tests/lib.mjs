// game/tests/lib.mjs
// Shared by every suite. The host is GAME_HOST, else the first argument,
// else the local dev server; anything but localhost is reached over TLS.
export const HOST = process.env.GAME_HOST ?? process.argv[2] ?? "127.0.0.1:8787";
const SECURE = !HOST.includes("127.0.0.1") && !HOST.includes("localhost");
export const HTTP = `${SECURE ? "https" : "http"}://${HOST}`;
const WS = `${SECURE ? "wss" : "ws"}://${HOST}`;

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The first failed check ends the suite with a non-zero exit.
let passed = 0;
export function ok(cond, label) {
  if (!cond) {
    console.error(`FAIL: ${label}`);
    process.exit(1);
  }
  passed++;
  console.log(`ok: ${label}`);
}

// Open sockets would keep the process alive, so a suite ends by exiting.
export function done() {
  console.log(`\nALL ${passed} CHECKS PASSED`);
  process.exit(0);
}

// A client that keeps every message in arrival order: `all` for searching,
// next() for reading them one at a time.
export function connect(room = "plaza", extra = "") {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${WS}/?room=${room}${extra}`);
    const c = { ws, all: [], read: 0, closed: null, send: (o) => ws.send(JSON.stringify(o)) };
    c.next = async (label) => {
      for (let t = 0; t < 3000; t += 20) {
        if (c.read < c.all.length) return c.all[c.read++];
        await sleep(20);
      }
      throw new Error(`timeout waiting for ${label}`);
    };
    ws.onmessage = (e) => c.all.push(JSON.parse(e.data));
    ws.onclose = (e) => { c.closed = e.code; };
    ws.onopen = () => resolve(c);
    ws.onerror = () => reject(new Error(`${room}: connect failed`));
    setTimeout(() => reject(new Error(`${room}: connect timeout`)), 3000);
  });
}

export const init = (c) => c.all.find((m) => m.type === "init");
