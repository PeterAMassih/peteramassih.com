// Smoke test for the v0 Room server. Node 22+ (native WebSocket).
// Usage: node smoke-test.mjs [host]  — defaults to the local dev server.
const HOST = process.argv[2] ?? "127.0.0.1:8787";
const SECURE = !HOST.includes("127.0.0.1") && !HOST.includes("localhost");
const URL_HTTP = `${SECURE ? "https" : "http"}://${HOST}`;
const URL_WS = `${SECURE ? "wss" : "ws"}://${HOST}`;

let passed = 0;
function ok(cond, label) {
  if (!cond) {
    console.error(`FAIL: ${label}`);
    process.exit(1);
  }
  passed++;
  console.log(`ok: ${label}`);
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, rej) => setTimeout(() => rej(new Error(`timeout: ${label}`)), ms)),
  ]);
}

function connect(name) {
  return withTimeout(
    new Promise((resolve, reject) => {
      const ws = new WebSocket(URL_WS);
      const inbox = [];
      const waiters = [];
      let closed = null;
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (waiters.length) waiters.shift()(m);
        else inbox.push(m);
      };
      ws.onclose = (e) => { closed = { code: e.code, reason: e.reason }; };
      ws.onopen = () =>
        resolve({
          ws,
          name,
          getClosed: () => closed,
          next: (label) =>
            withTimeout(
              inbox.length
                ? Promise.resolve(inbox.shift())
                : new Promise((r) => waiters.push(r)),
              3000,
              `${name} waiting for ${label}`,
            ),
        });
      ws.onerror = () => reject(new Error(`${name}: socket error on connect`));
    }),
    3000,
    `${name} connect`,
  );
}

// 1. Plain HTTP is refused, except the one presence endpoint.
const res = await fetch(URL_HTTP);
ok(res.status === 426, `plain GET refused with 426 (got ${res.status})`);
const presence = await (await fetch(`${URL_HTTP}/presence`)).json();
ok(Number.isInteger(presence.count) && presence.count >= 0,
  `presence answers a headcount (${presence.count})`);

// 2. A connects, gets init containing itself.
const A = await connect("A");
const initA = await A.next("init");
ok(initA.type === "init" && typeof initA.id === "string", "A got init with an id");
ok(initA.players.some((p) => p.id === initA.id), "A's init includes A itself");
const aCount = initA.players.length;

// 3. B connects: B's init has one more player; A hears joined.
const B = await connect("B");
const initB = await B.next("init");
ok(initB.players.length === aCount + 1, `B's init lists ${aCount + 1} players`);
const joinedA = await A.next("joined");
ok(joinedA.type === "joined" && joinedA.player.id === initB.id, "A heard B join");

// 4. B moves; A receives the exact position.
B.ws.send(JSON.stringify({ type: "move", x: 100.5, y: 200 }));
const moved = await A.next("moved");
ok(moved.type === "moved" && moved.id === initB.id && moved.x === 100.5 && moved.y === 200,
  "A saw B move to (100.5, 200)");

// 5. Out-of-bounds positions come back clamped by the server.
B.ws.send(JSON.stringify({ type: "move", x: 99999, y: -50 }));
const clamped = await A.next("clamped move");
ok(clamped.x === 640 && clamped.y === 0, `server clamped (99999,-50) to (${clamped.x},${clamped.y})`);

// 6. Malformed JSON gets that client disconnected with 1002.
const C = await connect("C");
const initC = await C.next("init");
await A.next("C joined"); // drain
C.ws.send("this is not json{{{");
await new Promise((r) => setTimeout(r, 500));
ok(C.getClosed()?.code === 1002, `garbage sender was closed with 1002 (got ${JSON.stringify(C.getClosed())})`);

// 7. Clean disconnect broadcasts left. The kick above also (correctly)
// broadcasts left for C, so read until B's arrives.
B.ws.close();
let sawCLeft = false;
let leftB = null;
for (let i = 0; i < 3 && !leftB; i++) {
  const m = await A.next("left");
  ok(m.type === "left", `A heard a leave (${m.id === initC.id ? "C" : "B"})`);
  if (m.id === initC.id) sawCLeft = true;
  if (m.id === initB.id) leftB = m;
}
ok(leftB !== null, "A heard B leave");
ok(sawCLeft, "kicked client C was also broadcast as left (no ghosts)");

A.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
