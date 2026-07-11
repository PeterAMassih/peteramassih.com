// v6b guestbook checks: persistence, upsert, room-gating, validation.
// Node 22+. Usage: node marks-test.mjs [host]
const HOST = process.argv[2] ?? "127.0.0.1:8787";
const SECURE = !HOST.includes("127.0.0.1") && !HOST.includes("localhost");
const URL_WS = `${SECURE ? "wss" : "ws"}://${HOST}`;

let passed = 0;
function ok(cond, label) {
  if (!cond) { console.error(`FAIL: ${label}`); process.exit(1); }
  passed++;
  console.log(`ok: ${label}`);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function connect(roomName, extra = "") {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${URL_WS}/?room=${roomName}${extra}`);
    const all = [];
    ws.onmessage = (e) => all.push(JSON.parse(e.data));
    ws.onopen = () => resolve({ ws, all, send: (o) => ws.send(JSON.stringify(o)) });
    ws.onerror = () => reject(new Error("connect failed"));
    setTimeout(() => reject(new Error("connect timeout")), 3000);
  });
}
const init = (c) => c.all.find((m) => m.type === "init");
const VID = "&vid=aaaa1111-2222-4333-8444-555566667777";
const NAME = "&name=scribbler";

// 1. A mark on the library wall is broadcast and persists for the next visitor.
const W = await connect("library", VID + NAME);
await sleep(300);
W.send({ type: "move", x: 300, y: 312 });
await sleep(200);
W.send({ type: "mark", text: "the wall remembers" });
await sleep(300);
ok(W.all.some((m) => m.type === "marked" && m.mark.text === "the wall remembers"),
  "mark broadcast to the room");
W.ws.close();
await sleep(400);

const R = await connect("library");
await sleep(300);
const wall = init(R).marks;
ok(init(R).marksOn === true, "library init says the wall is on");
const mine = wall.find((m) => m.name === "scribbler");
ok(mine && mine.text === "the wall remembers", "mark persisted for the next visitor");
ok(Math.abs(mine.x - 300) < 1, `mark pinned where its author stood (x=${mine.x})`);
R.ws.close();

// 2. Re-marking replaces, never duplicates.
const W2 = await connect("library", VID + NAME);
await sleep(300);
W2.send({ type: "mark", text: "second thoughts" });
await sleep(300);
W2.ws.close();
await sleep(400);
const R2 = await connect("library");
await sleep(300);
const entries = init(R2).marks.filter((m) => m.name === "scribbler");
ok(entries.length === 1 && entries[0].text === "second thoughts",
  "re-marking replaced the old line (no duplicates)");
R2.ws.close();

// 3. Other rooms have no wall: marks are dropped, init says off.
const P = await connect("plaza", VID + NAME);
await sleep(300);
ok(init(P).marksOn === false, "plaza init says no wall");
P.send({ type: "mark", text: "graffiti in the plaza" });
await sleep(300);
ok(!P.all.some((m) => m.type === "marked"), "plaza mark was dropped");
P.ws.close();

// 4. Overlong text is truncated to 40 chars; vid-less writers are ignored.
const L = await connect("library", "&vid=bbbb1111-2222-4333-8444-555566667777&name=rambler");
await sleep(300);
L.send({ type: "mark", text: "x".repeat(120) });
await sleep(300);
const long = L.all.find((m) => m.type === "marked" && m.mark.name === "rambler");
ok(long && long.mark.text.length === 40, `long mark truncated to ${long?.mark.text.length}`);
const anon = await connect("library"); // no vid on the handshake
await sleep(300);
anon.send({ type: "mark", text: "ghost writer" });
await sleep(300);
ok(!anon.all.some((m) => m.type === "marked" && m.mark.text === "ghost writer"),
  "vid-less mark ignored");
L.ws.close();
anon.ws.close();

console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
