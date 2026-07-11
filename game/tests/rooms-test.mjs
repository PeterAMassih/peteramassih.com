// v5 room checks: isolation, per-room rules, quiet room, name whitelist.
// Node 22+. Usage: node rooms-test.mjs [host]
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
    ws.onerror = () => reject(new Error(`${roomName}: connect failed`));
    setTimeout(() => reject(new Error(`${roomName}: connect timeout`)), 3000);
  });
}
const init = (c) => c.all.find((m) => m.type === "init");

// 1. Rooms are isolated worlds: chat in the arena never reaches the plaza.
const P = await connect("plaza");
const A = await connect("arena");
await sleep(300);
ok(init(P).room === "plaza", "plaza client landed in plaza");
ok(init(A).room === "arena", "arena client landed in arena");
ok(!init(P).players.some((pl) => pl.id === init(A).id), "plaza roster does not contain the arena player");
A.send({ type: "chat", text: "arena secret" });
await sleep(300);
ok(!P.all.some((m) => m.type === "chat"), "arena chat never reached the plaza");

// 2. Per-room rules arrive in init.
ok(init(P).gravity === 1 && init(P).brawl === true, "plaza: normal gravity, brawl on");
const M = await connect("moon");
await sleep(300);
ok(init(M).gravity === 0.35, `moon gravity is 0.35 (got ${init(M).gravity})`);

// 3. The library is a quiet room: punches produce nothing at all.
const L1 = await connect("library");
const L2 = await connect("library");
await sleep(300);
L1.send({ type: "move", x: 100, y: 312 });
L2.send({ type: "move", x: 115, y: 312 });
await sleep(300);
L1.send({ type: "punch", facing: 1 });
await sleep(300);
ok(!L2.all.some((m) => m.type === "swung" || m.type === "hit"), "no swing, no hit in the library");
ok(init(L1).brawl === false, "library init says brawl off");

// 4. Unknown room names fall back to the plaza (no arbitrary objects minted).
const X = await connect("basement");
await sleep(300);
ok(init(X).room === "plaza", `unknown room name lands in the plaza (got ${init(X).room})`);

// 5. Identity travels: name and color on the handshake are honored,
//    junk colors fall back to a palette pick.
const I = await connect("plaza", "&name=traveler&color=%232fa98a");
await sleep(300);
const selfI = init(I).players.find((p) => p.id === init(I).id);
ok(selfI.name === "traveler", "handshake name honored across a join");
ok(selfI.color === "#2fa98a", "handshake color honored across a join");
const J = await connect("plaza", "&name=x&color=hotpink");
await sleep(300);
const selfJ = init(J).players.find((p) => p.id === init(J).id);
ok(/^guest_\d{3}$/.test(selfJ.name), "invalid handshake name falls back to a guest name");
ok(selfJ.color.startsWith("#") && selfJ.color !== "hotpink", "unknown color falls back to the palette");

// 6. Per-room knockback arrives in init.
ok(init(P).kb === 1, "plaza knockback is 1x");
ok(init(A).kb === 1.6, `arena knockback is 1.6x (got ${init(A).kb})`);

// 7. The room remembers where a visitor stood: leave at (222, 200),
//    return with the same vid, resume there.
const V1 = await connect("plaza", "&vid=deadbeef-cafe-4000-8000-123456789abc");
await sleep(300);
V1.send({ type: "move", x: 222, y: 200 });
await sleep(300);
V1.ws.close();
await sleep(400);
const V2 = await connect("plaza", "&vid=deadbeef-cafe-4000-8000-123456789abc");
await sleep(300);
const selfV = init(V2).players.find((p) => p.id === init(V2).id);
ok(selfV.x === 222 && selfV.y === 200, `returning visitor resumed at (${selfV.x}, ${selfV.y})`);
V2.ws.close();

for (const c of [P, A, M, L1, L2, X, I, J]) c.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
