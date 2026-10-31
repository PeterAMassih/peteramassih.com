// Smoke test: the HTTP surface, joins, moves, clamping, kicks, leaves.
// Node 22+. Usage: node smoke-test.mjs [host]
import { HTTP, connect, ok, done, sleep } from "./lib.mjs";

// 1. Plain HTTP is refused, except the one presence endpoint.
const res = await fetch(HTTP);
ok(res.status === 426, `plain GET refused with 426 (got ${res.status})`);
const presence = await (await fetch(`${HTTP}/presence`)).json();
ok(Number.isInteger(presence.count) && presence.count >= 0,
  `presence answers a headcount (${presence.count})`);

// 2. A connects, gets init containing itself.
const A = await connect();
const initA = await A.next("init");
ok(initA.type === "init" && typeof initA.id === "string", "A got init with an id");
ok(initA.players.some((p) => p.id === initA.id), "A's init includes A itself");
const aCount = initA.players.length;

// 3. B connects: B's init has one more player; A hears joined.
const B = await connect();
const initB = await B.next("init");
ok(initB.players.length === aCount + 1, `B's init lists ${aCount + 1} players`);
const joinedA = await A.next("joined");
ok(joinedA.type === "joined" && joinedA.player.id === initB.id, "A heard B join");

// 4. B moves; A receives the exact position.
B.send({ type: "move", x: 100.5, y: 200 });
const moved = await A.next("moved");
ok(moved.type === "moved" && moved.id === initB.id && moved.x === 100.5 && moved.y === 200,
  "A saw B move to (100.5, 200)");

// 5. Out-of-bounds positions come back clamped by the server.
B.send({ type: "move", x: 99999, y: -50 });
const clamped = await A.next("clamped move");
ok(clamped.x === 640 && clamped.y === 0, `server clamped (99999,-50) to (${clamped.x},${clamped.y})`);

// 6. Malformed JSON gets that client disconnected with 1002.
const C = await connect();
const initC = await C.next("init");
await A.next("C joined"); // drain
C.ws.send("this is not json{{{");
await sleep(500);
ok(C.closed === 1002, `garbage sender was closed with 1002 (got ${C.closed})`);

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
done();
