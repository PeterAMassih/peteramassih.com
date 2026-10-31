// game/tests/names-test.mjs
// Names come from the server's fixed word lists, derived from the visitor id:
// stable, never typed, and the id itself never leaves the room.
// Node 22+. Usage: node names-test.mjs [host]
import { connect, init, ok, done, sleep } from "./lib.mjs";

const NAME = /^[a-z]+ [a-z]+$/;
const vid = crypto.randomUUID();
const selfName = (c) => init(c).players.find((p) => p.id === init(c).id).name;

// 1. The same visitor id gets the same name, visit after visit.
const A = await connect("plaza", `&vid=${vid}`);
await sleep(300);
const name = selfName(A);
ok(NAME.test(name), `a name from the lists ("${name}")`);
A.ws.close();
await sleep(200);
const A2 = await connect("arena", `&vid=${vid}`);
await sleep(300);
ok(selfName(A2) === name, "the same vid keeps its name in another room");

// 2. A name from the client is ignored, on the handshake and as a message.
const W = await connect("arena");
await sleep(300);
const M = await connect("arena", `&vid=${crypto.randomUUID()}&name=Mallory`);
await sleep(300);
M.send({ type: "name", name: "Mallory" });
await sleep(300);
const joined = W.all.find((m) => m.type === "joined" && m.player.id === init(M).id);
ok(NAME.test(selfName(M)) && joined?.player.name === selfName(M), "the handshake name is ignored");
ok(!W.all.some((m) => JSON.stringify(m).includes("Mallory")), "no typed name reaches anyone");

// 3. Boards carry derived names: a real moon run by this vid.
const R = await connect("moon", `&vid=${vid}`);
await sleep(300);
ok(init(R).raceTop.every((r) => NAME.test(r.name) && !("vid" in r)), "the moon board shows names, no ids");
R.send({ type: "move", x: 60, y: 312 });
await sleep(1700);
R.send({ type: "move", x: 335, y: 142 });
await sleep(300);
const fin = R.all.find((m) => m.type === "race" && m.phase === "finish");
ok(fin && fin.top.every((r) => NAME.test(r.name) && !("vid" in r)), "the finish board shows names, no ids");
// Unless five faster runs fill the board. Matching the time alone is not
// enough: another visitor's run can tie to the millisecond.
ok(fin.top.some((r) => r.ms === fin.ms && r.name === name) || (fin.top.length === 5 && fin.top[4].ms <= fin.ms),
  "this run is listed under the visitor's name");

// 4. The visitor id never appears in anything the room sent.
ok(![A, A2, W, M, R].some((c) => JSON.stringify(c.all).includes(vid)), "no vid in any broadcast");

for (const c of [A2, W, M, R]) c.ws.close();
done();
