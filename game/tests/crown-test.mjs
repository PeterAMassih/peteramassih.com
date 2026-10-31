// Crown checks: spawn, pickup by walking, drop on hit, re-pickup,
// drop on disconnect, and absence outside the arena. Node 22+.
import { connect, init, ok, done, sleep } from "./lib.mjs";

const lastCrown = (c) => [...c.all].reverse().find((m) => m.type === "crown")?.crown;

// 1. The arena spawns a crown on the hill; the plaza has none.
const A = await connect("arena", "&vid=eeee1111-2222-4333-8444-555566667777");
const B = await connect("arena");
const P = await connect("plaza");
await sleep(300);
ok(init(P).crown === null || init(P).crown === undefined, "no crown in the plaza");
ok(Array.isArray(init(A).reignTop), "arena init carries the reigns board");
const c0 = init(A).crown;
ok(c0 && c0.wearer === null, "arena crown starts lying in wait");

// 2. Walking into it picks it up.
const idA = init(A).id;
const idB = init(B).id;
A.send({ type: "move", x: c0.x, y: c0.y });
await sleep(300);
ok(lastCrown(B)?.wearer === idA, "A walked into the crown and wears it");

// 3. Punching the wearer knocks it off; it lands on the floor nearby.
//    (Wait first so the reign is long enough to count.)
await sleep(1300);
B.send({ type: "move", x: c0.x + 15, y: c0.y });
await sleep(250);
B.send({ type: "punch", facing: -1 });
await sleep(300);
const dropped = lastCrown(A);
ok(dropped && dropped.wearer === null, "the hit knocked the crown off");
const reign = A.all.find((m) => m.type === "reign" && m.id === idA);
ok(reign && reign.ms >= 1200, `the reign was measured (${reign?.ms}ms)`);
// The board persists between runs, so the invariant is: longest first, and
// never shorter at the top than the reign we just witnessed.
ok(reign.top[0]?.ms >= reign.ms && reign.top.every((r, i, a) => !i || a[i - 1].ms >= r.ms),
  "the reigns board is recorded, longest first");
const expectedX = Math.min(Math.max(c0.x - 36, 20), 620); // server clamps the same way
ok(dropped.y === 312 && Math.abs(dropped.x - expectedX) < 1,
  `it fell to the floor beside the fight (x=${dropped.x})`);

// 4. The attacker can walk over and claim it.
B.send({ type: "move", x: dropped.x, y: 312 });
await sleep(300);
ok(lastCrown(A)?.wearer === idB, "B claimed the fallen crown");

// 5. A wearer who leaves forfeits: the crown lands on one of the tiers.
const PERCHES = [[320, 177], [120, 250], [520, 250]];
B.ws.close();
await sleep(400);
const abandoned = lastCrown(A);
ok(abandoned && abandoned.wearer === null &&
  PERCHES.some(([px, py]) => abandoned.x === px && abandoned.y === py),
  `the leaver's crown landed on a tier (${abandoned?.x},${abandoned?.y})`);

A.ws.close();
P.ws.close();
done();
