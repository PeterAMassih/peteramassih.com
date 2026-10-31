// Moon-run checks: server-side timing, leaderboard, cheat rejection,
// best-per-visitor persistence. Node 22+. Usage: node race-test.mjs [host]
import { connect, init, ok, done, sleep } from "./lib.mjs";

const finishes = (c) => c.all.filter((m) => m.type === "race" && m.phase === "finish");

const START = { x: 60, y: 312 };
const FINISH = { x: 335, y: 142 };

// 1. A real run: enter start, climb (1.7s), reach the goal.
const A = await connect("moon", "&vid=cccc1111-2222-4333-8444-555566667777");
await sleep(300);
ok(Array.isArray(init(A).raceTop), "moon init carries the leaderboard");
const idA = init(A).id;
A.send({ type: "move", ...START });
await sleep(300);
ok(A.all.some((m) => m.type === "race" && m.phase === "start" && m.id === idA),
  "entering the start flag arms the run");
await sleep(1700);
A.send({ type: "move", ...FINISH });
await sleep(300);
const f1 = finishes(A)[0];
ok(f1 && f1.id === idA, "reaching the goal finishes the run");
ok(f1.ms >= 1500 && f1.ms < 4000, `server timed it (${f1.ms}ms)`);
// The board persists between runs, so the invariants are: fastest first, and
// never slower at the top than any run we just witnessed.
ok(f1.top[0]?.ms <= f1.ms && f1.top.every((r, i, a) => !i || a[i - 1].ms <= r.ms),
  "the board is recorded, fastest first");

// 2. A teleport claim (start to goal instantly) is not a run.
const B = await connect("moon", "&vid=dddd1111-2222-4333-8444-555566667777");
await sleep(300);
B.send({ type: "move", ...START });
await sleep(150);
B.send({ type: "move", ...FINISH });
await sleep(300);
ok(finishes(B).length === 0, "instant start-to-goal is rejected");

// 3. A slower second run finishes but does not overwrite the better time.
const bestMs = f1.ms;
A.send({ type: "move", ...START });
await sleep(2600);
A.send({ type: "move", ...FINISH });
await sleep(300);
const f2 = finishes(A)[1];
ok(f2 && f2.ms > bestMs, `second run was slower (${f2.ms}ms)`);
ok(f2.top[0].ms <= bestMs && !f2.top.some((r) => r.ms === f2.ms),
  "the board keeps the best time, not the last");

// 4. The plaza has no stopwatch.
const P = await connect("plaza");
await sleep(300);
P.send({ type: "move", ...START });
await sleep(200);
P.send({ type: "move", ...FINISH });
await sleep(300);
ok(!P.all.some((m) => m.type === "race"), "no race events outside the moon");

A.ws.close();
B.ws.close();
P.ws.close();
done();
