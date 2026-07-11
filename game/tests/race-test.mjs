// Moon-run checks: server-side timing, leaderboard, cheat rejection,
// best-per-visitor persistence. Node 22+. Usage: node race-test.mjs [host]
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
const finishes = (c) => c.all.filter((m) => m.type === "race" && m.phase === "finish");

const START = { x: 60, y: 312 };
const FINISH = { x: 335, y: 142 };

// 1. A real run: enter start, climb (1.7s), reach the goal.
const A = await connect("moon", "&vid=cccc1111-2222-4333-8444-555566667777&name=runner");
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
ok(f1.top.some((r) => r.name === "runner"), "runner is on the board");

// 2. A teleport claim (start to goal instantly) is not a run.
const B = await connect("moon", "&vid=dddd1111-2222-4333-8444-555566667777&name=cheater");
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
const onBoard = f2.top.find((r) => r.name === "runner");
// The stored best may predate this session (the table persists), so the
// invariant is: never worse than any run we just witnessed.
ok(onBoard && onBoard.ms <= bestMs, "the board keeps the best time, not the last");

// 4. The presence endpoint volunteers the moon record for the homepage.
const p = await (await fetch(`${SECURE ? "https" : "http"}://${HOST}/presence`)).json();
ok(Number.isInteger(p.record) && p.record <= bestMs,
  `presence carries the moon record (${p.record}ms)`);

// 5. The plaza has no stopwatch.
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
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
