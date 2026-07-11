// Brawl checks: swing broadcast, hit resolution, facing, range, immunity.
// Node 22+. Usage: node punch-test.mjs [host]
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

function connect(name) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(URL_WS);
    const all = [];
    ws.onmessage = (e) => all.push(JSON.parse(e.data));
    ws.onopen = () => resolve({ ws, all, send: (o) => ws.send(JSON.stringify(o)) });
    ws.onerror = () => reject(new Error(`${name}: connect failed`));
    setTimeout(() => reject(new Error(`${name}: connect timeout`)), 3000);
  });
}
const hits = (c) => c.all.filter((m) => m.type === "hit");
const swings = (c) => c.all.filter((m) => m.type === "swung");

const A = await connect("A");
const B = await connect("B");
await sleep(300);
const idA = A.all.find((m) => m.type === "init").id;
const idB = B.all.find((m) => m.type === "init").id;

// Stand adjacent: B is 20px to A's right.
A.send({ type: "move", x: 100, y: 312 });
B.send({ type: "move", x: 120, y: 312 });
await sleep(300);

// 1. Punch right: swing seen by all, B is hit with dir 1.
A.send({ type: "punch", facing: 1 });
await sleep(300);
ok(swings(B).some((m) => m.id === idA), "B saw A swing");
const h = hits(B).find((m) => m.attacker === idA);
ok(h && h.target === idB && h.dir === 1, "server resolved the hit on B, knocked right");

// 2. Immunity: an immediate second punch swings but cannot hit.
A.send({ type: "punch", facing: 1 });
await sleep(300);
ok(swings(B).length >= 2, "second swing broadcast");
ok(hits(B).length === 1, "immune target cannot be chain-hit");

// 3. Facing matters: B stands to the right, A punches LEFT.
await sleep(2300); // let immunity lapse
A.send({ type: "punch", facing: -1 });
await sleep(300);
ok(hits(B).length === 1, "punching away from the target misses");

// 4. Range matters: B walks out of reach.
B.send({ type: "move", x: 200, y: 312 });
await sleep(200);
A.send({ type: "punch", facing: 1 });
await sleep(300);
ok(hits(B).length === 1, "punching from 100px away misses");

// 5. Back in range, immunity long gone: it lands again.
B.send({ type: "move", x: 118, y: 312 });
await sleep(200);
A.send({ type: "punch", facing: 1 });
await sleep(300);
ok(hits(B).length === 2, "after immunity lapses the next punch lands");

// 6. Overlapping sprites can hit each other (the phone bug): B stands
//    exactly on A, punch in either direction still lands.
await sleep(2300);
B.send({ type: "move", x: 100, y: 312 });
await sleep(200);
A.send({ type: "punch", facing: -1 });
await sleep(300);
ok(hits(B).length === 3, "punch lands on a fully overlapping target");

A.ws.close();
B.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
