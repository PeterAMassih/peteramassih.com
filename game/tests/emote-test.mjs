// v2 emote checks. Node 22+. Usage: node emote-test.mjs [host]
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
    ws.onopen = () => resolve({ ws, all });
    ws.onerror = () => reject(new Error(`${name}: connect failed`));
    setTimeout(() => reject(new Error(`${name}: connect timeout`)), 3000);
  });
}

const A = await connect("A");
const B = await connect("B");
await sleep(300);
const initB = B.all.find((m) => m.type === "init");

// 1. A wave reaches the other player and echoes to the sender.
B.ws.send(JSON.stringify({ type: "emote", kind: "wave" }));
await sleep(300);
ok(A.all.some((m) => m.type === "emote" && m.id === initB.id && m.kind === "wave"), "A saw B wave");
ok(B.all.some((m) => m.type === "emote" && m.kind === "wave"), "B got the wave echo");

// 2. Unknown kinds are dropped.
B.ws.send(JSON.stringify({ type: "emote", kind: "backflip" }));
await sleep(300);
ok(!A.all.some((m) => m.type === "emote" && m.kind === "backflip"), "unknown emote kind dropped");

// 3. Emote budget is 10 per 5s; one spent, so 9 of 14 hearts get through.
for (let i = 0; i < 14; i++) B.ws.send(JSON.stringify({ type: "emote", kind: "heart" }));
await sleep(500);
const hearts = A.all.filter((m) => m.type === "emote" && m.kind === "heart").length;
ok(hearts === 9, `emote rate limit let 9 of 14 hearts through (saw ${hearts})`);

// 4. Emote spam does not eat the chat budget.
B.ws.send(JSON.stringify({ type: "chat", text: "still chatting" }));
await sleep(300);
ok(A.all.some((m) => m.type === "chat" && m.text === "still chatting"), "chat unaffected by emote spam");

A.ws.close();
B.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
