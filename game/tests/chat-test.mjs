// v1 chat checks: names, echo, rate limit, truncation. Node 22+.
// Usage: node chat-test.mjs [host]
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
    ws.onopen = () => resolve({ ws, all, name });
    ws.onerror = () => reject(new Error(`${name}: connect failed`));
    setTimeout(() => reject(new Error(`${name}: connect timeout`)), 3000);
  });
}

const A = await connect("A");
const B = await connect("B");
await sleep(400);

// 1. New joiners get a guest name; every player carries a valid name.
const initA = A.all.find((m) => m.type === "init");
const selfA = initA.players.find((p) => p.id === initA.id);
ok(/^guest_\d{3}$/.test(selfA.name), "new joiner got a guest_NNN name");
ok(initA.players.every((p) => /^[\w-]{2,16}$/.test(p.name)), "every player name is valid");

// 2. Chat reaches the other player AND echoes to the sender.
const initB = B.all.find((m) => m.type === "init");
B.ws.send(JSON.stringify({ type: "chat", text: "hello world" }));
await sleep(400);
ok(A.all.some((m) => m.type === "chat" && m.id === initB.id && m.text === "hello world"),
  "A received B's chat");
ok(B.all.some((m) => m.type === "chat" && m.id === initB.id && m.text === "hello world"),
  "B got the server echo of their own chat");

// 3. Rate limit: burst budget is 4 per 5s and one is already spent,
//    so of 6 rapid messages only 3 should get through.
for (let i = 1; i <= 6; i++) B.ws.send(JSON.stringify({ type: "chat", text: `spam ${i}` }));
await sleep(600);
const spamSeen = A.all.filter((m) => m.type === "chat" && m.text.startsWith("spam")).length;
ok(spamSeen === 3, `rate limit let 3 of 6 rapid messages through (saw ${spamSeen})`);

// 4. Oversized text is truncated to 120 chars; blank text is dropped.
await sleep(5100); // let the window reset
B.ws.send(JSON.stringify({ type: "chat", text: "x".repeat(300) }));
B.ws.send(JSON.stringify({ type: "chat", text: "   " }));
await sleep(400);
const long = A.all.find((m) => m.type === "chat" && m.text.startsWith("xxx"));
ok(long && long.text.length === 120, `long message truncated to ${long?.text.length} chars`);
ok(!A.all.some((m) => m.type === "chat" && m.text.trim() === ""), "blank message was dropped");

A.ws.close();
B.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
