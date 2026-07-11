// Library hangman checks: guessing via chat, duplicates, wrong-word
// attempts, chat passthrough, room gating, and a full round to completion.
// Node 22+. Usage: node hang-test.mjs [host]
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
const hangs = (c) => c.all.filter((m) => m.type === "hang");
const state = (c) => (hangs(c).length ? hangs(c)[hangs(c).length - 1].state : init(c).hang);

// 1. The library has a word going; other rooms do not.
const A = await connect("library", "&name=guesser_a");
const B = await connect("library", "&name=guesser_b");
const P = await connect("plaza");
await sleep(300);
ok(init(P).hang === null, "no hangman outside the library");
const h0 = init(A).hang;
ok(h0 && h0.masked.length >= 4 && h0.lives === 8, `library has a word (${h0.masked})`);

// 2. A lone letter in chat is a guess, visible to everyone.
A.send({ type: "chat", text: "e" });
await sleep(300);
const g1 = hangs(B)[0];
ok(g1 && g1.by === init(A).id && g1.guess === "e", "B saw A guess the letter e");
ok(g1.state.masked.includes("e") || g1.state.wrongL.includes("e"),
  "the guess landed in the mask or the misses");

// 3. Guessing the same letter again changes nothing.
const before = hangs(B).length;
A.send({ type: "chat", text: "e" });
await sleep(300);
ok(hangs(B).length === before, "duplicate guess is ignored");

// 4. A wrong word of the right length costs a life.
const missesBefore = state(B).misses;
A.send({ type: "chat", text: "z".repeat(state(B).masked.length) });
await sleep(300);
ok(state(B).misses === missesBefore + 1, "a wrong word attempt costs a miss");

// 5. Ordinary sentences are still chat, not guesses.
B.send({ type: "chat", text: "hello library" });
await sleep(300);
ok(A.all.some((m) => m.type === "chat" && m.text === "hello library"),
  "multi-word chat passes through untouched");

// 6. Letters in the plaza are just chat.
P.send({ type: "chat", text: "e" });
await sleep(300);
ok(!P.all.some((m) => m.type === "hang"), "a lone letter in the plaza is only chat");

// 7. Drive the round to an end: two guessers alternate through the alphabet
//    until the word is solved or the gallows completes, then a fresh word.
let ended = null;
for (const c of "abcdefghijklmnopqrstuvwxyz") {
  (c.charCodeAt(0) % 2 ? A : B).send({ type: "chat", text: c });
  await sleep(120);
  ended = hangs(B).find((m) => m.event === "solve" || m.event === "fail");
  if (ended) break;
}
ok(ended, `the round ended (${ended?.event} by ${ended?.state.last?.name})`);
ok(ended.state.last && typeof ended.state.last.won === "boolean",
  "the outcome is recorded with a verdict");
ok(!("word" in ended.state.last),
  "the word itself never leaves the server, even after the round");
ok(!ended.state.masked.replace(/_/g, "").length && ended.state.misses === 0,
  "a fresh word is already waiting");

A.ws.close();
B.ws.close();
P.ws.close();
console.log(`\nALL ${passed} CHECKS PASSED`);
process.exit(0);
