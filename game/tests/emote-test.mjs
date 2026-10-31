// Emote checks. Node 22+. Usage: node emote-test.mjs [host]
import { connect, init, ok, done, sleep } from "./lib.mjs";

const A = await connect();
const B = await connect();
await sleep(300);
const idB = init(B).id;

// 1. A wave reaches the other player and echoes to the sender.
B.send({ type: "emote", kind: "wave" });
await sleep(300);
ok(A.all.some((m) => m.type === "emote" && m.id === idB && m.kind === "wave"), "A saw B wave");
ok(B.all.some((m) => m.type === "emote" && m.kind === "wave"), "B got the wave echo");

// 2. Unknown kinds are dropped.
B.send({ type: "emote", kind: "backflip" });
await sleep(300);
ok(!A.all.some((m) => m.type === "emote" && m.kind === "backflip"), "unknown emote kind dropped");

// 3. Emote budget is 10 per 5s; one spent, so 9 of 14 hearts get through.
for (let i = 0; i < 14; i++) B.send({ type: "emote", kind: "heart" });
await sleep(500);
const hearts = A.all.filter((m) => m.type === "emote" && m.kind === "heart").length;
ok(hearts === 9, `emote rate limit let 9 of 14 hearts through (saw ${hearts})`);

// 4. Emote spam does not eat the move budget.
B.send({ type: "move", x: 300, y: 312 });
await sleep(300);
ok(A.all.some((m) => m.type === "moved" && m.id === idB && m.x === 300), "moves unaffected by emote spam");

A.ws.close();
B.ws.close();
done();
