// game/tests/hang-test.mjs
// The library's hangman: one lowercase letter per guess, nothing else, only
// in the library, wrong letters sorted, guesses rate-limited.
// Node 22+. Usage: node hang-test.mjs [host]
import { connect, init, ok, done, sleep } from "./lib.mjs";

const LETTERS = [..."abcdefghijklmnopqrstuvwxyz"];
const hangs = (c) => c.all.filter((m) => m.type === "hang");
const sorted = (s) => [...s].sort().join("") === s;

// 1. The library sends the word as a mask, never the word itself.
const L = await connect("library");
await sleep(300);
const start = init(L).hang;
ok(start && /^[a-z_]+$/.test(start.masked) && !("word" in start), "library init carries the masked word only");

// 2. A letter nobody has tried changes the state: it lands in the mask, in
// the wrong letters, or ends the round.
const fresh = LETTERS.find((l) => !start.masked.includes(l) && !start.wrongL.includes(l));
L.send({ type: "guess", letter: fresh });
await sleep(300);
const h1 = hangs(L)[0];
ok(h1 && h1.guess === fresh && h1.by === init(L).id, `guessing "${fresh}" is broadcast`);
ok(h1.event !== "guess" || h1.state.masked.includes(fresh) || h1.state.wrongL.includes(fresh),
  "the guess updates the mask or the wrong letters");

// 3. Anything but one lowercase letter is ignored: no word attempts, no text.
for (const letter of ["A", "1", "ab", "", "é", " ", "<b>", 5, null]) L.send({ type: "guess", letter });
L.send({ type: "chat", text: "hello" });
L.send({ type: "guess" });
await sleep(400);
ok(hangs(L).length === 1, "non-letters, words and chat are ignored");

// 4. Wrong letters come back sorted, whatever order they were guessed in.
// Rare letters, guessed in reverse alphabetical order, are mostly misses.
for (const letter of ["z", "x", "w", "v", "q", "k", "j"]) L.send({ type: "guess", letter });
await sleep(500);
ok(hangs(L).every((m) => sorted(m.state.wrongL)), "wrong letters are always sorted");
ok(hangs(L).some((m) => m.state.wrongL.length >= 2), "at least two wrong letters were shown");

// 5. Other rooms have no word, and their guesses go nowhere.
const P = await connect("plaza");
await sleep(300);
ok(init(P).hang === null, "the plaza has no hangman");
P.send({ type: "guess", letter: "e" });
await sleep(300);
ok(hangs(P).length === 0 && hangs(L).every((m) => m.by !== init(P).id), "a plaza guess changes nothing");

// 6. Twenty guesses per five seconds per socket. Two passes over the alphabet
// would broadcast far more than that without the limit, since rounds end and
// letters free up again.
const G = await connect("library");
await sleep(300);
for (let i = 0; i < 52; i++) G.send({ type: "guess", letter: LETTERS[i % 26] });
await sleep(800);
const mine = hangs(G).filter((m) => m.by === init(G).id).length;
ok(mine >= 1 && mine <= 20, `a burst of 52 guesses broadcast ${mine} (limit 20)`);

L.ws.close();
P.ws.close();
G.ws.close();
done();
