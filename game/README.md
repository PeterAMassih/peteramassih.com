# play — a tiny multiplayer world on the free tier

The game at [peteramassih.com/play](https://peteramassih.com/play). Every
character is a live visitor. Four rooms, four games: hang out on the plaza,
fight for the crown in the arena, write on the library wall, race the clock
on the moon. The whole thing runs on Cloudflare's free tier and costs
nothing while nobody plays — which is most of the time, and the design
treats that as a feature.

## Architecture

```
Browser  /play  (canvas client: rendering, input, physics, prediction)
   |  one WebSocket per player
   v
Worker  game/src/index.ts  (upgrade, origin check, room whitelist)
   |  idFromName(room)
   v
Room  game/src/room.ts  (one Durable Object per room: the authority)
   |- in-memory Map      live players (rebuilt after hibernation)
   |- socket attachments the durable copy of each player
   '- SQLite             marks, race times, reigns, positions, crown
```

One `Room` class, four named instances. The Worker maps `?room=` onto a
whitelist and hands the socket to that room's object. Walking through a
door is the client closing one socket and opening the next.

## The two rules that keep it free

1. **The server never ticks.** No timers, no alarms, no game loop. Every
   piece of logic runs inside an event the runtime already delivers: a
   message, a close, a join. An idle room hibernates (WebSocket
   Hibernation API) and bills nothing, even with players connected.
2. **Physics runs in the browser.** Your machine simulates your own
   gravity, jumps, and knockback, and reports positions. The server clamps
   them, referees everything contested (punches, pickups, stopwatches),
   and relays the rest.

Everything else follows from those two. The crown has no respawn timer
because a pickup is a move and a drop is a hit. The race has no clock
because the server timestamps two moves and subtracts. The position sweep
runs piggybacked on a join, never on a schedule.

## Protocol

Client to room:

| type | payload | notes |
|---|---|---|
| `move` | `x, y` | 10 Hz max while moving, clamped and rate-limited server-side |
| `chat` | `text` | 120 chars, 4 per 5s, never stored |
| `emote` | `kind` | `wave` or `heart`, looser budget |
| `name` | `name` | `[\w-]{2,16}`, also accepted on the handshake URL |
| `punch` | `facing` | server resolves the hit from its own positions |
| `mark` | `text` | 40 chars, library only, one per visitor |

Room to clients: `init` (you, the roster, room rules, boards, crown),
`joined`, `moved`, `named`, `chat`, `emote`, `swung`, `hit`, `crown`,
`reign`, `race`, `marked`, `left`.

The hit broadcast carries attacker, target, and direction; the *target's
own client* applies the knockback impulse. One movement authority per
sprite, no exceptions.

## What persists, and for how long

| data | where | retention |
|---|---|---|
| chat | nowhere | seconds of RAM, then gone |
| your name and color | your browser (localStorage) + handshake | yours to keep |
| last position per room | room storage, keyed by anonymous visitor id | 30 days, then swept |
| library marks | `marks` table | forever, one per visitor, 500 cap |
| moon run times | `races` table | forever, your best only, 200 cap |
| crown reigns | `reigns` table | forever, your longest only, 200 cap |
| the crown itself | room storage | survives hibernation |

The visitor id is a random UUID the browser generates once and presents on
connect. It is never broadcast to other players.

## Hibernation, honestly

The in-memory player map dies with every eviction. The durable copy of
each player rides on their own WebSocket via `serializeAttachment`, and the
constructor rebuilds the world from `getWebSockets()` on wake. Anything
that must outlive the sockets themselves (crown, boards, positions) lives
in storage. Transient state (rate limiters, active runs, immunity timers)
is deliberately allowed to die: a fresh burst allowance after a nap is
harmless, a phantom one is not.

## Budgets

Free tier gives each day: 100k requests (inbound WebSocket messages count
20:1, broadcasts are free) and 13,000 GB-s of duration (hibernating rooms
accrue none). A full room of twenty frantic players spends requests about
36k/hour, and needs hours of sustained play to threaten the duration
budget. Every message type is rate-limited; the room caps at 32 players.

## Tests

Eight suites, ~90 checks, run against a real server (local or production):

```
npx wrangler dev -c game/wrangler.jsonc          # terminal 1
node game/tests/smoke-test.mjs                   # terminal 2, repeat per suite
node game/tests/crown-test.mjs [host]            # host arg targets production
```

They exercise the protocol over real WebSockets: clamping, rate limits,
hit resolution, immunity, room isolation, mark upsert, server-side race
timing, reign recording. The boards persist between runs, so assertions
are written against invariants, not fresh-state assumptions.

## Dev and deploy

```
npx wrangler dev -c game/wrangler.jsonc          # local server on :8787
npx wrangler deploy -c game/wrangler.jsonc       # ship the server
```

The `-c` flag is load-bearing: the site's Astro adapter generates a
deploy-redirect config at the repo root that otherwise collides with this
nested project. The client ships with the site build (`src/scripts/play.js`
on the `/play` page); server and client deploy independently, server first
when the protocol grows.

## Reading order

1. `game/src/index.ts` — the front door, 50 lines.
2. `game/src/room.ts` — the whole server. Read `fetch` (a join), then
   `webSocketMessage` (the protocol), then `drop` (a leave). Everything
   else is a helper.
3. `src/scripts/play.js` — the client, top to bottom: constants and room
   table, sprites, socket lifecycle, input, physics (`frame`), rendering
   (`draw`).
