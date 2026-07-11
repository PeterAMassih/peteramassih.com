// game/src/room.ts
// One Room instance is one shared world. Strictly event-driven: no timers, no
// tick loop, so an idle room hibernates and bills nothing.
import { DurableObject } from "cloudflare:workers";

// Mirrored in the client (src/scripts/play.js). Kept as plain constants on
// both sides rather than a shared package: two deploy targets, one line each.
const WORLD = { w: 640, h: 360 };

// Every room is one instance of this class, picked by name in the Worker.
// The server owns the rules (gravity is echoed to clients, brawl is enforced
// here); looks and doors live client-side. Unknown names never reach us —
// the Worker whitelists against this same list.
export const ROOM_NAMES = ["plaza", "arena", "library", "moon"] as const;
interface RaceZones {
	sx: number;
	sy: number;
	fx: number;
	fy: number;
}

const ROOM_RULES: Record<
	string,
	{
		gravity: number; brawl: boolean; kb: number; marks: boolean; crown: boolean;
		hangman?: boolean; race?: RaceZones;
	}
> = {
	plaza: { gravity: 1, brawl: true, kb: 1, marks: false, crown: false },
	arena: { gravity: 1, brawl: true, kb: 1.6, marks: false, crown: true }, // king of the hill
	library: { gravity: 1, brawl: false, kb: 1, marks: true, crown: false, hangman: true },
	moon: {
		gravity: 0.35, brawl: true, kb: 1, marks: false, crown: false,
		// The moon run: floor flag to the top rock. The server does the timing —
		// it stamps the move that leaves the start and the move that reaches the
		// goal, so a client cannot lie about its time (only about ~100ms of it).
		race: { sx: 60, sy: 312, fx: 335, fy: 142 },
	},
};

const RACE_R = 22; // how close a move must be to a flag to count
const RACE_MIN_MS = 1500; // faster than this is a teleport claim, not a run
const RACE_TOP = 5;
const RACE_KEPT = 200;

// The library's hangman. A lone letter in chat is a guess, a word of the
// right length is an attempt, and the puzzle persists across hibernation —
// a half-solved word waits for the next stranger. Words from this world.
const HANG_WORDS = [
	"pixel", "crown", "plaza", "arena", "gravity", "socket", "worker", "durable",
	"object", "tensor", "gradient", "neuron", "library", "hibernate", "sprite",
	"canvas", "emote", "guestbook", "record", "reign", "punch", "bubble",
	"platform", "visitor", "moon", "kernel", "matrix", "entropy", "lambda",
	"vector", "cache", "quantize", "diffusion", "token", "epoch", "layer",
];
const HANG_LIVES = 8;
const GUESS_BURST = 20; // guessing is bursty in co-op; looser than chat

interface Hang {
	word: string;
	found: string; // letters guessed right
	wrongL: string; // letters guessed wrong, for display
	misses: number; // wrong letters plus failed word attempts
	last: { name: string; won: boolean } | null;
}

// The crown needs no tick: a pickup is a move, a drop is a hit or a leave.
// When forfeited it lands on a random arena tier (coordinates mirror the
// client's platform layout) — always elevated, never quite where you left it.
const CROWN_PERCHES = [
	{ x: 320, y: 177 }, // the hill
	{ x: 120, y: 250 }, // left tier
	{ x: 520, y: 250 }, // right tier
];
const CROWN_SPAWN = CROWN_PERCHES[0];
const CROWN_REACH = { x: 20, y: 26 };

interface Crown {
	wearer: string | null;
	x: number;
	y: number;
	since?: number; // when the current wearer took it — reigns are this minus now
}

const REIGN_MIN_MS = 1000; // shorter than this is a fumble, not a reign
const REIGN_TOP = 5;

// The guestbook: one mark per visitor per room (writing again replaces your
// old one), capped at 500 with the oldest pruned — self-cleaning by design.
// Positions expire instead: ignored after 30 days, swept opportunistically.
const MARK_MAX_LEN = 40;
const MARK_BURST = 3;
const MARKS_KEPT = 500;
const MARKS_SENT = 60; // what a joiner sees: the freshest wallful
const POS_TTL_MS = 30 * 24 * 3600 * 1000;
const SWEEP_EVERY_MS = 24 * 3600 * 1000;

const PALETTE = [
	"#e6482e", // red
	"#3b82c4", // blue
	"#2fa98a", // teal
	"#7b5cd6", // purple
	"#e5a33b", // amber
	"#d4537e", // pink
	"#6cae44", // green
	"#f2f2f2", // white
];

// Chat guards: silently drop past this rate, hard-cap the length.
const CHAT_MAX_LEN = 120;
const CHAT_WINDOW_MS = 5000;
const CHAT_BURST = 4;

// Emotes are cheaper than chat and spamming hearts is part of the fun,
// so they get their own, looser budget.
const EMOTES = ["wave", "heart"] as const;
const EMOTE_BURST = 10;

// Names: word characters and dashes only, so they stay renderable on the
// canvas and safe to log. Renames are rare; the tight budget stops flicker.
const NAME_RE = /^[\w-]{2,16}$/;
const NAME_BURST = 3;

// The brawl. A punch hits the nearest player in front within reach, knocked
// back by their own client on the hit broadcast. Immunity after a hit means
// nobody can be chain-punched — that is what keeps it funny instead of mean.
// STUN_MS + IMMUNE_MS are mirrored in the client for the tumble and blink.
const PUNCH_RANGE_X = 30;
const PUNCH_RANGE_Y = 26;
const PUNCH_BURST = 20; // paired with the client's 260ms cooldown
const STUN_MS = 700;
const IMMUNE_MS = 1500;

// The honest client sends at most 10 moves/s; the budget leaves headroom for
// bursts but stops a hostile script from amplifying through the broadcast.
const MOVE_BURST = 80;
// Well above the 2-20 design target; insurance against a scripted mass-join
// (each join costs an O(n) broadcast and a slice of per-connection memory).
const MAX_PLAYERS = 32;

interface Player {
	id: string;
	name: string;
	x: number;
	y: number;
	color: string;
}

export class Room extends DurableObject<Env> {
	// Live state is an in-memory cache. The durable copy of each player rides on
	// their socket via serializeAttachment, so a room woken from hibernation can
	// rebuild everything from the sockets the runtime kept open for it.
	private players = new Map<WebSocket, Player>();
	// Anonymous visitor ids (from the handshake, never broadcast): the key the
	// room remembers positions under, so a return visit resumes where you left.
	private vids = new Map<WebSocket, string>();
	// Which room this object is. First use of durable storage: derived from
	// the URL on first join, persisted, and reloaded after hibernation — the
	// punch handler needs the rules even when the wake-up event is a message.
	private roomName: string | null = null;
	private crown: Crown | null = null;
	private hang: Hang | null = null;
	// Active runs, playerId -> started-at. In-memory only: a run is seconds
	// long, and one lost to a mid-run hibernation is no loss at all.
	private runs = new Map<string, number>();
	// Chat and emote timestamps per socket, for rate limiting. Deliberately not
	// in the attachment: losing them across hibernation just grants a fresh burst.
	private chatTimes = new Map<WebSocket, number[]>();
	private emoteTimes = new Map<WebSocket, number[]>();
	private nameTimes = new Map<WebSocket, number[]>();
	private punchTimes = new Map<WebSocket, number[]>();
	private moveTimes = new Map<WebSocket, number[]>();
	private markTimes = new Map<WebSocket, number[]>();
	private guessTimes = new Map<WebSocket, number[]>();
	// Transient combat state, by player id. Deliberately not persisted: a room
	// that hibernated mid-brawl waking up with immunity cleared is harmless.
	private immuneUntil = new Map<string, number>();

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		for (const ws of ctx.getWebSockets()) {
			const att = ws.deserializeAttachment() as { p: Player; vid: string | null };
			this.players.set(ws, att.p);
			if (att.vid) this.vids.set(ws, att.vid);
		}
		ctx.blockConcurrencyWhile(async () => {
			this.roomName = ((await ctx.storage.get("room")) as string | undefined) ?? null;
			this.crown = ((await ctx.storage.get("crown")) as Crown | undefined) ?? null;
			this.hang = ((await ctx.storage.get("hang")) as Hang | undefined) ?? null;
			// A wearer who vanished while the room slept forfeits: back to the spawn.
			if (this.crown?.wearer && ![...this.players.values()].some((p) => p.id === this.crown?.wearer)) {
				this.crown = { wearer: null, ...CROWN_SPAWN };
			}
			ctx.storage.sql.exec(
				"CREATE TABLE IF NOT EXISTS marks (vid TEXT PRIMARY KEY, name TEXT, text TEXT, x REAL, ts INTEGER)",
			);
			ctx.storage.sql.exec(
				"CREATE TABLE IF NOT EXISTS races (vid TEXT PRIMARY KEY, name TEXT, ms INTEGER, ts INTEGER)",
			);
			ctx.storage.sql.exec(
				"CREATE TABLE IF NOT EXISTS reigns (vid TEXT PRIMARY KEY, name TEXT, ms INTEGER, ts INTEGER)",
			);
		});
	}

	private topReigns() {
		return this.ctx.storage.sql
			.exec("SELECT name, ms FROM reigns ORDER BY ms DESC LIMIT ?", REIGN_TOP)
			.toArray();
	}

	// Close out the current wearer's reign: record it if it is their longest,
	// tell the room how long they lasted. Called before the crown moves on.
	// The name comes from the caller: on the leave path the player is already
	// out of the map, so a lookup here would come up empty.
	private endReign(vid: string | null, name: string): void {
		const c = this.crown;
		if (!c?.wearer || !c.since) return;
		const ms = Date.now() - c.since;
		if (ms < REIGN_MIN_MS) return;
		if (vid) {
			const best = this.ctx.storage.sql.exec("SELECT ms FROM reigns WHERE vid = ?", vid).toArray()[0];
			if (!best || (best.ms as number) < ms) {
				this.ctx.storage.sql.exec(
					"INSERT OR REPLACE INTO reigns (vid, name, ms, ts) VALUES (?, ?, ?, ?)",
					vid, name, ms, Date.now(),
				);
				this.ctx.storage.sql.exec(
					"DELETE FROM reigns WHERE vid NOT IN (SELECT vid FROM reigns ORDER BY ms DESC LIMIT ?)",
					RACE_KEPT,
				);
			}
		}
		this.broadcast({ type: "reign", id: c.wearer, ms, top: this.topReigns() });
	}

	private wsOf(id: string): WebSocket | null {
		for (const [ws, p] of this.players) if (p.id === id) return ws;
		return null;
	}

	private newWord(not: string): string {
		let w = not;
		while (w === not) w = HANG_WORDS[Math.floor(Math.random() * HANG_WORDS.length)];
		return w;
	}

	// What clients see: the mask, never the word itself.
	private hangView() {
		const h = this.hang;
		if (!h) return null;
		return {
			masked: [...h.word].map((c) => (h.found.includes(c) ? c : "_")).join(""),
			wrongL: h.wrongL,
			misses: h.misses,
			lives: HANG_LIVES,
			// Projected, not passed through: rounds persisted before the word
			// was dropped from the verdict still carry it in storage.
			last: h.last ? { name: h.last.name, won: h.last.won } : null,
		};
	}

	private handleGuess(player: Player, t: string): void {
		const h = this.hang;
		if (!h) return;
		if (t.length === 1) {
			if (h.found.includes(t) || h.wrongL.includes(t)) return; // already tried
			if (h.word.includes(t)) h.found += t;
			else {
				h.wrongL += t;
				h.misses++;
			}
		} else if (t === h.word) {
			h.found = [...new Set(h.word)].join("");
		} else {
			h.misses++;
		}
		const solved = [...h.word].every((c) => h.found.includes(c));
		const failed = h.misses >= HANG_LIVES;
		let event = "guess";
		if (solved || failed) {
			event = solved ? "solve" : "fail";
			h.last = { name: player.name, won: solved };
			h.word = this.newWord(h.word);
			h.found = "";
			h.wrongL = "";
			h.misses = 0;
		}
		void this.ctx.storage.put("hang", h);
		this.broadcast({
			type: "hang",
			event,
			by: player.id,
			guess: t.length === 1 ? t : "a word",
			state: this.hangView(),
		});
	}

	private topRuns() {
		return this.ctx.storage.sql
			.exec("SELECT name, ms FROM races ORDER BY ms ASC LIMIT ?", RACE_TOP)
			.toArray();
	}

	private setCrown(crown: Crown): void {
		this.crown = crown;
		void this.ctx.storage.put("crown", crown);
		// held rides along so clients can count reigns on their own clock —
		// comparing raw timestamps across machines invites clock-skew nonsense.
		this.broadcast({ type: "crown", crown, held: crown.since ? Date.now() - crown.since : null });
	}

	// One attachment shape everywhere: the broadcastable player plus the
	// private visitor id, both surviving hibernation on the socket itself.
	private attach(ws: WebSocket, player: Player): void {
		ws.serializeAttachment({ p: player, vid: this.vids.get(ws) ?? null });
	}

	private get rules() {
		return ROOM_RULES[this.roomName ?? "plaza"] ?? ROOM_RULES.plaza;
	}

	async fetch(request: Request): Promise<Response> {
		// Anything that is not a socket upgrade is a presence ping. Answered
		// before any join state is touched, so a count can never mis-name a
		// fresh room or mint a crown. Race rooms also volunteer their record:
		// an empty world with a standing record is a challenge, not a void.
		if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
			const record = this.rules.race
				? ((this.ctx.storage.sql
						.exec("SELECT ms FROM races ORDER BY ms ASC LIMIT 1")
						.toArray()[0]?.ms as number | undefined) ?? null)
				: null;
			return Response.json({ count: this.players.size, record });
		}
		if (this.players.size >= MAX_PLAYERS) {
			return new Response("Room is full.", { status: 503 });
		}
		const url = new URL(request.url);
		if (!this.roomName) {
			this.roomName = url.searchParams.get("room") ?? "plaza";
			void this.ctx.storage.put("room", this.roomName);
		}
		if (this.rules.crown && !this.crown) {
			this.crown = { wearer: null, ...CROWN_SPAWN };
			void this.ctx.storage.put("crown", this.crown);
		}
		if (this.rules.hangman && !this.hang) {
			this.hang = { word: this.newWord(""), found: "", wrongL: "", misses: 0, last: null };
			void this.ctx.storage.put("hang", this.hang);
		}
		const [client, server] = Object.values(new WebSocketPair());

		// Identity travels with the visitor: the client passes its remembered
		// name and color so walking through a door does not reroll the sprite.
		// Both are validated — unknown colors and bad names fall back to fresh.
		const wantName = url.searchParams.get("name");
		const wantColor = url.searchParams.get("color");
		const player: Player = {
			id: crypto.randomUUID(),
			name: wantName && NAME_RE.test(wantName)
				? wantName
				: `guest_${100 + Math.floor(Math.random() * 900)}`,
			x: 40 + Math.floor(Math.random() * (WORLD.w - 80)),
			// Spawn strictly above the client's floor line (y=312) so nobody
			// stands sunk into the ground until their first jump.
			y: 40 + Math.floor(Math.random() * 240),
			color: wantColor && PALETTE.includes(wantColor)
				? wantColor
				: PALETTE[Math.floor(Math.random() * PALETTE.length)],
		};

		// This room remembers you: a returning visitor id resumes at the spot
		// it left from. One storage read per join, one write per leave — the
		// whole persistence bill for a visit is two operations.
		const vid = url.searchParams.get("vid");
		const validVid = vid && /^[0-9a-f-]{8,40}$/i.test(vid) ? vid : null;
		if (validVid) {
			const saved = (await this.ctx.storage.get(`pos:${validVid}`)) as
				| { x: number; y: number; t?: number }
				| undefined;
			if (saved && Date.now() - (saved.t ?? 0) < POS_TTL_MS) {
				player.x = clamp(saved.x, 0, WORLD.w);
				player.y = clamp(saved.y, 0, WORLD.h);
			}
		}
		void this.sweepPositions(); // hygiene rides along on a join, never a timer

		// acceptWebSocket (not accept) opts into hibernation: the runtime holds
		// the connection open while the object itself can be evicted between
		// messages. This is the whole free-tier economy of the design.
		this.ctx.acceptWebSocket(server);
		if (validVid) this.vids.set(server, validVid);
		this.attach(server, player);
		this.players.set(server, player);

		server.send(
			JSON.stringify({
				type: "init",
				id: player.id,
				room: this.roomName,
				gravity: this.rules.gravity,
				brawl: this.rules.brawl,
				kb: this.rules.kb,
				crown: this.crown,
				crownHeld: this.crown?.since ? Date.now() - this.crown.since : null,
				reignTop: this.rules.crown ? this.topReigns() : [],
				raceTop: this.rules.race ? this.topRuns() : [],
				marksOn: this.rules.marks,
				hang: this.hangView(),
				marks: this.rules.marks
					? this.ctx.storage.sql
							.exec("SELECT name, text, x, ts FROM marks ORDER BY ts DESC LIMIT ?", MARKS_SENT)
							.toArray()
					: [],
				players: [...this.players.values()],
			}),
		);
		this.broadcast({ type: "joined", player }, server);

		return new Response(null, { status: 101, webSocket: client });
	}

	async webSocketMessage(ws: WebSocket, raw: string | ArrayBuffer): Promise<void> {
		const player = this.players.get(ws);
		if (!player || typeof raw !== "string") return;

		let msg: {
			type?: unknown;
			x?: unknown;
			y?: unknown;
			text?: unknown;
			kind?: unknown;
			name?: unknown;
			facing?: unknown;
		};
		try {
			msg = JSON.parse(raw);
		} catch {
			ws.close(1002, "malformed message"); // a broken or hostile client, not us
			return;
		}

		if (msg.type === "move") {
			if (!Number.isFinite(msg.x) || !Number.isFinite(msg.y)) return;
			if (!this.underLimit(this.moveTimes, ws, MOVE_BURST)) return;
			// The server owns the truth: positions are clamped, never trusted raw.
			player.x = clamp(msg.x as number, 0, WORLD.w);
			player.y = clamp(msg.y as number, 0, WORLD.h);
			this.attach(ws, player); // keep the durable copy current for wake-ups
			this.broadcast({ type: "moved", id: player.id, x: player.x, y: player.y }, ws);
			// Walking into the lying crown picks it up. A move is the only way
			// to reach it, so pickup needs no clock of its own.
			if (
				this.crown && !this.crown.wearer &&
				Math.abs(player.x - this.crown.x) < CROWN_REACH.x &&
				Math.abs(player.y - this.crown.y) < CROWN_REACH.y
			) {
				this.setCrown({ wearer: player.id, x: 0, y: 0, since: Date.now() });
			}

			// The race, timed entirely by the server's own clock. Every move
			// inside the start zone re-arms the timer, so the run effectively
			// starts on the move that leaves the flag.
			const race = this.rules.race;
			if (race) {
				const now = Date.now();
				if (Math.abs(player.x - race.sx) < RACE_R && Math.abs(player.y - race.sy) < RACE_R) {
					const had = this.runs.has(player.id);
					this.runs.set(player.id, now);
					if (!had) this.broadcast({ type: "race", phase: "start", id: player.id });
				} else if (
					this.runs.has(player.id) &&
					Math.abs(player.x - race.fx) < RACE_R &&
					Math.abs(player.y - race.fy) < RACE_R
				) {
					const ms = now - (this.runs.get(player.id) ?? now);
					this.runs.delete(player.id);
					if (ms >= RACE_MIN_MS) {
						const vid = this.vids.get(ws);
						if (vid) {
							const best = this.ctx.storage.sql
								.exec("SELECT ms FROM races WHERE vid = ?", vid)
								.toArray()[0];
							if (!best || (best.ms as number) > ms) {
								this.ctx.storage.sql.exec(
									"INSERT OR REPLACE INTO races (vid, name, ms, ts) VALUES (?, ?, ?, ?)",
									vid, player.name, ms, now,
								);
								this.ctx.storage.sql.exec(
									"DELETE FROM races WHERE vid NOT IN (SELECT vid FROM races ORDER BY ms ASC LIMIT ?)",
									RACE_KEPT,
								);
							}
						}
						this.broadcast({ type: "race", phase: "finish", id: player.id, ms, top: this.topRuns() });
					}
				}
			}
		} else if (msg.type === "chat") {
			if (typeof msg.text !== "string") return;
			const text = msg.text.trim().slice(0, CHAT_MAX_LEN);
			if (!text) return;
			// In the library a lone letter is a hangman guess, and a word of
			// exactly the right length is an attempt. Everything else is chat.
			if (this.rules.hangman && this.hang) {
				const t = text.toLowerCase();
				if (/^[a-z]$/.test(t) || (/^[a-z]+$/.test(t) && t.length === this.hang.word.length)) {
					if (!this.underLimit(this.guessTimes, ws, GUESS_BURST)) return;
					this.handleGuess(player, t);
					return;
				}
			}
			if (!this.underLimit(this.chatTimes, ws, CHAT_BURST)) return;
			// Everyone gets it, sender included — your own bubble is the server echo,
			// so what you see is exactly what the room saw.
			this.broadcast({ type: "chat", id: player.id, text });
		} else if (msg.type === "emote") {
			if (!EMOTES.includes(msg.kind as (typeof EMOTES)[number])) return;
			if (!this.underLimit(this.emoteTimes, ws, EMOTE_BURST)) return;
			this.broadcast({ type: "emote", id: player.id, kind: msg.kind });
		} else if (msg.type === "name") {
			if (typeof msg.name !== "string" || !NAME_RE.test(msg.name)) return;
			if (!this.underLimit(this.nameTimes, ws, NAME_BURST)) return;
			player.name = msg.name;
			this.attach(ws, player); // renames survive hibernation too
			this.broadcast({ type: "named", id: player.id, name: player.name });
		} else if (msg.type === "mark") {
			if (!this.rules.marks || typeof msg.text !== "string") return;
			const vid = this.vids.get(ws);
			if (!vid) return; // marks need a rememberable author
			const text = msg.text.trim().slice(0, MARK_MAX_LEN);
			if (!text) return;
			if (!this.underLimit(this.markTimes, ws, MARK_BURST)) return;
			const ts = Date.now();
			// One mark per visitor: writing again replaces yours. The cap prunes
			// the oldest beyond 500, so the wall can never grow without bound.
			this.ctx.storage.sql.exec(
				"INSERT OR REPLACE INTO marks (vid, name, text, x, ts) VALUES (?, ?, ?, ?, ?)",
				vid, player.name, text, player.x, ts,
			);
			this.ctx.storage.sql.exec(
				"DELETE FROM marks WHERE vid NOT IN (SELECT vid FROM marks ORDER BY ts DESC LIMIT ?)",
				MARKS_KEPT,
			);
			this.broadcast({ type: "marked", mark: { name: player.name, text, x: player.x, ts } });
		} else if (msg.type === "punch") {
			if (!this.rules.brawl) return; // the quiet room stays quiet
			const dir = msg.facing === -1 ? -1 : 1;
			if (!this.underLimit(this.punchTimes, ws, PUNCH_BURST)) return;
			// Everyone sees the swing, hit or miss.
			this.broadcast({ type: "swung", id: player.id, dir });

			// The server decides what connected, from its own positions.
			const now = Date.now();
			let target: Player | null = null;
			for (const other of this.players.values()) {
				if (other.id === player.id) continue;
				const dx = other.x - player.x;
				if (Math.abs(other.y - player.y) > PUNCH_RANGE_Y) continue;
				if (Math.abs(dx) > PUNCH_RANGE_X) continue; // too far
				// Overlapping players are always hittable; only a target clearly
				// behind the swing is excluded. Sprites stand inside each other
				// all the time, and a punch that whiffs at zero range feels broken.
				if (dx * dir < -6) continue;
				if ((this.immuneUntil.get(other.id) ?? 0) > now) continue;
				if (!target || Math.abs(dx) < Math.abs(target.x - player.x)) target = other;
			}
			if (target) {
				this.immuneUntil.set(target.id, now + STUN_MS + IMMUNE_MS);
				this.broadcast({ type: "hit", attacker: player.id, target: target.id, dir });
				// The crown is held only as long as you can defend it.
				if (this.crown?.wearer === target.id) {
					const tws = this.wsOf(target.id);
					this.endReign(tws ? (this.vids.get(tws) ?? null) : null, target.name);
					this.setCrown({
						wearer: null,
						x: clamp(target.x + dir * 36, 20, WORLD.w - 20),
						y: 312, // it falls to the floor; the client draws it there
					});
				}
			}
		}
	}

	// Deletes positions nobody has claimed in 30 days. Runs at most daily and
	// only when a join has already woken the room — never on its own clock.
	private async sweepPositions(): Promise<void> {
		const now = Date.now();
		const sweptAt = ((await this.ctx.storage.get("sweptAt")) as number | undefined) ?? 0;
		if (now - sweptAt < SWEEP_EVERY_MS) return;
		await this.ctx.storage.put("sweptAt", now);
		const positions = await this.ctx.storage.list({ prefix: "pos:" });
		const stale: string[] = [];
		for (const [key, value] of positions) {
			if (now - ((value as { t?: number }).t ?? 0) >= POS_TTL_MS) stale.push(key);
		}
		if (stale.length) await this.ctx.storage.delete(stale);
	}

	// Sliding-window rate limit; returns false (and records nothing) when over.
	private underLimit(times: Map<WebSocket, number[]>, ws: WebSocket, burst: number): boolean {
		const now = Date.now();
		const recent = (times.get(ws) ?? []).filter((t) => now - t < CHAT_WINDOW_MS);
		if (recent.length >= burst) return false;
		recent.push(now);
		times.set(ws, recent);
		return true;
	}

	async webSocketClose(ws: WebSocket): Promise<void> {
		this.drop(ws);
	}

	async webSocketError(ws: WebSocket): Promise<void> {
		this.drop(ws);
	}

	private drop(ws: WebSocket): void {
		const player = this.players.get(ws);
		if (!player) return;
		// The room remembers where you were standing; see you next visit.
		const vid = this.vids.get(ws);
		if (vid) {
			void this.ctx.storage.put(`pos:${vid}`, { x: player.x, y: player.y, t: Date.now() });
		}
		this.players.delete(ws);
		this.vids.delete(ws);
		this.chatTimes.delete(ws);
		this.emoteTimes.delete(ws);
		this.nameTimes.delete(ws);
		this.punchTimes.delete(ws);
		this.moveTimes.delete(ws);
		this.markTimes.delete(ws);
		this.guessTimes.delete(ws);
		this.immuneUntil.delete(player.id);
		this.runs.delete(player.id);
		this.broadcast({ type: "left", id: player.id });
		// A wearer who leaves forfeits: the crown lands on a random tier rather
		// than lying jammed against whichever door they walked out through.
		if (this.crown?.wearer === player.id) {
			this.endReign(vid ?? null, player.name); // vid captured before the maps were cleared
			const perch = CROWN_PERCHES[Math.floor(Math.random() * CROWN_PERCHES.length)];
			this.setCrown({ wearer: null, ...perch });
		}
	}

	private broadcast(msg: object, except?: WebSocket): void {
		const data = JSON.stringify(msg);
		for (const ws of this.players.keys()) {
			if (ws === except) continue;
			try {
				ws.send(data);
			} catch {
				// Socket died before its close event landed. Full drop, not a bare
				// map delete: otherwise no "left" is ever broadcast and every other
				// client renders a frozen ghost until they refresh.
				this.drop(ws);
			}
		}
	}
}

function clamp(v: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, v));
}
