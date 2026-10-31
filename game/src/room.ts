// game/src/room.ts
// One Room instance is one shared world. Strictly event-driven: no timers, no
// tick loop, so an idle room hibernates and bills nothing.
import { DurableObject } from "cloudflare:workers";

// Mirrored in the client (src/scripts/play.js). Kept as plain constants on
// both sides rather than a shared package: two deploy targets, one line each.
const WORLD = { w: 640, h: 360 };

// Every room is one instance of this class, picked by name in the Worker,
// which whitelists names against this table. The server owns the rules
// (gravity is echoed to clients, brawl is enforced here); looks and doors
// live client-side.
export const ROOM_RULES: Record<
	string,
	{ gravity: number; brawl: boolean; kb: number; crown: boolean; hangman?: boolean; race?: { sx: number; sy: number; fx: number; fy: number } }
> = {
	plaza: { gravity: 1, brawl: true, kb: 1, crown: false },
	arena: { gravity: 1, brawl: true, kb: 1.6, crown: true }, // king of the hill
	library: { gravity: 1, brawl: false, kb: 1, crown: false, hangman: true },
	moon: {
		gravity: 0.35, brawl: true, kb: 1, crown: false,
		// The moon run: floor flag to the top rock. The server does the timing —
		// it stamps the move that leaves the start and the move that reaches the
		// goal, so a client cannot lie about its time (only about ~100ms of it).
		race: { sx: 60, sy: 312, fx: 335, fy: 142 },
	},
};

const RACE_R = 22; // how close a move must be to a flag to count
const RACE_MIN_MS = 1500; // faster than this is a teleport claim, not a run

// The two all-time boards: each visitor keeps one row, their best, and the
// better direction is the sort order (the fastest run, the longest reign).
const BOARDS = { races: "ASC", reigns: "DESC" } as const;
const BOARD_TOP = 5; // rows a client sees
const BOARD_KEPT = 200; // rows a board keeps

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

// The library's hangman. A guess is one letter, picked from buttons, never
// typed text, and the puzzle persists across hibernation: a half-solved word
// waits for the next stranger. Words from this world.
const HANG_WORDS = [
	"pixel", "crown", "plaza", "arena", "gravity", "socket", "worker", "durable",
	"object", "tensor", "gradient", "neuron", "library", "hibernate", "sprite",
	"canvas", "emote", "guestbook", "record", "reign", "punch", "bubble",
	"platform", "visitor", "moon", "kernel", "matrix", "entropy", "lambda",
	"vector", "cache", "quantize", "diffusion", "token", "epoch", "layer",
];
const HANG_LIVES = 8;

interface Hang {
	word: string;
	found: string; // letters guessed right
	wrongL: string; // letters guessed wrong, in guess order
	misses: number;
	// Rounds stored before names came from the fixed list carry a typed
	// `name` here instead of `by`; it is never read.
	last: { by?: string; won: boolean } | null;
}

// Nobody types a name: a stable hash of the visitor id picks one word from
// each list, so you keep your name across visits and rooms, and no visitor
// can put words of their own in front of another. Every pairing was read
// through once to be sure it is harmless.
const ADJECTIVES = [
	"amber", "azure", "calm", "coral", "cozy", "gentle", "indigo", "jade",
	"lilac", "mint", "plum", "quiet", "sage", "sunny", "swift", "teal",
];
const ANIMALS = [
	"crane", "finch", "fox", "gecko", "hare", "heron", "koala", "lemur",
	"lynx", "otter", "owl", "panda", "puffin", "robin", "seal", "wren",
];

// FNV-1a: tiny, stable across runtimes, and spreads similar UUIDs apart.
export function nameOf(key: string): string {
	let h = 2166136261;
	for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619);
	h >>>= 0;
	return `${ADJECTIVES[h % ADJECTIVES.length]} ${ANIMALS[(h >>> 4) % ANIMALS.length]}`;
}

// Positions expire: ignored after 30 days, swept opportunistically.
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

const EMOTES = ["wave", "heart"] as const;

// Messages allowed per socket per window, by type; past that they are
// silently dropped.
const WINDOW_MS = 5000;
const BURST = {
	// The honest client sends at most 10 moves/s; the budget leaves headroom for
	// bursts but stops a hostile script from amplifying through the broadcast.
	move: 80,
	emote: 10, // spamming hearts is part of the fun
	punch: 20, // paired with the client's 260ms cooldown
	guess: 20, // guessing is bursty in co-op
};

// The brawl. A punch hits the nearest player in front within reach, knocked
// back by their own client on the hit broadcast. Immunity after a hit means
// nobody can be chain-punched — that is what keeps it funny instead of mean.
// STUN_MS + IMMUNE_MS are mirrored in the client for the tumble and blink.
const PUNCH_RANGE_X = 30;
const PUNCH_RANGE_Y = 26;
const STUN_MS = 700;
const IMMUNE_MS = 1500;

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

// What rides on each socket: the broadcastable player, plus the anonymous
// visitor id from the handshake (never broadcast), the key the room
// remembers positions and best times under.
interface Conn {
	p: Player;
	vid: string | null;
}

export class Room extends DurableObject<Env> {
	// Live state is an in-memory cache. The durable copy of each connection
	// rides on its socket via serializeAttachment, so a room woken from
	// hibernation can rebuild everything from the sockets the runtime kept open.
	private conns = new Map<WebSocket, Conn>();
	private crown: Crown | null = null;
	private hang: Hang | null = null;
	// Active runs, playerId -> started-at. In-memory only: a run is seconds
	// long, and one lost to a mid-run hibernation is no loss at all.
	private runs = new Map<string, number>();
	// Rate-limit timestamps per socket and message type. Deliberately not in
	// the attachment: losing them across hibernation just grants a fresh burst.
	private times = new Map<WebSocket, Record<string, number[]>>();
	// Transient combat state, by player id. Deliberately not persisted: a room
	// that hibernated mid-brawl waking up with immunity cleared is harmless.
	private immuneUntil = new Map<string, number>();

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		for (const ws of ctx.getWebSockets()) this.conns.set(ws, ws.deserializeAttachment() as Conn);
		ctx.blockConcurrencyWhile(async () => {
			this.crown = ((await ctx.storage.get("crown")) as Crown | undefined) ?? null;
			this.hang = ((await ctx.storage.get("hang")) as Hang | undefined) ?? null;
			// A wearer who vanished while the room slept forfeits: back to the spawn.
			if (this.crown?.wearer && ![...this.conns.values()].some((c) => c.p.id === this.crown?.wearer)) {
				this.crown = { wearer: null, ...CROWN_SPAWN };
			}
			for (const board of Object.keys(BOARDS)) {
				ctx.storage.sql.exec(`CREATE TABLE IF NOT EXISTS ${board} (vid TEXT PRIMARY KEY, ms INTEGER, ts INTEGER)`);
			}
		});
	}

	// The Worker's presence headcount, over RPC.
	count(): number {
		return this.conns.size;
	}

	// The visitor's row changes only when the new time beats it.
	private recordBest(board: keyof typeof BOARDS, vid: string, ms: number): void {
		const sql = this.ctx.storage.sql;
		const beats = BOARDS[board] === "ASC" ? "<" : ">";
		sql.exec(
			`INSERT INTO ${board} (vid, ms, ts) VALUES (?, ?, ?)
			 ON CONFLICT (vid) DO UPDATE SET ms = excluded.ms, ts = excluded.ts WHERE excluded.ms ${beats} ${board}.ms`,
			vid, ms, Date.now(),
		);
		sql.exec(
			`DELETE FROM ${board} WHERE vid NOT IN (SELECT vid FROM ${board} ORDER BY ms ${BOARDS[board]} LIMIT ?)`,
			BOARD_KEPT,
		);
	}

	// Names are derived from the row's visitor id at read time. The old typed
	// names still sit in a `name` column on these tables and are never read,
	// and the vid itself never leaves the room.
	private top(board: keyof typeof BOARDS) {
		return this.ctx.storage.sql
			.exec<{ vid: string; ms: number }>(`SELECT vid, ms FROM ${board} ORDER BY ms ${BOARDS[board]} LIMIT ?`, BOARD_TOP)
			.toArray()
			.map((r) => ({ name: nameOf(r.vid), ms: r.ms }));
	}

	private newWord(not: string): string {
		let w = not;
		while (w === not) w = HANG_WORDS[Math.floor(Math.random() * HANG_WORDS.length)];
		return w;
	}

	// What clients see: the mask, never the word. Wrong letters go out sorted,
	// so the order of guesses cannot be used to spell something.
	private hangView() {
		const h = this.hang;
		if (!h) return null;
		return {
			masked: [...h.word].map((c) => (h.found.includes(c) ? c : "_")).join(""),
			wrongL: [...h.wrongL].sort().join(""),
			misses: h.misses,
			lives: HANG_LIVES,
			last: h.last?.by ? { name: h.last.by, won: h.last.won } : null,
		};
	}

	private guess(player: Player, letter: string): void {
		const h = this.hang;
		if (!h || h.found.includes(letter) || h.wrongL.includes(letter)) return; // already tried
		if (h.word.includes(letter)) h.found += letter;
		else {
			h.wrongL += letter;
			h.misses++;
		}
		const solved = [...h.word].every((c) => h.found.includes(c));
		const failed = h.misses >= HANG_LIVES;
		let event = "guess";
		if (solved || failed) {
			event = solved ? "solve" : "fail";
			this.hang = { word: this.newWord(h.word), found: "", wrongL: "", misses: 0, last: { by: player.name, won: solved } };
		}
		void this.ctx.storage.put("hang", this.hang);
		this.broadcast({ type: "hang", event, by: player.id, guess: letter, state: this.hangView() });
	}

	// Close out the current wearer's reign: record it if it is their longest,
	// tell the room how long they lasted. Called before the crown moves on.
	private endReign(vid: string | null): void {
		const c = this.crown;
		if (!c?.wearer || !c.since) return;
		const ms = Date.now() - c.since;
		if (ms < REIGN_MIN_MS) return;
		if (vid) this.recordBest("reigns", vid, ms);
		this.broadcast({ type: "reign", id: c.wearer, ms, top: this.top("reigns") });
	}

	private setCrown(crown: Crown): void {
		this.crown = crown;
		void this.ctx.storage.put("crown", crown);
		// held rides along so clients can count reigns on their own clock —
		// comparing raw timestamps across machines invites clock-skew nonsense.
		this.broadcast({ type: "crown", crown, held: crown.since ? Date.now() - crown.since : null });
	}

	// ctx.id.name is the name the Worker passed to getByName, so the rules are
	// known even when the wake-up event is a message rather than a join.
	private get rules() {
		return ROOM_RULES[this.ctx.id.name ?? "plaza"];
	}

	async fetch(request: Request): Promise<Response> {
		if (this.conns.size >= MAX_PLAYERS) {
			return new Response("Room is full.", { status: 503 });
		}
		if (this.rules.crown && !this.crown) {
			this.crown = { wearer: null, ...CROWN_SPAWN };
			void this.ctx.storage.put("crown", this.crown);
		}
		if (this.rules.hangman && !this.hang) {
			this.hang = { word: this.newWord(""), found: "", wrongL: "", misses: 0, last: null };
			void this.ctx.storage.put("hang", this.hang);
		}
		const url = new URL(request.url);
		const [client, server] = Object.values(new WebSocketPair());

		const vid = url.searchParams.get("vid");
		const validVid = vid && /^[0-9a-f-]{8,40}$/i.test(vid) ? vid : null;

		// The client passes its remembered color so walking through a door
		// does not reroll the sprite. Unknown colors fall back to a fresh pick.
		// A visitor without an id still gets a name, just not a lasting one.
		const wantColor = url.searchParams.get("color");
		const id = crypto.randomUUID();
		const player: Player = {
			id,
			name: nameOf(validVid ?? id),
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
		const conn = { p: player, vid: validVid };
		this.conns.set(server, conn);
		server.serializeAttachment(conn);

		server.send(
			JSON.stringify({
				type: "init",
				id: player.id,
				room: this.ctx.id.name,
				gravity: this.rules.gravity,
				brawl: this.rules.brawl,
				kb: this.rules.kb,
				crown: this.crown,
				crownHeld: this.crown?.since ? Date.now() - this.crown.since : null,
				reignTop: this.rules.crown ? this.top("reigns") : [],
				raceTop: this.rules.race ? this.top("races") : [],
				hang: this.hangView(),
				players: [...this.conns.values()].map((c) => c.p),
			}),
		);
		this.broadcast({ type: "joined", player }, server);

		return new Response(null, { status: 101, webSocket: client });
	}

	async webSocketMessage(ws: WebSocket, raw: string | ArrayBuffer): Promise<void> {
		const conn = this.conns.get(ws);
		if (!conn || typeof raw !== "string") return;
		const player = conn.p;

		let msg: { type?: unknown; x?: unknown; y?: unknown; kind?: unknown; facing?: unknown; letter?: unknown };
		try {
			// Valid JSON can still be null, which has no fields to read.
			msg = JSON.parse(raw) ?? {};
		} catch {
			ws.close(1002, "malformed message"); // a broken or hostile client, not us
			return;
		}

		if (msg.type === "move") {
			if (!Number.isFinite(msg.x) || !Number.isFinite(msg.y)) return;
			if (!this.underLimit(ws, "move")) return;
			// The server owns the truth: positions are clamped, never trusted raw.
			player.x = clamp(msg.x as number, 0, WORLD.w);
			player.y = clamp(msg.y as number, 0, WORLD.h);
			ws.serializeAttachment(conn); // keep the durable copy current for wake-ups
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
						if (conn.vid) this.recordBest("races", conn.vid, ms);
						this.broadcast({ type: "race", phase: "finish", id: player.id, ms, top: this.top("races") });
					}
				}
			}
		} else if (msg.type === "emote") {
			if (!EMOTES.includes(msg.kind as (typeof EMOTES)[number])) return;
			if (!this.underLimit(ws, "emote")) return;
			this.broadcast({ type: "emote", id: player.id, kind: msg.kind });
		} else if (msg.type === "guess") {
			if (!this.hang || typeof msg.letter !== "string" || !/^[a-z]$/.test(msg.letter)) return;
			if (!this.underLimit(ws, "guess")) return;
			this.guess(player, msg.letter);
		} else if (msg.type === "punch") {
			if (!this.rules.brawl) return; // the quiet room stays quiet
			const dir = msg.facing === -1 ? -1 : 1;
			if (!this.underLimit(ws, "punch")) return;
			// Everyone sees the swing, hit or miss.
			this.broadcast({ type: "swung", id: player.id, dir });

			// The server decides what connected, from its own positions.
			const now = Date.now();
			let target: Conn | null = null;
			for (const other of this.conns.values()) {
				const o = other.p;
				if (o.id === player.id) continue;
				const dx = o.x - player.x;
				if (Math.abs(o.y - player.y) > PUNCH_RANGE_Y) continue;
				if (Math.abs(dx) > PUNCH_RANGE_X) continue; // too far
				// Overlapping players are always hittable; only a target clearly
				// behind the swing is excluded. Sprites stand inside each other
				// all the time, and a punch that whiffs at zero range feels broken.
				if (dx * dir < -6) continue;
				if ((this.immuneUntil.get(o.id) ?? 0) > now) continue;
				if (!target || Math.abs(dx) < Math.abs(target.p.x - player.x)) target = other;
			}
			if (target) {
				const t = target.p;
				this.immuneUntil.set(t.id, now + STUN_MS + IMMUNE_MS);
				this.broadcast({ type: "hit", attacker: player.id, target: t.id, dir });
				// The crown is held only as long as you can defend it.
				if (this.crown?.wearer === t.id) {
					this.endReign(target.vid);
					this.setCrown({
						wearer: null,
						x: clamp(t.x + dir * 36, 20, WORLD.w - 20),
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
	private underLimit(ws: WebSocket, type: keyof typeof BURST): boolean {
		const now = Date.now();
		const mine = this.times.get(ws) ?? {};
		const recent = (mine[type] ?? []).filter((t) => now - t < WINDOW_MS);
		if (recent.length >= BURST[type]) return false;
		recent.push(now);
		mine[type] = recent;
		this.times.set(ws, mine);
		return true;
	}

	async webSocketClose(ws: WebSocket): Promise<void> {
		this.drop(ws);
	}

	async webSocketError(ws: WebSocket): Promise<void> {
		this.drop(ws);
	}

	private drop(ws: WebSocket): void {
		const conn = this.conns.get(ws);
		if (!conn) return;
		const player = conn.p;
		// The room remembers where you were standing; see you next visit.
		if (conn.vid) {
			void this.ctx.storage.put(`pos:${conn.vid}`, { x: player.x, y: player.y, t: Date.now() });
		}
		this.conns.delete(ws);
		this.times.delete(ws);
		this.immuneUntil.delete(player.id);
		this.runs.delete(player.id);
		this.broadcast({ type: "left", id: player.id });
		// A wearer who leaves forfeits: the crown lands on a random tier rather
		// than lying jammed against whichever door they walked out through.
		if (this.crown?.wearer === player.id) {
			this.endReign(conn.vid);
			const perch = CROWN_PERCHES[Math.floor(Math.random() * CROWN_PERCHES.length)];
			this.setCrown({ wearer: null, ...perch });
		}
	}

	private broadcast(msg: object, except?: WebSocket): void {
		const data = JSON.stringify(msg);
		for (const ws of this.conns.keys()) {
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
