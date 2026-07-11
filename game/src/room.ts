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
const ROOM_RULES: Record<
	string,
	{ gravity: number; brawl: boolean; kb: number; marks: boolean }
> = {
	plaza: { gravity: 1, brawl: true, kb: 1, marks: false },
	arena: { gravity: 1, brawl: true, kb: 1.6, marks: false }, // hits launch you
	library: { gravity: 1, brawl: false, kb: 1, marks: true }, // quiet; the guestbook
	moon: { gravity: 0.35, brawl: true, kb: 1, marks: false }, // low-g comedy
};

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
	// Chat and emote timestamps per socket, for rate limiting. Deliberately not
	// in the attachment: losing them across hibernation just grants a fresh burst.
	private chatTimes = new Map<WebSocket, number[]>();
	private emoteTimes = new Map<WebSocket, number[]>();
	private nameTimes = new Map<WebSocket, number[]>();
	private punchTimes = new Map<WebSocket, number[]>();
	private moveTimes = new Map<WebSocket, number[]>();
	private markTimes = new Map<WebSocket, number[]>();
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
			ctx.storage.sql.exec(
				"CREATE TABLE IF NOT EXISTS marks (vid TEXT PRIMARY KEY, name TEXT, text TEXT, x REAL, ts INTEGER)",
			);
		});
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
		if (this.players.size >= MAX_PLAYERS) {
			return new Response("Room is full.", { status: 503 });
		}
		const url = new URL(request.url);
		if (!this.roomName) {
			this.roomName = url.searchParams.get("room") ?? "plaza";
			void this.ctx.storage.put("room", this.roomName);
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
				marksOn: this.rules.marks,
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
		} else if (msg.type === "chat") {
			if (typeof msg.text !== "string") return;
			const text = msg.text.trim().slice(0, CHAT_MAX_LEN);
			if (!text) return;
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
		this.immuneUntil.delete(player.id);
		this.broadcast({ type: "left", id: player.id });
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
