// game/src/room.ts
// One Room instance is one shared world. Strictly event-driven: no timers, no
// tick loop, so an idle room hibernates and bills nothing.
import { DurableObject } from "cloudflare:workers";

// Mirrored in the client (src/scripts/play.js). Kept as plain constants on
// both sides rather than a shared package: two deploy targets, one line each.
const WORLD = { w: 640, h: 360 };

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
	// Chat and emote timestamps per socket, for rate limiting. Deliberately not
	// in the attachment: losing them across hibernation just grants a fresh burst.
	private chatTimes = new Map<WebSocket, number[]>();
	private emoteTimes = new Map<WebSocket, number[]>();

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		for (const ws of ctx.getWebSockets()) {
			this.players.set(ws, ws.deserializeAttachment() as Player);
		}
	}

	async fetch(_request: Request): Promise<Response> {
		const [client, server] = Object.values(new WebSocketPair());

		const player: Player = {
			id: crypto.randomUUID(),
			name: `guest_${100 + Math.floor(Math.random() * 900)}`,
			x: 40 + Math.floor(Math.random() * (WORLD.w - 80)),
			y: 40 + Math.floor(Math.random() * (WORLD.h - 80)),
			color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
		};

		// acceptWebSocket (not accept) opts into hibernation: the runtime holds
		// the connection open while the object itself can be evicted between
		// messages. This is the whole free-tier economy of the design.
		this.ctx.acceptWebSocket(server);
		server.serializeAttachment(player);
		this.players.set(server, player);

		server.send(
			JSON.stringify({ type: "init", id: player.id, players: [...this.players.values()] }),
		);
		this.broadcast({ type: "joined", player }, server);

		return new Response(null, { status: 101, webSocket: client });
	}

	async webSocketMessage(ws: WebSocket, raw: string | ArrayBuffer): Promise<void> {
		const player = this.players.get(ws);
		if (!player || typeof raw !== "string") return;

		let msg: { type?: unknown; x?: unknown; y?: unknown; text?: unknown; kind?: unknown };
		try {
			msg = JSON.parse(raw);
		} catch {
			ws.close(1002, "malformed message"); // a broken or hostile client, not us
			return;
		}

		if (msg.type === "move") {
			if (!Number.isFinite(msg.x) || !Number.isFinite(msg.y)) return;
			// The server owns the truth: positions are clamped, never trusted raw.
			player.x = clamp(msg.x as number, 0, WORLD.w);
			player.y = clamp(msg.y as number, 0, WORLD.h);
			ws.serializeAttachment(player); // keep the durable copy current for wake-ups
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
		}
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
		this.players.delete(ws);
		this.chatTimes.delete(ws);
		this.emoteTimes.delete(ws);
		this.broadcast({ type: "left", id: player.id });
	}

	private broadcast(msg: object, except?: WebSocket): void {
		const data = JSON.stringify(msg);
		for (const ws of this.players.keys()) {
			if (ws === except) continue;
			try {
				ws.send(data);
			} catch {
				this.players.delete(ws); // socket died before its close event landed
			}
		}
	}
}

function clamp(v: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, v));
}
