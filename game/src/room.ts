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

interface Player {
	id: string;
	x: number;
	y: number;
	color: string;
}

export class Room extends DurableObject<Env> {
	// Live state is an in-memory cache. The durable copy of each player rides on
	// their socket via serializeAttachment, so a room woken from hibernation can
	// rebuild everything from the sockets the runtime kept open for it.
	private players = new Map<WebSocket, Player>();

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

		let msg: { type?: unknown; x?: unknown; y?: unknown };
		try {
			msg = JSON.parse(raw);
		} catch {
			ws.close(1002, "malformed message"); // a broken or hostile client, not us
			return;
		}
		if (msg.type !== "move" || !Number.isFinite(msg.x) || !Number.isFinite(msg.y)) return;

		// The server owns the truth: positions are clamped, never trusted raw.
		player.x = clamp(msg.x as number, 0, WORLD.w);
		player.y = clamp(msg.y as number, 0, WORLD.h);
		ws.serializeAttachment(player); // keep the durable copy current for wake-ups

		this.broadcast({ type: "moved", id: player.id, x: player.x, y: player.y }, ws);
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
