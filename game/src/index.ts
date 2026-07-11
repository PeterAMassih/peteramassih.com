// game/src/index.ts
// Thin front door: upgrade the WebSocket and hand the visitor to the room.
import { Room } from "./room";

export { Room };

// Browsers always send Origin on WebSocket handshakes, so a present-but-foreign
// value means another site is hotlinking the server. An absent Origin (curl,
// scripts) passes: it is fakeable either way, and real abuse control is rate
// limiting, which arrives with chat in v1.
function originAllowed(origin: string | null): boolean {
	if (origin === null) return true;
	let hostname: string;
	try {
		hostname = new URL(origin).hostname;
	} catch {
		return false;
	}
	return (
		hostname === "peteramassih.com" ||
		hostname === "www.peteramassih.com" ||
		hostname === "localhost" ||
		hostname === "127.0.0.1" ||
		hostname.endsWith(".trycloudflare.com")
	);
}

export default {
	async fetch(request, env): Promise<Response> {
		if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
			return new Response("This endpoint speaks WebSocket only.", { status: 426 });
		}
		if (!originAllowed(request.headers.get("Origin"))) {
			return new Response("Forbidden origin.", { status: 403 });
		}
		// Every visitor lands in the same world for v0. More named rooms later.
		return env.ROOM.get(env.ROOM.idFromName("main")).fetch(request);
	},
} satisfies ExportedHandler<Env>;
