// game/src/index.ts
// Thin front door: upgrade the WebSocket and hand the visitor to their room.
import { Room, ROOM_RULES } from "./room";

export { Room };

// Browsers always send Origin on WebSocket handshakes, so a present-but-foreign
// value means another site is hotlinking the server. An absent Origin (curl,
// scripts) passes: it is fakeable either way, and the real abuse control is
// the room's per-message rate limiting.
const ORIGINS = new Set(["peteramassih.com", "www.peteramassih.com", "localhost", "127.0.0.1"]);

export default {
	async fetch(request, env): Promise<Response> {
		const url = new URL(request.url);
		// The one HTTP endpoint: how many visitors are in the world right now.
		// Public and cacheable for half a minute; feeds the /play headcount.
		if (url.pathname === "/presence") {
			const counts = await Promise.all(Object.keys(ROOM_RULES).map((name) => env.ROOM.getByName(name).count()));
			return Response.json(
				{ count: counts.reduce((a, n) => a + n, 0) },
				{ headers: { "Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=30" } },
			);
		}
		if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
			return new Response("This endpoint speaks WebSocket only.", { status: 426 });
		}
		const origin = request.headers.get("Origin");
		if (origin !== null && !ORIGINS.has(URL.parse(origin)?.hostname ?? "")) {
			return new Response("Forbidden origin.", { status: 403 });
		}
		// One Durable Object per room, picked by name. Unknown names fall back
		// to the plaza rather than minting objects for arbitrary strings.
		const wanted = url.searchParams.get("room") ?? "plaza";
		return env.ROOM.getByName(Object.hasOwn(ROOM_RULES, wanted) ? wanted : "plaza").fetch(request);
	},
} satisfies ExportedHandler<Env>;
