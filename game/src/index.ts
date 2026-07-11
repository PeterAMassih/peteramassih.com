// game/src/index.ts
// Thin front door: upgrade the WebSocket and hand the visitor to their room.
import { Room, ROOM_NAMES } from "./room";

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
		// The one HTTP endpoint: how many visitors are in the world right now.
		// Public, cacheable for half a minute, feeds the homepage invitation.
		if (new URL(request.url).pathname === "/presence") {
			const rooms = await Promise.all(
				ROOM_NAMES.map(async (name) => {
					const res = await env.ROOM.get(env.ROOM.idFromName(name)).fetch("https://room/count");
					return (await res.json()) as { count: number; record: number | null };
				}),
			);
			return Response.json(
				{
					count: rooms.reduce((a, r) => a + r.count, 0),
					record: rooms.find((r) => r.record !== null)?.record ?? null,
				},
				{
					headers: {
						"Access-Control-Allow-Origin": "*",
						"Cache-Control": "public, max-age=30",
					},
				},
			);
		}
		if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
			return new Response("This endpoint speaks WebSocket only.", { status: 426 });
		}
		if (!originAllowed(request.headers.get("Origin"))) {
			return new Response("Forbidden origin.", { status: 403 });
		}
		// One Durable Object per room, picked by name. Unknown names fall back
		// to the plaza rather than minting objects for arbitrary strings.
		const wanted = new URL(request.url).searchParams.get("room") ?? "plaza";
		const room = (ROOM_NAMES as readonly string[]).includes(wanted) ? wanted : "plaza";
		return env.ROOM.get(env.ROOM.idFromName(room)).fetch(request);
	},
} satisfies ExportedHandler<Env>;
