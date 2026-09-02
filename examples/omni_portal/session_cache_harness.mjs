import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
require("./static/session_cache.js");

const { createSessionCache, memoryStorage } = globalThis.OmniSessionCacheFactory;
let now = 10_000;
const storage = memoryStorage();
const cache = createSessionCache({ storage, now: () => now, ttlMs: 300_000 });
const scope = "test-session-scope";

await cache.save(scope, {
  history: [{ role: "user", content: "remember this" }],
  messages: [{ role: "user", content: "remember this", media: [{ kind: "video", data: "AAAA" }] }],
});
assert.equal((await cache.load(scope)).history[0].content, "remember this");

now += 120_000;
await cache.markLeft(scope);
now += 299_999;
assert.equal((await cache.load(scope)).messages[0].media[0].kind, "video");
now += 2;
assert.equal(await cache.load(scope), null);

await cache.save(scope, { history: [{ role: "assistant", content: "new" }] });
await cache.clear(scope);
assert.equal(await cache.load(scope), null);

console.log(JSON.stringify({ status: "passed", ttl_ms: cache.ttlMs }));
