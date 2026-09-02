(() => {
  "use strict";

  const DEFAULT_TTL_MS = 5 * 60 * 1000;
  const DATABASE_NAME = "robit-omni-portal";
  const STORE_NAME = "sessions";

  function memoryStorage() {
    const records = new Map();
    return {
      get: key => Promise.resolve(records.get(key) || null),
      put: record => {
        records.set(record.key, record);
        return Promise.resolve();
      },
      delete: key => {
        records.delete(key);
        return Promise.resolve();
      },
    };
  }

  function indexedDbStorage(indexedDB) {
    let databasePromise = null;
    const database = () => {
      if (databasePromise) return databasePromise;
      databasePromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(DATABASE_NAME, 1);
        request.addEventListener("upgradeneeded", () => {
          if (!request.result.objectStoreNames.contains(STORE_NAME)) {
            request.result.createObjectStore(STORE_NAME, { keyPath: "key" });
          }
        });
        request.addEventListener("success", () => resolve(request.result), { once: true });
        request.addEventListener("error", () => reject(request.error), { once: true });
      });
      return databasePromise;
    };
    const transact = async (mode, action) => {
      const db = await database();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE_NAME, mode);
        const request = action(transaction.objectStore(STORE_NAME));
        request.addEventListener("success", () => resolve(request.result), { once: true });
        request.addEventListener("error", () => reject(request.error), { once: true });
        transaction.addEventListener("abort", () => reject(transaction.error), { once: true });
      });
    };
    return {
      get: key => transact("readonly", store => store.get(key)),
      put: record => transact("readwrite", store => store.put(record)).then(() => undefined),
      delete: key => transact("readwrite", store => store.delete(key)).then(() => undefined),
    };
  }

  function createSessionCache({ storage, now, ttlMs, indexedDB } = {}) {
    const clock = typeof now === "function" ? now : () => Date.now();
    const lifetime = Number.isFinite(ttlMs) && ttlMs > 0 ? ttlMs : DEFAULT_TTL_MS;
    const backend = storage || (
      indexedDB
        ? indexedDbStorage(indexedDB)
        : memoryStorage()
    );
    const scopeKey = scope => {
      const value = String(scope || "").trim();
      if (!value) throw new Error("browser session cache scope is missing");
      return value;
    };
    const validRecord = async scope => {
      const key = scopeKey(scope);
      const record = await backend.get(key);
      if (!record) return null;
      if (!Number.isFinite(record.expiresAt) || record.expiresAt <= clock()) {
        await backend.delete(key);
        return null;
      }
      return record;
    };
    return {
      ttlMs: lifetime,
      async load(scope) {
        const record = await validRecord(scope);
        return record ? record.snapshot : null;
      },
      async save(scope, snapshot) {
        const timestamp = clock();
        await backend.put({
          key: scopeKey(scope),
          version: 1,
          snapshot,
          touchedAt: timestamp,
          leftAt: null,
          expiresAt: timestamp + lifetime,
        });
      },
      async touch(scope) {
        const record = await validRecord(scope);
        if (!record) return false;
        const timestamp = clock();
        record.touchedAt = timestamp;
        record.leftAt = null;
        record.expiresAt = timestamp + lifetime;
        await backend.put(record);
        return true;
      },
      async markLeft(scope) {
        const record = await validRecord(scope);
        if (!record) return false;
        const timestamp = clock();
        record.leftAt = timestamp;
        record.expiresAt = timestamp + lifetime;
        await backend.put(record);
        return true;
      },
      clear(scope) {
        return backend.delete(scopeKey(scope));
      },
    };
  }

  const root = typeof window === "object" ? window : globalThis;
  root.OmniSessionCacheFactory = Object.freeze({
    DEFAULT_TTL_MS,
    createSessionCache,
    memoryStorage,
  });
  root.OmniSessionCache = createSessionCache({ indexedDB: root.indexedDB });
})();
