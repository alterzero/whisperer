import { loadLiteRtLm, Engine } from "@litert-lm/core";

const MODEL_URL =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm";
const DB_NAME = "whisperer-models";
const STORE_NAME = "models";
const CACHE_KEY = "gemma-4-e4b-web";

let engine = null;
let liteRtLoaded = false;

// --- IndexedDB cache ---

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getCachedModel() {
  try {
    const db = await openDB();
    return await new Promise((resolve, reject) => {
      const req = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(CACHE_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;
  }
}

async function cacheModel(blob) {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const req = db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).put(blob, CACHE_KEY);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch { }
}

async function loadModelStream() {
  const cached = await getCachedModel();
  if (cached) {
    post("status", "Loading model from cache...");
    return cached.stream();
  }

  post("status", "Downloading Gemma 4 E4B model (~3 GB)...");
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Download failed (HTTP ${response.status})`);

  const total = parseInt(response.headers.get("content-length") || "0", 10);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.round((received / total) * 100);
      self.postMessage({ type: "progress", progress: pct });
      post("status", `Downloading model... ${pct}% (${(received / 1e9).toFixed(2)}/${(total / 1e9).toFixed(2)} GB)`);
    }
  }

  const blob = new Blob(chunks);
  post("status", "Saving model to cache...");
  await cacheModel(blob);
  return blob.stream();
}

// --- Message handler ---

self.onmessage = async (e) => {
  const { type, text, wasmPath, language } = e.data;

  if (type === "load") {
    try {
      post("status", "Loading LiteRT-LM runtime...");
      if (!liteRtLoaded) {
        await loadLiteRtLm(wasmPath);
        liteRtLoaded = true;
      }

      const modelStream = await loadModelStream();
      post("status", "Initializing model...");

      engine = await Engine.create({
        model: modelStream,
        mainExecutorSettings: { maxNumTokens: 131072 },
      });

      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: errMsg(err) });
    }
  }

  if (type === "summarize") {
    if (!engine) {
      self.postMessage({ type: "error", message: "Summarizer model not loaded" });
      return;
    }

    try {
      post("status", "Summarizing...");

      const langNote = language ? ` You MUST write the entire summary in ${language}.` : "";
      const conversation = await engine.createConversation({
        preface: {
          messages: [{
            role: "system",
            content: `You are a concise meeting notes assistant. Always respond with structured bullet points.${langNote}`,
          }],
        },
      });

      const MAX_INPUT_CHARS = 500000;
      const trimmed = text.length > MAX_INPUT_CHARS ? text.slice(0, MAX_INPUT_CHARS) + "\n[...transcription truncated]" : text;

      const langLine = language ? `\nIMPORTANT: Write the summary in ${language}, matching the language of the transcription.\n` : "";
      const prompt = `Analyze the following transcription and provide a structured summary with these sections:

**Important Points:**
- List the key points discussed

**Decisions:**
- List any decisions that were made

**Action Items:**
- List any tasks, follow-ups, or action items mentioned

Be concise and use bullet points. If a section has no relevant content, write "None identified."
${langLine}
Transcription:
${trimmed}`;

      let summary = "";
      const stream = conversation.sendMessageStreaming(prompt);
      for await (const chunk of stream) {
        for (const item of chunk.content) {
          if (item.type === "text") {
            summary += item.text;
            self.postMessage({ type: "stream", partial: summary });
          }
        }
      }

      self.postMessage({ type: "result", summary: summary.trim() || "No summary generated" });
    } catch (err) {
      self.postMessage({ type: "error", message: errMsg(err) });
    }
  }
};

function post(type, message) {
  self.postMessage({ type, message });
}

function errMsg(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  return err.message || String(err);
}
