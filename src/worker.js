import { pipeline, AutoModel, AutoProcessor, env } from "@huggingface/transformers";

env.backends.onnx.wasm.wasmPaths = new URL("./", import.meta.url).href;

let transcriber = null;

// Diarization state
let embeddingModel = null;
let embeddingProcessor = null;
let isDiarizationReady = false;

const SAMPLE_RATE = 16000;
// Multi-scale windows (in seconds) centered on each chunk for robust embeddings.
// Averaging across scales captures both precision (small) and stability (large).
const EMBED_SCALES = [1.5, 3.0];

async function detectWebGPU() {
  try {
    if (!navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

function progressCallback(prefix) {
  return (progress) => {
    const file = progress.file?.split("/").pop() || "";
    if (progress.status === "progress") {
      const pct = Math.round(progress.progress || 0);
      self.postMessage({ type: "progress", progress: progress.progress });
      self.postMessage({ type: "status", message: `${prefix}${file}... ${pct}%` });
    } else if (progress.status === "initiate") {
      self.postMessage({ type: "status", message: `${prefix}${file}...` });
    }
  };
}

self.onmessage = async (e) => {
  const { type, audio, model, language } = e.data;

  if (type === "load") {
    try {
      self.postMessage({ type: "status", message: "Downloading model files..." });

      const modelId = model || "onnx-community/whisper-tiny";
      const wasmOpts = {
        device: "wasm",
        dtype: model?.includes("medium") || model?.includes("large") ? "q4" : "q8",
        progress_callback: progressCallback("Downloading "),
      };

      if (await detectWebGPU()) {
        try {
          transcriber = await pipeline("automatic-speech-recognition", modelId, {
            device: "webgpu",
            dtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
            progress_callback: progressCallback("Downloading "),
          });
        } catch {
          self.postMessage({ type: "status", message: "WebGPU unavailable, using CPU..." });
          transcriber = await pipeline("automatic-speech-recognition", modelId, wasmOpts);
        }
      } else {
        transcriber = await pipeline("automatic-speech-recognition", modelId, wasmOpts);
      }

      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: errMsg(err) });
    }
  }

  if (type === "load-diarization") {
    try {
      self.postMessage({ type: "status", message: "Loading speaker embedding model..." });

      embeddingProcessor = await AutoProcessor.from_pretrained(
        "onnx-community/wespeaker-voxceleb-resnet34-LM",
        { progress_callback: progressCallback("Embedding: ") }
      );
      embeddingModel = await AutoModel.from_pretrained(
        "onnx-community/wespeaker-voxceleb-resnet34-LM",
        { device: "wasm", dtype: "fp32", progress_callback: progressCallback("Embedding: ") }
      );

      isDiarizationReady = true;
      self.postMessage({ type: "diarization-ready" });
    } catch (err) {
      self.postMessage({ type: "diarization-error", message: errMsg(err) });
    }
  }

  if (type === "transcribe-live") {
    if (!transcriber) {
      self.postMessage({ type: "live-error", message: "Model not loaded" });
      return;
    }

    try {
      const opts = {
        return_timestamps: true,
        task: "transcribe",
      };
      if (audio.length > 30 * SAMPLE_RATE) {
        opts.chunk_length_s = 30;
        opts.stride_length_s = 5;
      }
      if (language && language !== "auto") opts.language = language;

      const timeOffset = e.data.timeOffset || 0;
      const result = await transcriber(audio, opts);

      const chunks = (result.chunks || []).map((c) => ({
        text: c.text,
        timestamp: [
          (c.timestamp[0] || 0) + timeOffset,
          (c.timestamp[1] ?? (c.timestamp[0] || 0) + 5) + timeOffset,
        ],
      }));

      self.postMessage({ type: "live-result", text: result.text || "", chunks });
    } catch (err) {
      self.postMessage({ type: "live-error", message: errMsg(err) });
    }
  }

  // --- Batch Diarization (post-recording) ---
  // Receives the full 16kHz audio + all Whisper chunks with timestamps.
  // Extracts an embedding for each chunk's audio segment, clusters them,
  // and assigns speaker labels.
  if (type === "diarize-batch") {
    try {
      const { audio, chunks } = e.data;
      const threshold = e.data.threshold || 0.35;

      const embeddings = [];
      const embeddingIndices = []; // which chunk indices have embeddings

      for (let i = 0; i < chunks.length; i++) {
        self.postMessage({ type: "diarization-progress", current: i + 1, total: chunks.length });

        const [start, end] = chunks[i].timestamp;
        const center = (start + end) / 2;

        // Multi-scale: compute embedding at each window size, average them
        const scaleEmbs = [];
        for (const scale of EMBED_SCALES) {
          const half = scale / 2;
          const s0 = Math.max(0, Math.floor((center - half) * SAMPLE_RATE));
          const s1 = Math.min(Math.ceil((center + half) * SAMPLE_RATE), audio.length);
          if (s1 - s0 < SAMPLE_RATE * 0.3) continue;
          const segment = audio.slice(s0, s1);
          const emb = await computeEmbedding(segment);
          if (emb) scaleEmbs.push(emb);
        }

        if (scaleEmbs.length === 0) continue;

        // Average across scales and L2 normalize
        const dim = scaleEmbs[0].length;
        const avg = new Float32Array(dim);
        for (const emb of scaleEmbs) for (let j = 0; j < dim; j++) avg[j] += emb[j];
        let norm = 0;
        for (let j = 0; j < dim; j++) { avg[j] /= scaleEmbs.length; norm += avg[j] * avg[j]; }
        norm = Math.sqrt(norm);
        if (norm > 0) for (let j = 0; j < dim; j++) avg[j] /= norm;

        embeddings.push(avg);
        embeddingIndices.push(i);
      }

      if (embeddings.length === 0) {
        self.postMessage({ type: "diarization-result", chunks });
        return;
      }

      const labels = clusterEmbeddings(embeddings, threshold);

      // Assign speakers to chunks that had embeddings
      const result = chunks.map((c) => ({ ...c }));
      for (let i = 0; i < embeddingIndices.length; i++) {
        result[embeddingIndices[i]].speaker = `Speaker ${labels[i] + 1}`;
      }

      // Short chunks without embeddings: inherit from nearest labeled chunk
      for (let i = 0; i < result.length; i++) {
        if (result[i].speaker) continue;
        let nearest = -1;
        let minDist = Infinity;
        for (let j = 0; j < result.length; j++) {
          if (!result[j].speaker) continue;
          const d = Math.abs(i - j);
          if (d < minDist) { minDist = d; nearest = j; }
        }
        if (nearest >= 0) result[i].speaker = result[nearest].speaker;
      }

      self.postMessage({ type: "diarization-result", chunks: result });
    } catch (err) {
      self.postMessage({ type: "diarization-error", message: errMsg(err) });
    }
  }
};

// --- Agglomerative Clustering ---
// Following whisper-diarization approach: store all embeddings, cluster them
// using agglomerative hierarchical clustering with cosine distance.

function clusterEmbeddings(embeddings, threshold) {
  const n = embeddings.length;
  if (n === 0) return [];
  if (n === 1) return [0];

  // Compute full cosine similarity matrix
  const sim = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    sim[i * n + i] = 1.0;
    for (let j = i + 1; j < n; j++) {
      const s = cosineSimilarity(embeddings[i], embeddings[j]);
      sim[i * n + j] = s;
      sim[j * n + i] = s;
    }
  }

  // Each embedding starts as its own cluster
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) labels[i] = i;

  // Average-linkage agglomerative clustering
  for (;;) {
    // Find the two most similar clusters
    let bestSim = -Infinity;
    let bestA = -1, bestB = -1;

    const clusters = new Map(); // clusterId -> [indices]
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      if (!clusters.has(c)) clusters.set(c, []);
      clusters.get(c).push(i);
    }

    const clusterIds = [...clusters.keys()];
    if (clusterIds.length <= 1) break;

    for (let ci = 0; ci < clusterIds.length; ci++) {
      for (let cj = ci + 1; cj < clusterIds.length; cj++) {
        const a = clusters.get(clusterIds[ci]);
        const b = clusters.get(clusterIds[cj]);

        // Average linkage: mean similarity between all pairs
        let total = 0;
        for (const ai of a) {
          for (const bi of b) {
            total += sim[ai * n + bi];
          }
        }
        const avg = total / (a.length * b.length);

        if (avg > bestSim) {
          bestSim = avg;
          bestA = clusterIds[ci];
          bestB = clusterIds[cj];
        }
      }
    }

    // Stop if most similar pair is below threshold
    if (bestSim < threshold) break;

    // Merge: relabel bestB → bestA
    for (let i = 0; i < n; i++) {
      if (labels[i] === bestB) labels[i] = bestA;
    }
  }

  // Renumber clusters to 0, 1, 2, ... in order of first appearance
  const seen = new Map();
  let nextId = 0;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const c = labels[i];
    if (!seen.has(c)) seen.set(c, nextId++);
    result[i] = seen.get(c);
  }
  return result;
}

// --- Embedding ---

async function computeEmbedding(audio) {
  const inputs = await embeddingProcessor(audio);
  const output = await embeddingModel(inputs);

  const raw = output.embeddings || output.last_hidden_state;
  if (!raw) {
    self.postMessage({ type: "diarization-warn", message: "No embedding output. Keys: " + Object.keys(output).join(",") });
    return null;
  }

  const embedding = Float32Array.from(raw.data);

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
  else return null;

  return embedding;
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function errMsg(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  return err.message || String(err);
}
