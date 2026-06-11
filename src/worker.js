import { pipeline, AutoModel, AutoProcessor, env } from "@huggingface/transformers";

env.backends.onnx.wasm.wasmPaths = new URL("./", import.meta.url).href;

let transcriber = null;

// Diarization state
let segmentationModel = null;
let segmentationProcessor = null;
let embeddingModel = null;
let embeddingProcessor = null;
let isDiarizationReady = false;
let speakerCentroids = []; // [{id, centroid: Float32Array, count}]
let similarityThreshold = 0.6;

const SAMPLE_RATE = 16000;

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

      transcriber = await pipeline(
        "automatic-speech-recognition",
        model || "onnx-community/whisper-tiny",
        {
          device: "wasm",
          dtype: model?.includes("medium") || model?.includes("large") ? "q4" : "q8",
          progress_callback: progressCallback("Downloading "),
        }
      );

      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: errMsg(err) });
    }
  }

  if (type === "load-diarization") {
    try {
      if (e.data.threshold) similarityThreshold = e.data.threshold;

      self.postMessage({ type: "status", message: "Loading diarization models..." });

      segmentationProcessor = await AutoProcessor.from_pretrained(
        "onnx-community/pyannote-segmentation-3.0",
        { progress_callback: progressCallback("Segmentation: ") }
      );
      segmentationModel = await AutoModel.from_pretrained(
        "onnx-community/pyannote-segmentation-3.0",
        { device: "wasm", dtype: "fp32", progress_callback: progressCallback("Segmentation: ") }
      );

      embeddingProcessor = await AutoProcessor.from_pretrained(
        "Xenova/wavlm-base-plus-sv",
        { progress_callback: progressCallback("Embedding: ") }
      );
      embeddingModel = await AutoModel.from_pretrained(
        "Xenova/wavlm-base-plus-sv",
        { device: "wasm", dtype: "q8", progress_callback: progressCallback("Embedding: ") }
      );

      isDiarizationReady = true;
      self.postMessage({ type: "diarization-ready" });
    } catch (err) {
      self.postMessage({ type: "diarization-error", message: errMsg(err) });
    }
  }

  if (type === "reset-speakers") {
    speakerCentroids = [];
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
      // Only needed for audio longer than Whisper's 30s window;
      // for short live chunks it just wastes compute on padding.
      if (audio.length > 30 * SAMPLE_RATE) {
        opts.chunk_length_s = 30;
        opts.stride_length_s = 5;
      }
      if (language && language !== "auto") opts.language = language;

      const result = await transcriber(audio, opts);
      const timeOffset = e.data.timeOffset || 0;
      const diarize = e.data.diarize && isDiarizationReady;

      let chunks = (result.chunks || []).map((c) => ({
        text: c.text,
        timestamp: [
          (c.timestamp[0] || 0) + timeOffset,
          (c.timestamp[1] ?? (c.timestamp[0] || 0) + 5) + timeOffset,
        ],
      }));

      if (diarize) {
        try {
          chunks = await assignSpeakers(audio, chunks, timeOffset);
        } catch (err) {
          self.postMessage({ type: "diarization-warn", message: errMsg(err) });
        }
      }

      self.postMessage({ type: "live-result", text: result.text || "", chunks });
    } catch (err) {
      self.postMessage({ type: "live-error", message: errMsg(err) });
    }
  }
};

// --- Speaker Diarization ---

const MIN_EMBED_SEC = 0.5; // minimum speech needed for a reliable embedding
const MAX_EMBED_SEC = 6; // cap embedding input length
const SEG_MAX_GAP_SEC = 0.5; // merge same-speaker segments separated by less
const SEG_MIN_DUR_SEC = 0.3; // drop segments shorter than this after merging

async function assignSpeakers(audio, whisperChunks, timeOffset) {
  // Run pyannote segmentation to get speaker activity per frame
  const segments = await getSegments(audio);
  if (segments.length === 0) return whisperChunks;

  // Group segments by local speaker and compute one embedding per local
  // speaker from their concatenated speech. Longer audio gives far more
  // reliable WavLM embeddings than tiny per-segment slices.
  const byLocal = new Map();
  for (const seg of segments) {
    if (!byLocal.has(seg.localSpeaker)) byLocal.set(seg.localSpeaker, []);
    byLocal.get(seg.localSpeaker).push(seg);
  }

  for (const segs of byLocal.values()) {
    const maxSamples = SAMPLE_RATE * MAX_EMBED_SEC;
    const parts = [];
    let total = 0;
    for (const seg of segs) {
      if (total >= maxSamples) break;
      const slice = audio.subarray(
        Math.floor(seg.start * SAMPLE_RATE),
        Math.floor(seg.end * SAMPLE_RATE)
      );
      parts.push(slice);
      total += slice.length;
    }
    if (total < SAMPLE_RATE * MIN_EMBED_SEC) continue; // too little speech

    const concat = new Float32Array(Math.min(total, maxSamples));
    let pos = 0;
    for (const p of parts) {
      const n = Math.min(p.length, concat.length - pos);
      concat.set(p.subarray(0, n), pos);
      pos += n;
      if (pos >= concat.length) break;
    }

    const label = await identifySpeaker(concat);
    for (const seg of segs) seg.speaker = label;
  }

  // Merge centroids that drifted close together, remap labels accordingly
  const remap = checkCentroidMerges();
  if (remap) {
    for (const seg of segments) {
      if (seg.speaker && remap[seg.speaker]) seg.speaker = remap[seg.speaker];
    }
  }

  return labelChunks(whisperChunks, segments, timeOffset);
}

// Assign speakers to whisper chunks; split a chunk's text proportionally
// when it significantly overlaps more than one speaker (speaker turn).
function labelChunks(whisperChunks, segments, timeOffset) {
  const out = [];

  for (const chunk of whisperChunks) {
    const cs = chunk.timestamp[0] - timeOffset;
    const ce = chunk.timestamp[1] - timeOffset;
    const chunkDur = Math.max(0.01, ce - cs);

    // Accumulate overlap per global speaker label
    const overlaps = new Map(); // label -> { dur, firstStart }
    for (const seg of segments) {
      if (!seg.speaker) continue;
      const o = Math.min(ce, seg.end) - Math.max(cs, seg.start);
      if (o <= 0) continue;
      const e = overlaps.get(seg.speaker) || { dur: 0, firstStart: Infinity };
      e.dur += o;
      e.firstStart = Math.min(e.firstStart, Math.max(seg.start, cs));
      overlaps.set(seg.speaker, e);
    }

    if (overlaps.size === 0) {
      // Fallback: nearest labeled segment by midpoint
      const chunkMid = (cs + ce) / 2;
      let best = null;
      let minDist = Infinity;
      for (const seg of segments) {
        if (!seg.speaker) continue;
        const dist = Math.abs(chunkMid - (seg.start + seg.end) / 2);
        if (dist < minDist) { minDist = dist; best = seg.speaker; }
      }
      out.push({ ...chunk, speaker: best });
      continue;
    }

    const entries = [...overlaps.entries()].sort((a, b) => a[1].firstStart - b[1].firstStart);
    const significant = entries.filter(([, v]) => v.dur >= 0.5 && v.dur >= chunkDur * 0.25);

    if (significant.length < 2) {
      // Single dominant speaker
      let best = null;
      let bestDur = 0;
      for (const [label, v] of entries) {
        if (v.dur > bestDur) { bestDur = v.dur; best = label; }
      }
      out.push({ ...chunk, speaker: best });
      continue;
    }

    // Multiple speakers within one chunk: split words proportionally to
    // each speaker's overlap duration, in order of first appearance.
    const words = chunk.text.trim().split(/\s+/).filter(Boolean);
    const totalDur = significant.reduce((s, [, v]) => s + v.dur, 0);
    let wi = 0;
    let t = chunk.timestamp[0];
    significant.forEach(([label, v], idx) => {
      const isLast = idx === significant.length - 1;
      const n = isLast
        ? words.length - wi
        : Math.max(1, Math.round((words.length * v.dur) / totalDur));
      const part = words.slice(wi, wi + n).join(" ");
      wi += n;
      const partDur = chunkDur * (v.dur / totalDur);
      const end = isLast ? chunk.timestamp[1] : t + partDur;
      if (part) out.push({ text: " " + part, timestamp: [t, end], speaker: label });
      t = end;
    });
  }

  return out;
}

async function getSegments(audio) {
  const inputs = await segmentationProcessor(audio);
  const { logits } = await segmentationModel(inputs);

  // logits shape: [1, numFrames, numClasses]
  // pyannote-segmentation-3.0 outputs 7 classes (powerset):
  // 0=none, 1=spk1, 2=spk2, 3=spk3, 4=spk1+spk2, 5=spk1+spk3, 6=spk2+spk3
  const data = logits.data;
  const numFrames = logits.dims[1];
  const numClasses = logits.dims[2];
  const frameDuration = audio.length / SAMPLE_RATE / numFrames;

  // Decode per-frame argmax into 3 boolean activity tracks
  const tracks = [new Uint8Array(numFrames), new Uint8Array(numFrames), new Uint8Array(numFrames)];
  for (let f = 0; f < numFrames; f++) {
    const offset = f * numClasses;
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < numClasses; c++) {
      if (data[offset + c] > maxVal) { maxVal = data[offset + c]; maxIdx = c; }
    }
    if (maxIdx === 1 || maxIdx === 4 || maxIdx === 5) tracks[0][f] = 1;
    if (maxIdx === 2 || maxIdx === 4 || maxIdx === 6) tracks[1][f] = 1;
    if (maxIdx === 3 || maxIdx === 5 || maxIdx === 6) tracks[2][f] = 1;
  }

  const segments = []; // [{start, end, localSpeaker}]
  const totalDuration = audio.length / SAMPLE_RATE;
  for (let s = 0; s < 3; s++) {
    const smoothed = majorityFilter(tracks[s], 5); // remove frame flicker
    let segs = trackToSegments(smoothed, frameDuration, s + 1, totalDuration);
    segs = mergeAndFilterSegments(segs, SEG_MAX_GAP_SEC, SEG_MIN_DUR_SEC);
    segments.push(...segs);
  }
  segments.sort((a, b) => a.start - b.start);
  return segments;
}

// Majority vote over a sliding window for a binary track
function majorityFilter(track, windowSize) {
  const half = Math.floor(windowSize / 2);
  const out = new Uint8Array(track.length);
  for (let i = 0; i < track.length; i++) {
    let ones = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(track.length - 1, i + half); j++) {
      ones += track[j];
      count++;
    }
    out[i] = ones * 2 > count ? 1 : 0;
  }
  return out;
}

function trackToSegments(track, frameDuration, localSpeaker, totalDuration) {
  const segs = [];
  let start = -1;
  for (let f = 0; f < track.length; f++) {
    if (track[f] && start < 0) start = f * frameDuration;
    else if (!track[f] && start >= 0) {
      segs.push({ start, end: f * frameDuration, localSpeaker });
      start = -1;
    }
  }
  if (start >= 0) segs.push({ start, end: totalDuration, localSpeaker });
  return segs;
}

function mergeAndFilterSegments(segs, maxGap, minDur) {
  const merged = [];
  for (const seg of segs) {
    const prev = merged[merged.length - 1];
    if (prev && seg.start - prev.end <= maxGap) prev.end = seg.end;
    else merged.push({ ...seg });
  }
  return merged.filter((s) => s.end - s.start >= minDur);
}

// Merge speaker centroids that became more similar than the threshold
// (fixes clusters split early on noisy embeddings). Returns a label remap
// and notifies the UI so already-displayed labels can be rewritten.
function checkCentroidMerges() {
  const remap = {};
  for (let i = 0; i < speakerCentroids.length; i++) {
    for (let j = i + 1; j < speakerCentroids.length; j++) {
      const a = speakerCentroids[i];
      const b = speakerCentroids[j];
      if (cosineSimilarity(a.centroid, b.centroid) >= similarityThreshold) {
        const total = a.count + b.count;
        for (let k = 0; k < a.centroid.length; k++) {
          a.centroid[k] = (a.centroid[k] * a.count + b.centroid[k] * b.count) / total;
        }
        a.count = total;
        remap[`Speaker ${b.id}`] = `Speaker ${a.id}`;
        speakerCentroids.splice(j, 1);
        j--;
      }
    }
  }

  if (Object.keys(remap).length === 0) return null;

  // Resolve chains (e.g. 3 -> 2 -> 1)
  for (const key of Object.keys(remap)) {
    let target = remap[key];
    while (remap[target]) target = remap[target];
    remap[key] = target;
  }
  self.postMessage({ type: "speakers-remap", map: remap });
  return remap;
}

async function identifySpeaker(audioSlice) {
  const inputs = await embeddingProcessor(audioSlice, { sampling_rate: SAMPLE_RATE });
  const output = await embeddingModel(inputs);

  // Extract embedding - WavLM outputs embeddings in last_hidden_state or embeddings
  let embedding;
  if (output.embeddings) {
    embedding = output.embeddings.data;
  } else if (output.last_hidden_state) {
    // Mean pooling over time dimension
    const hs = output.last_hidden_state;
    const [, frames, dim] = hs.dims;
    embedding = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      let sum = 0;
      for (let f = 0; f < frames; f++) sum += hs.data[f * dim + d];
      embedding[d] = sum / frames;
    }
  } else {
    return null;
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;

  // Match against known speakers
  let bestSim = -1;
  let bestIdx = -1;
  for (let i = 0; i < speakerCentroids.length; i++) {
    const sim = cosineSimilarity(embedding, speakerCentroids[i].centroid);
    if (sim > bestSim) { bestSim = sim; bestIdx = i; }
  }

  if (bestSim >= similarityThreshold && bestIdx >= 0) {
    // Update running centroid
    const spk = speakerCentroids[bestIdx];
    const w = spk.count / (spk.count + 1);
    for (let i = 0; i < spk.centroid.length; i++) {
      spk.centroid[i] = w * spk.centroid[i] + (1 - w) * embedding[i];
    }
    spk.count++;
    return `Speaker ${spk.id}`;
  }

  // New speaker
  const id = speakerCentroids.length + 1;
  speakerCentroids.push({ id, centroid: embedding, count: 1 });
  return `Speaker ${id}`;
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
