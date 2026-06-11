import { pipeline, AutoModel, AutoProcessor, AutoModelForAudioFrameClassification, env } from "@huggingface/transformers";

env.backends.onnx.wasm.wasmPaths = new URL("./", import.meta.url).href;

let transcriber = null;

// Diarization state
let segmentationModel = null;
let segmentationProcessor = null;
let embeddingModel = null;
let embeddingProcessor = null;
let isDiarizationReady = false;
let speakerCentroids = []; // [{id, centroid: Float32Array, count}]
let nextSpeakerId = 1;
let similarityThreshold = 0.6;

const SAMPLE_RATE = 16000;

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
      if (e.data.threshold) similarityThreshold = e.data.threshold;

      self.postMessage({ type: "status", message: "Loading diarization models..." });

      // Use official pyannote processor (includes post_process_speaker_diarization)
      segmentationProcessor = await AutoProcessor.from_pretrained(
        "onnx-community/pyannote-segmentation-3.0",
        { progress_callback: progressCallback("Segmentation: ") }
      );
      segmentationModel = await AutoModelForAudioFrameClassification.from_pretrained(
        "onnx-community/pyannote-segmentation-3.0",
        { device: "wasm", dtype: "fp32", progress_callback: progressCallback("Segmentation: ") }
      );

      // WeSpeaker: pyannote's recommended speaker embedding model (256-dim)
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

  if (type === "reset-speakers") {
    speakerCentroids = [];
    nextSpeakerId = 1;
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

      const timeOffset = e.data.timeOffset || 0;
      const diarize = e.data.diarize && isDiarizationReady;

      // Kick off segmentation in parallel with ASR — it doesn't depend on
      // the transcription result, only on the audio.
      const segmentsPromise = diarize
        ? getSegments(audio).catch((err) => {
            self.postMessage({ type: "diarization-warn", message: errMsg(err) });
            return [];
          })
        : null;

      const result = await transcriber(audio, opts);

      let chunks = (result.chunks || []).map((c) => ({
        text: c.text,
        timestamp: [
          (c.timestamp[0] || 0) + timeOffset,
          (c.timestamp[1] ?? (c.timestamp[0] || 0) + 5) + timeOffset,
        ],
      }));

      if (segmentsPromise) {
        try {
          const segments = await segmentsPromise;
          if (segments.length > 0) {
            chunks = await assignSpeakers(audio, chunks, timeOffset, segments);
          }
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

async function assignSpeakers(audio, whisperChunks, timeOffset, segments) {
  if (segments.length === 0) return whisperChunks;

  // Extract speech audio from all segments, compute one embedding for this chunk
  const maxSamples = SAMPLE_RATE * MAX_EMBED_SEC;
  const parts = [];
  let total = 0;
  for (const seg of segments) {
    if (total >= maxSamples) break;
    const slice = audio.subarray(
      Math.floor(seg.start * SAMPLE_RATE),
      Math.floor(seg.end * SAMPLE_RATE)
    );
    parts.push(slice);
    total += slice.length;
  }

  if (total >= SAMPLE_RATE * MIN_EMBED_SEC) {
    const concat = new Float32Array(Math.min(total, maxSamples));
    let pos = 0;
    for (const p of parts) {
      const n = Math.min(p.length, concat.length - pos);
      concat.set(p.subarray(0, n), pos);
      pos += n;
      if (pos >= concat.length) break;
    }

    const embedding = await computeEmbedding(concat);
    if (embedding) {
      const label = matchOrCreateSpeaker(embedding);
      for (const seg of segments) seg.speaker = label;

      // Merge centroids that converged (fixes duplicate speakers from
      // noisy early embeddings). Remap updates displayed labels.
      const remap = checkCentroidMerges(0.92);
      if (remap) {
        for (const seg of segments) {
          if (seg.speaker && remap[seg.speaker]) seg.speaker = remap[seg.speaker];
        }
      }
    } else {
      self.postMessage({ type: "diarization-warn", message: "No embedding produced for chunk" });
    }
  }

  return labelChunks(whisperChunks, segments, timeOffset);
}

function matchOrCreateSpeaker(embedding) {
  let bestSim = -1;
  let bestIdx = -1;
  for (let i = 0; i < speakerCentroids.length; i++) {
    const sim = cosineSimilarity(embedding, speakerCentroids[i].centroid);
    if (sim > bestSim) { bestSim = sim; bestIdx = i; }
  }

  if (bestSim >= similarityThreshold && bestIdx >= 0) {
    return updateCentroid(speakerCentroids[bestIdx], embedding);
  }

  const id = nextSpeakerId++;
  speakerCentroids.push({ id, centroid: embedding, count: 1 });
  return `Speaker ${id}`;
}

function updateCentroid(spk, embedding) {
  const w = Math.min(0.8, spk.count / (spk.count + 1));
  for (let i = 0; i < spk.centroid.length; i++) {
    spk.centroid[i] = w * spk.centroid[i] + (1 - w) * embedding[i];
  }
  // Re-normalize to unit length after update
  let norm = 0;
  for (let i = 0; i < spk.centroid.length; i++) norm += spk.centroid[i] * spk.centroid[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < spk.centroid.length; i++) spk.centroid[i] /= norm;
  spk.count++;
  return `Speaker ${spk.id}`;
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

  // Use pyannote's official post-processing: softmax → argmax per frame,
  // merge consecutive same-speaker frames into segments with confidence.
  // Returns [{id, start, end, confidence}] where id is a powerset class.
  const result = segmentationProcessor.post_process_speaker_diarization(logits, audio.length);
  const raw = result[0] || [];

  // Filter: skip non-speech (id=0) and low-confidence / short segments
  return raw
    .filter((s) => s.id !== 0 && s.end - s.start >= SEG_MIN_DUR_SEC)
    .map((s) => ({ start: s.start, end: s.end, localSpeaker: s.id }));
}

// Merge speaker centroids that converged to near-identical embeddings.
// Returns a label remap or null if nothing changed.
function checkCentroidMerges(mergeThreshold) {
  const remap = {};
  for (let i = 0; i < speakerCentroids.length; i++) {
    for (let j = i + 1; j < speakerCentroids.length; j++) {
      const a = speakerCentroids[i];
      const b = speakerCentroids[j];
      if (cosineSimilarity(a.centroid, b.centroid) >= mergeThreshold) {
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

  for (const key of Object.keys(remap)) {
    let target = remap[key];
    while (remap[target]) target = remap[target];
    remap[key] = target;
  }
  self.postMessage({ type: "speakers-remap", map: remap });
  return remap;
}

async function computeEmbedding(audioSlice) {
  const inputs = await embeddingProcessor(audioSlice);
  const output = await embeddingModel(inputs);

  // WeSpeaker outputs speaker embeddings in last_hidden_state [1, 256]
  const raw = output.embeddings || output.last_hidden_state;
  if (!raw) return null;

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
