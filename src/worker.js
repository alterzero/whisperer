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
        chunk_length_s: 30,
        stride_length_s: 5,
      };
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
        } catch {}
      }

      self.postMessage({ type: "live-result", text: result.text || "", chunks });
    } catch (err) {
      self.postMessage({ type: "live-error", message: errMsg(err) });
    }
  }
};

// --- Speaker Diarization ---

async function assignSpeakers(audio, whisperChunks, timeOffset) {
  // Run pyannote segmentation to get speaker activity per frame
  const segments = await getSegments(audio);
  if (segments.length === 0) return whisperChunks;

  // Extract embedding for each speaker segment and assign global IDs
  for (const seg of segments) {
    const slice = audio.slice(
      Math.floor(seg.start * SAMPLE_RATE),
      Math.floor(seg.end * SAMPLE_RATE)
    );
    if (slice.length < SAMPLE_RATE * 0.3) continue; // skip very short segments
    seg.speaker = await identifySpeaker(slice);
  }

  // Align whisper chunks with speaker segments
  return whisperChunks.map((chunk) => {
    const chunkStart = chunk.timestamp[0] - timeOffset;
    const chunkEnd = chunk.timestamp[1] - timeOffset;
    const chunkMid = (chunkStart + chunkEnd) / 2;

    // Find the speaker segment that best overlaps with this chunk
    let bestSpeaker = null;
    let bestOverlap = 0;

    for (const seg of segments) {
      if (!seg.speaker) continue;
      const overlapStart = Math.max(chunkStart, seg.start);
      const overlapEnd = Math.min(chunkEnd, seg.end);
      const overlap = Math.max(0, overlapEnd - overlapStart);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestSpeaker = seg.speaker;
      }
    }

    // Fallback: find nearest segment by midpoint
    if (!bestSpeaker) {
      let minDist = Infinity;
      for (const seg of segments) {
        if (!seg.speaker) continue;
        const segMid = (seg.start + seg.end) / 2;
        const dist = Math.abs(chunkMid - segMid);
        if (dist < minDist) { minDist = dist; bestSpeaker = seg.speaker; }
      }
    }

    return { ...chunk, speaker: bestSpeaker || null };
  });
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

  // For each frame, determine which local speakers are active
  const segments = []; // [{start, end, localSpeaker}]
  const active = {}; // localSpeaker -> {start}

  for (let f = 0; f < numFrames; f++) {
    const offset = f * numClasses;

    // Find class with highest logit using softmax-argmax
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < numClasses; c++) {
      if (data[offset + c] > maxVal) { maxVal = data[offset + c]; maxIdx = c; }
    }

    // Decode which speakers are active for this class
    const activeSpeakers = new Set();
    if (maxIdx === 1 || maxIdx === 4 || maxIdx === 5) activeSpeakers.add(1);
    if (maxIdx === 2 || maxIdx === 4 || maxIdx === 6) activeSpeakers.add(2);
    if (maxIdx === 3 || maxIdx === 5 || maxIdx === 6) activeSpeakers.add(3);

    const time = f * frameDuration;

    // Close segments for speakers no longer active
    for (const spk of Object.keys(active)) {
      if (!activeSpeakers.has(Number(spk))) {
        segments.push({ start: active[spk].start, end: time, localSpeaker: Number(spk) });
        delete active[spk];
      }
    }

    // Open segments for newly active speakers
    for (const spk of activeSpeakers) {
      if (!active[spk]) active[spk] = { start: time };
    }
  }

  // Close any remaining open segments
  const totalDuration = audio.length / SAMPLE_RATE;
  for (const spk of Object.keys(active)) {
    segments.push({ start: active[spk].start, end: totalDuration, localSpeaker: Number(spk) });
    delete active[spk];
  }

  return segments;
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
