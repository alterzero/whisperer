const STORAGE_KEY = "whisperer_transcriptions";
const MODEL_KEY = "whisperer_selected_model";
const SOURCE_KEY = "whisperer_audio_source";
const LANG_KEY = "whisperer_language";
const CONFIG_KEY = "whisperer_config";

const CONFIG_DEFAULTS = {
  liveIntervalMs: 5000,
  maxHistory: 50,
  maxTokens: 16384,
  chunkChars: 20000,
  systemPrompt: "You are a concise meeting notes assistant. Always respond with structured bullet points.",
  sections: [
    { title: "Important Points", instruction: "List the key points discussed" },
    { title: "Decisions", instruction: "List any decisions that were made" },
    { title: "Action Items", instruction: "List any tasks, follow-ups, or action items mentioned" },
  ],
  diarizationEnabled: false,
  diarizationThreshold: 0.6,
  vadThreshold: 0.01,
};

// Live chunking: keep a short audio tail uncommitted and re-send it with the
// next chunk so Whisper gets context and words are never cut mid-chunk.
const LIVE_OVERLAP_SEC = 1.0;
const LIVE_MAX_DEFER_SEC = 3.0;

let config = { ...CONFIG_DEFAULTS };

// --- State ---

let worker = null;
let summarizerWorker = null;
let isModelReady = false;
let isSummarizerReady = false;
let isSummarizerLoading = false;
let pendingSummarize = false;
let currentChunks = [];
let isDiarizationReady = false;
let isDiarizationLoading = false;

// Recording state
let isRecording = false;
let activeStreams = [];
let captureCtx = null;
let workletNode = null;
let pcmBuffers = [];
let pcmSampleCount = 0;
let pcmOffset = 0;
let liveSentSamples = 0; // start of audio not yet sent (may include uncommitted tail)
let committedSamples = 0; // audio before this point is already in liveText
let lastSeenSamples = 0; // total samples observed at last send/skip
let lastChunkStartSample = 0;
let lastChunkEndSample = 0;
let liveInterval = null;
let dynamicIntervalMs = CONFIG_DEFAULTS.liveIntervalMs; // grows if inference is slower than the configured interval
let chunkSentAt = 0;
let liveText = "";
let liveChunks = [];
let isLiveProcessing = false;
let isFinalizing = false;
let liveProcessingDone = null; // resolve callback for awaiting processing
let nativeSampleRate = 48000;
let timerInterval = null;
let recordingStartTime = null;

// --- DOM ---

const $ = (id) => document.getElementById(id);
const modelSelect = $("model-select");
const loadModelBtn = $("load-model-btn");
const modelStatus = $("model-status");
const modelStatusText = $("model-status-text");
const progressContainer = $("progress-container");
const progressBar = $("progress-bar");
const progressText = $("progress-text");
const languageSelect = $("language-select");
const sourceSelect = $("source-select");
const recordBtn = $("record-btn");
const recordLabel = $("record-label");
const recordingIndicator = $("recording-indicator");
const timerEl = $("timer");
const transcriptionSection = $("transcription-section");
const transcriptionResult = $("transcription-result");
const copyBtn = $("copy-btn");
const downloadTxtBtn = $("download-txt-btn");
const downloadSrtBtn = $("download-srt-btn");
const downloadVttBtn = $("download-vtt-btn");
const summarizeBtn = $("summarize-btn");
const clearHistoryBtn = $("clear-history-btn");
const historyList = $("history-list");
const summarySection = $("summary-section");
const summaryStatus = $("summary-status");
const summaryProgressContainer = $("summary-progress-container");
const summaryProgressBar = $("summary-progress-bar");
const summaryProgressText = $("summary-progress-text");
const summaryResult = $("summary-result");
const diarizationToggle = $("diarization-toggle");
const diarizationStatus = $("diarization-status");
const copySummaryBtn = $("copy-summary-btn");
const downloadSummaryBtn = $("download-summary-btn");

// --- Init ---

document.addEventListener("DOMContentLoaded", async () => {
  const stored = await chrome.storage.local.get([MODEL_KEY, SOURCE_KEY, LANG_KEY, CONFIG_KEY]);
  if (stored[MODEL_KEY]) modelSelect.value = stored[MODEL_KEY];
  if (stored[SOURCE_KEY]) sourceSelect.value = stored[SOURCE_KEY];
  if (stored[LANG_KEY]) languageSelect.value = stored[LANG_KEY];
  if (stored[CONFIG_KEY]) config = { ...CONFIG_DEFAULTS, ...stored[CONFIG_KEY] };

  await renderHistory();

  loadModelBtn.addEventListener("click", loadModel);
  recordBtn.addEventListener("click", toggleRecording);
  copyBtn.addEventListener("click", () => copyToClipboard(transcriptionResult, copyBtn));
  downloadTxtBtn.addEventListener("click", () => downloadTextFile(transcriptionResult.textContent, "whisperer-transcription"));
  downloadSrtBtn.addEventListener("click", () => downloadSubtitle("srt"));
  downloadVttBtn.addEventListener("click", () => downloadSubtitle("vtt"));
  summarizeBtn.addEventListener("click", handleSummarize);
  copySummaryBtn.addEventListener("click", () => copyToClipboard(summaryResult, copySummaryBtn));
  downloadSummaryBtn.addEventListener("click", () => downloadTextFile(summaryResult.textContent, "whisperer-summary"));
  clearHistoryBtn.addEventListener("click", clearHistory);

  modelSelect.addEventListener("change", () => {
    chrome.storage.local.set({ [MODEL_KEY]: modelSelect.value });
    isModelReady = false;
    recordBtn.disabled = true;
    setModelStatus("Model not loaded");
    modelStatus.className = "model-status";
    loadModelBtn.textContent = "Load";
    loadModelBtn.disabled = false;
  });
  sourceSelect.addEventListener("change", () => chrome.storage.local.set({ [SOURCE_KEY]: sourceSelect.value }));
  languageSelect.addEventListener("change", () => chrome.storage.local.set({ [LANG_KEY]: languageSelect.value }));

  diarizationToggle.checked = config.diarizationEnabled;
  diarizationToggle.addEventListener("change", () => {
    config.diarizationEnabled = diarizationToggle.checked;
    chrome.storage.local.set({ [CONFIG_KEY]: config });
    if (config.diarizationEnabled && !isDiarizationReady && !isDiarizationLoading) {
      loadDiarization();
    }
  });

  worker = new Worker("worker.bundle.js", { type: "module" });
  worker.onmessage = handleWorkerMessage;
  worker.onerror = (err) => setModelStatus("Worker error: " + err.message, "error");

  summarizerWorker = new Worker("summarizer.bundle.js");
  summarizerWorker.onmessage = handleSummarizerMessage;
  summarizerWorker.onerror = (err) => { summaryStatus.textContent = "Worker error: " + err.message; summarizeBtn.disabled = false; };
});

// --- Whisper Worker ---

function handleWorkerMessage(e) {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      setModelStatus(msg.message, "in-progress");
      break;
    case "progress":
      progressContainer.classList.remove("hidden");
      progressBar.style.width = Math.round(msg.progress || 0) + "%";
      progressText.textContent = Math.round(msg.progress || 0) + "%";
      break;
    case "ready":
      isModelReady = true;
      progressContainer.classList.add("hidden");
      setModelStatus("Ready", "ready");
      recordBtn.disabled = false;
      loadModelBtn.textContent = "Loaded";
      loadModelBtn.disabled = true;
      break;
    case "live-result":
      handleLiveResult(msg.text, msg.chunks);
      break;
    case "live-error":
      isLiveProcessing = false;
      // Skip the failed audio so we don't retry it in a loop
      liveSentSamples = Math.max(liveSentSamples, lastChunkEndSample);
      committedSamples = Math.max(committedSamples, liveSentSamples);
      trimPcmBuffers();
      if (liveProcessingDone) { liveProcessingDone(); liveProcessingDone = null; }
      if (isFinalizing) { isFinalizing = false; finalizeLiveTranscription(); }
      break;
    case "diarization-ready":
      isDiarizationReady = true;
      isDiarizationLoading = false;
      diarizationStatus.textContent = "Ready";
      diarizationStatus.className = "diarization-status ready";
      break;
    case "diarization-error":
      isDiarizationLoading = false;
      diarizationStatus.textContent = "Error: " + msg.message;
      diarizationStatus.className = "diarization-status error";
      break;
    case "diarization-warn":
      // Non-fatal: a single chunk failed diarization, transcription continues
      diarizationStatus.textContent = "Warning: " + msg.message;
      diarizationStatus.className = "diarization-status error";
      break;
    case "speakers-remap": {
      // Worker merged speaker clusters; rewrite labels in displayed text
      let changed = false;
      for (const c of liveChunks) {
        if (c.speaker && msg.map[c.speaker]) { c.speaker = msg.map[c.speaker]; changed = true; }
      }
      for (const c of currentChunks) {
        if (c.speaker && msg.map[c.speaker]) { c.speaker = msg.map[c.speaker]; changed = true; }
      }
      if (changed && liveChunks.length > 0) {
        liveText = buildDisplayText(liveChunks);
        transcriptionResult.textContent = liveText;
      }
      break;
    }
    case "error":
      setModelStatus("Error: " + msg.message, "error");
      progressContainer.classList.add("hidden");
      resetRecordButton();
      if (!isModelReady) { loadModelBtn.textContent = "Load"; loadModelBtn.disabled = false; }
      break;
  }
}

function setModelStatus(text, state) {
  modelStatusText.textContent = text;
  modelStatus.className = "model-status" + (state ? " " + state : "");
}

function loadModel() {
  loadModelBtn.disabled = true;
  loadModelBtn.textContent = "Loading...";
  progressBar.style.width = "0%";
  progressText.textContent = "0%";
  worker.postMessage({ type: "load", model: modelSelect.value });
}

function loadDiarization() {
  isDiarizationLoading = true;
  diarizationStatus.textContent = "Loading...";
  diarizationStatus.className = "diarization-status loading";
  worker.postMessage({ type: "load-diarization", threshold: config.diarizationThreshold });
}

// --- Summarizer Worker ---

function handleSummarizerMessage(e) {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      summaryStatus.textContent = msg.message;
      break;
    case "progress":
      summaryProgressContainer.classList.remove("hidden");
      summaryProgressBar.style.width = Math.round(msg.progress || 0) + "%";
      summaryProgressText.textContent = Math.round(msg.progress || 0) + "%";
      break;
    case "ready":
      isSummarizerReady = true;
      isSummarizerLoading = false;
      summaryProgressContainer.classList.add("hidden");
      summaryStatus.textContent = "Model ready";
      if (pendingSummarize) { pendingSummarize = false; doSummarize(); }
      break;
    case "stream":
      summaryResult.textContent = msg.partial;
      summarySection.classList.remove("hidden");
      summaryStatus.textContent = "Generating...";
      summaryResult.scrollTop = summaryResult.scrollHeight;
      break;
    case "result":
      summaryResult.textContent = msg.summary;
      summarySection.classList.remove("hidden");
      summaryStatus.textContent = "";
      summarizeBtn.disabled = false;
      break;
    case "error":
      summaryStatus.textContent = "Error: " + msg.message;
      summaryProgressContainer.classList.add("hidden");
      summarizeBtn.disabled = false;
      isSummarizerLoading = false;
      break;
  }
}

function handleSummarize() {
  if (!transcriptionResult.textContent) return;
  summarizeBtn.disabled = true;
  summarySection.classList.remove("hidden");
  summaryResult.textContent = "";

  if (!isSummarizerReady) {
    if (!isSummarizerLoading) {
      isSummarizerLoading = true;
      summaryStatus.textContent = "Loading summarizer model...";
      summarizerWorker.postMessage({ type: "load", wasmPath: chrome.runtime.getURL(""), maxTokens: config.maxTokens });
    }
    pendingSummarize = true;
    return;
  }
  doSummarize();
}

function doSummarize() {
  const text = transcriptionResult.textContent;
  if (!text) { summarizeBtn.disabled = false; return; }
  summaryStatus.textContent = "Summarizing...";
  const sel = languageSelect.selectedOptions[0];
  const language = sel && sel.value !== "auto" ? sel.textContent.trim() : "";
  summarizerWorker.postMessage({
    type: "summarize", text, language,
    chunkChars: config.chunkChars,
    systemPrompt: config.systemPrompt,
    sections: config.sections,
  });
}

// --- Audio Streams ---

async function getAudioStream() {
  const source = sourceSelect.value;
  if (source === "mic") return navigator.mediaDevices.getUserMedia({ audio: true });
  if (source === "system") return getSystemAudioStream();

  const [mic, sys] = await Promise.all([
    navigator.mediaDevices.getUserMedia({ audio: true }),
    getSystemAudioStream(),
  ]);
  const ctx = new AudioContext();
  const dest = ctx.createMediaStreamDestination();
  ctx.createMediaStreamSource(mic).connect(dest);
  ctx.createMediaStreamSource(sys).connect(dest);
  const mixed = dest.stream;
  mixed._cleanup = () => { ctx.close(); mic.getTracks().forEach((t) => t.stop()); sys.getTracks().forEach((t) => t.stop()); };
  return mixed;
}

async function getSystemAudioStream() {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
  stream.getVideoTracks().forEach((t) => t.stop());
  if (stream.getAudioTracks().length === 0) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error("No audio track. Make sure to check 'Share audio' in the picker.");
  }
  return stream;
}

function stopAllStreams() {
  for (const s of activeStreams) {
    if (s._cleanup) s._cleanup();
    else s.getTracks().forEach((t) => t.stop());
  }
  activeStreams = [];
}

// --- Recording ---

function toggleRecording() {
  isRecording ? stopRecording() : startRecording();
}

async function startRecording() {
  try {
    const stream = await getAudioStream();
    activeStreams.push(stream);

    pcmBuffers = []; pcmSampleCount = 0; pcmOffset = 0; liveSentSamples = 0;
    committedSamples = 0; lastSeenSamples = 0; lastChunkStartSample = 0; lastChunkEndSample = 0;
    liveText = ""; liveChunks = [];
    isLiveProcessing = false; isFinalizing = false; liveProcessingDone = null;
    worker.postMessage({ type: "reset-speakers" });

    await setupPcmCapture(stream);
    dynamicIntervalMs = config.liveIntervalMs;
    chunkSentAt = Date.now();
    liveInterval = setInterval(processLiveChunk, 500);

    isRecording = true;
    recordingStartTime = Date.now();
    startTimer();

    transcriptionResult.textContent = "";
    transcriptionSection.classList.remove("hidden");
    setSubtitleButtons(false);
    recordBtn.classList.add("recording");
    recordLabel.textContent = "Stop Recording";
    recordingIndicator.classList.remove("hidden");
  } catch (err) {
    stopAllStreams();
    stopPcmCapture();
    setModelStatus(err.name === "NotAllowedError" ? "Permission denied. Allow access and try again." : err.message, "error");
  }
}

function stopRecording() {
  isRecording = false;
  clearInterval(liveInterval);
  liveInterval = null;

  recordBtn.classList.remove("recording");
  recordLabel.textContent = "";
  const spinner = document.createElement("span");
  spinner.className = "loading";
  recordLabel.append(spinner, " Finalizing...");
  recordBtn.disabled = true;

  stopPcmCapture();
  stopAllStreams();
  stopTimer();
  processRemainingLiveAudio();
}

// --- PCM Capture ---

async function setupPcmCapture(stream) {
  captureCtx = new AudioContext();
  nativeSampleRate = captureCtx.sampleRate;

  await captureCtx.audioWorklet.addModule(chrome.runtime.getURL("pcm-processor.js"));

  const source = captureCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(captureCtx, "pcm-processor");
  workletNode.port.onmessage = (e) => {
    pcmBuffers.push(e.data);
    pcmSampleCount += e.data.length;
  };

  source.connect(workletNode);
  workletNode.connect(captureCtx.destination);
}

function stopPcmCapture() {
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (captureCtx) { captureCtx.close(); captureCtx = null; }
}

async function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  try {
    // OfflineAudioContext resamples with proper anti-aliasing filtering,
    // unlike naive linear interpolation which aliases high frequencies.
    const len = Math.ceil((samples.length * toRate) / fromRate);
    const ctx = new OfflineAudioContext(1, len, toRate);
    const buf = ctx.createBuffer(1, samples.length, fromRate);
    buf.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    const rendered = await ctx.startRendering();
    return rendered.getChannelData(0);
  } catch {
    return resampleLinear(samples, fromRate, toRate);
  }
}

function resampleLinear(samples, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const len = Math.round(samples.length / ratio);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const idx = i * ratio;
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const f = idx - lo;
    out[i] = samples[lo] * (1 - f) + samples[hi] * f;
  }
  return out;
}

function isSilent(samples, threshold) {
  if (!threshold || samples.length === 0) return false;
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    sum += v * v;
    const a = Math.abs(v);
    if (a > peak) peak = a;
  }
  const rms = Math.sqrt(sum / samples.length);
  return rms < threshold && peak < threshold * 3;
}

function extractPcm(startSample) {
  let offset = pcmOffset;
  const parts = [];
  for (const buf of pcmBuffers) {
    const end = offset + buf.length;
    if (end > startSample) parts.push(buf.subarray(Math.max(0, startSample - offset)));
    offset = end;
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const result = new Float32Array(total);
  let pos = 0;
  for (const p of parts) { result.set(p, pos); pos += p.length; }
  return result;
}

function trimPcmBuffers() {
  let offset = pcmOffset;
  let cut = 0;
  for (let i = 0; i < pcmBuffers.length; i++) {
    const next = offset + pcmBuffers[i].length;
    if (next <= liveSentSamples) { cut = i + 1; offset = next; }
    else break;
  }
  if (cut > 0) { pcmOffset = offset; pcmBuffers.splice(0, cut); }
}

// --- Live Transcription ---

async function sendLiveChunk(rawPcm, timeOffset) {
  isLiveProcessing = true;
  const resampled = await resample(rawPcm, nativeSampleRate, 16000);
  worker.postMessage(
    { type: "transcribe-live", audio: resampled, language: languageSelect.value, timeOffset, diarize: config.diarizationEnabled && isDiarizationReady },
    [resampled.buffer]
  );
}

function processLiveChunk() {
  if (isLiveProcessing) return;
  if (Date.now() - chunkSentAt < dynamicIntervalMs) return;
  if (pcmSampleCount - lastSeenSamples < nativeSampleRate) return;

  const rawPcm = extractPcm(liveSentSamples);
  lastSeenSamples = pcmSampleCount;

  // VAD: skip silent audio entirely (avoids Whisper hallucinations + saves compute)
  if (isSilent(rawPcm, config.vadThreshold)) {
    liveSentSamples = Math.max(liveSentSamples, pcmSampleCount - Math.floor(nativeSampleRate * LIVE_OVERLAP_SEC));
    committedSamples = Math.max(committedSamples, liveSentSamples);
    trimPcmBuffers();
    return;
  }

  lastChunkStartSample = liveSentSamples;
  lastChunkEndSample = pcmSampleCount;
  chunkSentAt = Date.now();
  const timeOffset = liveSentSamples / nativeSampleRate;
  sendLiveChunk(rawPcm, timeOffset);
}

function handleLiveResult(text, chunks) {
  isLiveProcessing = false;
  if (liveProcessingDone) { liveProcessingDone(); liveProcessingDone = null; }

  // Adapt chunk cadence to actual inference speed so slow machines don't
  // pile up unprocessed audio (keeps latency and timestamps stable).
  const processingMs = Date.now() - chunkSentAt;
  dynamicIntervalMs = Math.min(30000, Math.max(config.liveIntervalMs, Math.round(processingMs * 1.2)));

  const timeOffset = lastChunkStartSample / nativeSampleRate;
  const dur = (lastChunkEndSample - lastChunkStartSample) / nativeSampleRate;
  const relEnd = (c) => (c.timestamp?.[1] ?? timeOffset + dur) - timeOffset;
  const relStart = (c) => (c.timestamp?.[0] ?? timeOffset) - timeOffset;

  let all = chunks || [];
  if (all.length === 0 && text?.trim()) {
    all = [{ text, timestamp: [timeOffset, timeOffset + dur] }];
  }

  // Drop content already committed in a previous pass (re-sent overlap audio)
  const committedRel = (committedSamples - lastChunkStartSample) / nativeSampleRate;
  all = all.filter((c) => relEnd(c) > committedRel + 0.15);

  let committed;
  if (isFinalizing) {
    committed = all;
    liveSentSamples = lastChunkEndSample;
    committedSamples = lastChunkEndSample;
  } else {
    // Defer chunks near the tail; they get re-transcribed with the next chunk
    // so words/sentences cut at the boundary are recognized with full context.
    const boundary = dur - LIVE_OVERLAP_SEC;
    const minStart = dur - LIVE_MAX_DEFER_SEC;
    committed = [];
    const deferred = [];
    for (const c of all) {
      // Commit if it ends before the tail, or if deferring would grow the
      // re-sent window beyond the cap.
      if (relEnd(c) <= boundary || relStart(c) < minStart) committed.push(c);
      else deferred.push(c);
    }
    let resendFrom = boundary;
    for (const c of deferred) resendFrom = Math.min(resendFrom, relStart(c));
    resendFrom = Math.max(0, Math.min(resendFrom, dur));

    let commitEndRel = resendFrom;
    for (const c of committed) commitEndRel = Math.max(commitEndRel, Math.min(relEnd(c), dur));

    liveSentSamples = Math.min(lastChunkEndSample, lastChunkStartSample + Math.floor(resendFrom * nativeSampleRate));
    committedSamples = Math.max(committedSamples, lastChunkStartSample + Math.floor(commitEndRel * nativeSampleRate));
  }
  trimPcmBuffers();

  if (committed.length > 0) {
    liveChunks.push(...committed);
    liveText = buildDisplayText(liveChunks);
    transcriptionResult.textContent = liveText;
    transcriptionResult.scrollTop = transcriptionResult.scrollHeight;
  }

  if (isFinalizing) { isFinalizing = false; finalizeLiveTranscription(); return; }
  if (isRecording) processLiveChunk();
}

async function processRemainingLiveAudio() {
  if (isLiveProcessing) {
    await new Promise((resolve) => { liveProcessingDone = resolve; });
  }

  const remaining = pcmSampleCount - committedSamples;
  if (remaining > nativeSampleRate * 0.5) {
    const rawPcm = extractPcm(liveSentSamples);
    if (isSilent(rawPcm, config.vadThreshold)) {
      finalizeLiveTranscription();
      return;
    }
    lastChunkStartSample = liveSentSamples;
    lastChunkEndSample = pcmSampleCount;
    const timeOffset = liveSentSamples / nativeSampleRate;
    isFinalizing = true;
    sendLiveChunk(rawPcm, timeOffset);
  } else {
    finalizeLiveTranscription();
  }
}

// Build display text from chunks, merging consecutive same-speaker chunks
// into a single labeled line.
function buildDisplayText(chunks) {
  if (!chunks.some((c) => c.speaker)) {
    return chunks.map((c) => c.text.trim()).filter(Boolean).join(" ");
  }
  const lines = [];
  let curSpeaker;
  for (const c of chunks) {
    const t = c.text.trim();
    if (!t) continue;
    const sp = c.speaker || null;
    if (lines.length > 0 && sp === curSpeaker) {
      lines[lines.length - 1] += " " + t;
    } else {
      lines.push(sp ? `[${sp}] ${t}` : t);
      curSpeaker = sp;
    }
  }
  return lines.join("\n");
}

async function finalizeLiveTranscription() {
  const text = liveText.trim();
  currentChunks = liveChunks;

  if (!text) { setModelStatus("No speech detected", "error"); resetRecordButton(); return; }

  transcriptionResult.textContent = text;
  transcriptionSection.classList.remove("hidden");
  setModelStatus("Ready", "ready");
  if (currentChunks.length > 0) setSubtitleButtons(true);

  await saveTranscription(text, currentChunks);
  await renderHistory();
  resetRecordButton();
}

// --- Subtitles ---

function formatSubTime(seconds, sep) {
  if (seconds == null || isNaN(seconds)) seconds = 0;
  const h = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const s = String(Math.floor(seconds % 60)).padStart(2, "0");
  const ms = String(Math.round((seconds % 1) * 1000)).padStart(3, "0");
  return `${h}:${m}:${s}${sep}${ms}`;
}

function downloadSubtitle(format) {
  if (currentChunks.length === 0) return;
  const sep = format === "srt" ? "," : ".";
  const lines = currentChunks.map((c, i) => {
    const start = c.timestamp?.[0] ?? 0;
    const end = c.timestamp?.[1] ?? start + 5;
    const time = `${formatSubTime(start, sep)} --> ${formatSubTime(end, sep)}`;
    const speaker = c.speaker ? `[${c.speaker}] ` : "";
    return format === "srt" ? `${i + 1}\n${time}\n${speaker}${c.text.trim()}\n` : `${time}\n${speaker}${c.text.trim()}\n`;
  }).join("\n");

  const content = format === "vtt" ? `WEBVTT\n\n${lines}` : lines;
  triggerDownload(content, format === "vtt" ? "text/vtt" : "text/plain", `whisperer-${dateStamp()}.${format}`);
}

function setSubtitleButtons(show) {
  downloadSrtBtn.classList.toggle("hidden", !show);
  downloadVttBtn.classList.toggle("hidden", !show);
}

// --- Timer ---

function startTimer() {
  timerEl.textContent = "00:00";
  timerInterval = setInterval(() => {
    const s = Math.floor((Date.now() - recordingStartTime) / 1000);
    timerEl.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}

function stopTimer() { clearInterval(timerInterval); timerInterval = null; }

// --- Storage ---

async function saveTranscription(text, chunks) {
  const result = await chrome.storage.local.get(STORAGE_KEY);
  const list = result[STORAGE_KEY] || [];
  list.unshift({ id: crypto.randomUUID(), text, chunks: chunks || [], source: sourceSelect.value, timestamp: new Date().toISOString() });
  if (list.length > config.maxHistory) list.length = config.maxHistory;
  await chrome.storage.local.set({ [STORAGE_KEY]: list });
}

async function getTranscriptions() {
  return (await chrome.storage.local.get(STORAGE_KEY))[STORAGE_KEY] || [];
}

async function deleteTranscription(id) {
  const list = (await getTranscriptions()).filter((t) => t.id !== id);
  await chrome.storage.local.set({ [STORAGE_KEY]: list });
  await renderHistory();
}

async function clearHistory() {
  await chrome.storage.local.remove(STORAGE_KEY);
  transcriptionSection.classList.add("hidden");
  setSubtitleButtons(false);
  await renderHistory();
}

// --- UI Helpers ---

function resetRecordButton() {
  recordBtn.disabled = !isModelReady;
  recordBtn.classList.remove("recording");
  recordLabel.textContent = "Start Recording";
  recordingIndicator.classList.add("hidden");
}

function copyToClipboard(el, btn) {
  const text = el.textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = "Copy"), 1500);
  });
}

function downloadTextFile(text, prefix) {
  if (!text) return;
  triggerDownload(text, "text/plain", `${prefix}-${dateStamp()}.txt`);
}

function triggerDownload(content, mimeType, filename) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function dateStamp() {
  return new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
}

// --- History ---

function createHistoryItem(t, transcriptions) {
  const srcLabels = { mic: "Mic", system: "System", both: "Mic+Sys" };
  const date = new Date(t.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const src = srcLabels[t.source] || "";
  const subs = t.chunks?.length > 0;

  const item = document.createElement("div");
  item.className = "history-item";
  item.dataset.id = t.id;

  const actions = document.createElement("div");
  actions.className = "item-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "item-btn save-btn";
  saveBtn.title = "Save TXT";
  saveBtn.textContent = "\uD83D\uDCBE";
  saveBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const entry = transcriptions.find((x) => String(x.id) === item.dataset.id);
    if (entry) downloadTextFile(entry.text, "whisperer-transcription");
  });

  const delBtn = document.createElement("button");
  delBtn.className = "item-btn delete-btn";
  delBtn.title = "Delete";
  delBtn.textContent = "\u00D7";
  delBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    deleteTranscription(item.dataset.id);
  });

  actions.append(saveBtn, delBtn);

  const ts = document.createElement("div");
  ts.className = "timestamp";
  ts.textContent = date + (src ? " \u00B7 " + src : "") + (subs ? " \u00B7 Subtitles" : "");

  const text = document.createElement("div");
  text.className = "text";
  text.textContent = t.text;

  item.append(actions, ts, text);

  item.addEventListener("click", (e) => {
    if (e.target.closest(".item-btn")) return;
    const entry = transcriptions.find((x) => String(x.id) === item.dataset.id);
    if (!entry) return;
    transcriptionResult.textContent = entry.text;
    transcriptionSection.classList.remove("hidden");
    currentChunks = entry.chunks || [];
    setSubtitleButtons(currentChunks.length > 0);
  });

  return item;
}

async function renderHistory() {
  const transcriptions = await getTranscriptions();

  historyList.textContent = "";

  if (transcriptions.length === 0) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = "No transcriptions yet";
    historyList.appendChild(p);
    return;
  }

  for (const t of transcriptions) {
    historyList.appendChild(createHistoryItem(t, transcriptions));
  }
}
