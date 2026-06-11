import { pipeline, env } from "@huggingface/transformers";

env.backends.onnx.wasm.wasmPaths = new URL("./", import.meta.url).href;

let transcriber = null;

const SAMPLE_RATE = 16000;

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
          dtype: model?.includes("medium") || model?.includes("large") ? "q4" : undefined,
          progress_callback: (progress) => {
            const file = progress.file?.split("/").pop() || "";
            if (progress.status === "progress") {
              const pct = Math.round(progress.progress || 0);
              self.postMessage({ type: "progress", progress: progress.progress });
              self.postMessage({ type: "status", message: `Downloading ${file}... ${pct}%` });
            } else if (progress.status === "initiate") {
              self.postMessage({ type: "status", message: `Loading ${file}...` });
            }
          },
        }
      );

      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: errMsg(err) });
    }
  }

  if (type === "transcribe-live") {
    if (!transcriber) {
      self.postMessage({ type: "live-error", message: "Model not loaded" });
      return;
    }

    try {
      const opts = { return_timestamps: true, task: "transcribe" };
      if (language && language !== "auto") opts.language = language;

      const result = await transcriber(audio, opts);
      const timeOffset = e.data.timeOffset || 0;

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
};

function errMsg(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  return err.message || String(err);
}
