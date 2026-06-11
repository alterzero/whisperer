# Whisperer - Local Speech to Text

A Chrome extension that transcribes speech to text entirely in the browser using OpenAI Whisper. No API keys, no cloud services — everything runs locally on your machine.

## Features

- **Real-time transcription** — see results as you speak, not after
- **Speaker diarization** — identify who is speaking using pyannote segmentation + WavLM embeddings
- **Multiple Whisper models** — from Tiny (~45 MB) to Large V3 (~800 MB), choose your speed/accuracy tradeoff
- **AI-powered summaries** — summarize transcriptions using Gemma 4 E4B (LiteRT-LM), also runs locally
- **99 languages** supported with auto-detection
- **Audio sources** — microphone, system audio (tab/screen capture), or both mixed
- **Subtitle export** — download transcriptions as SRT or VTT with timestamps and speaker labels
- **History** — automatically saves transcriptions to local storage
- **Configurable** — settings page for tokens, chunk size, summary sections, diarization threshold
- **Model caching** — Gemma 4 model is cached in IndexedDB after first download (~3 GB)

## How It Works

| Component | Library | Model | Size |
|-----------|---------|-------|------|
| Speech-to-text | [Transformers.js](https://github.com/huggingface/transformers.js) | Whisper (ONNX) | 45 MB–800 MB |
| Speaker diarization | [Transformers.js](https://github.com/huggingface/transformers.js) | pyannote-segmentation-3.0 + WavLM | ~104 MB |
| AI Summarizer | [@litert-lm/core](https://www.npmjs.com/package/@litert-lm/core) | Gemma 4 E4B | ~3 GB |

All models run in Web Workers to keep the UI responsive. Whisper and diarization use WebAssembly; the summarizer uses WebGPU.

## Requirements

- **Chrome 120+** (or Chromium-based browser)
- **WebGPU support** required for the AI summarizer (Gemma 4)

## Setup

```bash
npm install
npm run build
```

## Installing in Chrome

### Option A: Download pre-built package

1. Download the [ready-to-install package](https://drive.google.com/file/d/1CyCxB-Rbdi97bZcmP4fj8ho6AUTuh-DX/view?usp=sharing) and extract the zip
2. Open Chrome and navigate to `chrome://extensions`
3. Enable **Developer mode** (toggle in the top-right corner)
4. Click **Load unpacked**
5. Select the extracted folder (the folder containing `manifest.json`)
6. The Whisperer icon will appear in your toolbar — click it to open

### Option B: Build from source

1. Clone the repository and build:
   ```bash
   git clone https://github.com/alterzero/whisperer.git
   cd whisperer
   npm install
   npm run build
   ```
2. Open Chrome and navigate to `chrome://extensions`
3. Enable **Developer mode** (toggle in the top-right corner)
4. Click **Load unpacked**
5. Select the `whisperer` project directory (the folder containing `manifest.json`)
6. The Whisperer icon will appear in your toolbar — click it to open

To update after pulling new changes:
```bash
npm install
npm run build
```
Then click the reload button on the extension card in `chrome://extensions`.

## Usage

1. **Load a model** — select a Whisper model size and click "Load"
2. **Choose language** — select the transcription language or leave on auto-detect
3. **Enable diarization** (optional) — check "Speaker Diarization" to identify speakers (downloads ~104 MB on first use)
4. **Record** — click "Start Recording" and grant microphone/screen access
5. **View results** — transcription appears in real-time with `[Speaker 1]`, `[Speaker 2]` labels when diarization is on
6. **Export** — copy to clipboard, save as TXT, or download SRT/VTT subtitles (includes speaker labels)
7. **Summarize** — click "Summarize" to generate structured meeting notes (downloads Gemma 4 on first use)
8. **Settings** — click the Settings button to configure tokens, chunk size, summary sections, and diarization threshold

## Project Structure

```
src/
  popup.js          UI logic, audio capture, recording state
  worker.js         Whisper ASR + speaker diarization web worker (ESM)
  summarizer.js     Gemma 4 LiteRT-LM summarizer web worker (IIFE)
  options.js        Settings page logic
build.js            esbuild bundler + WASM file copier
popup.html          Extension UI
popup.css           Styles
options.html        Settings page
options.css         Settings styles
pcm-processor.js    AudioWorklet processor for PCM capture
background.js       Opens popup in a new tab on icon click
manifest.json       Chrome MV3 manifest
```

## Build Details

The build script (`build.js`) does two things:

1. **Copies WASM files** from `node_modules` to the extension root (required by Chrome CSP — no CDN loads allowed)
2. **Bundles source files** with esbuild:
   - `popup.js` → `popup.bundle.js` (IIFE)
   - `worker.js` → `worker.bundle.js` (ESM module worker)
   - `summarizer.js` → `summarizer.bundle.js` (IIFE classic worker, required for `importScripts` compatibility)
   - `options.js` → `options.bundle.js` (IIFE)

## License

MIT
