# Whisperer - Local Speech to Text

A Chrome extension that transcribes speech to text entirely in the browser using OpenAI Whisper. No API keys, no cloud services — everything runs locally on your machine.

## Features

- **Real-time transcription** — see results as you speak, not after
- **Multiple Whisper models** — from Tiny (~45 MB) to Large V3 (~800 MB), choose your speed/accuracy tradeoff
- **AI-powered summaries** — summarize transcriptions using Gemma 4 E4B (LiteRT-LM), also runs locally
- **99 languages** supported with auto-detection
- **Audio sources** — microphone, system audio (tab/screen capture), or both mixed
- **Subtitle export** — download transcriptions as SRT or VTT with timestamps
- **History** — automatically saves transcriptions to local storage
- **Model caching** — Gemma 4 model is cached in IndexedDB after first download (~3 GB)

## How It Works

| Component | Library | Runtime |
|-----------|---------|---------|
| Speech-to-text | [Transformers.js](https://github.com/huggingface/transformers.js) (Whisper ONNX) | WebAssembly |
| AI Summarizer | [@litert-lm/core](https://www.npmjs.com/package/@litert-lm/core) (Gemma 4 E4B) | WebGPU |

Both models run in dedicated Web Workers to keep the UI responsive.

## Requirements

- **Chrome 120+** (or Chromium-based browser)
- **WebGPU support** required for the AI summarizer (Gemma 4)

## Setup

```bash
npm install
npm run build
```

## Installing in Chrome

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
3. **Record** — click "Start Recording" and grant microphone/screen access
4. **View results** — transcription appears in real-time during recording
5. **Export** — copy to clipboard, save as TXT, or download SRT/VTT subtitles
6. **Summarize** — click "Summarize" to generate structured meeting notes (downloads Gemma 4 on first use)

## Project Structure

```
src/
  popup.js          UI logic, audio capture, recording state
  worker.js         Whisper ASR web worker (ESM)
  summarizer.js     Gemma 4 LiteRT-LM summarizer web worker (IIFE)
build.js            esbuild bundler + WASM file copier
popup.html          Extension UI
popup.css           Styles
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

## License

MIT
