const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

// ONNX runtime files that must be in the extension root
// (Chrome CSP blocks dynamic imports from CDN)
const ONNX_FILES = [
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

const LITERT_WASM_FILES = [
  "litertlm_wasm_internal.js",
  "litertlm_wasm_internal.wasm",
  "litertlm_wasm_compat_internal.js",
  "litertlm_wasm_compat_internal.wasm",
];

function copyOnnxFiles() {
  const srcDir = path.join(__dirname, "node_modules/onnxruntime-web/dist");
  for (const file of ONNX_FILES) {
    const src = path.join(srcDir, file);
    const dest = path.join(__dirname, file);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
    }
  }
  console.log("Copied ONNX runtime files.");
}

function copyLiteRtFiles() {
  const srcDir = path.join(__dirname, "node_modules/@litert-lm/core/wasm");
  for (const file of LITERT_WASM_FILES) {
    const src = path.join(srcDir, file);
    const dest = path.join(__dirname, file);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
    }
  }
  console.log("Copied LiteRT-LM WASM files.");
}

async function build() {
  // Copy WASM files to extension root
  copyOnnxFiles();
  copyLiteRtFiles();

  // Build popup (IIFE for regular script tag)
  await esbuild.build({
    entryPoints: ["src/popup.js"],
    bundle: true,
    outfile: "popup.bundle.js",
    format: "iife",
    target: ["chrome120"],
    minify: true,
  });

  // Build workers as ESM (native import.meta support, no polyfills)
  await esbuild.build({
    entryPoints: ["src/worker.js"],
    bundle: true,
    outfile: "worker.bundle.js",
    format: "esm",
    target: ["chrome120"],
    minify: true,
  });

  await esbuild.build({
    entryPoints: ["src/summarizer.js"],
    bundle: true,
    outfile: "summarizer.bundle.js",
    format: "iife",
    target: ["chrome120"],
    minify: true,
  });

  console.log("Build complete.");
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
