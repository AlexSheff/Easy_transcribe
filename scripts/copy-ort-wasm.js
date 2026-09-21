// Copies the ONNX Runtime WebAssembly binaries into public/ort-wasm/ so the optional
// in-browser Whisper backend loads them from this app instead of a CDN.
import { cpSync, mkdirSync, readdirSync, existsSync } from 'node:fs';

const SRC = 'node_modules/onnxruntime-web/dist';
const DEST = 'public/ort-wasm';

if (!existsSync(SRC)) {
  console.warn('[copy-ort-wasm] onnxruntime-web not installed; the in-browser backend will be unavailable.');
} else {
  mkdirSync(DEST, { recursive: true });
  const files = readdirSync(SRC).filter((f) => /^ort-wasm.*\.(wasm|js)$/.test(f));
  for (const f of files) cpSync(`${SRC}/${f}`, `${DEST}/${f}`);
  console.log(`[copy-ort-wasm] copied ${files.length} files to ${DEST}`);
}
