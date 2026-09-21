/**
 * Loader for the optional in-browser Whisper backend (@xenova/transformers).
 *
 * Nothing is fetched from the network:
 *  - the library itself comes from node_modules (bundled by Vite),
 *  - ONNX Runtime WASM binaries are served from /ort-wasm/ (copied by scripts/copy-ort-wasm.js),
 *  - model weights are read from `localModelPath` (default: /models/) unless the user
 *    explicitly turns off "Block Remote Model Downloads".
 */

export interface TransformersLoaderOptions {
  allowRemoteModels?: boolean;
  localModelPath?: string;
}

let cachedModule: any = null;

export async function loadTransformersModule(options?: TransformersLoaderOptions) {
  if (!cachedModule) {
    try {
      cachedModule = await import('@xenova/transformers');
    } catch (e: unknown) {
      const reason = e instanceof Error ? e.message : String(e);
      throw new Error(
        `Could not load @xenova/transformers from local node_modules (${reason}). ` +
        `Run "npm install" and restart the app.`
      );
    }
  }

  const env = cachedModule.env;
  env.allowLocalModels = true;
  env.allowRemoteModels = options?.allowRemoteModels ?? false;
  env.localModelPath = options?.localModelPath || '/models/';
  // Never load the ONNX Runtime WASM from a CDN (the library's default).
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.wasmPaths = '/ort-wasm/';
  }
  return cachedModule;
}
