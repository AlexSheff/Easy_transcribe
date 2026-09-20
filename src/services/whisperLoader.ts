/**
 * Dynamic loader for @xenova/transformers
 * Provides controlled offline/local model loading and prevents runtime failures.
 */

export interface TransformersLoaderOptions {
  allowRemoteModels?: boolean;
  localModelPath?: string;
}

let cachedModule: any = null;

export async function loadTransformersModule(options?: TransformersLoaderOptions) {
  const allowRemote = options?.allowRemoteModels ?? false;

  if (cachedModule) {
    if (cachedModule.env) {
      cachedModule.env.allowLocalModels = true;
      cachedModule.env.allowRemoteModels = allowRemote;
      if (options?.localModelPath) {
        cachedModule.env.localModelPath = options.localModelPath;
      }
    }
    return cachedModule;
  }

  // Dynamic import executed at runtime to shield Vite from static resolution failures
  const dynamicImport = new Function('specifier', 'return import(specifier)');

  let mod: any = null;
  let loadError: string | null = null;

  // 1. Try local node_modules
  try {
    mod = await dynamicImport('@xenova/transformers');
  } catch (e: any) {
    loadError = e?.message || String(e);
  }

  // 2. Fallback to CDN ESM only if remote models are allowed or local import failed
  if (!mod || !mod.pipeline) {
    try {
      mod = await dynamicImport('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    } catch (cdnErr: any) {
      console.warn('CDN fallback also failed:', cdnErr);
    }
  }

  if (!mod || !mod.pipeline) {
    throw new Error(
      `Package @xenova/transformers is not found in local modules and could not be loaded.\n` +
      `Reason: ${loadError || 'Module missing'}.\n\n` +
      `Resolution:\n` +
      `1. Run 'install.bat' in the project root directory (or run: npm install)\n` +
      `2. Restart the application using 'run.bat'`
    );
  }

  if (mod.env) {
    mod.env.allowLocalModels = true;
    mod.env.allowRemoteModels = allowRemote;
    if (options?.localModelPath) {
      mod.env.localModelPath = options.localModelPath;
    }
  }

  cachedModule = mod;
  return mod;
}

