import React, { useState } from 'react';
import { LocalEngineConfig, EngineBackendType } from '../types';
import { ShieldCheck, CheckCircle2, XCircle, RefreshCw, Cpu, HardDrive } from 'lucide-react';

interface ModelSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: LocalEngineConfig;
  onSaveConfig: (cfg: LocalEngineConfig) => void;
  onAddLog: (type: 'SYSTEM' | 'PROC' | 'DONE' | 'ERROR' | 'INFO', msg: string) => void;
}

export const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onSaveConfig,
  onAddLog,
}) => {
  const [localPath, setLocalPath] = useState(config.localModelPath || '/models/');
  const [backend, setBackend] = useState<EngineBackendType>(config.backend);
  const [serverUrl, setServerUrl] = useState(config.localServerUrl || 'http://127.0.0.1:8000');
  const [blockRemote, setBlockRemote] = useState(config.blockRemoteDownloads);
  const [device, setDevice] = useState<'cuda' | 'cpu' | 'auto'>(config.device || 'auto');
  const [beamSize, setBeamSize] = useState(config.beamSize || 5);
  const [minSilenceMs, setMinSilenceMs] = useState(config.minSilenceMs || 250);
  const [maxSpeakers, setMaxSpeakers] = useState(config.maxSpeakers ?? 3);
  const [exactSpeakers, setExactSpeakers] = useState(config.exactSpeakers ?? false);

  const [testingStatus, setTestingStatus] = useState<'idle' | 'testing' | 'online' | 'offline'>('idle');
  const [testMessage, setTestMessage] = useState('');

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setTestingStatus('testing');
    setTestMessage('Pinging local server endpoint...');
    const targetUrl = (serverUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      const res = await fetch(`${targetUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        setTestingStatus('online');
        const diarizer = data.diarizer_type || 'unknown';
        const models = Array.isArray(data.whisper_models) && data.whisper_models.length ? data.whisper_models.join(', ') : 'none found';
        setTestMessage(`Server online. Whisper models: ${models}. Diarizer: ${diarizer}.`);
        onAddLog('DONE', `Local backend at ${targetUrl} online. Models: ${models}. Diarizer: ${diarizer}.`);
      } else {
        setTestingStatus('offline');
        setTestMessage(`Server returned HTTP ${res.status}`);
      }
    } catch (err: unknown) {
      setTestingStatus('offline');
      setTestMessage('Unable to connect. Start the server with: python server_faster_whisper.py');
      onAddLog('INFO', `Local backend test: server at ${targetUrl} is not currently responding.`);
    }
  };

  const handleSave = () => {
    const updated: LocalEngineConfig = {
      ...config,
      localModelPath: localPath.trim() || '/models/',
      backend,
      localServerUrl: (serverUrl || 'http://127.0.0.1:8000').trim(),
      blockRemoteDownloads: blockRemote,
      device,
      beamSize,
      minSilenceMs,
      maxSpeakers,
      exactSpeakers,
    };
    onSaveConfig(updated);
    onAddLog('SYSTEM', `Engine settings updated: Backend = ${backend}, MinSilence = ${minSilenceMs}ms, Speakers = ${exactSpeakers ? '' : '≤'}${maxSpeakers}`);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
      <div
        id="model-settings-modal"
        className="w-full max-w-2xl bg-[#111115] border border-[#262630] rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262630] bg-[#16161c]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#00ffcc]/10 text-[#00ffcc] border border-[#00ffcc]/30">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">LOCAL ENGINE & MODEL CONFIGURATION</h2>
              <p className="text-xs text-[#888899]">
                Strict offline execution with zero remote downloads
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#666677] hover:text-white transition-colors p-1 rounded-md"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm text-[#cccccc]">
          {/* Strict Offline Protection Banner */}
          <div className="p-4 rounded-lg bg-[#1a1a24] border border-[#2a2a38] flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-[#00ffcc] mt-0.5 shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white">Block Remote Model Downloads</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={blockRemote}
                    onChange={(e) => setBlockRemote(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-[#333344] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#00ffcc]"></div>
                </label>
              </div>
              <p className="text-xs text-[#888899] mt-1">
                Applies to the in-browser backend: when enabled, model weights are never downloaded and must exist locally. The Python server is always offline (HF_HUB_OFFLINE=1).
              </p>
            </div>
          </div>

          {/* Engine Backend Selection */}
          <div className="space-y-3">
            <label className="block text-xs font-mono uppercase tracking-wider text-[#9999aa]">
              Transcription Execution Backend
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div
                onClick={() => setBackend('local-faster-whisper')}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  backend === 'local-faster-whisper'
                    ? 'border-[#00ffcc] bg-[#00ffcc]/10 text-white'
                    : 'border-[#262630] bg-[#14141a] text-[#888899] hover:border-[#444455]'
                }`}
              >
                <div className="flex items-center gap-2 font-semibold text-xs mb-1 text-white">
                  <Cpu className="w-4 h-4 text-[#00ffcc]" />
                  <span>Faster-Whisper</span>
                </div>
                <p className="text-[11px] leading-tight text-[#9999aa]">
                  Recommended. Reads models from your Hugging Face cache; includes speaker diarization.
                </p>
              </div>

              <div
                onClick={() => setBackend('offline-transformers')}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  backend === 'offline-transformers'
                    ? 'border-[#00ffcc] bg-[#00ffcc]/10 text-white'
                    : 'border-[#262630] bg-[#14141a] text-[#888899] hover:border-[#444455]'
                }`}
              >
                <div className="flex items-center gap-2 font-semibold text-xs mb-1 text-white">
                  <HardDrive className="w-4 h-4 text-[#00ffcc]" />
                  <span>WASM / WebGPU</span>
                </div>
                <p className="text-[11px] leading-tight text-[#9999aa]">
                  Advanced. Runs in the browser from model files placed in public/models/. No speaker diarization.
                </p>
              </div>

            </div>
          </div>

          {/* Local faster-whisper Bridge Connection Details */}
          {backend === 'local-faster-whisper' && (
            <div className="p-4 rounded-lg bg-[#0d0d12] border border-[#2a2a38] space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white">Local Server Endpoint</span>
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={testingStatus === 'testing'}
                  className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-[#1e1e2a] hover:bg-[#2a2a3c] text-white border border-[#333348] transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${testingStatus === 'testing' ? 'animate-spin text-[#00ffcc]' : ''}`} />
                  <span>Test Connection</span>
                </button>
              </div>

              <input
                type="text"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000"
                className="w-full px-3 py-1.5 bg-[#050508] border border-[#333348] rounded text-white font-mono text-xs focus:outline-none focus:border-[#00ffcc]"
              />

              {/* Status feedback */}
              {testingStatus !== 'idle' && (
                <div className="flex items-center gap-2 text-xs">
                  {testingStatus === 'online' && <CheckCircle2 className="w-4 h-4 text-[#00ffcc]" />}
                  {testingStatus === 'offline' && <XCircle className="w-4 h-4 text-[#ff0055]" />}
                  <span className={testingStatus === 'online' ? 'text-[#00ffcc]' : 'text-[#ff6688]'}>
                    {testMessage}
                  </span>
                </div>
              )}

              {/* Instructions */}
              <div className="text-[11px] text-[#777788] bg-[#14141c] p-2.5 rounded border border-[#222230] space-y-1">
                <div className="font-semibold text-[#aaaaee]">How to start your local faster-whisper + diarization server:</div>
                <div>1. Run: <code className="text-[#00ffcc]">pip install -r requirements.txt</code></div>
                <div>2. Run <code className="text-[#00ffcc]">python server_faster_whisper.py</code> (or use <code className="text-[#00ffcc]">start.bat</code>).</div>
                <div>3. Server runs 100% offline at http://127.0.0.1:8000 using your cached models.</div>
              </div>
            </div>
          )}

          {/* VAD tuning */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-mono text-[#9999aa]">VAD Min Silence (Turns)</label>
              <span className="text-xs font-mono text-[#00ffcc]">{minSilenceMs} ms</span>
            </div>
            <input
              type="range"
              min="150"
              max="600"
              step="25"
              value={minSilenceMs}
              onChange={(e) => setMinSilenceMs(parseInt(e.target.value, 10))}
              className="w-full accent-[#00ffcc] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-[#666677] mt-0.5">
              <span>150ms (finer turns, more segments)</span>
              <span>600ms (merged turns, fewer segments)</span>
            </div>
          </div>

          {backend === 'offline-transformers' && (
            <div className="space-y-2">
              <label className="block text-xs font-mono uppercase tracking-wider text-[#9999aa]">
                Browser Model Path
              </label>
              <input
                type="text"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
                placeholder="/models/"
                className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white font-mono text-xs focus:outline-none focus:border-[#00ffcc] transition-colors"
              />
              <p className="text-[11px] text-[#777788]">
                URL path served by this app (folder <code className="text-[#00ffcc]">public/models/</code>), e.g. <code className="text-[#00ffcc]">public/models/Xenova/whisper-base/</code>.
              </p>
            </div>
          )}

          {/* Speaker count (Python backend) */}
          {backend === 'local-faster-whisper' && (
            <div className="grid grid-cols-2 gap-4 items-end">
              <div>
                <label className="block text-xs font-mono text-[#9999aa] mb-1">Max Speakers</label>
                <input
                  type="number"
                  min="1"
                  max="12"
                  value={maxSpeakers}
                  onChange={(e) => setMaxSpeakers(Math.max(1, Math.min(12, parseInt(e.target.value, 10) || 1)))}
                  className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
                />
              </div>
              <label className="flex items-center gap-2 text-xs text-[#cccccc] cursor-pointer pb-2">
                <input
                  type="checkbox"
                  checked={exactSpeakers}
                  onChange={(e) => setExactSpeakers(e.target.checked)}
                  className="accent-[#00ffcc]"
                />
                <span>Exactly this many (skip auto-detect)</span>
              </label>
              <p className="col-span-2 text-[10px] text-[#777788] -mt-2">
                Auto mode picks 1 to Max Speakers based on how clearly the voices separate. If it under-splits a conversation you know has N people, tick "Exactly".
              </p>
            </div>
          )}

          {/* Hardware & Inference Options */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-[#9999aa] mb-1">Compute Device</label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value as any)}
                className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
              >
                <option value="auto">Auto (CUDA if available, else CPU)</option>
                <option value="cuda">CUDA (NVIDIA GPU)</option>
                <option value="cpu">CPU (AVX2 / Multi-thread)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-mono text-[#9999aa] mb-1">Beam Search Size</label>
              <input
                type="number"
                min="1"
                max="10"
                value={beamSize}
                onChange={(e) => setBeamSize(parseInt(e.target.value, 10) || 5)}
                className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[#262630] bg-[#16161c] flex items-center justify-between">
          <div className="text-xs text-[#777788] flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#00ffcc]"></span>
            <span>Local mode active: No remote downloads</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-medium text-[#aaaaaa] hover:text-white bg-[#22222c] hover:bg-[#2a2a38] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-5 py-2 rounded-lg text-xs font-bold text-black bg-[#00ffcc] hover:bg-[#00e6b8] transition-all shadow-md shadow-[#00ffcc]/20"
            >
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
