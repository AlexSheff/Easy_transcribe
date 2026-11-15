import React, { useState, useEffect } from 'react';
import { LocalEngineConfig, EngineBackendType, ComputeTypeOption, MarkdownPreset } from '../types';
import { ShieldCheck, CheckCircle2, XCircle, RefreshCw, Cpu, HardDrive, SkipForward, Zap } from 'lucide-react';

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
  const [computeType, setComputeType] = useState<ComputeTypeOption>(config.computeType || 'auto');
  const [beamSize, setBeamSize] = useState(config.beamSize || 5);
  const [minSilenceMs, setMinSilenceMs] = useState(config.minSilenceMs || 250);
  const [vadThreshold, setVadThreshold] = useState(config.vadThreshold ?? 0.35);
  const [maxSpeakers, setMaxSpeakers] = useState(config.maxSpeakers ?? 3);
  const [exactSpeakers, setExactSpeakers] = useState(config.exactSpeakers ?? false);
  const [minSilhouette, setMinSilhouette] = useState(config.minSilhouette ?? 0.08);
  const [skipExisting, setSkipExisting] = useState(config.skipExisting ?? true);
  const [useBatched, setUseBatched] = useState(config.useBatched ?? true);
  const [normalizeAudio, setNormalizeAudio] = useState(config.normalizeAudio ?? true);
  const [wordTimestamps, setWordTimestamps] = useState(config.wordTimestamps ?? true);
  const [initialPrompt, setInitialPrompt] = useState(config.initialPrompt || '');
  const [hotwords, setHotwords] = useState(config.hotwords || '');
  const [autoSavePreset, setAutoSavePreset] = useState<MarkdownPreset>(config.autoSavePreset || 'standard');

  const [testingStatus, setTestingStatus] = useState<'idle' | 'testing' | 'online' | 'offline'>('idle');
  const [testMessage, setTestMessage] = useState('');

  useEffect(() => {
    if (isOpen) {
      setLocalPath(config.localModelPath || '/models/');
      setBackend(config.backend);
      setServerUrl(config.localServerUrl || 'http://127.0.0.1:8000');
      setBlockRemote(config.blockRemoteDownloads);
      setDevice(config.device || 'auto');
      setComputeType(config.computeType || 'auto');
      setBeamSize(config.beamSize || 5);
      setMinSilenceMs(config.minSilenceMs || 250);
      setVadThreshold(config.vadThreshold ?? 0.35);
      setMaxSpeakers(config.maxSpeakers ?? 3);
      setExactSpeakers(config.exactSpeakers ?? false);
      setMinSilhouette(config.minSilhouette ?? 0.08);
      setSkipExisting(config.skipExisting ?? true);
      setUseBatched(config.useBatched ?? true);
      setNormalizeAudio(config.normalizeAudio ?? true);
      setWordTimestamps(config.wordTimestamps ?? true);
      setInitialPrompt(config.initialPrompt || '');
      setHotwords(config.hotwords || '');
      setAutoSavePreset(config.autoSavePreset || 'standard');
      setTestingStatus('idle');
      setTestMessage('');
    }
  }, [isOpen, config]);

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
    } catch {
      setTestingStatus('offline');
      setTestMessage('Unable to connect. Start the server with: python server_faster_whisper.py');
      onAddLog('INFO', `Local backend test: server at ${targetUrl} is not currently responding.`);
    }
  };

  const handleSave = () => {
    const safeBeam = Math.max(1, Math.min(10, Number.isFinite(beamSize) ? beamSize : 5));
    const safeSilence = Math.max(100, Math.min(1000, Number.isFinite(minSilenceMs) ? minSilenceMs : 250));
    const safeVad = Math.max(0.1, Math.min(0.9, Number.isFinite(vadThreshold) ? vadThreshold : 0.35));
    const safeSilhouette = Math.max(0.02, Math.min(0.35, Number.isFinite(minSilhouette) ? minSilhouette : 0.08));
    const safeMaxSpeakers = Math.max(1, Math.min(12, Number.isFinite(maxSpeakers) ? maxSpeakers : 3));

    const updated: LocalEngineConfig = {
      ...config,
      localModelPath: localPath.trim() || '/models/',
      backend,
      localServerUrl: (serverUrl || 'http://127.0.0.1:8000').trim(),
      blockRemoteDownloads: blockRemote,
      device,
      computeType,
      beamSize: safeBeam,
      minSilenceMs: safeSilence,
      vadThreshold: safeVad,
      maxSpeakers: safeMaxSpeakers,
      exactSpeakers,
      minSilhouette: safeSilhouette,
      skipExisting,
      useBatched,
      normalizeAudio,
      wordTimestamps,
      initialPrompt: initialPrompt.trim(),
      hotwords: hotwords.trim(),
      autoSavePreset,
    };
    onSaveConfig(updated);
    onAddLog(
      'SYSTEM',
      `Settings updated: Backend=${backend}, Compute=${computeType}, Batched=${useBatched ? 'ON' : 'OFF'}, VAD=${safeVad}/${safeSilence}ms, Speakers=${exactSpeakers ? '' : '≤'}${safeMaxSpeakers} (sil≥${safeSilhouette})`
    );
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
                Strict offline execution, speed tuning, VAD & diarization sensitivity
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#666677] hover:text-white transition-colors p-1 rounded-md cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5 text-sm text-[#cccccc]">
          {/* Strict Offline Protection Banner */}
          <div className="p-3.5 rounded-lg bg-[#1a1a24] border border-[#2a2a38] flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-[#00ffcc] mt-0.5 shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white text-xs">Block Remote Model Downloads</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={blockRemote}
                    onChange={(e) => setBlockRemote(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-[#333344] rounded-full peer peer-checked:after:translate-x-5 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#00ffcc]"></div>
                </label>
              </div>
              <p className="text-[11px] text-[#888899] mt-1">
                Applies to browser WASM/WebGPU mode. Python server is always offline (<code className="text-[#00ffcc]">HF_HUB_OFFLINE=1</code>).
              </p>
            </div>
          </div>

          {/* Skip Already Transcribed Files Toggle */}
          <div className="p-3.5 rounded-lg bg-[#1a1a24] border border-[#2a2a38] flex items-start gap-3">
            <SkipForward className="w-5 h-5 text-[#00ffcc] mt-0.5 shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white text-xs">Skip Already Transcribed Files (outputs)</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={skipExisting}
                    onChange={(e) => setSkipExisting(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-[#333344] rounded-full peer peer-checked:after:translate-x-5 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#00ffcc]"></div>
                </label>
              </div>
              <p className="text-[11px] text-[#888899] mt-1">
                Checks <code className="text-[#00ffcc]">outputs/</code> and <code className="text-[#00ffcc]">transcripts/</code> before processing.
              </p>
            </div>
          </div>

          {/* Engine Backend Selection */}
          <div className="space-y-2">
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
                  <span>Faster-Whisper (Recommended)</span>
                </div>
                <p className="text-[11px] leading-tight text-[#9999aa]">
                  Direct media streaming (PyAV/ffmpeg) + ECAPA diarization + real-time progress.
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
                  <span>WASM / WebGPU (Browser)</span>
                </div>
                <p className="text-[11px] leading-tight text-[#9999aa]">
                  Runs inside browser from <code className="text-[#00ffcc]">public/models/</code>. Single speaker only.
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
                  className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-[#1e1e2a] hover:bg-[#2a2a3c] text-white border border-[#333348] transition-colors cursor-pointer"
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

              {testingStatus !== 'idle' && (
                <div className="flex items-center gap-2 text-xs">
                  {testingStatus === 'online' && <CheckCircle2 className="w-4 h-4 text-[#00ffcc]" />}
                  {testingStatus === 'offline' && <XCircle className="w-4 h-4 text-[#ff0055]" />}
                  <span className={testingStatus === 'online' ? 'text-[#00ffcc]' : 'text-[#ff6688]'}>
                    {testMessage}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Speed & Audio Quality Toggles (Python Backend) */}
          {backend === 'local-faster-whisper' && (
            <div className="p-4 rounded-lg bg-[#14141c] border border-[#262630] space-y-3">
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#00ffcc]">
                <Zap className="w-3.5 h-3.5" />
                <span>Speed & Recognition Enhancements</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <label className="flex items-start gap-2.5 cursor-pointer bg-[#0e0e14] p-2.5 rounded-lg border border-[#222230]">
                  <input
                    type="checkbox"
                    checked={useBatched}
                    onChange={(e) => setUseBatched(e.target.checked)}
                    className="accent-[#00ffcc] mt-0.5"
                  />
                  <div>
                    <div className="font-semibold text-white">Batched Whisper Pipeline</div>
                    <div className="text-[10px] text-[#777788]">Faster on GPU via BatchedInferencePipeline (auto-fallback if unsupported)</div>
                  </div>
                </label>

                <label className="flex items-start gap-2.5 cursor-pointer bg-[#0e0e14] p-2.5 rounded-lg border border-[#222230]">
                  <input
                    type="checkbox"
                    checked={normalizeAudio}
                    onChange={(e) => setNormalizeAudio(e.target.checked)}
                    className="accent-[#00ffcc] mt-0.5"
                  />
                  <div>
                    <div className="font-semibold text-white">Loudness Normalization</div>
                    <div className="text-[10px] text-[#777788]">Boosts quiet mic recordings to ~-18 dBFS before VAD</div>
                  </div>
                </label>

                <label className="flex items-start gap-2.5 cursor-pointer bg-[#0e0e14] p-2.5 rounded-lg border border-[#222230]">
                  <input
                    type="checkbox"
                    checked={wordTimestamps}
                    onChange={(e) => setWordTimestamps(e.target.checked)}
                    className="accent-[#00ffcc] mt-0.5"
                  />
                  <div>
                    <div className="font-semibold text-white">Word Timestamps & Anti-Loop</div>
                    <div className="text-[10px] text-[#777788]">Enables clickable words + silence hallucination guard (slower)</div>
                  </div>
                </label>

                <div className="bg-[#0e0e14] p-2.5 rounded-lg border border-[#222230]">
                  <label className="block text-[11px] font-semibold text-white mb-1">Auto-Save Markdown Preset</label>
                  <select
                    value={autoSavePreset}
                    onChange={(e) => setAutoSavePreset(e.target.value as MarkdownPreset)}
                    className="w-full px-2 py-1 bg-[#07070b] border border-[#333344] rounded text-white text-xs font-mono focus:outline-none focus:border-[#00ffcc]"
                  >
                    <option value="standard">Standard (.md)</option>
                    <option value="obsidian">Obsidian Note (YAML + Callout)</option>
                    <option value="meeting">Meeting Protocol</option>
                    <option value="clean">Clean Dialogue</option>
                    <option value="timestamps">Timecoded Script</option>
                  </select>
                </div>
              </div>

              {/* Custom Vocabulary / Prompt & Hotwords */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block text-[11px] font-mono text-[#9999aa] mb-1">
                    Vocabulary Prompt (initial_prompt)
                  </label>
                  <input
                    type="text"
                    value={initialPrompt}
                    onChange={(e) => setInitialPrompt(e.target.value)}
                    placeholder="Names, brands, terms..."
                    className="w-full px-3 py-1.5 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono text-[#9999aa] mb-1">
                    Hotwords (boosted terms)
                  </label>
                  <input
                    type="text"
                    value={hotwords}
                    onChange={(e) => setHotwords(e.target.value)}
                    placeholder="Kubernetes, PostgreSQL, ..."
                    className="w-full px-3 py-1.5 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
                  />
                </div>
              </div>
            </div>
          )}

          {/* VAD Tuning */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-mono text-[#9999aa]">VAD Min Silence (Turns)</label>
                <span className="text-xs font-mono text-[#00ffcc]">{minSilenceMs} ms</span>
              </div>
              <input
                type="range"
                min="100"
                max="1000"
                step="25"
                value={minSilenceMs}
                onChange={(e) => setMinSilenceMs(parseInt(e.target.value, 10))}
                className="w-full accent-[#00ffcc] cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-[#666677] mt-0.5">
                <span>100ms (finer turns)</span>
                <span>1000ms (merged)</span>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-mono text-[#9999aa]">VAD Speech Threshold</label>
                <span className="text-xs font-mono text-[#00ffcc]">{vadThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.15"
                max="0.85"
                step="0.05"
                value={vadThreshold}
                onChange={(e) => setVadThreshold(parseFloat(e.target.value))}
                className="w-full accent-[#00ffcc] cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-[#666677] mt-0.5">
                <span>0.15 (catches quiet speech)</span>
                <span>0.85 (strict)</span>
              </div>
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
                URL path served by this app (folder <code className="text-[#00ffcc]">public/models/</code>).
              </p>
            </div>
          )}

          {/* Speaker count & Sensitivity (Python backend) */}
          {backend === 'local-faster-whisper' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
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

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-mono text-[#9999aa]">Split Sensitivity</label>
                  <span className="text-xs font-mono text-[#00ffcc]">{minSilhouette.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="0.30"
                  step="0.01"
                  value={minSilhouette}
                  onChange={(e) => setMinSilhouette(parseFloat(e.target.value))}
                  className="w-full accent-[#00ffcc] cursor-pointer"
                />
              </div>

              <label className="flex items-center gap-2 text-xs text-[#cccccc] cursor-pointer pb-2">
                <input
                  type="checkbox"
                  checked={exactSpeakers}
                  onChange={(e) => setExactSpeakers(e.target.checked)}
                  className="accent-[#00ffcc]"
                />
                <span>Exact speaker count</span>
              </label>
            </div>
          )}

          {/* Hardware & Inference Options */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-mono text-[#9999aa] mb-1">Compute Device</label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value as any)}
                className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
              >
                <option value="auto">Auto (CUDA if available)</option>
                <option value="cuda">CUDA (NVIDIA GPU)</option>
                <option value="cpu">CPU (Multi-thread)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-mono text-[#9999aa] mb-1">Quantization / Precision</label>
              <select
                value={computeType}
                onChange={(e) => setComputeType(e.target.value as ComputeTypeOption)}
                className="w-full px-3 py-2 bg-[#09090d] border border-[#333344] rounded-lg text-white text-xs focus:outline-none focus:border-[#00ffcc]"
              >
                <option value="auto">Auto (float16 GPU / int8 CPU)</option>
                <option value="float16">float16 (Fast GPU)</option>
                <option value="int8_float16">int8_float16 (Low VRAM GPU)</option>
                <option value="int8">int8 (Fastest CPU / Low RAM)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-mono text-[#9999aa] mb-1">Beam Search Size</label>
              <input
                type="number"
                min="1"
                max="10"
                value={beamSize}
                onChange={(e) => setBeamSize(Math.max(1, Math.min(10, parseInt(e.target.value, 10) || 5)))}
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
              className="px-4 py-2 rounded-lg text-xs font-medium text-[#aaaaaa] hover:text-white bg-[#22222c] hover:bg-[#2a2a38] transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-5 py-2 rounded-lg text-xs font-bold text-black bg-[#00ffcc] hover:bg-[#00e6b8] transition-all shadow-md shadow-[#00ffcc]/20 cursor-pointer"
            >
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
