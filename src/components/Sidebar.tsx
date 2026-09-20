import React from 'react';
import { WhisperModelSize } from '../types';
import { 
  Cpu, 
  FolderOpen, 
  RefreshCw, 
  Mic, 
  Sparkles, 
  Sliders, 
  Layers, 
  Download, 
  Radio
} from 'lucide-react';

interface SidebarProps {
  model: WhisperModelSize;
  onModelChange: (model: WhisperModelSize) => void;
  onImportClick: () => void;
  onConvertClick: () => void;
  onRecordClick: () => void;
  onOpenModelSettings: () => void;
  onOpenSpeakers: () => void;
  onOpenSemantic: () => void;
  onOpenExport: () => void;
  hasActiveTranscript: boolean;
  isProcessing: boolean;
  engineBackend?: string;
  blockRemoteDownloads?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  model,
  onModelChange,
  onImportClick,
  onConvertClick,
  onRecordClick,
  onOpenModelSettings,
  onOpenSpeakers,
  onOpenSemantic,
  onOpenExport,
  hasActiveTranscript,
  isProcessing,
  engineBackend = 'local-faster-whisper',
  blockRemoteDownloads = true,
}) => {
  return (
    <aside 
      id="sidebar" 
      className="w-72 bg-[#111111] border-r border-[#222222] p-5 flex flex-col justify-between shrink-0 h-screen overflow-y-auto"
    >
      <div className="space-y-6">
        {/* Brand / Logo */}
        <div className="flex items-center space-x-3 pb-2 border-b border-[#222222]">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#00ffcc] to-[#0099ff] flex items-center justify-center p-0.5 shadow-[0_0_15px_rgba(0,255,204,0.3)]">
            <img 
              src="/assets/logo.png" 
              alt="Logo" 
              className="w-full h-full object-contain rounded-md"
              onError={(e) => {
                // Fallback if image fails
                (e.target as HTMLImageElement).style.display = 'none';
              }} 
            />
          </div>
          <div>
            <div className="text-[#00ffcc] font-black text-xl tracking-[0.15em]">Easy TScribe</div>
            <div className="text-[10px] text-[#666666] tracking-wider uppercase">Transcriber Node v1.0</div>
          </div>
        </div>

        {/* Model Selector & Engine Settings */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-[#888888] uppercase tracking-wider block">
              Whisper Engine
            </label>
            <button
              onClick={onOpenModelSettings}
              className="text-[10px] text-[#00ffcc] hover:underline flex items-center gap-1 font-mono uppercase cursor-pointer"
              title="Configure local model directory and execution backend"
            >
              <Sliders className="w-3 h-3" />
              <span>Model Path</span>
            </button>
          </div>
          <div className="relative">
            <select
              id="model-selector"
              value={model}
              disabled={isProcessing}
              onChange={(e) => onModelChange(e.target.value as WhisperModelSize)}
              className="w-full bg-[#1a1a1a] border border-[#333333] hover:border-[#00ffcc]/50 rounded-lg px-3 py-2.5 text-sm text-[#eeeeee] font-mono focus:outline-none focus:border-[#00ffcc] disabled:opacity-50 cursor-pointer"
            >
              <option value="tiny">tiny (39 MB · Ultra Fast)</option>
              <option value="base">base (74 MB · Balanced)</option>
              <option value="small">small (244 MB · Accurate)</option>
              <option value="medium">medium (769 MB · Systran Local)</option>
              <option value="large-v3">large-v3 (1.5 GB · Studio Precision)</option>
            </select>
          </div>
          <div className="text-[11px] text-[#555555] flex items-center justify-between px-1">
            <span>{engineBackend === 'local-faster-whisper' ? 'faster-whisper' : 'Offline Engine'}</span>
            <span className={blockRemoteDownloads ? 'text-[#00ffcc]' : 'text-[#ffaa00]'}>
              {blockRemoteDownloads ? '● Local Only' : '○ Hybrid'}
            </span>
          </div>
        </div>

        {/* Main Action Buttons Matching PySide6 UI */}
        <div className="space-y-3 pt-2">
          <button
            id="import-media-btn"
            onClick={onImportClick}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-[#00ffcc] to-[#0099ff] hover:brightness-110 active:scale-[0.98] text-[#000000] font-extrabold text-xs tracking-wider uppercase py-3 px-4 rounded-lg shadow-md transition-all flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <FolderOpen className="w-4 h-4 text-black" />
            <span>Import Media</span>
          </button>

          <button
            id="convert-wav-btn"
            onClick={onConvertClick}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-[#ffcc00] to-[#ff9900] hover:brightness-110 active:scale-[0.98] text-[#000000] font-extrabold text-xs tracking-wider uppercase py-3 px-4 rounded-lg shadow-md transition-all flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <RefreshCw className="w-4 h-4 text-black" />
            <span>Convert to WAV</span>
          </button>

          <button
            id="record-mic-btn"
            onClick={onRecordClick}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-[#ff0055] to-[#ff55aa] hover:brightness-110 active:scale-[0.98] text-[#ffffff] font-extrabold text-xs tracking-wider uppercase py-3 px-4 rounded-lg shadow-md transition-all flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Mic className="w-4 h-4 text-white" />
            <span>Record Mic</span>
          </button>
        </div>

        {/* Transcripts Directory Status */}
        <div className="pt-2">
          <div className="bg-[#141418] border border-[#22222e] rounded-lg p-2.5 text-xs">
            <div className="flex items-center justify-between text-[#888899] mb-1">
              <span className="font-mono text-[10px] uppercase tracking-wider">Storage Target</span>
              <span className="text-[10px] text-[#00ffcc] font-mono">Auto-Save</span>
            </div>
            <div className="font-mono text-[11px] text-white truncate flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00ffcc] shrink-0"></span>
              <span className="truncate">./transcripts/*.md</span>
            </div>
          </div>
        </div>

        {/* Session Tools (Speaker DB & Semantic Clusters & Export) */}
        {hasActiveTranscript && (
          <div className="pt-3 border-t border-[#222222] space-y-2">
            <div className="text-[10px] uppercase font-bold text-[#666666] tracking-wider px-1">
              Active Session Tools
            </div>
            <button
              onClick={onOpenSpeakers}
              className="w-full bg-[#161616] hover:bg-[#222222] text-[#cccccc] hover:text-white text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-between border border-[#262626] transition cursor-pointer"
            >
              <div className="flex items-center space-x-2">
                <Sliders className="w-3.5 h-3.5 text-[#00ffcc]" />
                <span>Speaker Manager</span>
              </div>
              <span className="text-[10px] text-[#00ffcc] font-mono">F0 Pitch</span>
            </button>

            <button
              onClick={onOpenSemantic}
              className="w-full bg-[#161616] hover:bg-[#222222] text-[#cccccc] hover:text-white text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-between border border-[#262626] transition cursor-pointer"
            >
              <div className="flex items-center space-x-2">
                <Layers className="w-3.5 h-3.5 text-[#ffcc00]" />
                <span>Semantic Clusters</span>
              </div>
              <span className="text-[10px] text-[#ffcc00] font-mono">K-Means</span>
            </button>

            <button
              onClick={onOpenExport}
              className="w-full bg-[#161616] hover:bg-[#222222] text-[#cccccc] hover:text-white text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-between border border-[#262626] transition cursor-pointer"
            >
              <div className="flex items-center space-x-2">
                <Download className="w-3.5 h-3.5 text-[#0099ff]" />
                <span>Export Intelligence</span>
              </div>
              <span className="text-[10px] text-[#0099ff] font-mono">.MD / .SRT</span>
            </button>
          </div>
        )}
      </div>

      {/* Hardware Info Card matching PySide6 */}
      <div className="pt-4 border-t border-[#222222] mt-4">
        <div 
          id="hw-card"
          className="bg-[#141414] border-l-2 border-[#00ffcc] rounded-r-lg p-3 space-y-1.5"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-xs text-[#00ffcc] font-bold">
              <Cpu className="w-3.5 h-3.5" />
              <span>ENGINE: WebGPU</span>
            </div>
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00ffcc] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00ffcc]"></span>
            </span>
          </div>
          <div className="text-[11px] text-[#888888] font-mono flex items-center justify-between">
            <span>Hardware:</span>
            <span className="text-[#cccccc]">CUDA / Wasm SIMD</span>
          </div>
          <div className="text-[11px] text-[#888888] font-mono flex items-center justify-between">
            <span>Memory:</span>
            <span className="text-[#00ffcc]">Session Ephemeral</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
