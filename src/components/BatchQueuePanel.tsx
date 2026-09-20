import React from 'react';
import { ProcessedFile, BatchFileItem } from '../types';
import { formatTimeSeconds, generateMarkdownExport, generateMasterMarkdownExport, downloadTextFile } from '../services/exporter';
import { 
  FileText, 
  Download, 
  Trash2, 
  Plus, 
  Play, 
  CheckCircle2, 
  Clock, 
  AlertCircle, 
  Loader2, 
  Layers,
  Sparkles,
  Music,
  Video
} from 'lucide-react';

interface BatchQueuePanelProps {
  queueItems: BatchFileItem[];
  processedFiles: ProcessedFile[];
  activeFile: ProcessedFile | null;
  isProcessing: boolean;
  onSelectFile: (file: ProcessedFile) => void;
  onAddFilesClick: () => void;
  onClearQueue: () => void;
  onRemoveItem: (id: string) => void;
}

export const BatchQueuePanel: React.FC<BatchQueuePanelProps> = ({
  queueItems,
  processedFiles,
  activeFile,
  isProcessing,
  onSelectFile,
  onAddFilesClick,
  onClearQueue,
  onRemoveItem,
}) => {
  const completedCount = queueItems.filter((i) => i.status === 'completed').length;
  const totalCount = queueItems.length;

  const handleDownloadSingleMd = (f: ProcessedFile) => {
    const md = generateMarkdownExport(f, f.speakers, { preset: 'standard' });
    const bName = f.filename.replace(/\.[^/.]+$/, '');
    downloadTextFile(md, `transcript_${bName}.md`, 'text/markdown');
  };

  const handleDownloadAllMd = () => {
    if (processedFiles.length === 0) return;
    processedFiles.forEach((f, idx) => {
      setTimeout(() => {
        const md = generateMarkdownExport(f, f.speakers, { preset: 'standard' });
        const bName = f.filename.replace(/\.[^/.]+$/, '');
        downloadTextFile(md, `transcript_${bName}.md`, 'text/markdown');
      }, idx * 200);
    });
  };

  const handleDownloadMasterMd = () => {
    if (processedFiles.length === 0) return;
    const master = generateMasterMarkdownExport(processedFiles);
    const dateStr = new Date().toISOString().substring(0, 10);
    downloadTextFile(master, `master_report_${dateStr}.md`, 'text/markdown');
  };

  const formatBytes = (bytes: number): string => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const isVideoFile = (filename: string) => {
    return /\.(mp4|mkv|avi|mov|webm)$/i.test(filename);
  };

  return (
    <div className="bg-[#121212] border border-[#222222] rounded-xl flex flex-col h-[520px] overflow-hidden shadow-xl">
      {/* Header Bar */}
      <div className="px-5 py-3.5 bg-[#161616] border-b border-[#222222] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-[#00ffcc]/15 border border-[#00ffcc]/30 flex items-center justify-center">
            <Layers className="w-4 h-4 text-[#00ffcc]" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              BATCH TRANSCRIPTION QUEUE ({completedCount}/{totalCount})
            </h3>
            <p className="text-[11px] text-[#777777]">
              Automated audio conversion, speaker diarization, and Markdown generation
            </p>
          </div>
        </div>

        {/* Global Batch Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onAddFilesClick}
            disabled={isProcessing}
            className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#333333] hover:border-[#00ffcc]/60 text-[#00ffcc] text-xs font-mono px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition cursor-pointer disabled:opacity-40"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>+ Add Files</span>
          </button>

          {processedFiles.length > 0 && (
            <>
              <button
                onClick={handleDownloadAllMd}
                className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#00ffcc]/40 text-[#00ffcc] text-xs font-mono font-bold px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition cursor-pointer shadow-xs"
                title="Download all transcriptions as individual .md files"
              >
                <Download className="w-3.5 h-3.5" />
                <span>All .MD ({processedFiles.length})</span>
              </button>

              <button
                onClick={handleDownloadMasterMd}
                className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] hover:brightness-110 text-black font-extrabold text-xs px-3.5 py-1.5 rounded-lg flex items-center space-x-1.5 transition shadow cursor-pointer"
                title="Combine all files into one master document with summaries and table of contents"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Master .MD Report</span>
              </button>
            </>
          )}

          {queueItems.length > 0 && !isProcessing && (
            <button
              onClick={onClearQueue}
              className="text-[#666666] hover:text-[#ff4444] p-1.5 rounded transition cursor-pointer"
              title="Clear queue"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Queue Item List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
        {queueItems.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8">
            <div className="w-14 h-14 rounded-2xl bg-[#181818] border border-[#282828] flex items-center justify-center mb-3 text-[#555]">
              <Layers className="w-7 h-7" />
            </div>
            <h4 className="text-sm font-bold text-white mb-1">Queue is empty</h4>
            <p className="text-xs text-[#666666] max-w-sm mb-4">
              Select multiple audio or video files or drag and drop recordings directly into the application
            </p>
            <button
              onClick={onAddFilesClick}
              className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] text-black font-extrabold text-xs px-4 py-2 rounded-lg transition hover:brightness-110 cursor-pointer shadow"
            >
              Select Files to Transcribe
            </button>
          </div>
        ) : (
          queueItems.map((item, idx) => {
            const isCurrent = isProcessing && item.status === 'processing';
            const isFinished = item.status === 'completed';
            const isError = item.status === 'error';
            const isQueued = item.status === 'queued';
            const isSelected = activeFile && item.result && activeFile.id === item.result.id;

            return (
              <div
                key={item.id}
                className={`p-3.5 rounded-xl border transition flex flex-col space-y-2 ${
                  isSelected
                    ? 'bg-[#181818] border-[#00ffcc]/60 shadow-[0_0_12px_rgba(0,255,204,0.1)]'
                    : isCurrent
                    ? 'bg-[#151515] border-[#0099ff]/50'
                    : 'bg-[#141414] border-[#222222] hover:border-[#333333]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 min-w-0">
                    <span className="text-[11px] font-mono text-[#555] w-5 text-right">{idx + 1}.</span>
                    <div className="w-7 h-7 rounded-lg bg-[#1e1e1e] border border-[#2a2a2a] flex items-center justify-center shrink-0">
                      {isVideoFile(item.name) ? (
                        <Video className="w-3.5 h-3.5 text-[#0099ff]" />
                      ) : (
                        <Music className="w-3.5 h-3.5 text-[#00ffcc]" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-mono font-bold text-white truncate max-w-[340px]">
                        {item.name}
                      </div>
                      <div className="text-[10px] text-[#666] font-mono flex items-center space-x-2">
                        <span>{formatBytes(item.size)}</span>
                        {item.result && (
                          <>
                            <span>•</span>
                            <span className="text-[#00ffcc]">{formatTimeSeconds(item.result.duration)}</span>
                            <span>•</span>
                            <span>{item.result.segments.length} segments</span>
                            <span>•</span>
                            <span>{Object.keys(item.result.speakers).length} speakers</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Right Status / Action Controls */}
                  <div className="flex items-center space-x-2 shrink-0">
                    {isCurrent && (
                      <span className="text-[11px] font-mono text-[#0099ff] flex items-center space-x-1.5">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>{item.stageMessage || 'Processing...'}</span>
                      </span>
                    )}

                    {isQueued && (
                      <span className="text-[11px] font-mono text-[#888] flex items-center space-x-1 bg-[#1a1a1a] px-2 py-0.5 rounded border border-[#282828]">
                        <Clock className="w-3 h-3 text-[#777]" />
                        <span>Queued</span>
                      </span>
                    )}

                    {isFinished && item.result && (
                      <div className="flex items-center space-x-1.5">
                        <button
                          onClick={() => onSelectFile(item.result!)}
                          className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#333] hover:border-[#00ffcc]/50 text-xs font-mono text-white px-2.5 py-1 rounded-md transition cursor-pointer"
                        >
                          Open
                        </button>
                        <button
                          onClick={() => handleDownloadSingleMd(item.result!)}
                          className="bg-[#00ffcc]/15 hover:bg-[#00ffcc]/25 border border-[#00ffcc]/40 text-[#00ffcc] text-xs font-mono px-2.5 py-1 rounded-md transition flex items-center space-x-1 cursor-pointer"
                          title="Download .md file"
                        >
                          <Download className="w-3 h-3" />
                          <span>.MD</span>
                        </button>
                      </div>
                    )}

                    {isError && (
                      <span className="text-[11px] font-mono text-[#ff5555] flex items-center space-x-1">
                        <AlertCircle className="w-3.5 h-3.5" />
                        <span>Error</span>
                      </span>
                    )}

                    {!isProcessing && (
                      <button
                        onClick={() => onRemoveItem(item.id)}
                        className="text-[#555] hover:text-[#ff5555] p-1 rounded transition cursor-pointer ml-1"
                        title="Remove from queue"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Progress bar for active file */}
                {isCurrent && (
                  <div className="w-full bg-[#1e1e1e] h-1.5 rounded-full overflow-hidden border border-[#2a2a2a]">
                    <div
                      className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] h-full transition-all duration-300"
                      style={{ width: `${Math.max(5, item.progress)}%` }}
                    />
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer Info */}
      <div className="px-5 py-2.5 bg-[#0e0e0e] border-t border-[#222222] flex items-center justify-between text-[11px] font-mono text-[#666666]">
        <div>
          Completed: <span className="text-[#00ffcc] font-bold">{completedCount}</span> of {totalCount}
        </div>
        <div className="flex items-center space-x-4">
          <span>Supported: MP4, MKV, AVI, MOV, MP3, WAV, FLAC, M4A</span>
          <span className="text-[#00ffcc]">Markdown .MD (Obsidian / Notion)</span>
        </div>
      </div>
    </div>
  );
};
