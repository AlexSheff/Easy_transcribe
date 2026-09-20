import React, { useState, useRef, useEffect } from 'react';
import { WhisperModelSize, ProcessedFile, LogEntry, SpeakerMetadata, BatchFileItem, LocalEngineConfig } from './types';
import { OfflineTranscriptionEngine } from './services/transcriptionEngine';
import { generateMarkdownExport } from './services/exporter';
import { Sidebar } from './components/Sidebar';
import { StatCards } from './components/StatCards';
import { StatusBox } from './components/StatusBox';
import { TerminalOutput } from './components/TerminalOutput';
import { TranscriptView } from './components/TranscriptView';
import { BatchQueuePanel } from './components/BatchQueuePanel';
import { ConverterModal } from './components/ConverterModal';
import { RecorderModal } from './components/RecorderModal';
import { SpeakerManagerModal } from './components/SpeakerManagerModal';
import { SemanticViewModal } from './components/SemanticViewModal';
import { ExportModal } from './components/ExportModal';
import { ModelSettingsModal } from './components/ModelSettingsModal';
import { Upload, Sparkles, SlidersHorizontal, Eye, AlertTriangle, X, FileText, Layers, Download, Sliders } from 'lucide-react';

export const App: React.FC = () => {
  const [model, setModel] = useState<WhisperModelSize>('medium');
  const [status, setStatus] = useState<string>('System Ready');
  const [progress, setProgress] = useState<number>(0);
  const [showProgress, setShowProgress] = useState<boolean>(false);
  const [queueCount, setQueueCount] = useState<number>(0);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [queueItems, setQueueItems] = useState<BatchFileItem[]>([]);

  // Engine configuration with local Systran model path and strict offline enforcement
  const [engineConfig, setEngineConfig] = useState<LocalEngineConfig>({
    localModelPath: 'C:\\Users\\admin_fdr\\.cache\\huggingface\\hub\\models--Systran--faster-whisper-medium\\snapshots\\08e178d48790749d25932bbc082711ddcfdfbc4f',
    backend: 'local-faster-whisper',
    localServerUrl: 'http://127.0.0.1:8000',
    blockRemoteDownloads: true,
    device: 'auto',
    beamSize: 5,
    vadSensitivity: 0.5,
  });

  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: 'init_1',
      timestamp: new Date().toLocaleTimeString(),
      type: 'SYSTEM',
      message: 'Easy TScribe Node initialized. Strict offline configuration active.'
    },
    {
      id: 'init_2',
      timestamp: new Date().toLocaleTimeString(),
      type: 'INFO',
      message: 'Local model target: Systran faster-whisper-medium. Remote downloads: BLOCKED.'
    }
  ]);

  const [activeFile, setActiveFile] = useState<ProcessedFile | null>(null);
  const [processedFiles, setProcessedFiles] = useState<ProcessedFile[]>([]);
  const [speakers, setSpeakers] = useState<Record<string, SpeakerMetadata>>({});
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'console' | 'queue' | 'transcript'>('console');

  // Modals
  const [isConverterOpen, setIsConverterOpen] = useState(false);
  const [isRecorderOpen, setIsRecorderOpen] = useState(false);
  const [isSpeakersOpen, setIsSpeakersOpen] = useState(false);
  const [isSemanticOpen, setIsSemanticOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [isModelSettingsOpen, setIsModelSettingsOpen] = useState(false);

  // Hidden File Input
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Pipeline Engine Instance
  const engineRef = useRef<OfflineTranscriptionEngine>(new OfflineTranscriptionEngine());
  const activeAudioUrlRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (activeAudioUrlRef.current) {
        URL.revokeObjectURL(activeAudioUrlRef.current);
      }
    };
  }, []);

  const addLog = (type: 'SYSTEM' | 'PROC' | 'DONE' | 'ERROR' | 'INFO', message: string) => {
    const entry: LogEntry = {
      id: 'log_' + Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toLocaleTimeString(),
      type,
      message,
    };
    setLogs((prev) => [...prev, entry]);
  };

  // Model change handler
  const handleModelChange = async (newModel: WhisperModelSize) => {
    if (newModel === model || isProcessing) return;
    setModel(newModel);
    setShowProgress(true);
    setProgress(20);
    setStatus(`Switching model configuration to [${newModel.toUpperCase()}]...`);
    addLog('SYSTEM', `Model size set to Whisper ${newModel.toUpperCase()}.`);

    await new Promise((r) => setTimeout(r, 300));
    setProgress(100);
    setStatus('System Ready');
    setShowProgress(false);
    addLog('DONE', `Engine configured for ${newModel.toUpperCase()} (Backend: ${engineConfig.backend}).`);
  };

  // Process multiple media files through the pipeline sequentially
  const enqueueAndProcessFiles = async (files: File[]) => {
    if (!files || files.length === 0 || isProcessing) return;
    setErrorBanner(null);
    setIsProcessing(true);
    setShowProgress(true);

    const total = files.length;
    setQueueCount(total);
    addLog('SYSTEM', `Batch queue initialized: ${total} file(s) received for transcription.`);

    // Build batch queue items
    const newItems: BatchFileItem[] = files.map((f, idx) => ({
      id: `queue_${Date.now()}_${idx}_${Math.random().toString(36).substring(2, 6)}`,
      file: f,
      name: f.name,
      size: f.size,
      status: 'queued',
      progress: 0,
      stageMessage: 'Queued for processing...'
    }));

    setQueueItems((prev) => [...prev, ...newItems]);
    if (total > 1 || activeTab === 'console') {
      setActiveTab('queue');
    }

    setErrorBanner(null);
    let hasErrors = false;

    for (let i = 0; i < total; i++) {
      const file = files[i];
      const batchItem = newItems[i];
      const currentIdx = i + 1;
      setProgress(5);
      setShowProgress(true);
      setStatus(`[${currentIdx}/${total}] Processing: ${file.name}...`);
      addLog('PROC', `[${currentIdx}/${total}] Started processing: ${file.name}...`);

      setQueueItems((prev) =>
        prev.map((item) =>
          item.id === batchItem.id
            ? { ...item, status: 'processing', progress: 5, stageMessage: 'Preparing 16kHz audio track...' }
            : item
        )
      );

      try {
        const result = await engineRef.current.processMediaFile(
          file,
          {
            modelSize: model,
            diarizationEnabled: true,
            voiceFingerprintEnabled: true,
            semanticClusteringEnabled: true,
            language: 'auto',
            fingerprintThreshold: 0.65,
            engineConfig,
          },
          (prog, stage) => {
            setProgress(prog);
            setStatus(`[${currentIdx}/${total}] ${stage}`);
            setQueueItems((prev) =>
              prev.map((item) =>
                item.id === batchItem.id
                  ? { ...item, progress: prog, stageMessage: stage }
                  : item
              )
            );
            if (prog === 35 || prog === 60 || prog === 75 || prog === 88) {
              addLog('PROC', `[${currentIdx}/${total}] ${stage}`);
            }
          }
        );

        if (activeAudioUrlRef.current && activeFile?.filename === file.name) {
          URL.revokeObjectURL(activeAudioUrlRef.current);
        }
        activeAudioUrlRef.current = result.audioUrl || null;

        setProcessedFiles((prev) => {
          const others = prev.filter((p) => p.filename !== file.name);
          return [...others, result];
        });
        setActiveFile(result);
        setSpeakers(result.speakers);
        setProcessedCount((prev) => prev + 1);
        setQueueCount(Math.max(0, total - currentIdx));

        setQueueItems((prev) =>
          prev.map((item) =>
            item.id === batchItem.id
              ? {
                  ...item,
                  status: 'completed',
                  progress: 100,
                  stageMessage: 'Completed successfully',
                  result
                }
              : item
          )
        );

        const outName = `transcript_${file.name.replace(/\.[^/.]+$/, '')}.md`;
        addLog('DONE', `Completed [${currentIdx}/${total}]: ${file.name} -> ${outName}`);
        addLog('DONE', `Diarization: identified ${Object.keys(result.speakers).length} distinct speaker(s).`);

        // Automatically persist standard markdown directly to project transcripts folder
        await saveTranscriptMarkdownToServer(result, file.name);
      } catch (err: unknown) {
        hasErrors = true;
        const msg = err instanceof Error ? err.message : String(err);
        addLog('ERROR', `Error processing ${file.name}:\n${msg}`);
        setErrorBanner(`Recognition Failed for "${file.name}":\n${msg}`);
        setProgress(0);
        setShowProgress(false);
        setIsProcessing(false);
        setStatus(`Recognition Failed: ${file.name}`);
        setQueueItems((prev) =>
          prev.map((item) =>
            item.id === batchItem.id
              ? {
                  ...item,
                  status: 'error',
                  progress: 0,
                  stageMessage: 'Recognition error',
                  error: msg
                }
              : item
          )
        );
      }
    }

    setShowProgress(false);
    setIsProcessing(false);
    if (hasErrors) {
      setStatus('Processing stopped due to error. See alert above.');
    } else {
      setStatus('All files in batch queue processed successfully');
    }
  };

  // Automatically save markdown file directly to project's transcripts folder
  const saveTranscriptMarkdownToServer = async (fileData: ProcessedFile, originalName: string) => {
    try {
      const mdContent = generateMarkdownExport(fileData, fileData.speakers, { preset: 'standard' });
      const bName = originalName.replace(/\.[^/.]+$/, '');
      const cleanName = `transcript_${bName}.md`;

      let autoSaved = false;

      // 1. Try local dev server endpoint
      try {
        const res = await fetch('/api/save-markdown', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: cleanName, content: mdContent })
        });
        if (res.ok) {
          const data = await res.json();
          addLog('DONE', `[AUTO-SAVE] Written to project folder: ${data.relativePath || 'transcripts/' + cleanName}`);
          autoSaved = true;
        }
      } catch {
        // Fallback to python server if running
      }

      // 2. If python backend is running, also ensure saved there
      if (!autoSaved && engineConfig.localServerUrl) {
        try {
          const res = await fetch(`${engineConfig.localServerUrl.replace(/\/+$/, '')}/api/save-markdown`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: cleanName, content: mdContent })
          });
          if (res.ok) {
            const data = await res.json();
            addLog('DONE', `[AUTO-SAVE] Written to project folder: ${data.relativePath || 'transcripts/' + cleanName}`);
          }
        } catch {
          // ignore
        }
      }
    } catch (saveErr) {
      console.warn('Auto-save error:', saveErr);
    }
  };

  const handleClearQueue = () => {
    setQueueItems([]);
  };

  const handleRemoveQueueItem = (id: string) => {
    setQueueItems((prev) => prev.filter((item) => item.id !== id));
  };

  // Handle file input selection
  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const fileList = Array.from(files);
    enqueueAndProcessFiles(fileList);
    e.target.value = '';
  };

  // Update Speaker Metadata
  const handleUpdateSpeaker = (id: string, updates: Partial<SpeakerMetadata>) => {
    engineRef.current.updateSpeaker(id, updates);
    const spks = engineRef.current.getSpeakers();
    setSpeakers(spks);
    setActiveFile((prev) => (prev ? { ...prev, speakers: spks } : null));
    setProcessedFiles((prev) =>
      prev.map((f) => (f.id === activeFile?.id ? { ...f, speakers: spks } : f))
    );
    addLog('INFO', `Updated speaker identity ${id} -> ${updates.name || ''}`);
  };

  // Merge Speakers
  const handleMergeSpeakers = (sourceId: string, targetId: string) => {
    engineRef.current.mergeSpeakers(sourceId, targetId);
    const spks = engineRef.current.getSpeakers();
    setSpeakers(spks);

    // Update current active file segments & speakers
    setActiveFile((prev) => {
      if (!prev) return null;
      const updatedSegments = prev.segments.map((seg) =>
        seg.speakerId === sourceId ? { ...seg, speakerId: targetId } : seg
      );
      return { ...prev, segments: updatedSegments, speakers: spks };
    });
    setProcessedFiles((prev) =>
      prev.map((f) => {
        if (f.id !== activeFile?.id) return f;
        const updatedSegments = f.segments.map((seg) =>
          seg.speakerId === sourceId ? { ...seg, speakerId: targetId } : seg
        );
        return { ...f, segments: updatedSegments, speakers: spks };
      })
    );
    addLog('INFO', `Merged speaker identity ${sourceId} into ${targetId}`);
  };

  // Reset Voice DB
  const handleResetDb = () => {
    engineRef.current.resetVoiceDb();
    setSpeakers({});
    setActiveFile((prev) => (prev ? { ...prev, speakers: {} } : null));
    addLog('SYSTEM', 'Voice database reset. Session voiceprints cleared.');
  };

  // Update segment text edit
  const handleUpdateSegmentText = (segId: string, newText: string) => {
    if (!activeFile) return;
    const updated = activeFile.segments.map((s) => (s.id === segId ? { ...s, text: newText } : s));
    const updatedFile = { ...activeFile, segments: updated };
    setActiveFile(updatedFile);
    setProcessedFiles((prev) =>
      prev.map((f) => (f.id === activeFile.id ? { ...f, segments: updated } : f))
    );
    addLog('INFO', `Edited segment ${segId}.`);
    saveTranscriptMarkdownToServer(updatedFile, activeFile.filename);
  };

  return (
    <div
      className="flex h-screen bg-[#0b0b0b] text-[#e0e0e0] overflow-hidden select-none relative"
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!isDragOver) setIsDragOver(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.currentTarget.contains(e.relatedTarget as Node)) return;
        setIsDragOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          enqueueAndProcessFiles(Array.from(e.dataTransfer.files));
        }
      }}
    >
      {/* Drag & Drop Visual Overlay */}
      {isDragOver && (
        <div className="fixed inset-0 bg-[#00ffcc]/10 border-4 border-dashed border-[#00ffcc] backdrop-blur-xs z-50 flex flex-col items-center justify-center pointer-events-none">
          <Upload className="w-16 h-16 text-[#00ffcc] animate-bounce mb-3" />
          <div className="text-xl font-mono font-bold text-white tracking-widest uppercase">
            Drop Media Files Here to Transcribe
          </div>
          <div className="text-sm text-[#00ffcc] font-mono mt-1">
            Batch queueing supported (MP4, MKV, AVI, MOV, MP3, WAV, FLAC, M4A)
          </div>
        </div>
      )}

      {/* Hidden File Input for IMPORT MEDIA - with multiple attribute */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFilesSelected}
        accept="video/*,audio/*,.mp4,.mkv,.avi,.mp3,.wav,.m4a,.webm,.flac"
        multiple
        className="hidden"
      />

      {/* Sidebar matching PySide6 */}
      <Sidebar
        model={model}
        onModelChange={handleModelChange}
        onImportClick={() => fileInputRef.current?.click()}
        onConvertClick={() => setIsConverterOpen(true)}
        onRecordClick={() => setIsRecorderOpen(true)}
        onOpenModelSettings={() => setIsModelSettingsOpen(true)}
        onOpenSpeakers={() => setIsSpeakersOpen(true)}
        onOpenSemantic={() => setIsSemanticOpen(true)}
        onOpenExport={() => setIsExportOpen(true)}
        hasActiveTranscript={Boolean(activeFile)}
        isProcessing={isProcessing}
        engineBackend={engineConfig.backend}
        blockRemoteDownloads={engineConfig.blockRemoteDownloads}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col p-6 lg:p-8 space-y-5 overflow-y-auto">
        {/* Dashboard Header: 3 Stat Cards */}
        <StatCards
          queueCount={queueCount}
          processedCount={processedCount}
          activeModel={model}
        />

        {/* Status Box & Progress Bar */}
        <StatusBox
          status={errorBanner ? errorBanner.split('\n')[0] : status}
          progress={progress}
          showProgress={showProgress && !errorBanner}
          isError={!!errorBanner}
          onDismissError={() => setErrorBanner(null)}
        />

        {/* Global Error Alert Banner */}
        {errorBanner && (
          <div className="bg-[#240d0d] border-2 border-[#ff4444]/70 rounded-xl p-4 text-xs font-mono text-[#ffdddd] flex items-start justify-between space-x-3 shadow-[0_0_25px_rgba(255,68,68,0.2)] animate-in fade-in duration-200">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="w-5 h-5 text-[#ff4444] shrink-0 mt-0.5" />
              <div className="space-y-1 overflow-x-auto">
                <div className="font-bold text-sm text-[#ff6666]">Recognition Error / Ошибка распознавания:</div>
                <div className="whitespace-pre-wrap leading-relaxed text-[#ffcccc] font-mono select-text">{errorBanner}</div>
              </div>
            </div>
            <button
              onClick={() => setErrorBanner(null)}
              className="text-[#aaaaaa] hover:text-white shrink-0 p-1.5 bg-[#ff4444]/20 hover:bg-[#ff4444]/40 rounded-lg cursor-pointer transition-colors"
              title="Close Warning"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* View Tabs Selector */}
        <div className="flex items-center justify-between border-b border-[#222222] pb-2">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab('console')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                activeTab === 'console'
                  ? 'bg-[#1a1a1a] text-[#00ffcc] border border-[#00ffcc]/30'
                  : 'text-[#777777] hover:text-white'
              }`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span>CONSOLE LOG</span>
            </button>

            <button
              onClick={() => setActiveTab('queue')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                activeTab === 'queue'
                  ? 'bg-[#1a1a1a] text-[#00ffcc] border border-[#00ffcc]/30'
                  : 'text-[#777777] hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>QUEUE & .MD</span>
              {queueItems.length > 0 && (
                <span className="text-[10px] bg-[#00ffcc]/20 text-[#00ffcc] px-1.5 py-0.5 rounded-full font-mono">
                  {queueItems.filter((i) => i.status === 'completed').length}/{queueItems.length}
                </span>
              )}
            </button>

            {activeFile && (
              <button
                onClick={() => setActiveTab('transcript')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                  activeTab === 'transcript'
                    ? 'bg-[#1a1a1a] text-[#00ffcc] border border-[#00ffcc]/30'
                    : 'text-[#777777] hover:text-white'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>TRANSCRIPT & DIALOGUE</span>
                <span className="text-[10px] bg-[#00ffcc]/20 text-[#00ffcc] px-1.5 py-0.5 rounded-full">
                  {activeFile.segments.length}
                </span>
              </button>
            )}
          </div>

          <div className="flex items-center space-x-3 text-xs text-[#666666] font-mono">
            <button
              onClick={() => setIsModelSettingsOpen(true)}
              className="text-[#00ffcc] hover:underline flex items-center gap-1 cursor-pointer"
            >
              <Sliders className="w-3 h-3" />
              <span>Local Model Config</span>
            </button>
            <span>·</span>
            <span>Diarization: Pitch F0 + AHC</span>
            <span>·</span>
            <span className="text-[#00ffcc]">Output: Markdown (.md)</span>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'console' && (
          <div className="space-y-4">
            {/* Terminal Output */}
            <TerminalOutput logs={logs} onClear={() => setLogs([])} />

            {/* Media Upload Banner if no file is currently loaded */}
            {!activeFile && (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-[#262626] hover:border-[#00ffcc]/60 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all bg-[#121215]/50 hover:bg-[#15151a] group"
              >
                <div className="w-12 h-12 rounded-xl bg-[#1c1c22] border border-[#2a2a35] flex items-center justify-center mb-3 group-hover:border-[#00ffcc]/40 group-hover:scale-105 transition-all">
                  <Upload className="w-6 h-6 text-[#888899] group-hover:text-[#00ffcc] transition" />
                </div>
                <span className="text-base font-bold text-white tracking-wide">
                  Select or Drop Audio / Video Files
                </span>
                <span className="text-xs text-[#888899] mt-1.5 max-w-md">
                  Supports batch processing of MP4, MKV, AVI, MOV, MP3, WAV, FLAC, M4A, OGG
                </span>
                <div className="mt-4 flex items-center gap-2 text-xs font-mono text-[#00ffcc] bg-[#00ffcc]/10 border border-[#00ffcc]/20 px-3.5 py-1.5 rounded-lg">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00ffcc]"></span>
                  <span>Auto-saves standard .md transcripts to project/transcripts/</span>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'queue' && (
          <div className="space-y-4">
            <BatchQueuePanel
              queueItems={queueItems}
              processedFiles={processedFiles}
              activeFile={activeFile}
              isProcessing={isProcessing}
              onSelectFile={(f) => {
                if (activeAudioUrlRef.current && activeAudioUrlRef.current !== f.audioUrl) {
                  URL.revokeObjectURL(activeAudioUrlRef.current);
                }
                activeAudioUrlRef.current = f.audioUrl || null;
                setActiveFile(f);
                setSpeakers(f.speakers);
                setActiveTab('transcript');
              }}
              onAddFilesClick={() => fileInputRef.current?.click()}
              onClearQueue={handleClearQueue}
              onRemoveItem={handleRemoveQueueItem}
            />
          </div>
        )}

        {activeTab === 'transcript' && activeFile && (
          <div className="space-y-3">
            {/* Multi-File Switcher Bar */}
            {processedFiles.length > 1 && (
              <div className="bg-[#111111] border border-[#222222] rounded-xl p-3 flex items-center space-x-3 overflow-x-auto scrollbar-thin">
                <div className="flex items-center space-x-1.5 text-xs font-mono font-bold text-[#00ffcc] uppercase tracking-wider shrink-0">
                  <FileText className="w-3.5 h-3.5" />
                  <span>Files ({processedFiles.length}):</span>
                </div>
                <div className="flex items-center space-x-2">
                  {processedFiles.map((pf) => {
                    const isSelected = activeFile.id === pf.id;
                    return (
                      <button
                        key={pf.id}
                        onClick={() => {
                          if (activeAudioUrlRef.current && activeAudioUrlRef.current !== pf.audioUrl) {
                            URL.revokeObjectURL(activeAudioUrlRef.current);
                          }
                          activeAudioUrlRef.current = pf.audioUrl || null;
                          setActiveFile(pf);
                          setSpeakers(pf.speakers);
                        }}
                        className={`text-xs font-mono px-3 py-1.5 rounded-lg border transition shrink-0 cursor-pointer flex items-center space-x-2 ${
                          isSelected
                            ? 'bg-[#00ffcc]/15 text-[#00ffcc] border-[#00ffcc]/60 shadow-[0_0_10px_rgba(0,255,204,0.15)] font-bold'
                            : 'bg-[#181818] text-[#999999] border-[#2a2a2a] hover:border-[#444444] hover:text-white'
                        }`}
                      >
                        <span className="truncate max-w-[200px]">{pf.filename}</span>
                        <span className="text-[10px] bg-[#222222] text-[#cccccc] px-1.5 py-0.5 rounded-full">
                          {pf.segments.length} seg
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <TranscriptView
              file={activeFile}
              speakers={speakers}
              onOpenSpeakers={() => setIsSpeakersOpen(true)}
              onOpenSemantic={() => setIsSemanticOpen(true)}
              onOpenExport={() => setIsExportOpen(true)}
              onUpdateSegmentText={handleUpdateSegmentText}
            />
          </div>
        )}
      </main>

      {/* Modals */}
      <ModelSettingsModal
        isOpen={isModelSettingsOpen}
        onClose={() => setIsModelSettingsOpen(false)}
        config={engineConfig}
        onSaveConfig={(cfg) => setEngineConfig(cfg)}
        onAddLog={addLog}
      />

      <ConverterModal
        isOpen={isConverterOpen}
        onClose={() => setIsConverterOpen(false)}
        onSendToTranscriber={(file) => {
          enqueueAndProcessFiles([file]);
        }}
        onLog={addLog}
      />

      <RecorderModal
        isOpen={isRecorderOpen}
        onClose={() => setIsRecorderOpen(false)}
        onRecordingComplete={(file) => {
          enqueueAndProcessFiles([file]);
        }}
        onLog={addLog}
      />

      <SpeakerManagerModal
        isOpen={isSpeakersOpen}
        onClose={() => setIsSpeakersOpen(false)}
        speakers={speakers}
        onUpdateSpeaker={handleUpdateSpeaker}
        onMergeSpeakers={handleMergeSpeakers}
        onResetDb={handleResetDb}
      />

      {activeFile && (
        <>
          <SemanticViewModal
            isOpen={isSemanticOpen}
            onClose={() => setIsSemanticOpen(false)}
            clusters={activeFile.clusters}
            segments={activeFile.segments}
            speakers={speakers}
          />

          <ExportModal
            isOpen={isExportOpen}
            onClose={() => setIsExportOpen(false)}
            file={activeFile}
            speakers={speakers}
            allFiles={processedFiles}
          />
        </>
      )}
    </div>
  );
};
