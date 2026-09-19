import React, { useState, useRef, useEffect } from 'react';
import { WhisperModelSize, ProcessedFile, LogEntry, SpeakerMetadata } from './types';
import { OfflineTranscriptionEngine } from './services/transcriptionEngine';
import { Sidebar } from './components/Sidebar';
import { StatCards } from './components/StatCards';
import { StatusBox } from './components/StatusBox';
import { TerminalOutput } from './components/TerminalOutput';
import { TranscriptView } from './components/TranscriptView';
import { ConverterModal } from './components/ConverterModal';
import { RecorderModal } from './components/RecorderModal';
import { SpeakerManagerModal } from './components/SpeakerManagerModal';
import { SemanticViewModal } from './components/SemanticViewModal';
import { ExportModal } from './components/ExportModal';
import { Upload, Sparkles, SlidersHorizontal, Eye, AlertTriangle, X } from 'lucide-react';

export const App: React.FC = () => {
  const [model, setModel] = useState<WhisperModelSize>('base');
  const [status, setStatus] = useState<string>('System Ready');
  const [progress, setProgress] = useState<number>(0);
  const [showProgress, setShowProgress] = useState<boolean>(false);
  const [queueCount, setQueueCount] = useState<number>(0);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: 'init_1',
      timestamp: new Date().toLocaleTimeString(),
      type: 'SYSTEM',
      message: 'NEUROMICON Transcriber Node initialized. WebGPU & Wasm engines online.'
    },
    {
      id: 'init_2',
      timestamp: new Date().toLocaleTimeString(),
      type: 'INFO',
      message: 'Active Whisper model: BASE. Diarization: Utterance-Chunked AHC. F0 boundary: 165Hz.'
    }
  ]);

  const [activeFile, setActiveFile] = useState<ProcessedFile | null>(null);
  const [speakers, setSpeakers] = useState<Record<string, SpeakerMetadata>>({});
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'console' | 'transcript'>('console');

  // Modals
  const [isConverterOpen, setIsConverterOpen] = useState(false);
  const [isRecorderOpen, setIsRecorderOpen] = useState(false);
  const [isSpeakersOpen, setIsSpeakersOpen] = useState(false);
  const [isSemanticOpen, setIsSemanticOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);

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
    setProgress(15);
    setStatus(`Synchronizing Whisper Engine [${newModel.toUpperCase()}]...`);
    addLog('SYSTEM', `Unloading previous model. Loading Whisper ${newModel.toUpperCase()} weights...`);

    // Simulated model download/load
    await new Promise((r) => setTimeout(r, 400));
    setProgress(55);
    setStatus(`Loading tensor graph into memory...`);
    await new Promise((r) => setTimeout(r, 400));
    setProgress(100);
    setStatus('System Ready');
    setShowProgress(false);
    addLog('DONE', `Whisper Engine [${newModel.toUpperCase()}] ready.`);
  };

  // Process a media file through the pipeline
  const processFile = async (file: File) => {
    setErrorBanner(null);
    setIsProcessing(true);
    setShowProgress(true);
    setProgress(5);
    setStatus(`[1/1] Processing: ${file.name}...`);
    addLog('SYSTEM', `Queue initialized: 1 files. Target: ${file.name}`);
    addLog('PROC', `[1/1] Processing: ${file.name}...`);

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
        },
        (prog, stage) => {
          setProgress(prog);
          setStatus(stage);
          if (prog === 35 || prog === 60 || prog === 70 || prog === 88) {
            addLog('PROC', stage);
          }
        }
      );

      if (activeAudioUrlRef.current) {
        URL.revokeObjectURL(activeAudioUrlRef.current);
      }
      activeAudioUrlRef.current = result.audioUrl || null;

      setActiveFile(result);
      setSpeakers(result.speakers);
      setProcessedCount((prev) => prev + 1);
      setQueueCount(0);
      setStatus('Batch Processing Finished');
      setShowProgress(false);
      setActiveTab('transcript');

      const outName = `transcript_${file.name.replace(/\.[^/.]+$/, '')}.md`;
      addLog('DONE', `Finished: ${file.name} -> ${outName}`);
      addLog('DONE', `Identified ${Object.keys(result.speakers).length} distinct speaker voiceprints.`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog('ERROR', `Error in ${file.name}:\n${msg}`);
      setErrorBanner(msg);
      setStatus('Operational Error');
      setShowProgress(false);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle file input selection
  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setQueueCount(files.length);
    processFile(files[0]);
    e.target.value = '';
  };

  // Load Demo Audio
  const handleLoadDemo = () => {
    const demoFile = engineRef.current.createDemoAudioFile();
    setQueueCount(1);
    processFile(demoFile);
  };

  // Update Speaker Metadata
  const handleUpdateSpeaker = (id: string, updates: Partial<SpeakerMetadata>) => {
    engineRef.current.updateSpeaker(id, updates);
    const spks = engineRef.current.getSpeakers();
    setSpeakers(spks);
    setActiveFile((prev) => (prev ? { ...prev, speakers: spks } : null));
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
    setActiveFile({ ...activeFile, segments: updated });
    addLog('INFO', `Edited segment ${segId}.`);
  };

  return (
    <div className="flex h-screen bg-[#0b0b0b] text-[#e0e0e0] overflow-hidden select-none">
      {/* Hidden File Input for IMPORT MEDIA */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFilesSelected}
        accept="video/*,audio/*,.mp4,.mkv,.avi,.mp3,.wav,.m4a,.webm,.flac"
        className="hidden"
      />

      {/* Sidebar matching PySide6 */}
      <Sidebar
        model={model}
        onModelChange={handleModelChange}
        onImportClick={() => fileInputRef.current?.click()}
        onConvertClick={() => setIsConverterOpen(true)}
        onRecordClick={() => setIsRecorderOpen(true)}
        onLoadDemoClick={handleLoadDemo}
        onOpenSpeakers={() => setIsSpeakersOpen(true)}
        onOpenSemantic={() => setIsSemanticOpen(true)}
        onOpenExport={() => setIsExportOpen(true)}
        hasActiveTranscript={Boolean(activeFile)}
        isProcessing={isProcessing}
      />

      {/* Main Content Area matching PySide6 window */}
      <main className="flex-1 flex flex-col p-6 lg:p-8 space-y-5 overflow-y-auto">
        {/* Dashboard Header: 3 Stat Cards */}
        <StatCards
          queueCount={queueCount}
          processedCount={processedCount}
          activeModel={model}
        />

        {/* Status Box & Progress Bar */}
        <StatusBox
          status={status}
          progress={progress}
          showProgress={showProgress}
        />

        {/* View Tabs Selector */}
        <div className="flex items-center justify-between border-b border-[#222222] pb-2">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setActiveTab('console')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                activeTab === 'console'
                  ? 'bg-[#1a1a1a] text-[#00ffcc] border border-[#00ffcc]/30'
                  : 'text-[#777777] hover:text-white'
              }`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span>SYSTEM CONSOLE LOG</span>
            </button>

            {activeFile && (
              <button
                onClick={() => setActiveTab('transcript')}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                  activeTab === 'transcript'
                    ? 'bg-[#1a1a1a] text-[#00ffcc] border border-[#00ffcc]/30'
                    : 'text-[#777777] hover:text-white'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>TRANSCRIPT & DIARIZATION VIEW</span>
                <span className="text-[10px] bg-[#00ffcc]/20 text-[#00ffcc] px-1.5 rounded-full">
                  {activeFile.segments.length}
                </span>
              </button>
            )}
          </div>

          <div className="flex items-center space-x-3 text-xs text-[#666666] font-mono">
            <span>Diarization: AHC + F0 Pitch</span>
            <span>·</span>
            <span>Format: 16kHz Mono</span>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'console' ? (
          <div className="space-y-4">
            {/* Error Troubleshooting Alert Banner */}
            {errorBanner && (
              <div className="bg-[#ff2222]/10 border border-[#ff4444]/40 rounded-xl p-4 text-xs font-mono text-[#ffdddd] flex items-start justify-between space-x-3">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-[#ff4444] shrink-0 mt-0.5" />
                  <div className="whitespace-pre-wrap leading-relaxed">{errorBanner}</div>
                </div>
                <button
                  onClick={() => setErrorBanner(null)}
                  className="text-[#888888] hover:text-white shrink-0 p-1 cursor-pointer"
                  title="Close"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Terminal Output */}
            <TerminalOutput logs={logs} onClear={() => setLogs([])} />

            {/* Quick Upload Banner if no file is currently loaded */}
            {!activeFile && (
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-[#222222] hover:border-[#00ffcc]/40 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition bg-[#111111]/30 group"
              >
                <Upload className="w-8 h-8 text-[#555555] group-hover:text-[#00ffcc] transition mb-2" />
                <span className="text-sm font-bold text-white">Click to Import Audio/Video Media</span>
                <span className="text-xs text-[#666666] mt-1">
                  Supports MP4, MKV, AVI, MOV, MP3, WAV, FLAC (Auto-converts and diarizes offline)
                </span>
                <div className="mt-4 flex items-center space-x-2">
                  <span className="text-[11px] text-[#555555]">Or test right now with:</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLoadDemo();
                    }}
                    className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#333333] text-[#00ffcc] text-xs font-semibold px-3 py-1 rounded-md flex items-center space-x-1.5 transition"
                  >
                    <Sparkles className="w-3 h-3" />
                    <span>Run Dialogue Demo</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          activeFile && (
            <TranscriptView
              file={activeFile}
              speakers={speakers}
              onOpenSpeakers={() => setIsSpeakersOpen(true)}
              onOpenSemantic={() => setIsSemanticOpen(true)}
              onOpenExport={() => setIsExportOpen(true)}
              onUpdateSegmentText={handleUpdateSegmentText}
            />
          )
        )}
      </main>

      {/* Modals */}
      <ConverterModal
        isOpen={isConverterOpen}
        onClose={() => setIsConverterOpen(false)}
        onSendToTranscriber={(file) => {
          setQueueCount(1);
          processFile(file);
        }}
        onLog={addLog}
      />

      <RecorderModal
        isOpen={isRecorderOpen}
        onClose={() => setIsRecorderOpen(false)}
        onRecordingComplete={(file) => {
          setQueueCount(1);
          processFile(file);
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
          />
        </>
      )}
    </div>
  );
};
