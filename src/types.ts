export type WhisperModelSize = 'tiny' | 'base' | 'small' | 'medium' | 'large-v3';

export type GenderType = 'Male' | 'Female' | 'Unknown';

export interface SpeakerMetadata {
  id: string;
  name: string;
  gender: GenderType;
  confidence: number;
  pitchF0?: number;
  sampleCount: number;
  color: string;
}

export interface WordToken {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface TranscriptSegment {
  id: string;
  start: number;
  end: number;
  text: string;
  speakerId: string;
  gender: GenderType;
  uncertain: boolean;
  confidence: number;
  words?: WordToken[];
  clusterId?: number;
}

export interface SemanticCluster {
  clusterId: number;
  topic: string;
  summary: string;
  segmentIds: string[];
}

export interface ProcessedFile {
  id: string;
  filename: string;
  fileSize: number;
  duration: number;
  audioBlob?: Blob;
  audioUrl?: string;
  segments: TranscriptSegment[];
  speakers: Record<string, SpeakerMetadata>;
  clusters: SemanticCluster[];
  processedAt: string;
  language: string;
  modelUsed: WhisperModelSize;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  type: 'SYSTEM' | 'PROC' | 'DONE' | 'ERROR' | 'INFO';
  message: string;
}

export interface ConversionItem {
  id: string;
  file: File;
  name: string;
  size: number;
  status: 'pending' | 'converting' | 'completed' | 'error';
  progress: number;
  convertedBlob?: Blob;
  convertedUrl?: string;
  error?: string;
}

export type MarkdownPreset = 'standard' | 'obsidian' | 'meeting' | 'clean' | 'timestamps';

export type EngineBackendType = 'local-faster-whisper' | 'offline-transformers' | 'local-acoustic';

export type DiarizationEngineType = 'speechbrain-ecapa' | 'pyannote-segmentation' | 'acoustic-cluster';

export type AsrEngineType = 'faster-whisper' | 'gigaam-v3';

export interface LocalEngineConfig {
  backend: EngineBackendType;
  localModelPath: string;
  localServerUrl?: string;
  blockRemoteDownloads: boolean;
  device?: 'cuda' | 'cpu' | 'auto';
  vadSensitivity?: number;
  minSilenceMs?: number;
  beamSize?: number;
  temperature?: number;
  language?: string;
  diarizationEngine?: DiarizationEngineType;
  asrEngine?: AsrEngineType;
  hfCacheDir?: string;
}

export interface BatchFileItem {
  id: string;
  file: File;
  name: string;
  size: number;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress: number;
  stageMessage?: string;
  error?: string;
  result?: ProcessedFile;
}

