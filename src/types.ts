export type WhisperModelSize = 'tiny' | 'base' | 'small' | 'medium';

export type GenderType = 'Male' | 'Female' | 'Unknown';

export interface SpeakerMetadata {
  id: string;
  name: string;
  gender: GenderType;
  confidence?: number;
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
  /** Average token probability (0-1) reported by Whisper; absent if the backend gives none. */
  confidence?: number;
  words?: WordToken[];
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

export type EngineBackendType = 'local-faster-whisper' | 'offline-transformers';

export interface LocalEngineConfig {
  backend: EngineBackendType;
  /** Python server URL (faster-whisper backend). */
  localServerUrl?: string;
  /** In-browser backend only: URL path under which model folders are served (default /models/). */
  localModelPath?: string;
  /** In-browser backend only: refuse to download model weights from the network. */
  blockRemoteDownloads: boolean;
  device?: 'cuda' | 'cpu' | 'auto';
  minSilenceMs?: number;
  beamSize?: number;
  /** Upper bound on detected speakers (Python backend). */
  maxSpeakers?: number;
  /** If true, force exactly `maxSpeakers` speakers instead of auto-detecting up to that number. */
  exactSpeakers?: boolean;
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

