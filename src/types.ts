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
