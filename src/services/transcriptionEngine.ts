import { ProcessedFile, TranscriptSegment, SpeakerMetadata, WhisperModelSize, LocalEngineConfig } from '../types';
import { convertMediaToWav } from './audioConverter';
import { loadTransformersModule } from './whisperLoader';

const SPEAKER_COLORS = [
  '#00ffcc', // Cyan
  '#ff0077', // Magenta / Rose
  '#ffcc00', // Amber
  '#0099ff', // Sky Blue
  '#a855f7', // Purple
  '#22c55e', // Emerald
  '#f97316', // Orange
];

/** Segments with an average token probability below this are flagged "(?)" in the UI. */
const UNCERTAIN_CONFIDENCE = 0.37;

/** Models the in-browser (transformers.js) backend can load. Others are not offered there. */
const BROWSER_MODELS: Partial<Record<WhisperModelSize, string>> = {
  tiny: 'Xenova/whisper-tiny',
  base: 'Xenova/whisper-base',
  small: 'Xenova/whisper-small',
  medium: 'Xenova/whisper-medium',
};

export interface PipelineOptions {
  modelSize: WhisperModelSize;
  /** ISO code ("ru", "en", ...) or "auto". */
  language: string;
  engineConfig?: LocalEngineConfig;
}

export class OfflineTranscriptionEngine {
  private voiceDb: Map<string, SpeakerMetadata> = new Map();
  private transcriberInstance: any = null;
  private currentModelName = '';

  public resetVoiceDb() {
    this.voiceDb.clear();
  }

  public getSpeakers(): Record<string, SpeakerMetadata> {
    const res: Record<string, SpeakerMetadata> = {};
    this.voiceDb.forEach((val, key) => {
      res[key] = { ...val };
    });
    return res;
  }

  public updateSpeaker(id: string, updates: Partial<SpeakerMetadata>) {
    const existing = this.voiceDb.get(id);
    if (existing) {
      this.voiceDb.set(id, { ...existing, ...updates });
    }
  }

  public mergeSpeakers(sourceId: string, targetId: string) {
    if (sourceId === targetId) return;
    const source = this.voiceDb.get(sourceId);
    const target = this.voiceDb.get(targetId);
    if (source && target) {
      target.sampleCount += source.sampleCount;
    }
    this.voiceDb.delete(sourceId);
  }

  /**
   * Main pipeline: decode/normalise audio in the browser, then transcribe with the selected backend.
   * Failures are thrown, never papered over with placeholder text.
   */
  public async processMediaFile(
    file: File,
    options: PipelineOptions,
    onProgress: (progress: number, stage: string) => void
  ): Promise<ProcessedFile> {
    this.voiceDb.clear();

    onProgress(5, `Extracting and normalizing audio track (16kHz Mono)...`);
    const { blob, duration, buffer } = await convertMediaToWav(file, 16000, (p, msg) => {
      onProgress(Math.floor(p * 0.3), msg);
    });

    const backend = options.engineConfig?.backend || 'local-faster-whisper';
    onProgress(35, `Starting speech recognition (${backend})...`);

    const segments =
      backend === 'local-faster-whisper'
        ? await this.transcribeWithPythonServer(file, blob, duration, options, onProgress)
        : await this.transcribeInBrowser(buffer, duration, options, onProgress);

    onProgress(100, `Transcription completed: ${segments.length} segments.`);

    return {
      id: 'file_' + Date.now(),
      filename: file.name,
      fileSize: file.size,
      duration,
      audioBlob: blob,
      audioUrl: URL.createObjectURL(blob),
      segments,
      speakers: this.getSpeakers(),
      processedAt: new Date().toISOString(),
      language: options.language || 'auto',
      modelUsed: options.modelSize,
    };
  }

  // ---------------------------------------------------------------------------
  // Backend 1: local Python server (faster-whisper + speaker diarization)
  // ---------------------------------------------------------------------------
  private async transcribeWithPythonServer(
    file: File,
    blob: Blob,
    duration: number,
    options: PipelineOptions,
    onProgress: (progress: number, stage: string) => void
  ): Promise<TranscriptSegment[]> {
    const cfg = options.engineConfig;
    const serverUrl = (cfg?.localServerUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');

    const query = new URLSearchParams({
      model: options.modelSize,
      min_silence_ms: String(cfg?.minSilenceMs ?? 250),
      beam_size: String(cfg?.beamSize ?? 5),
      device: cfg?.device ?? 'auto',
      language: options.language || 'auto',
      max_speakers: String(cfg?.maxSpeakers ?? 3),
      exact_speakers: cfg?.exactSpeakers ? '1' : '0',
      filename: file.name,
    });
    onProgress(35, `Connecting to local faster-whisper server at ${serverUrl}...`);

    let resp: Response;
    try {
      resp = await fetch(`${serverUrl}/transcribe?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: blob,
      });
    } catch (networkErr: any) {
      throw new Error(
        `Local server is unreachable at ${serverUrl} (${networkErr?.message || 'connection refused'}). ` +
        `Start it with "python server_faster_whisper.py" (see README).`
      );
    }

    if (!resp.ok) {
      let errMessage = `HTTP ${resp.status}`;
      try {
        const errData = await resp.json();
        errMessage = errData.error || JSON.stringify(errData);
      } catch {
        /* keep HTTP status */
      }
      throw new Error(`Local server error (${resp.status}): ${errMessage}`);
    }

    let data: any;
    try {
      data = await resp.json();
    } catch (jsonErr: any) {
      throw new Error(`Invalid JSON response from local server: ${jsonErr.message}`);
    }
    if (data.status === 'error' || data.error) {
      throw new Error(`Local server error: ${data.error}`);
    }
    if (!Array.isArray(data.segments) || data.segments.length === 0) {
      throw new Error('No speech was detected. Check the audio, or lower "VAD Min Silence" in engine settings.');
    }

    onProgress(65, `Received ${data.segments.length} segments (${data.diarizer_type || 'unknown'} diarizer).`);

    if (data.speakers && typeof data.speakers === 'object') {
      Object.entries(data.speakers).forEach(([spkId, m]: [string, any]) => {
        this.voiceDb.set(spkId, {
          id: spkId,
          name: m.name || spkId,
          gender: 'Unknown',
          pitchF0: m.pitchF0 || undefined,
          sampleCount: m.segmentsCount || 1,
          color: m.color || SPEAKER_COLORS[this.voiceDb.size % SPEAKER_COLORS.length],
        });
      });
    }

    return data.segments.map((s: any, idx: number): TranscriptSegment => {
      const start = Math.max(0, Number(s.start || 0));
      const end = Math.min(duration, Math.max(start + 0.3, Number(s.end || duration)));
      const confidence = typeof s.confidence === 'number' ? s.confidence : undefined;
      return {
        id: `seg_${idx + 1}`,
        start,
        end,
        text: String(s.text || '').trim(),
        speakerId: s.speakerId || 'speaker_001',
        gender: 'Unknown',
        uncertain: confidence !== undefined && confidence < UNCERTAIN_CONFIDENCE,
        confidence,
        words: Array.isArray(s.words)
          ? s.words.map((w: any) => ({ word: w.word, start: w.start, end: w.end, confidence: w.probability }))
          : [],
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Backend 2: in-browser Whisper (transformers.js). No diarization.
  // ---------------------------------------------------------------------------
  private async getWhisperPipeline(
    modelSize: WhisperModelSize,
    engineConfig: LocalEngineConfig | undefined,
    onProgress: (progress: number, stage: string) => void
  ) {
    const modelId = BROWSER_MODELS[modelSize];
    if (!modelId) {
      throw new Error(
        `The in-browser backend does not support the "${modelSize}" model. ` +
        `Choose tiny/base/small/medium, or use the Faster-Whisper backend.`
      );
    }
    const blockRemote = engineConfig?.blockRemoteDownloads ?? true;

    if (!this.transcriberInstance || this.currentModelName !== modelId) {
      onProgress(35, `Loading in-browser Whisper [${modelId}]...`);
      const { pipeline } = await loadTransformersModule({
        allowRemoteModels: !blockRemote,
        localModelPath: engineConfig?.localModelPath,
      });
      try {
        this.transcriberInstance = await pipeline('automatic-speech-recognition', modelId, {
          progress_callback: (item: any) => {
            if (item.status === 'progress' && item.progress !== undefined) {
              const pct = Math.min(99, Math.round(item.progress));
              onProgress(35 + Math.floor(pct * 0.25), `Loading model weights (${item.file || ''}): ${pct}%`);
            }
          },
        });
        this.currentModelName = modelId;
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        throw new Error(
          `Could not load "${modelId}" (${msg}). ` +
          (blockRemote
            ? `Remote downloads are blocked, so the model files must exist under ${engineConfig?.localModelPath || '/models/'} ` +
              `(i.e. public/models/${modelId}/). Alternatively use the Faster-Whisper backend, which reads your Hugging Face cache.`
            : `Check your connection or place the model files locally.`)
        );
      }
    }
    return this.transcriberInstance;
  }

  private async transcribeInBrowser(
    buffer: AudioBuffer,
    duration: number,
    options: PipelineOptions,
    onProgress: (progress: number, stage: string) => void
  ): Promise<TranscriptSegment[]> {
    const transcriber = await this.getWhisperPipeline(options.modelSize, options.engineConfig, onProgress);

    const channelData = buffer.getChannelData(0); // already 16 kHz mono

    onProgress(60, `Running in-browser transcription...`);
    const lang =
      options.language && options.language !== 'auto'
        ? ({ ru: 'russian', en: 'english' } as Record<string, string>)[options.language] || options.language
        : null;

    const result = await transcriber(channelData, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: true,
      task: 'transcribe',
      language: lang,
    });

    const segments: TranscriptSegment[] = [];
    for (const chunk of result?.chunks ?? []) {
      const text = String(chunk.text || '').trim();
      if (!text) continue;
      const idx = segments.length + 1;
      const rawStart = Array.isArray(chunk.timestamp) ? Number(chunk.timestamp[0] ?? 0) : 0;
      const rawEnd = Array.isArray(chunk.timestamp) && chunk.timestamp[1] != null ? Number(chunk.timestamp[1]) : rawStart + 3;
      const start = Math.max(0, rawStart);
      segments.push({
        id: `seg_${idx}`,
        start,
        end: Math.min(duration, Math.max(start + 0.3, rawEnd)),
        text,
        speakerId: 'speaker_001',
        gender: 'Unknown',
        uncertain: false,
        words: [],
      });
    }

    if (segments.length === 0) {
      throw new Error('In-browser Whisper returned no text. Check the audio or try another model.');
    }

    // No diarization in this backend: everything is one speaker, and we say so.
    this.voiceDb.set('speaker_001', {
      id: 'speaker_001',
      name: 'Speaker 001',
      gender: 'Unknown',
      sampleCount: segments.length,
      color: SPEAKER_COLORS[0],
    });
    onProgress(70, `Note: speaker diarization is only available with the Faster-Whisper backend.`);
    return segments;
  }
}
