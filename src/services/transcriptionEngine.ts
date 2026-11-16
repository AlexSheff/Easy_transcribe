import {
  ProcessedFile,
  TranscriptSegment,
  SpeakerMetadata,
  WhisperModelSize,
  LocalEngineConfig,
  SavedTranscriptInfo,
} from '../types';
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
  force?: boolean;
  signal?: AbortSignal;
  onJobStart?: (cancelFn: () => void) => void;
}

export interface ExistingTranscriptMatch {
  exists: boolean;
  filename?: string;
  relativePath?: string;
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
   * Queries the local Python server to see if transcripts for the given filenames
   * already exist in outputs/ or transcripts/. Fails soft ({}) if server is offline.
   */
  public async checkExistingTranscripts(
    filenames: string[],
    serverUrl?: string
  ): Promise<Record<string, ExistingTranscriptMatch>> {
    if (!filenames.length) return {};
    const baseUrl = (serverUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);
    try {
      const resp = await fetch(`${baseUrl}/api/check-transcripts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames }),
        signal: controller.signal,
      });
      if (!resp.ok) return {};
      const data = await resp.json();
      return (data && typeof data.results === 'object' && data.results) || {};
    } catch {
      return {};
    } finally {
      clearTimeout(timeoutId);
    }
  }

  public async cancelJob(jobId: string, serverUrl?: string): Promise<void> {
    const baseUrl = (serverUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    try {
      await fetch(`${baseUrl}/api/cancel?job_id=${encodeURIComponent(jobId)}`, {
        method: 'POST',
      });
    } catch {
      /* server might already be closed or offline */
    }
  }

  public async listSavedTranscripts(serverUrl?: string): Promise<SavedTranscriptInfo[]> {
    const baseUrl = (serverUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    const resp = await fetch(`${baseUrl}/api/transcripts`);
    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }
    const data = await resp.json();
    return Array.isArray(data?.items) ? data.items : [];
  }

  public async loadSavedTranscript(
    relativePath: string,
    serverUrl?: string
  ): Promise<{ filename: string; relativePath: string; content: string }> {
    const baseUrl = (serverUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    const resp = await fetch(`${baseUrl}/api/transcript?path=${encodeURIComponent(relativePath)}`);
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try {
        const err = await resp.json();
        if (err?.error) msg = err.error;
      } catch {
        /* ignore */
      }
      throw new Error(`Could not load transcript (${msg})`);
    }
    return resp.json();
  }

  /**
   * Main pipeline: streams raw media directly to faster-whisper (no slow browser decoding),
   * or decodes in browser when running the in-browser transformers.js backend.
   */
  public async processMediaFile(
    file: File,
    options: PipelineOptions,
    onProgress: (progress: number, stage: string) => void
  ): Promise<ProcessedFile> {
    this.voiceDb.clear();
    if (options.signal?.aborted) {
      throw new Error('CANCELLED');
    }

    const backend = options.engineConfig?.backend || 'local-faster-whisper';

    let segments: TranscriptSegment[];
    let duration = 0;
    let audioBlob: Blob = file;
    let diarizerType: string | undefined;

    if (backend === 'local-faster-whisper') {
      onProgress(10, `Uploading ${file.name} directly to local engine...`);
      const serverResult = await this.transcribeWithPythonServer(file, options, onProgress);
      segments = serverResult.segments;
      duration = serverResult.duration;
      diarizerType = serverResult.diarizerType;
    } else {
      onProgress(5, `Extracting and normalizing audio track (16kHz Mono)...`);
      const converted = await convertMediaToWav(file, 16000, (p, msg) => {
        onProgress(Math.floor(p * 0.3), msg);
      });
      if (options.signal?.aborted) {
        throw new Error('CANCELLED');
      }
      audioBlob = converted.blob;
      duration = converted.duration;
      onProgress(35, `Starting speech recognition (${backend})...`);
      segments = await this.transcribeInBrowser(converted.buffer, duration, options, onProgress);
    }

    onProgress(100, `Transcription completed: ${segments.length} segments.`);

    return {
      id: `file_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      filename: file.name,
      fileSize: file.size,
      duration,
      audioBlob,
      audioUrl: URL.createObjectURL(audioBlob),
      segments,
      speakers: this.getSpeakers(),
      processedAt: new Date().toISOString(),
      language: options.language || 'auto',
      modelUsed: options.modelSize,
      diarizerType,
    };
  }

  // ---------------------------------------------------------------------------
  // Backend 1: local Python server (faster-whisper + speaker diarization)
  // ---------------------------------------------------------------------------
  private async transcribeWithPythonServer(
    file: File,
    options: PipelineOptions,
    onProgress: (progress: number, stage: string) => void
  ): Promise<{ segments: TranscriptSegment[]; duration: number; diarizerType?: string }> {
    const cfg = options.engineConfig;
    const serverUrl = (cfg?.localServerUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
    const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const query = new URLSearchParams({
      model: options.modelSize,
      min_silence_ms: String(cfg?.minSilenceMs ?? 250),
      vad_threshold: String(cfg?.vadThreshold ?? 0.35),
      beam_size: String(cfg?.beamSize ?? 5),
      device: cfg?.device ?? 'auto',
      compute_type: cfg?.computeType ?? 'auto',
      language: options.language || 'auto',
      max_speakers: String(cfg?.maxSpeakers ?? 3),
      exact_speakers: cfg?.exactSpeakers ? '1' : '0',
      min_silhouette: String(cfg?.minSilhouette ?? 0.08),
      skip_existing: !options.force && (cfg?.skipExisting ?? true) ? '1' : '0',
      use_batched: (cfg?.useBatched ?? true) ? '1' : '0',
      normalize_audio: (cfg?.normalizeAudio ?? true) ? '1' : '0',
      word_timestamps: (cfg?.wordTimestamps ?? true) ? '1' : '0',
      filename: file.name,
      job_id: jobId,
    });
    if (cfg?.initialPrompt?.trim()) {
      query.set('initial_prompt', cfg.initialPrompt.trim());
    }
    if (cfg?.hotwords?.trim()) {
      query.set('hotwords', cfg.hotwords.trim());
    }

    options.onJobStart?.(() => {
      void this.cancelJob(jobId, serverUrl);
    });

    onProgress(12, `Connecting to local faster-whisper server at ${serverUrl}...`);

    let lastPolledPct = 12;
    const pollTimer = setInterval(async () => {
      if (options.signal?.aborted) return;
      try {
        const pRes = await fetch(`${serverUrl}/api/progress?job_id=${encodeURIComponent(jobId)}`);
        if (!pRes.ok) return;
        const pData = await pRes.json();
        if (pData?.active && typeof pData.progress === 'number' && pData.progress >= lastPolledPct) {
          lastPolledPct = pData.progress;
          onProgress(pData.progress, pData.stage || 'Processing...');
        }
      } catch {
        /* ignore transient polling failures */
      }
    }, 450);

    let resp: Response;
    try {
      resp = await fetch(`${serverUrl}/transcribe?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': file.type || 'application/octet-stream' },
        body: file,
        signal: options.signal,
      });
    } catch (networkErr: any) {
      clearInterval(pollTimer);
      if (networkErr?.name === 'AbortError' || options.signal?.aborted) {
        throw new Error('CANCELLED');
      }
      throw new Error(
        `Local server is unreachable at ${serverUrl} (${networkErr?.message || 'connection refused'}). ` +
        `Start it with "python server_faster_whisper.py" (see README).`
      );
    } finally {
      clearInterval(pollTimer);
    }

    if (resp.status === 499) {
      throw new Error('CANCELLED');
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
    if (data.status === 'cancelled') {
      throw new Error('CANCELLED');
    }
    if (data.status === 'error' || data.error) {
      throw new Error(`Local server error: ${data.error}`);
    }
    if (data.status === 'skipped' || data.skipped) {
      throw new Error(`SKIPPED:${data.existing_path || data.filename || 'outputs/'}`);
    }
    if (!Array.isArray(data.segments) || data.segments.length === 0) {
      throw new Error('No speech was detected. Check the audio, or lower "VAD Threshold" / "VAD Min Silence" in engine settings.');
    }

    const lastEnd = data.segments.reduce((m: number, s: any) => Math.max(m, Number(s.end || 0)), 0);
    const duration = Math.max(Number(data.duration || 0), lastEnd);

    onProgress(98, `Received ${data.segments.length} segments (${data.diarizer_type || 'unknown'} diarizer).`);

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

    const segments = data.segments.map((s: any, idx: number): TranscriptSegment => {
      const start = Math.max(0, Number(s.start || 0));
      const end = Math.min(Math.max(duration, start + 0.3), Math.max(start + 0.3, Number(s.end || start + 0.3)));
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

    return {
      segments,
      duration,
      diarizerType: typeof data.diarizer_type === 'string' ? data.diarizer_type : undefined,
    };
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
