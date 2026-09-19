/**
 * Client-Side Audio/Video to 16kHz Mono WAV Converter.
 * Replicates audio_processor.py's FFmpeg pipeline:
 * 'ffmpeg -y -i input -ar 16000 -ac 1 -c:a pcm_s16le output.wav'
 * Works 100% offline in browser using Web Audio API.
 */

/**
 * Client-Side Audio/Video to 16kHz Mono WAV Converter.
 * Replicates audio_processor.py's FFmpeg pipeline:
 * 'ffmpeg -y -i input -ar 16000 -ac 1 -c:a pcm_s16le output.wav'
 * Works 100% offline in browser using Web Audio API.
 */

async function readMediaFileBuffer(
  file: File,
  onProgress?: (progress: number, stage: string) => void
): Promise<ArrayBuffer> {
  onProgress?.(5, `Reading binary stream (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`);

  // Attempt 1: Direct modern arrayBuffer()
  try {
    return await file.arrayBuffer();
  } catch (err1) {
    // Attempt 2: Slice to obtain a fresh Blob reference
    try {
      return await file.slice(0, file.size).arrayBuffer();
    } catch {
      // Attempt 3: Classic FileReader
      try {
        return await new Promise<ArrayBuffer>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            if (reader.result instanceof ArrayBuffer) {
              resolve(reader.result);
            } else {
              reject(new Error('FileReader returned empty result'));
            }
          };
          reader.onerror = () => reject(reader.error || new Error('FileReader failed'));
          reader.readAsArrayBuffer(file);
        });
      } catch {
        // Attempt 4: Brief 600ms delay retry in case Windows/OBS is finishing writing the file
        onProgress?.(8, 'File is busy in Windows. Retrying in 600ms...');
        await new Promise((r) => setTimeout(r, 600));

        try {
          return await file.slice(0, file.size).arrayBuffer();
        } catch (finalErr: unknown) {
          const errMsg = finalErr instanceof Error ? finalErr.message : String(finalErr);
          const isLocked =
            errMsg.includes('permission') ||
            errMsg.includes('could not be read') ||
            errMsg.includes('NotReadableError') ||
            (err1 instanceof Error && err1.name === 'NotReadableError');

          if (isLocked) {
            throw new Error(
              `Файл «${file.name}» заблокирован другой программой или системой Windows (NotReadableError).\n\n` +
              `Как это быстро исправить:\n` +
              `1. Закройте программу, которая создала или открыла файл (OBS Studio, плеер "Кино и ТВ", VLC, видеоредактор).\n` +
              `2. Если в OBS всё ещё идёт запись — нажмите "Остановить запись".\n` +
              `3. Самый быстрый способ: скопируйте этот файл в проводнике (Ctrl+C, затем Ctrl+V) и перетащите созданную копию файла в программу.`
            );
          }
          throw finalErr;
        }
      }
    }
  }
}

export async function convertMediaToWav(
  file: File,
  targetSampleRate = 16000,
  onProgress?: (progress: number, stage: string) => void
): Promise<{ blob: Blob; duration: number; buffer: AudioBuffer }> {
  const arrayBuffer = await readMediaFileBuffer(file, onProgress);

  onProgress?.(30, 'Decoding multi-format audio track (Web Audio API)...');
  const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const offlineCtx = new AudioContextClass();
  
  let audioBuffer: AudioBuffer;
  try {
    audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
  } catch {
    await offlineCtx.close().catch(() => {});
    throw new Error(`Failed to decode audio track from «${file.name}». Убедитесь, что видео/аудио файл содержит звуковую дорожку.`);
  }

  try {
    onProgress?.(60, 'Resampling to 16kHz Mono PCM...');
    const duration = audioBuffer.duration;
    const numChannels = 1; // Downmix to Mono as required by Whisper & Diarization
    const totalSamples = Math.max(1, Math.floor(duration * targetSampleRate));

    // Render via OfflineAudioContext for high quality resampled mono
    const resampleCtx = new OfflineAudioContext(numChannels, totalSamples, targetSampleRate);
    const source = resampleCtx.createBufferSource();
    source.buffer = audioBuffer;

    // Web Audio automatically downmixes multi-channel audio to mono destination (0.5*L + 0.5*R)
    source.connect(resampleCtx.destination);
    source.start(0);
    const renderedBuffer = await resampleCtx.startRendering();

    onProgress?.(85, 'Encoding 16-bit PCM WAV container...');
    const wavBlob = encodeWAV(renderedBuffer, targetSampleRate);
    
    onProgress?.(100, 'Conversion complete');

    return {
      blob: wavBlob,
      duration,
      buffer: renderedBuffer
    };
  } finally {
    await offlineCtx.close().catch(() => {});
  }
}

/**
 * Encodes an AudioBuffer into a standard 16-bit PCM RIFF WAV Blob.
 */
function encodeWAV(buffer: AudioBuffer, sampleRate: number): Blob {
  const channelData = buffer.getChannelData(0);
  const bytesPerSample = 2; // 16-bit
  const blockAlign = 1 * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = channelData.length * bytesPerSample;
  const bufferSize = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(bufferSize);
  const view = new DataView(arrayBuffer);

  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // FMT sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // SubChunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, 1, true); // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // BitsPerSample

  // DATA sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM samples (clamp float -1..1 to int16 -32768..32767)
  let offset = 44;
  for (let i = 0; i < channelData.length; i++) {
    const raw = channelData[i];
    const s = Number.isFinite(raw) ? Math.max(-1, Math.min(1, raw)) : 0;
    const sample = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, sample, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
