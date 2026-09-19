import { GenderType } from '../types';

/**
 * Offline pitch (F0) analysis inspired by voice_fingerprint.py's Welch / autocorrelation methods.
 * Threshold: 165.0 Hz.
 * Dominant F0 < 165 Hz -> Male
 * Dominant F0 >= 165 Hz -> Female
 */
export function estimateSegmentPitch(
  audioBuffer: AudioBuffer,
  startSec: number,
  endSec: number
): { gender: GenderType; pitchF0: number } {
  const sampleRate = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);

  const startSample = Math.max(0, Math.floor(startSec * sampleRate));
  const endSample = Math.min(channelData.length, Math.floor(endSec * sampleRate));
  const segmentLength = endSample - startSample;

  // Need at least 0.2s of audio
  if (segmentLength < sampleRate * 0.2) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  // Slice segment (limit to max 3 seconds around the middle of the segment for cleaner pitch detection)
  const maxAnalyzeSamples = Math.min(segmentLength, sampleRate * 3);
  const offsetWithinSegment = Math.floor((segmentLength - maxAnalyzeSamples) / 2);
  const buffer = new Float32Array(maxAnalyzeSamples);
  
  let sumSquares = 0;
  for (let i = 0; i < maxAnalyzeSamples; i++) {
    const val = channelData[startSample + offsetWithinSegment + i];
    buffer[i] = val;
    sumSquares += val * val;
  }

  // Energy gate: check RMS
  const rms = Math.sqrt(sumSquares / maxAnalyzeSamples);
  if (rms < 0.005) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < buffer.length; i++) {
    const abs = Math.abs(buffer[i]);
    if (abs > maxVal) maxVal = abs;
  }
  if (maxVal === 0) {
    return { gender: 'Unknown', pitchF0: 0 };
  }
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] /= maxVal;
  }

  // Autocorrelation within human voice range (60 Hz to 350 Hz)
  const minPeriod = Math.floor(sampleRate / 350); // ~45 samples at 16k
  const maxPeriod = Math.floor(sampleRate / 60);  // ~266 samples at 16k

  let bestPeriod = 0;
  let maxCorr = -1;

  // Center-clipped autocorrelation to suppress formant harmonics
  const clipThreshold = 0.2;
  const clipped = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    if (buffer[i] > clipThreshold) clipped[i] = buffer[i] - clipThreshold;
    else if (buffer[i] < -clipThreshold) clipped[i] = buffer[i] + clipThreshold;
    else clipped[i] = 0;
  }

  const windowSize = Math.min(clipped.length - maxPeriod, Math.floor(sampleRate * 0.06));
  if (windowSize <= 0) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  const correlations = new Float32Array(maxPeriod + 2);

  for (let period = minPeriod; period <= maxPeriod; period++) {
    let sum = 0;
    for (let i = 0; i < windowSize; i++) {
      sum += clipped[i] * clipped[i + period];
    }
    correlations[period] = sum;
    if (sum > maxCorr) {
      maxCorr = sum;
      bestPeriod = period;
    }
  }

  if (bestPeriod <= minPeriod || bestPeriod >= maxPeriod || maxCorr <= 0.01) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  // Parabolic interpolation for sub-sample accuracy
  const y1 = correlations[bestPeriod - 1] || maxCorr;
  const y2 = maxCorr;
  const y3 = correlations[bestPeriod + 1] || maxCorr;
  const denom = (2 * y2 - y1 - y3);
  const delta = denom !== 0 ? (0.5 * (y1 - y3)) / denom : 0;
  const refinedPeriod = Math.max(minPeriod, Math.min(maxPeriod, bestPeriod + delta));

  const pitchF0 = Math.round(sampleRate / refinedPeriod);

  // Biological vocal range validation
  if (pitchF0 >= 65 && pitchF0 <= 330) {
    const gender: GenderType = pitchF0 < 165 ? 'Male' : 'Female';
    return { gender, pitchF0 };
  }

  return { gender: 'Unknown', pitchF0: 0 };
}
