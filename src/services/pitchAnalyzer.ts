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

  const startSample = Math.floor(startSec * sampleRate);
  const endSample = Math.min(channelData.length, Math.floor(endSec * sampleRate));
  const segmentLength = endSample - startSample;

  // Need at least 0.2s of audio
  if (segmentLength < sampleRate * 0.2) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  // Slice segment (limit to max 3 seconds for speed)
  const maxAnalyzeSamples = Math.min(segmentLength, sampleRate * 3);
  const buffer = new Float32Array(maxAnalyzeSamples);
  for (let i = 0; i < maxAnalyzeSamples; i++) {
    buffer[i] = channelData[startSample + i];
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
  const minPeriod = Math.floor(sampleRate / 350); // ~126 samples at 44.1k, ~45 at 16k
  const maxPeriod = Math.floor(sampleRate / 60);  // ~735 samples at 44.1k, ~266 at 16k

  let bestPeriod = 0;
  let maxCorr = -1;

  // Center clipped autocorrelation
  const clipThreshold = 0.2;
  const clipped = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    if (buffer[i] > clipThreshold) clipped[i] = buffer[i] - clipThreshold;
    else if (buffer[i] < -clipThreshold) clipped[i] = buffer[i] + clipThreshold;
    else clipped[i] = 0;
  }

  const windowSize = Math.min(clipped.length - maxPeriod, Math.floor(sampleRate * 0.05));
  if (windowSize <= 0) {
    return { gender: 'Unknown', pitchF0: 0 };
  }

  for (let period = minPeriod; period <= maxPeriod; period++) {
    let sum = 0;
    for (let i = 0; i < windowSize; i++) {
      sum += clipped[i] * clipped[i + period];
    }
    if (sum > maxCorr) {
      maxCorr = sum;
      bestPeriod = period;
    }
  }

  if (bestPeriod === 0 || maxCorr <= 0.01) {
    // Default fallback
    return { gender: 'Unknown', pitchF0: 0 };
  }

  const pitchF0 = Math.round(sampleRate / bestPeriod);

  // Boundary check
  if (pitchF0 >= 60 && pitchF0 <= 320) {
    const gender: GenderType = pitchF0 < 165 ? 'Male' : 'Female';
    return { gender, pitchF0 };
  }

  return { gender: 'Unknown', pitchF0: 0 };
}
