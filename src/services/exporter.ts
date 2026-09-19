import { ProcessedFile, SpeakerMetadata } from '../types';

export function formatTimeSeconds(seconds: number): string {
  const safeSec = Math.max(0, isFinite(seconds) ? Math.floor(seconds) : 0);
  const m = Math.floor(safeSec / 60);
  const s = safeSec % 60;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  return `${h.toString().padStart(2, '0')}:${remM.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export function formatSrtTime(seconds: number): string {
  const safeSec = Math.max(0, isFinite(seconds) ? seconds : 0);
  const totalSecInt = Math.floor(safeSec);
  const h = Math.floor(totalSecInt / 3600);
  const m = Math.floor((totalSecInt % 3600) / 60);
  const s = totalSecInt % 60;
  const ms = Math.min(999, Math.floor((safeSec - totalSecInt) * 1000));
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

/**
 * Exports to exact Markdown format from exporter.py and README.md
 */
export function generateMarkdownExport(file: ProcessedFile, speakers: Record<string, SpeakerMetadata>): string {
  const lines: string[] = [];
  lines.push(`# Transcript: ${file.filename}`);
  lines.push(`Date: ${file.processedAt.replace('T', ' ').substring(0, 19)}`);
  lines.push(`Total Segments: ${file.segments.length}`);
  lines.push(`Whisper Engine: ${file.modelUsed.toUpperCase()}`);
  lines.push(`Speakers Identified: ${Object.keys(speakers).length}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  // Speaker Summary
  lines.push('### Speaker Roster');
  Object.values(speakers).forEach((spk) => {
    const f0Info = spk.pitchF0 ? ` (F0: ~${spk.pitchF0}Hz)` : '';
    lines.push(`- **${spk.name}** [${spk.gender}]${f0Info} - Confidence: ${(spk.confidence * 100).toFixed(0)}%`);
  });
  lines.push('');
  lines.push('---');
  lines.push('');

  // Segments
  lines.push('### Dialogue Transcript');
  lines.push('');
  file.segments.forEach((seg) => {
    const timeStr = formatTimeSeconds(seg.start);
    const spk = speakers[seg.speakerId];
    const spkName = spk ? spk.name : seg.speakerId;
    lines.push(`**${timeStr}** (${spkName}): ${seg.text}`);
    lines.push('');
  });

  // Semantic Clusters
  if (file.clusters && file.clusters.length > 0) {
    lines.push('---');
    lines.push('');
    lines.push('### Semantic Clusters');
    lines.push('');
    file.clusters.forEach((c) => {
      lines.push(`#### Block ${c.clusterId}: ${c.topic}`);
      lines.push(`*${c.summary}*`);
      lines.push('');
    });
  }

  return lines.join('\n');
}

export function generateSrtExport(file: ProcessedFile, speakers: Record<string, SpeakerMetadata>): string {
  const blocks: string[] = [];
  file.segments.forEach((seg, idx) => {
    const spk = speakers[seg.speakerId];
    const spkName = spk ? spk.name : seg.speakerId;
    blocks.push(`${idx + 1}`);
    blocks.push(`${formatSrtTime(seg.start)} --> ${formatSrtTime(seg.end)}`);
    blocks.push(`[${spkName}] ${seg.text}`);
    blocks.push('');
  });
  return blocks.join('\n');
}

export function generateJsonExport(file: ProcessedFile, speakers: Record<string, SpeakerMetadata>): string {
  return JSON.stringify(
    {
      file: {
        name: file.filename,
        duration: file.duration,
        processedAt: file.processedAt,
        model: file.modelUsed,
      },
      speakers,
      segments: file.segments,
      clusters: file.clusters,
    },
    null,
    2
  );
}

export function generatePlainTextExport(file: ProcessedFile, speakers: Record<string, SpeakerMetadata>): string {
  return file.segments
    .map((seg) => {
      const spk = speakers[seg.speakerId];
      const spkName = spk ? spk.name : seg.speakerId;
      return `[${formatTimeSeconds(seg.start)}] ${spkName}: ${seg.text}`;
    })
    .join('\n\n');
}

export function downloadTextFile(content: string, filename: string, mimeType = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
