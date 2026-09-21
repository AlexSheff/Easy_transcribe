import { ProcessedFile, SpeakerMetadata } from '../types';

export type MarkdownPreset = 'standard' | 'obsidian' | 'meeting' | 'clean' | 'timestamps';

export interface MarkdownExportOptions {
  preset?: MarkdownPreset;
  includePitch?: boolean;
  includeTimestamps?: boolean;
  includeFrontmatter?: boolean;
}

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
 * Generates high-quality, professional Markdown formatted for Obsidian, Notion, or text editors
 */
export function generateMarkdownExport(
  file: ProcessedFile,
  speakers: Record<string, SpeakerMetadata>,
  options: MarkdownExportOptions = {}
): string {
  const preset = options.preset || 'standard';
  const includePitch = options.includePitch !== false;
  const includeTimestamps = options.includeTimestamps !== false;
  const lines: string[] = [];

  const baseName = file.filename.replace(/\.[^/.]+$/, '');
  const cleanDate = file.processedAt ? file.processedAt.replace('T', ' ').substring(0, 19) : new Date().toISOString().replace('T', ' ').substring(0, 19);
  const totalDurationStr = formatTimeSeconds(file.duration);

  // Compute speaker statistics (speaking time, segment count)
  const speakerStats: Record<string, { seconds: number; segments: number }> = {};
  file.segments.forEach((seg) => {
    if (!speakerStats[seg.speakerId]) {
      speakerStats[seg.speakerId] = { seconds: 0, segments: 0 };
    }
    speakerStats[seg.speakerId].seconds += Math.max(0, seg.end - seg.start);
    speakerStats[seg.speakerId].segments += 1;
  });

  // Sort speakers by segment count descending (dominant speaker first)
  const sortedSpeakers = Object.values(speakers).sort((a, b) => {
    const countA = speakerStats[a.id]?.segments || a.sampleCount || 0;
    const countB = speakerStats[b.id]?.segments || b.sampleCount || 0;
    return countB - countA;
  });

  // --- PRESET: STANDARD (Clean Title, Date, Total Segments, Speaker Breakdown) ---
  if (preset === 'standard') {
    lines.push(`# Transcript: ${file.filename}`);
    lines.push(`Date: ${cleanDate}`);
    lines.push(`Total Segments: ${file.segments.length}`);
    lines.push('');
    lines.push('## Speakers');
    lines.push('');
    sortedSpeakers.forEach((spk) => {
      const stats = speakerStats[spk.id] || { seconds: 0, segments: spk.sampleCount || 0 };
      lines.push(`- ${spk.name}: ${stats.segments} segment(s)`);
    });
    lines.push('');
    lines.push('## Transcript');
    lines.push('');
    file.segments.forEach((seg) => {
      const spk = speakers[seg.speakerId];
      const spkName = spk ? spk.name : seg.speakerId;
      const t = formatTimeSeconds(seg.start);
      lines.push(`**[${t}] ${spkName}:** ${seg.text}`);
      lines.push('');
    });
    return lines.join('\n').trim();
  }

  // --- PRESET: CLEAN DIALOGUE ---
  if (preset === 'clean') {
    lines.push(`# ${baseName}`);
    lines.push('');
    file.segments.forEach((seg) => {
      const spk = speakers[seg.speakerId];
      const spkName = spk ? spk.name : seg.speakerId;
      if (includeTimestamps) {
        lines.push(`**${spkName}** [${formatTimeSeconds(seg.start)}]: ${seg.text}`);
      } else {
        lines.push(`**${spkName}**: ${seg.text}`);
      }
      lines.push('');
    });
    return lines.join('\n').trim();
  }

  // --- PRESET: TIMECODED SCRIPT ---
  if (preset === 'timestamps') {
    lines.push(`# ${file.filename} (Timecoded Transcript)`);
    lines.push(`- **Date:** ${cleanDate}`);
    lines.push(`- **Total Duration:** ${totalDurationStr}`);
    lines.push('');
    lines.push('---');
    lines.push('');
    file.segments.forEach((seg, idx) => {
      const spk = speakers[seg.speakerId];
      const spkName = spk ? spk.name : seg.speakerId;
      const tStart = formatTimeSeconds(seg.start);
      const tEnd = formatTimeSeconds(seg.end);
      lines.push(`\`[${tStart} - ${tEnd}]\` **${spkName}** (#${idx + 1}):`);
      lines.push(`> ${seg.text}`);
      lines.push('');
    });
    return lines.join('\n').trim();
  }

  // --- PRESET: OBSIDIAN / SECOND BRAIN (Default) ---
  if (preset === 'obsidian') {
    lines.push('---');
    lines.push(`title: "${baseName}"`);
    lines.push(`date: ${cleanDate.substring(0, 10)}`);
    lines.push(`time: "${cleanDate.substring(11)}"`);
    lines.push(`file_source: "${file.filename}"`);
    lines.push(`duration: "${totalDurationStr}"`);
    lines.push(`model: "${file.modelUsed}"`);
    lines.push(`speakers_count: ${Object.keys(speakers).length}`);
    lines.push(`tags:`);
    lines.push(`  - transcript`);
    lines.push(`  - audio_note`);
    lines.push(`  - speech_to_text`);
    lines.push('---');
    lines.push('');
    lines.push(`# 🎙️ ${baseName}`);
    lines.push('');
    lines.push(`> [!info] Recording Overview`);
    lines.push(`> - **File:** \`${file.filename}\``);
    lines.push(`> - **Duration:** \`${totalDurationStr}\` | **Segments:** ${file.segments.length}`);
    lines.push(`> - **Engine:** \`Whisper ${file.modelUsed.toUpperCase()}\``);
    lines.push(`> - **Processed:** ${cleanDate}`);
    lines.push('');
    lines.push('## 👥 Speakers');
    lines.push('');
    lines.push('| Speaker | Speech Time | Share | Voice Pitch (F0) |');
    lines.push('|:---|:---|:---|:---|');
    Object.values(speakers).forEach((spk) => {
      const stats = speakerStats[spk.id] || { seconds: 0, segments: 0 };
      const dur = formatTimeSeconds(stats.seconds);
      const share = file.duration > 0 ? ((stats.seconds / file.duration) * 100).toFixed(1) : '0.0';
      const f0 = spk.pitchF0 ? `~${spk.pitchF0} Hz` : '—';
      lines.push(`| **${spk.name}** | \`${dur}\` | ${share}% | ${f0} |`);
    });
    lines.push('');


    lines.push('## 💬 Transcript');
    lines.push('');
    file.segments.forEach((seg) => {
      const spk = speakers[seg.speakerId];
      const spkName = spk ? spk.name : seg.speakerId;
      const t = formatTimeSeconds(seg.start);
      lines.push(`**\`${t}\`** **${spkName}**:`);
      lines.push(`${seg.text}`);
      lines.push('');
    });

    return lines.join('\n').trim();
  }

  // --- PRESET: MEETING PROTOCOL (Corporate / Formal) ---
  lines.push(`# Meeting Transcript: ${baseName}`);
  lines.push('');
  lines.push(`**Source:** ${file.filename}  `);
  lines.push(`**Generated:** ${cleanDate}  `);
  lines.push(`**Total Duration:** ${totalDurationStr}  `);
  lines.push(`**Engine:** Whisper [${file.modelUsed.toUpperCase()}]  `);
  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('### 1. Participants');
  Object.values(speakers).forEach((spk) => {
    const stats = speakerStats[spk.id] || { seconds: 0, segments: 0 };
    const dur = formatTimeSeconds(stats.seconds);
    const share = file.duration > 0 ? ((stats.seconds / file.duration) * 100).toFixed(1) : '0';
    const f0Info = includePitch && spk.pitchF0 ? ` (F0: ${spk.pitchF0} Hz)` : '';
    lines.push(`- **${spk.name}**${f0Info} — ${dur} speech (${share}% share, ${stats.segments} utterances)`);
  });
  lines.push('');

  lines.push('---');
  lines.push('');
  lines.push('### 2. Full Verbatim Transcript');
  lines.push('');
  file.segments.forEach((seg) => {
    const spk = speakers[seg.speakerId];
    const spkName = spk ? spk.name : seg.speakerId;
    const timeStr = formatTimeSeconds(seg.start);
    lines.push(`**[${timeStr}] ${spkName}:** ${seg.text}`);
    lines.push('');
  });

  return lines.join('\n').trim();
}

/**
 * Generates an all-in-one Master Markdown report combining multiple transcribed files
 */
export function generateMasterMarkdownExport(
  files: ProcessedFile[],
  speakersMap?: Record<string, SpeakerMetadata>
): string {
  const lines: string[] = [];
  const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 19);

  lines.push('---');
  lines.push(`title: "Batch Master Transcription Report (${files.length} files)"`);
  lines.push(`date: ${dateStr.substring(0, 10)}`);
  lines.push(`total_files: ${files.length}`);
  lines.push('tags: [transcript_collection, batch_transcription, master_report]');
  lines.push('---');
  lines.push('');
  lines.push(`# 📚 Batch Transcription Master Report: ${files.length} file(s)`);
  lines.push(`*Generated: ${dateStr}*`);
  lines.push('');
  lines.push('> [!summary] Batch Summary');
  const totalSec = files.reduce((acc, f) => acc + (f.duration || 0), 0);
  const totalSegs = files.reduce((acc, f) => acc + f.segments.length, 0);
  lines.push(`> - **Total Audio/Video Files:** ${files.length}`);
  lines.push(`> - **Total Cumulative Duration:** \`${formatTimeSeconds(totalSec)}\``);
  lines.push(`> - **Total Utterances / Segments:** ${totalSegs}`);
  lines.push('');

  lines.push('## 📑 Table of Contents');
  lines.push('');
  files.forEach((f, idx) => {
    const anchor = f.filename.toLowerCase().replace(/[^a-z0-9]/gi, '-');
    lines.push(`${idx + 1}. [${f.filename}](#${anchor}) — \`${formatTimeSeconds(f.duration)}\` (${f.segments.length} segments)`);
  });
  lines.push('');
  lines.push('---');
  lines.push('');

  files.forEach((f, idx) => {
    const spks = f.speakers || speakersMap || {};
    const singleMd = generateMarkdownExport(f, spks, { preset: 'obsidian' });
    lines.push(`## Document ${idx + 1}: ${f.filename}`);
    lines.push('');
    lines.push(singleMd);
    lines.push('');
    lines.push('---');
    lines.push('');
  });

  return lines.join('\n').trim();
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
