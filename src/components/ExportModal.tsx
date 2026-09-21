import React, { useState } from 'react';
import { ProcessedFile, SpeakerMetadata } from '../types';
import { 
  generateMarkdownExport, 
  generateMasterMarkdownExport,
  generateJsonExport, 
  generateSrtExport, 
  generatePlainTextExport, 
  downloadTextFile,
  MarkdownPreset
} from '../services/exporter';
import { X, Download, Copy, Check, FileText, Code, AlignLeft, Subtitles, Layers, FileCode } from 'lucide-react';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  file: ProcessedFile;
  speakers: Record<string, SpeakerMetadata>;
  allFiles?: ProcessedFile[];
}

type ExportFormat = 'md' | 'srt' | 'json' | 'txt';

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  file,
  speakers,
  allFiles = [],
}) => {
  const [format, setFormat] = useState<ExportFormat>('md');
  const [mdPreset, setMdPreset] = useState<MarkdownPreset>('standard');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  let content = '';
  let filename = '';
  let mimeType = 'text/plain';

  const baseName = file.filename.replace(/\.[^/.]+$/, '');

  switch (format) {
    case 'md':
      content = generateMarkdownExport(file, speakers, { preset: mdPreset });
      filename = `transcript_${baseName}.md`;
      mimeType = 'text/markdown';
      break;
    case 'srt':
      content = generateSrtExport(file, speakers);
      filename = `${baseName}.srt`;
      mimeType = 'text/plain';
      break;
    case 'json':
      content = generateJsonExport(file, speakers);
      filename = `transcript_${baseName}.json`;
      mimeType = 'application/json';
      break;
    case 'txt':
      content = generatePlainTextExport(file, speakers);
      filename = `transcript_${baseName}.txt`;
      mimeType = 'text/plain';
      break;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    downloadTextFile(content, filename, mimeType);
  };

  const handleDownloadAllMd = () => {
    if (!allFiles || allFiles.length === 0) return;
    allFiles.forEach((f, idx) => {
      setTimeout(() => {
        const md = generateMarkdownExport(f, f.speakers || speakers, { preset: mdPreset });
        const bName = f.filename.replace(/\.[^/.]+$/, '');
        downloadTextFile(md, `transcript_${bName}.md`, 'text/markdown');
      }, idx * 250);
    });
  };

  const handleDownloadMasterMd = () => {
    if (!allFiles || allFiles.length === 0) return;
    const masterMd = generateMasterMarkdownExport(allFiles, speakers);
    const dateStr = new Date().toISOString().substring(0, 10);
    downloadTextFile(masterMd, `master_transcription_${dateStr}.md`, 'text/markdown');
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl w-full max-w-4xl overflow-hidden shadow-2xl flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="px-6 py-4 bg-[#111111] border-b border-[#222222] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Download className="w-5 h-5 text-[#00ffcc]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">EXPORT TRANSCRIPT / MARKDOWN (.MD)</h2>
              <p className="text-xs text-[#888888]">
                Export professional markdown for Obsidian, Notion, team wikis, and reports
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#888888] hover:text-white p-1 rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Format Selector Bar */}
        <div className="px-6 py-3 bg-[#161616] border-b border-[#222222] flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setFormat('md')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                format === 'md' ? 'bg-[#00ffcc] text-black shadow-[0_0_10px_rgba(0,255,204,0.3)]' : 'bg-[#222] text-[#888] hover:text-white'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>MARKDOWN (.MD)</span>
            </button>

            <button
              onClick={() => setFormat('srt')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                format === 'srt' ? 'bg-[#00ffcc] text-black' : 'bg-[#222] text-[#888] hover:text-white'
              }`}
            >
              <Subtitles className="w-3.5 h-3.5" />
              <span>SUBTITLES (.SRT)</span>
            </button>

            <button
              onClick={() => setFormat('json')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                format === 'json' ? 'bg-[#00ffcc] text-black' : 'bg-[#222] text-[#888] hover:text-white'
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              <span>JSON (.JSON)</span>
            </button>

            <button
              onClick={() => setFormat('txt')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition cursor-pointer ${
                format === 'txt' ? 'bg-[#00ffcc] text-black' : 'bg-[#222] text-[#888] hover:text-white'
              }`}
            >
              <AlignLeft className="w-3.5 h-3.5" />
              <span>TEXT (.TXT)</span>
            </button>
          </div>

          <div className="flex items-center space-x-2">
            {allFiles && allFiles.length > 1 && (
              <>
                <button
                  onClick={handleDownloadAllMd}
                  className="bg-[#1c1c1c] hover:bg-[#252525] border border-[#333333] text-[#00ffcc] px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition cursor-pointer"
                  title="Download all transcriptions as individual .md files"
                >
                  <Download className="w-3.5 h-3.5 text-[#00ffcc]" />
                  <span>All ({allFiles.length} .MD)</span>
                </button>

                <button
                  onClick={handleDownloadMasterMd}
                  className="bg-[#1c1c1c] hover:bg-[#252525] border border-[#0099ff]/50 text-[#0099ff] px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition cursor-pointer"
                  title="Generate a consolidated Master .MD document covering all files with table of contents"
                >
                  <Layers className="w-3.5 h-3.5 text-[#0099ff]" />
                  <span>Master .MD</span>
                </button>
              </>
            )}

            <button
              onClick={handleCopy}
              className="bg-[#222222] hover:bg-[#333333] text-white px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-[#00ffcc]" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] hover:brightness-110 text-black font-extrabold px-4 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition cursor-pointer shadow"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download File</span>
            </button>
          </div>
        </div>

        {/* Preset Selector Bar (Only for Markdown) */}
        {format === 'md' && (
          <div className="px-6 py-2 bg-[#121212] border-b border-[#222222] flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <span className="text-[#777777] font-mono">Format Preset:</span>
              <div className="flex items-center space-x-1 bg-[#1a1a1a] p-0.5 rounded-lg border border-[#2a2a2a]">
                <button
                  onClick={() => setMdPreset('standard')}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    mdPreset === 'standard' ? 'bg-[#00ffcc]/20 text-[#00ffcc] font-bold' : 'text-[#999] hover:text-white'
                  }`}
                  title="Standard layout with # Transcript, Date, Total Segments, and ## Speakers list"
                >
                  📄 Standard
                </button>
                <button
                  onClick={() => setMdPreset('obsidian')}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    mdPreset === 'obsidian' ? 'bg-[#00ffcc]/20 text-[#00ffcc] font-bold' : 'text-[#999] hover:text-white'
                  }`}
                  title="Obsidian format with YAML metadata, tags, and callout blocks"
                >
                  📘 Obsidian Note
                </button>
                <button
                  onClick={() => setMdPreset('meeting')}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    mdPreset === 'meeting' ? 'bg-[#00ffcc]/20 text-[#00ffcc] font-bold' : 'text-[#999] hover:text-white'
                  }`}
                  title="Meeting protocol with speaker percentage breakdown"
                >
                  💼 Meeting Minutes
                </button>
                <button
                  onClick={() => setMdPreset('clean')}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    mdPreset === 'clean' ? 'bg-[#00ffcc]/20 text-[#00ffcc] font-bold' : 'text-[#999] hover:text-white'
                  }`}
                  title="Clean dialogue without extraneous metadata"
                >
                  📝 Clean Dialogue
                </button>
                <button
                  onClick={() => setMdPreset('timestamps')}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    mdPreset === 'timestamps' ? 'bg-[#00ffcc]/20 text-[#00ffcc] font-bold' : 'text-[#999] hover:text-white'
                  }`}
                  title="Script with exact timestamps for every line"
                >
                  ⏱️ Timecoded Script
                </button>
              </div>
            </div>
            <div className="text-[#666666] font-mono text-[11px]">
              {mdPreset === 'obsidian' && 'YAML Frontmatter + Tags + Callout Cards'}
              {mdPreset === 'meeting' && 'Speaker Chronometry + Verbatim Protocol'}
              {mdPreset === 'clean' && 'Minimalist Dialogue for Quick Sharing'}
              {mdPreset === 'timestamps' && 'Line-by-line intervals [00:00 - 00:05]'}
            </div>
          </div>
        )}

        {/* Content Preview */}
        <div className="p-6 overflow-y-auto flex-1 bg-[#0a0a0a]">
          <pre className="font-mono text-xs text-[#cccccc] whitespace-pre-wrap leading-relaxed select-text bg-[#0e0e0e] p-4 rounded-lg border border-[#1f1f1f]">
            {content}
          </pre>
        </div>
      </div>
    </div>
  );
};
