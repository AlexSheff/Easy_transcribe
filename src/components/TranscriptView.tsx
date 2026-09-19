import React, { useState, useRef, useEffect } from 'react';
import { ProcessedFile, SpeakerMetadata, TranscriptSegment } from '../types';
import { formatTimeSeconds } from '../services/exporter';
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Search, 
  Clock, 
  User, 
  Sliders, 
  Download, 
  Layers, 
  Check, 
  Edit3
} from 'lucide-react';

interface TranscriptViewProps {
  file: ProcessedFile;
  speakers: Record<string, SpeakerMetadata>;
  onOpenSpeakers: () => void;
  onOpenSemantic: () => void;
  onOpenExport: () => void;
  onUpdateSegmentText: (segmentId: string, newText: string) => void;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({
  file,
  speakers,
  onOpenSpeakers,
  onOpenSemantic,
  onOpenExport,
  onUpdateSegmentText,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [editingSegId, setEditingSegId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleEnded = () => setIsPlaying(false);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
    };
  }, [file.audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    }
  };

  const jumpToTime = (timeSec: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = timeSec;
    if (!isPlaying) {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const target = parseFloat(e.target.value);
    setCurrentTime(target);
    if (audioRef.current) {
      audioRef.current.currentTime = target;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const startEdit = (seg: TranscriptSegment) => {
    setEditingSegId(seg.id);
    setEditText(seg.text);
  };

  const saveEdit = (segId: string) => {
    onUpdateSegmentText(segId, editText);
    setEditingSegId(null);
  };

  const filteredSegments = file.segments.filter((seg) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    const spk = speakers[seg.speakerId]?.name.toLowerCase() || '';
    return seg.text.toLowerCase().includes(q) || spk.includes(q);
  });

  return (
    <div className="bg-[#141414] border border-[#222222] rounded-xl flex flex-col h-[520px] overflow-hidden">
      {/* Hidden audio element */}
      {file.audioUrl && (
        <audio ref={audioRef} src={file.audioUrl} preload="auto" />
      )}

      {/* Header & Controls Bar */}
      <div className="p-4 border-b border-[#222222] bg-[#111111] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-[#00ffcc] tracking-wide uppercase">ACTIVE TRANSCRIPT</span>
            <span className="text-xs text-[#555555]">·</span>
            <span className="text-xs font-mono text-[#aaaaaa] truncate max-w-xs">{file.filename}</span>
          </div>
          <div className="text-[11px] text-[#666666] font-mono mt-0.5">
            Duration: {formatTimeSeconds(file.duration)} · Segments: {file.segments.length} · Model: {file.modelUsed.toUpperCase()}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onOpenSpeakers}
            className="bg-[#1c1c1c] hover:bg-[#282828] border border-[#333333] text-xs font-medium text-[#cccccc] hover:text-white px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition cursor-pointer"
          >
            <Sliders className="w-3.5 h-3.5 text-[#00ffcc]" />
            <span>Speakers ({Object.keys(speakers).length})</span>
          </button>

          {file.clusters && file.clusters.length > 0 && (
            <button
              onClick={onOpenSemantic}
              className="bg-[#1c1c1c] hover:bg-[#282828] border border-[#333333] text-xs font-medium text-[#cccccc] hover:text-white px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition cursor-pointer"
            >
              <Layers className="w-3.5 h-3.5 text-[#ffcc00]" />
              <span>Topics ({file.clusters.length})</span>
            </button>
          )}

          <button
            onClick={onOpenExport}
            className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] hover:brightness-110 text-black font-extrabold text-xs px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition shadow cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Media Player Scrubber */}
      {file.audioUrl && (
        <div className="px-4 py-3 bg-[#0d0d0d] border-b border-[#222222] flex items-center space-x-4">
          <button
            onClick={togglePlay}
            className="w-9 h-9 rounded-full bg-[#00ffcc] hover:bg-[#33ffdd] active:scale-95 text-black flex items-center justify-center transition shrink-0 cursor-pointer shadow-[0_0_10px_rgba(0,255,204,0.3)]"
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-black" /> : <Play className="w-4 h-4 fill-black ml-0.5" />}
          </button>

          <span className="text-xs font-mono text-[#00ffcc] w-14 text-right">
            {formatTimeSeconds(currentTime)}
          </span>

          <input
            type="range"
            min={0}
            max={file.duration || 1}
            step={0.05}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-1.5 bg-[#222222] rounded-lg appearance-none cursor-pointer accent-[#00ffcc]"
          />

          <span className="text-xs font-mono text-[#666666] w-14">
            {formatTimeSeconds(file.duration)}
          </span>

          <button
            onClick={toggleMute}
            className="text-[#888888] hover:text-white transition cursor-pointer"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
        </div>
      )}

      {/* Filter / Search Bar */}
      <div className="px-4 py-2 bg-[#121212] border-b border-[#222222] flex items-center justify-between">
        <div className="relative flex-1 max-w-sm">
          <Search className="w-3.5 h-3.5 text-[#555555] absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search words or speakers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#181818] border border-[#2a2a2a] rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#555555] focus:outline-none focus:border-[#00ffcc]"
          />
        </div>
        <div className="text-[11px] text-[#666666] font-mono">
          Showing {filteredSegments.length} of {file.segments.length} turns
        </div>
      </div>

      {/* Segments List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 divide-y divide-[#1e1e1e]/60">
        {filteredSegments.length === 0 ? (
          <div className="text-center py-12 text-[#555555] text-xs">
            No segments found matching "{searchQuery}"
          </div>
        ) : (
          filteredSegments.map((seg) => {
            const isCurrentlyActive = currentTime >= seg.start && currentTime <= seg.end;
            const spk = speakers[seg.speakerId];
            const spkName = spk?.name || seg.speakerId;
            const spkColor = spk?.color || '#00ffcc';
            const gender = spk?.gender || seg.gender;

            return (
              <div
                key={seg.id}
                className={`pt-3 first:pt-0 transition-all rounded-lg p-2.5 ${
                  isCurrentlyActive ? 'bg-[#1a1a1a] border-l-2 border-[#00ffcc]' : 'hover:bg-[#161616]'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center space-x-2">
                    {/* Timestamp clickable badge */}
                    <button
                      onClick={() => jumpToTime(seg.start)}
                      className="inline-flex items-center space-x-1 font-mono text-xs px-2 py-0.5 rounded bg-[#1e1e1e] hover:bg-[#00ffcc] hover:text-black text-[#00ffcc] transition cursor-pointer"
                      title="Click to play segment"
                    >
                      <Clock className="w-3 h-3" />
                      <span>{formatTimeSeconds(seg.start)}</span>
                    </button>

                    {/* Speaker badge */}
                    <div 
                      className="inline-flex items-center space-x-1 text-xs font-bold px-2 py-0.5 rounded border"
                      style={{ 
                        borderColor: `${spkColor}40`, 
                        backgroundColor: `${spkColor}15`,
                        color: spkColor 
                      }}
                    >
                      <User className="w-3 h-3" />
                      <span>{spkName}</span>
                    </div>

                    {/* Gender badge from F0 pitch analysis */}
                    <span 
                      className={`text-[10px] uppercase font-mono px-1.5 py-0.2 rounded ${
                        gender === 'Male' 
                          ? 'bg-blue-950/70 text-blue-300 border border-blue-800/40' 
                          : gender === 'Female'
                          ? 'bg-pink-950/70 text-pink-300 border border-pink-800/40'
                          : 'bg-neutral-800 text-neutral-400'
                      }`}
                    >
                      {gender}
                    </span>

                    {spk?.pitchF0 && (
                      <span className="text-[10px] font-mono text-[#666666]">
                        {spk.pitchF0}Hz
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-2">
                    {editingSegId !== seg.id ? (
                      <button
                        onClick={() => startEdit(seg)}
                        className="text-[#666666] hover:text-[#cccccc] text-xs transition p-1 cursor-pointer"
                        title="Edit text"
                      >
                        <Edit3 className="w-3 h-3" />
                      </button>
                    ) : (
                      <button
                        onClick={() => saveEdit(seg.id)}
                        className="text-[#00ffcc] hover:text-white text-xs font-mono flex items-center space-x-1 bg-[#222222] px-2 py-0.5 rounded transition cursor-pointer"
                      >
                        <Check className="w-3 h-3" />
                        <span>Save</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Content */}
                {editingSegId === seg.id ? (
                  <textarea
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    className="w-full bg-[#0a0a0a] border border-[#333333] rounded p-2 text-xs text-white focus:outline-none focus:border-[#00ffcc] font-sans resize-none"
                    rows={2}
                  />
                ) : (
                  <p className="text-sm text-[#e0e0e0] leading-relaxed pl-1 select-text">
                    {seg.uncertain && <span className="text-[#ffcc00] font-mono mr-1">[(?)]</span>}
                    {seg.text}
                  </p>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
