import React, { useState } from 'react';
import { SpeakerMetadata, GenderType } from '../types';
import { X, User, Sliders, Merge, Trash2, Check } from 'lucide-react';

interface SpeakerManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  speakers: Record<string, SpeakerMetadata>;
  onUpdateSpeaker: (id: string, updates: Partial<SpeakerMetadata>) => void;
  onMergeSpeakers: (sourceId: string, targetId: string) => void;
  onResetDb: () => void;
}

export const SpeakerManagerModal: React.FC<SpeakerManagerModalProps> = ({
  isOpen,
  onClose,
  speakers,
  onUpdateSpeaker,
  onMergeSpeakers,
  onResetDb,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editGender, setEditGender] = useState<GenderType>('Unknown');
  
  const [mergeSource, setMergeSource] = useState<string>('');
  const [mergeTarget, setMergeTarget] = useState<string>('');

  if (!isOpen) return null;

  const speakerList = Object.values(speakers);

  const startEdit = (spk: SpeakerMetadata) => {
    setEditingId(spk.id);
    setEditName(spk.name);
    setEditGender(spk.gender);
  };

  const saveEdit = (id: string) => {
    onUpdateSpeaker(id, { name: editName, gender: editGender });
    setEditingId(null);
  };

  const handleMerge = () => {
    if (!mergeSource || !mergeTarget || mergeSource === mergeTarget) return;
    onMergeSpeakers(mergeSource, mergeTarget);
    setMergeSource('');
    setMergeTarget('');
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="px-6 py-4 bg-[#111111] border-b border-[#222222] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-[#00ffcc]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">VOICE DATABASE · SPEAKER ROSTER</h2>
              <p className="text-xs text-[#666666]">Per-session ECAPA-TDNN & F0 pitch clustering management</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#888888] hover:text-white p-1 rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6 overflow-y-auto flex-1">
          {/* Speaker List */}
          <div className="space-y-3">
            <div className="text-xs font-mono uppercase text-[#888888] tracking-wider">
              Identified Speaker Entities ({speakerList.length})
            </div>

            {speakerList.length === 0 ? (
              <div className="text-center py-8 text-xs text-[#555555]">
                No speakers registered in database. Process an audio file to populate.
              </div>
            ) : (
              <div className="space-y-2">
                {speakerList.map((spk) => (
                  <div
                    key={spk.id}
                    className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-3 flex items-center justify-between gap-3"
                  >
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs shrink-0"
                        style={{ backgroundColor: `${spk.color}25`, color: spk.color }}
                      >
                        <User className="w-4 h-4" />
                      </div>

                      {editingId === spk.id ? (
                        <div className="flex items-center space-x-2 flex-1">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="bg-[#111] border border-[#00ffcc] rounded px-2 py-1 text-xs text-white focus:outline-none flex-1"
                          />
                          <select
                            value={editGender}
                            onChange={(e) => setEditGender(e.target.value as GenderType)}
                            className="bg-[#111] border border-[#333] rounded px-2 py-1 text-xs text-white"
                          >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Unknown">Unknown</option>
                          </select>
                          <button
                            onClick={() => saveEdit(spk.id)}
                            className="bg-[#00ffcc] text-black px-2 py-1 rounded text-xs font-bold cursor-pointer"
                          >
                            <Check className="w-3 h-3" />
                          </button>
                        </div>
                      ) : (
                        <div className="min-w-0">
                          <div className="text-xs font-bold text-white flex items-center space-x-2">
                            <span>{spk.name}</span>
                            <span className="text-[10px] text-[#666666] font-mono">({spk.id})</span>
                          </div>
                          <div className="text-[11px] text-[#888888] font-mono mt-0.5 space-x-2">
                            <span>Gender: <strong className="text-[#cccccc]">{spk.gender}</strong></span>
                            {spk.pitchF0 && <span>· F0: <strong className="text-[#00ffcc]">{spk.pitchF0}Hz</strong></span>}
                            <span>· Turns: {spk.sampleCount}</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {editingId !== spk.id && (
                      <button
                        onClick={() => startEdit(spk)}
                        className="text-xs text-[#00ffcc] hover:underline font-mono px-2 py-1 cursor-pointer"
                      >
                        Rename
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Merge Identities Section */}
          {speakerList.length > 1 && (
            <div className="bg-[#181818] border border-[#2a2a2a] rounded-lg p-4 space-y-3">
              <div className="flex items-center space-x-2 text-xs font-bold text-white">
                <Merge className="w-4 h-4 text-[#ffcc00]" />
                <span>Merge Speaker Identities</span>
              </div>
              <p className="text-[11px] text-[#777777]">
                Combine split speaker clusters into a single unified voice fingerprint.
              </p>
              <div className="flex flex-col sm:flex-row items-center gap-3">
                <select
                  value={mergeSource}
                  onChange={(e) => setMergeSource(e.target.value)}
                  className="bg-[#111111] border border-[#333333] rounded px-3 py-1.5 text-xs text-white w-full sm:w-1/2"
                >
                  <option value="">Select source speaker...</option>
                  {speakerList.map((s) => (
                    <option key={s.id} value={s.id}>{s.name} ({s.id})</option>
                  ))}
                </select>

                <span className="text-xs text-[#555555] font-mono">merge into &rarr;</span>

                <select
                  value={mergeTarget}
                  onChange={(e) => setMergeTarget(e.target.value)}
                  className="bg-[#111111] border border-[#333333] rounded px-3 py-1.5 text-xs text-white w-full sm:w-1/2"
                >
                  <option value="">Select target speaker...</option>
                  {speakerList.map((s) => (
                    <option key={s.id} value={s.id}>{s.name} ({s.id})</option>
                  ))}
                </select>

                <button
                  onClick={handleMerge}
                  disabled={!mergeSource || !mergeTarget || mergeSource === mergeTarget}
                  className="w-full sm:w-auto bg-[#ffcc00] hover:bg-[#ffdd33] disabled:opacity-40 text-black font-extrabold text-xs px-4 py-1.5 rounded transition cursor-pointer"
                >
                  Merge
                </button>
              </div>
            </div>
          )}

          {/* Reset Database Button */}
          <div className="pt-2 flex justify-between items-center border-t border-[#222222]">
            <span className="text-[11px] text-[#666666]">
              DB is isolated to this session for clean identification.
            </span>
            <button
              onClick={onResetDb}
              className="text-xs text-[#ff5555] hover:text-[#ff7777] font-mono flex items-center space-x-1 p-2 rounded hover:bg-[#ff5555]/10 transition cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Voice DB</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
