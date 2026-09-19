import React from 'react';
import { SemanticCluster, TranscriptSegment, SpeakerMetadata } from '../types';
import { formatTimeSeconds } from '../services/exporter';
import { X, Layers, MessageSquare, Clock } from 'lucide-react';

interface SemanticViewModalProps {
  isOpen: boolean;
  onClose: () => void;
  clusters: SemanticCluster[];
  segments: TranscriptSegment[];
  speakers: Record<string, SpeakerMetadata>;
}

export const SemanticViewModal: React.FC<SemanticViewModalProps> = ({
  isOpen,
  onClose,
  clusters,
  segments,
  speakers,
}) => {
  if (!isOpen) return null;

  const segmentMap = new Map(segments.map((s) => [s.id, s]));

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl w-full max-w-3xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="px-6 py-4 bg-[#111111] border-b border-[#222222] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-[#ffcc00]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">SEMANTIC CLUSTERING BLOCKS</h2>
              <p className="text-xs text-[#666666]">Sentence embeddings & K-Means topic boundaries</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#888888] hover:text-white p-1 rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5 overflow-y-auto flex-1">
          {clusters.length === 0 ? (
            <div className="text-center py-12 text-[#555555] text-xs">
              No semantic clusters generated for this transcript.
            </div>
          ) : (
            clusters.map((cluster) => {
              const clusterSegments = cluster.segmentIds
                .map((id) => segmentMap.get(id))
                .filter((s): s is TranscriptSegment => Boolean(s));

              return (
                <div
                  key={cluster.clusterId}
                  className="bg-[#181818] border border-[#262626] hover:border-[#333333] rounded-xl p-5 space-y-3 transition-all"
                >
                  <div className="flex items-center justify-between border-b border-[#222222] pb-3">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-mono font-bold text-[#ffcc00] px-2 py-0.5 bg-[#ffcc00]/10 rounded border border-[#ffcc00]/20">
                          CLUSTER #{cluster.clusterId}
                        </span>
                        <h3 className="text-sm font-bold text-white">{cluster.topic}</h3>
                      </div>
                      <p className="text-xs text-[#888888] mt-1">{cluster.summary}</p>
                    </div>
                    <span className="text-xs font-mono text-[#666666] shrink-0">
                      {clusterSegments.length} turns
                    </span>
                  </div>

                  {/* Segments under this cluster */}
                  <div className="space-y-2 pt-1">
                    {clusterSegments.map((seg) => {
                      const spk = speakers[seg.speakerId];
                      const spkName = spk?.name || seg.speakerId;
                      const spkColor = spk?.color || '#00ffcc';

                      return (
                        <div
                          key={seg.id}
                          className="bg-[#121212] border border-[#202020] rounded-lg p-3 text-xs flex items-start space-x-3"
                        >
                          <div className="flex items-center space-x-1 font-mono text-[11px] text-[#666666] shrink-0 mt-0.5">
                            <Clock className="w-3 h-3" />
                            <span>{formatTimeSeconds(seg.start)}</span>
                          </div>

                          <div className="flex-1 min-w-0">
                            <span 
                              className="font-bold mr-2 text-xs"
                              style={{ color: spkColor }}
                            >
                              {spkName}:
                            </span>
                            <span className="text-[#cccccc]">{seg.text}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
