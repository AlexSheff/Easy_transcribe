import React, { useState, useEffect } from 'react';
import { SavedTranscriptInfo } from '../types';
import { Archive, Search, RefreshCw, ExternalLink, X, FileText, AlertCircle } from 'lucide-react';

interface SavedOutputsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFetchList: () => Promise<SavedTranscriptInfo[]>;
  onOpenTranscript: (relativePath: string) => Promise<void>;
}

export const SavedOutputsModal: React.FC<SavedOutputsModalProps> = ({
  isOpen,
  onClose,
  onFetchList,
  onOpenTranscript,
}) => {
  const [items, setItems] = useState<SavedTranscriptInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [openingPath, setOpeningPath] = useState<string | null>(null);

  const loadList = async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await onFetchList();
      setItems(list);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      void loadList();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const filtered = items.filter((it) => {
    if (!query.trim()) return true;
    const q = query.toLowerCase();
    return it.filename.toLowerCase().includes(q) || it.relativePath.toLowerCase().includes(q);
  });

  const formatBytes = (bytes: number) => {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleOpen = async (relPath: string) => {
    setOpeningPath(relPath);
    setError(null);
    try {
      await onOpenTranscript(relPath);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOpeningPath(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
      <div className="w-full max-w-2xl bg-[#111115] border border-[#262630] rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262630] bg-[#16161c]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#00ffcc]/10 text-[#00ffcc] border border-[#00ffcc]/30">
              <Archive className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">SAVED OUTPUTS & TRANSCRIPTS</h2>
              <p className="text-xs text-[#888899]">
                Files stored in <code className="text-[#00ffcc]">outputs/</code> and{' '}
                <code className="text-[#00ffcc]">transcripts/</code>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#666677] hover:text-white transition-colors p-1 rounded-md cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Refresh */}
        <div className="px-5 py-3 bg-[#131318] border-b border-[#24242e] flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 text-[#666677] absolute left-3 top-2.5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter saved transcripts..."
              className="w-full bg-[#0a0a0f] border border-[#2c2c38] rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#555566] focus:outline-none focus:border-[#00ffcc]"
            />
          </div>
          <button
            onClick={loadList}
            disabled={loading}
            className="px-3 py-1.5 bg-[#1c1c24] hover:bg-[#262632] border border-[#303040] rounded-lg text-xs font-mono text-[#cccccc] hover:text-white flex items-center gap-1.5 transition cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-[#00ffcc]' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-2.5">
          {error && (
            <div className="p-3 rounded-lg bg-[#241014] border border-[#ff4455]/40 text-xs text-[#ff99aa] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading && items.length === 0 ? (
            <div className="py-12 text-center text-xs font-mono text-[#777788]">
              Scanning outputs/ and transcripts/...
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-12 text-center text-xs text-[#666677]">
              {items.length === 0
                ? 'No saved transcripts found yet in outputs/ or transcripts/.'
                : `No files matching "${query}".`}
            </div>
          ) : (
            filtered.map((item) => {
              const folder = item.relativePath.split('/')[0] || 'outputs';
              const isOpening = openingPath === item.relativePath;
              return (
                <div
                  key={item.relativePath}
                  className="p-3 rounded-lg bg-[#16161c] border border-[#24242f] hover:border-[#00ffcc]/40 flex items-center justify-between gap-3 transition"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <FileText className="w-4 h-4 text-[#00ffcc] shrink-0" />
                    <div className="min-w-0">
                      <div className="text-xs font-mono font-semibold text-white truncate">
                        {item.filename}
                      </div>
                      <div className="text-[10px] font-mono text-[#777788] flex items-center gap-2 mt-0.5">
                        <span
                          className={`px-1.5 py-0.2 rounded ${
                            folder === 'outputs'
                              ? 'bg-[#00ffcc]/15 text-[#00ffcc]'
                              : 'bg-[#0099ff]/15 text-[#66bbff]'
                          }`}
                        >
                          {folder}/
                        </span>
                        <span>{formatBytes(item.size)}</span>
                        <span>·</span>
                        <span>{item.modifiedAt.replace('T', ' ').slice(0, 16)}</span>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => handleOpen(item.relativePath)}
                    disabled={isOpening}
                    className="px-3 py-1.5 rounded-lg bg-[#00ffcc]/15 hover:bg-[#00ffcc]/25 border border-[#00ffcc]/40 text-xs font-mono text-[#00ffcc] flex items-center gap-1.5 shrink-0 transition cursor-pointer disabled:opacity-50"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                    <span>{isOpening ? 'Opening...' : 'Open in Viewer'}</span>
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
