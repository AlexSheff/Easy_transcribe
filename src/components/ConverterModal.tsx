import React, { useState, useRef, useEffect } from 'react';
import { ConversionItem } from '../types';
import { convertMediaToWav } from '../services/audioConverter';
import { X, RefreshCw, Upload, Download, CheckCircle, AlertCircle, ArrowRight } from 'lucide-react';

interface ConverterModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSendToTranscriber: (file: File) => void;
  onLog: (type: 'SYSTEM' | 'PROC' | 'DONE' | 'ERROR' | 'INFO', message: string) => void;
}

export const ConverterModal: React.FC<ConverterModalProps> = ({
  isOpen,
  onClose,
  onSendToTranscriber,
  onLog,
}) => {
  const [items, setItems] = useState<ConversionItem[]>([]);
  const [isConvertingAll, setIsConvertingAll] = useState(false);
  const itemsRef = useRef<ConversionItem[]>(items);
  itemsRef.current = items;

  useEffect(() => {
    return () => {
      itemsRef.current.forEach((it: ConversionItem) => {
        if (it.convertedUrl) URL.revokeObjectURL(it.convertedUrl);
      });
    };
  }, []);

  if (!isOpen) return null;

  const handleClearAll = () => {
    items.forEach((it) => {
      if (it.convertedUrl) URL.revokeObjectURL(it.convertedUrl);
    });
    setItems([]);
  };

  const handleRemoveItem = (id: string) => {
    const it = items.find((i) => i.id === id);
    if (it?.convertedUrl) {
      URL.revokeObjectURL(it.convertedUrl);
    }
    setItems((prev) => prev.filter((i) => i.id !== id));
  };

  const handleFilesAdded = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const newItems: ConversionItem[] = Array.from(files).map((f) => ({
      id: 'conv_' + Math.random().toString(36).substring(2, 9),
      file: f,
      name: f.name,
      size: f.size,
      status: 'pending',
      progress: 0,
    }));

    setItems((prev) => [...prev, ...newItems]);
    onLog('INFO', `Added ${newItems.length} media file(s) to WAV conversion queue.`);
  };

  const convertSingle = async (item: ConversionItem) => {
    setItems((prev) =>
      prev.map((it) => (it.id === item.id ? { ...it, status: 'converting', progress: 10 } : it))
    );
    onLog('PROC', `Converting ${item.name} -> 16kHz Mono WAV...`);

    try {
      const { blob } = await convertMediaToWav(item.file, 16000, (p) => {
        setItems((prev) =>
          prev.map((it) => (it.id === item.id ? { ...it, progress: p } : it))
        );
      });

      const convertedUrl = URL.createObjectURL(blob);
      setItems((prev) =>
        prev.map((it) =>
          it.id === item.id
            ? { ...it, status: 'completed', progress: 100, convertedBlob: blob, convertedUrl }
            : it
        )
      );
      onLog('DONE', `Finished converting: ${item.name} -> ${item.name.replace(/\.[^/.]+$/, '')}.wav`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setItems((prev) =>
        prev.map((it) =>
          it.id === item.id ? { ...it, status: 'error', error: msg } : it
        )
      );
      onLog('ERROR', `Error converting ${item.name}: ${msg}`);
    }
  };

  const convertAll = async () => {
    setIsConvertingAll(true);
    const pending = items.filter((i) => i.status === 'pending');
    for (const it of pending) {
      await convertSingle(it);
    }
    setIsConvertingAll(false);
  };

  const handleDownloadWav = (item: ConversionItem) => {
    if (!item.convertedBlob) return;
    const a = document.createElement('a');
    a.href = item.convertedUrl!;
    a.download = `${item.name.replace(/\.[^/.]+$/, '')}_16khz.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleSendToTranscribe = (item: ConversionItem) => {
    if (!item.convertedBlob) return;
    const wavFile = new File(
      [item.convertedBlob],
      `${item.name.replace(/\.[^/.]+$/, '')}_16khz.wav`,
      { type: 'audio/wav' }
    );
    onSendToTranscriber(wavFile);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="px-6 py-4 bg-[#111111] border-b border-[#222222] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <RefreshCw className="w-5 h-5 text-[#ffcc00]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">MEDIA TO WAV CONVERTER</h2>
              <p className="text-xs text-[#666666]">MP4, MKV, AVI, MOV, WEBM, MP3 &rarr; 16kHz Mono WAV</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#888888] hover:text-white p-1 rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Dropzone & List */}
        <div className="p-6 space-y-5 overflow-y-auto flex-1">
          {/* File input / Drag & drop */}
          <label className="border-2 border-dashed border-[#333333] hover:border-[#ffcc00] rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition bg-[#111111]/50 group">
            <Upload className="w-8 h-8 text-[#666666] group-hover:text-[#ffcc00] transition mb-2" />
            <span className="text-xs font-bold text-white">Click or drag media files here to convert</span>
            <span className="text-[11px] text-[#666666] mt-1">Supports MP4, MKV, AVI, WEBM, MP3, M4A, FLAC</span>
            <input
              type="file"
              multiple
              accept="video/*,audio/*,.mp4,.mkv,.avi,.webm,.mov,.mp3,.wav,.m4a"
              onChange={(e) => handleFilesAdded(e.target.files)}
              className="hidden"
            />
          </label>

          {/* Queue Actions */}
          {items.length > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-[#888888]">
                {items.length} file(s) in queue
              </span>
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleClearAll}
                  disabled={isConvertingAll}
                  className="text-xs text-[#888888] hover:text-white px-2 py-1 cursor-pointer"
                >
                  Clear All
                </button>
                <button
                  onClick={convertAll}
                  disabled={isConvertingAll || items.every((i) => i.status === 'completed')}
                  className="bg-gradient-to-r from-[#ffcc00] to-[#ff9900] text-black font-extrabold text-xs px-3 py-1.5 rounded-lg disabled:opacity-40 transition cursor-pointer"
                >
                  Convert All to WAV
                </button>
              </div>
            </div>
          )}

          {/* Items List */}
          <div className="space-y-2">
            {items.map((item) => (
              <div
                key={item.id}
                className="bg-[#181818] border border-[#262626] rounded-lg p-3 flex items-center justify-between gap-3 text-xs"
              >
                <div className="min-w-0 flex-1">
                  <div className="text-white font-medium truncate">{item.name}</div>
                  <div className="text-[11px] text-[#666666] font-mono mt-0.5">
                    {(item.size / (1024 * 1024)).toFixed(2)} MB · Status: {item.status}
                  </div>
                  {item.status === 'converting' && (
                    <div className="w-full bg-[#222] h-1 rounded-full mt-2 overflow-hidden">
                      <div
                        className="bg-[#ffcc00] h-full transition-all duration-200"
                        style={{ width: `${item.progress}%` }}
                      />
                    </div>
                  )}
                  {item.error && (
                    <div className="text-[11px] text-[#ff5555] mt-1">{item.error}</div>
                  )}
                </div>

                <div className="flex items-center space-x-2 shrink-0">
                  {item.status === 'pending' && (
                    <button
                      onClick={() => convertSingle(item)}
                      className="bg-[#222222] hover:bg-[#333333] text-white px-2.5 py-1 rounded cursor-pointer"
                    >
                      Convert
                    </button>
                  )}

                  {item.status === 'completed' && (
                    <>
                      <button
                        onClick={() => handleDownloadWav(item)}
                        className="bg-[#222222] hover:bg-[#333333] text-[#ffcc00] px-2.5 py-1 rounded flex items-center space-x-1 cursor-pointer"
                        title="Download WAV"
                      >
                        <Download className="w-3 h-3" />
                        <span>WAV</span>
                      </button>

                      <button
                        onClick={() => handleSendToTranscribe(item)}
                        className="bg-[#00ffcc] hover:bg-[#33ffdd] text-black font-bold px-2.5 py-1 rounded flex items-center space-x-1 cursor-pointer"
                        title="Load into Transcriber Node"
                      >
                        <span>Transcribe</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </>
                  )}

                  <button
                    onClick={() => handleRemoveItem(item.id)}
                    disabled={item.status === 'converting'}
                    className="text-[#666666] hover:text-[#ff5555] p-1 rounded transition cursor-pointer disabled:opacity-20"
                    title="Remove from queue"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
