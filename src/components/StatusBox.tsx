import React from 'react';
import { AlertTriangle, X, Square } from 'lucide-react';

interface StatusBoxProps {
  status: string;
  progress: number;
  showProgress: boolean;
  isError?: boolean;
  isProcessing?: boolean;
  onDismissError?: () => void;
  onCancel?: () => void;
}

export const StatusBox: React.FC<StatusBoxProps> = ({
  status,
  progress,
  showProgress,
  isError,
  isProcessing,
  onDismissError,
  onCancel
}) => {
  return (
    <div 
      id="status_box" 
      className={`rounded-xl p-4 space-y-2.5 transition-all ${
        isError 
          ? 'bg-[#220d0d] border border-[#ff4444]/50 shadow-[0_0_15px_rgba(255,68,68,0.15)]' 
          : 'bg-[#141414] border border-[#222222]'
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-medium flex items-center space-x-2 min-w-0">
          {isError ? (
            <AlertTriangle className="w-4 h-4 text-[#ff4444] shrink-0 animate-bounce" />
          ) : showProgress ? (
            <span className="inline-block w-2 h-2 rounded-full bg-[#00ffcc] animate-pulse shrink-0"></span>
          ) : null}
          <span className={`${isError ? 'text-[#ff9999]' : 'text-white'} truncate`}>{status}</span>
        </div>
        
        {isError && onDismissError && (
          <button
            onClick={onDismissError}
            className="text-xs text-[#ffaaaa] hover:text-white bg-[#ff4444]/20 hover:bg-[#ff4444]/30 px-2 py-1 rounded-md flex items-center space-x-1 cursor-pointer transition-colors shrink-0"
          >
            <span>Dismiss</span>
            <X className="w-3 h-3" />
          </button>
        )}

        {!isError && showProgress && (
          <div className="flex items-center space-x-2.5 shrink-0">
            <span className="text-xs text-[#00ffcc] font-mono font-bold">
              {Math.round(progress)}%
            </span>
            {isProcessing && onCancel && (
              <button
                onClick={onCancel}
                className="text-[11px] font-mono text-[#ff7788] hover:text-white bg-[#ff0055]/15 hover:bg-[#ff0055]/30 border border-[#ff0055]/30 px-2 py-0.5 rounded flex items-center space-x-1 cursor-pointer transition-colors"
                title="Cancel active transcription"
              >
                <Square className="w-2.5 h-2.5 fill-current" />
                <span>Cancel</span>
              </button>
            )}
          </div>
        )}
      </div>

      {!isError && showProgress && (
        <div className="w-full bg-[#222222] rounded-full h-1.5 overflow-hidden">
          <div 
            className="bg-gradient-to-r from-[#00ffcc] to-[#0099ff] h-full transition-all duration-300 rounded-full"
            style={{ width: `${Math.max(3, Math.min(100, progress))}%` }}
          />
        </div>
      )}
    </div>
  );
};
