import React from 'react';

interface StatusBoxProps {
  status: string;
  progress: number;
  showProgress: boolean;
}

export const StatusBox: React.FC<StatusBoxProps> = ({
  status,
  progress,
  showProgress,
}) => {
  return (
    <div 
      id="status_box" 
      className="bg-[#141414] border border-[#222222] rounded-xl p-4 space-y-2.5 transition-all"
    >
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-white flex items-center space-x-2">
          {showProgress && (
            <span className="inline-block w-2 h-2 rounded-full bg-[#00ffcc] animate-pulse"></span>
          )}
          <span>{status}</span>
        </div>
        {showProgress && (
          <span className="text-xs text-[#00ffcc] font-mono font-bold">
            {Math.round(progress)}%
          </span>
        )}
      </div>

      {showProgress && (
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
