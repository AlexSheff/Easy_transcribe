import React from 'react';
import { WhisperModelSize } from '../types';

interface StatCardsProps {
  queueCount: number;
  processedCount: number;
  activeModel: WhisperModelSize;
}

export const StatCards: React.FC<StatCardsProps> = ({
  queueCount,
  processedCount,
  activeModel,
}) => {
  return (
    <div id="dashboard" className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {/* QUEUE */}
      <div 
        id="stat-queue-card"
        className="bg-[#141414] border border-[#222222] rounded-xl p-5 flex flex-col justify-between transition-all hover:border-[#333333]"
      >
        <span className="text-[10px] font-black uppercase text-[#00ffcc] tracking-widest">
          QUEUE
        </span>
        <span className="text-3xl lg:text-4xl font-black text-white font-mono mt-1">
          {queueCount}
        </span>
      </div>

      {/* PROCESSED */}
      <div 
        id="stat-processed-card"
        className="bg-[#141414] border border-[#222222] rounded-xl p-5 flex flex-col justify-between transition-all hover:border-[#333333]"
      >
        <span className="text-[10px] font-black uppercase text-[#00ffcc] tracking-widest">
          PROCESSED
        </span>
        <span className="text-3xl lg:text-4xl font-black text-white font-mono mt-1">
          {processedCount}
        </span>
      </div>

      {/* ACTIVE MODEL */}
      <div 
        id="stat-model-card"
        className="bg-[#141414] border border-[#222222] rounded-xl p-5 flex flex-col justify-between transition-all hover:border-[#333333]"
      >
        <span className="text-[10px] font-black uppercase text-[#00ffcc] tracking-widest">
          ACTIVE MODEL
        </span>
        <span className="text-3xl lg:text-4xl font-black text-white font-mono mt-1 truncate">
          {activeModel.toUpperCase()}
        </span>
      </div>
    </div>
  );
};
