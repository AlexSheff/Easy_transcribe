import React, { useEffect, useRef } from 'react';
import { LogEntry } from '../types';
import { Terminal, Trash2, Copy, Check } from 'lucide-react';

interface TerminalOutputProps {
  logs: LogEntry[];
  onClear: () => void;
}

export const TerminalOutput: React.FC<TerminalOutputProps> = ({ logs, onClear }) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = React.useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleCopyLogs = () => {
    const text = logs.map((l) => `[${l.timestamp}] [${l.type}] ${l.message}`).join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col bg-[#0d0d0d] border border-[#222222] rounded-xl overflow-hidden min-h-[220px] max-h-[360px]">
      {/* Header Bar */}
      <div className="bg-[#141414] px-4 py-2 border-b border-[#222222] flex items-center justify-between text-xs text-[#888888] select-none">
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-[#00ffcc]" />
          <span className="font-mono font-bold tracking-wider text-[#cccccc]">NODE CONSOLE STREAM</span>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={handleCopyLogs}
            className="hover:text-[#00ffcc] transition flex items-center space-x-1 cursor-pointer"
            title="Copy logs"
          >
            {copied ? <Check className="w-3 h-3 text-[#00ffcc]" /> : <Copy className="w-3 h-3" />}
            <span className="text-[11px] font-mono">{copied ? 'Copied' : 'Copy'}</span>
          </button>
          <button
            onClick={onClear}
            className="hover:text-[#ff5555] transition flex items-center space-x-1 cursor-pointer"
            title="Clear logs"
          >
            <Trash2 className="w-3 h-3" />
            <span className="text-[11px] font-mono">Clear</span>
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div className="p-4 overflow-y-auto font-mono text-xs text-[#aaaaaa] space-y-1.5 flex-1 select-text">
        {logs.length === 0 ? (
          <div className="text-[#555555] italic">Ready for instructions...</div>
        ) : (
          logs.map((log) => {
            let badgeColor = 'text-[#00ffcc]';
            let msgColor = 'text-[#dddddd]';

            if (log.type === 'PROC') {
              badgeColor = 'text-[#0099ff]';
              msgColor = 'text-[#cccccc]';
            } else if (log.type === 'DONE') {
              badgeColor = 'text-[#00ffcc] font-bold';
              msgColor = 'text-[#00ffcc]';
            } else if (log.type === 'ERROR') {
              badgeColor = 'text-[#ff5555] font-bold';
              msgColor = 'text-[#ff7777]';
            } else if (log.type === 'INFO') {
              badgeColor = 'text-[#ffcc00]';
              msgColor = 'text-[#eedd88]';
            }

            return (
              <div key={log.id} className="leading-relaxed break-words flex items-start space-x-2">
                <span className="text-[#555555] select-none text-[11px] shrink-0 font-mono">
                  [{log.timestamp}]
                </span>
                <span className={`shrink-0 font-bold ${badgeColor}`}>
                  [{log.type}]
                </span>
                <span className={`${msgColor} flex-1`}>{log.message}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};
