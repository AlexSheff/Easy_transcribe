import React, { useState, useRef, useEffect } from 'react';
import { X, Mic, Square, Play, ArrowRight, Volume2 } from 'lucide-react';

interface RecorderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onRecordingComplete: (audioFile: File) => void;
  onLog: (type: 'SYSTEM' | 'PROC' | 'DONE' | 'ERROR' | 'INFO', message: string) => void;
}

export const RecorderModal: React.FC<RecorderModalProps> = ({
  isOpen,
  onClose,
  onRecordingComplete,
  onLog,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [liveTranscript, setLiveTranscript] = useState<string[]>([]);
  const [vadActive, setVadActive] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recognitionRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      stopRecording();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
    }
    return () => {
      stopRecording();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
    };
  }, [isOpen]);

  const startRecording = async () => {
    try {
      audioChunksRef.current = [];
      setLiveTranscript([]);
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
      setAudioUrl(null);
      setRecordedBlob(null);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Audio Context for VAD & visualizer
      const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      // MediaRecorder with robust format support
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/mp4')
        ? 'audio/mp4'
        : '';

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const actualType = mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: actualType });
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        setRecordedBlob(blob);
        setAudioUrl(url);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        if (audioContextRef.current) {
          audioContextRef.current.close().catch(() => {});
          audioContextRef.current = null;
        }
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      setRecordDuration(0);
      onLog('SYSTEM', 'Real-time microphone capture session initiated with VAD.');

      // Timer
      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        setRecordDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 500);

      // Visualizer loop
      drawWaveform();

      // Web Speech API for real-time live preview (if supported in browser)
      const SpeechRecognition = (window as unknown as { SpeechRecognition?: any; webkitSpeechRecognition?: any }).SpeechRecognition ||
        (window as unknown as { webkitSpeechRecognition?: any }).webkitSpeechRecognition;
      if (SpeechRecognition) {
        try {
          const rec = new SpeechRecognition();
          rec.continuous = true;
          rec.interimResults = true;
          rec.lang = navigator.language || 'ru-RU';
          rec.onresult = (event: any) => {
            let current = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
              current += event.results[i][0].transcript;
            }
            if (current.trim()) {
              setLiveTranscript((prev) => [...prev.slice(-4), current.trim()]);
              setVadActive(true);
            }
          };
          rec.start();
          recognitionRef.current = rec;
        } catch {
          // Speech recognition is optional progressive enhancement
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      onLog('ERROR', `Microphone access error: ${msg}`);
      alert('Could not access microphone. Please ensure microphone permissions are granted.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch {}
      recognitionRef.current = null;
    }
    setVadActive(false);
    onLog('SYSTEM', 'Microphone capture stopped. Ready for transcription.');
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      // Calculate average volume for VAD
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
      const avg = sum / bufferLength;
      setVadActive(avg > 15);

      ctx.fillStyle = '#0d0d0d';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        // Gradient color based on intensity
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#00ffcc');
        gradient.addColorStop(1, '#ff0055');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };

    render();
  };

  const handleSendToTranscribe = () => {
    if (!recordedBlob) return;
    const now = new Date().toISOString().replace(/[:.]/g, '-');
    const file = new File([recordedBlob], `mic_recording_${now}.wav`, { type: 'audio/wav' });
    onRecordingComplete(file);
    onClose();
  };

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-[#141414] border border-[#2a2a2a] rounded-xl w-full max-w-xl overflow-hidden shadow-2xl flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 bg-[#111111] border-b border-[#222222] flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Mic className="w-5 h-5 text-[#ff0055]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">REAL-TIME MICROPHONE RECORDING</h2>
              <p className="text-xs text-[#666666]">Live audio capture with Voice Activity Detection (VAD)</p>
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
        <div className="p-6 space-y-6 flex flex-col items-center">
          {/* Visualizer canvas */}
          <div className="w-full bg-[#0d0d0d] border border-[#222222] rounded-xl p-3 flex flex-col items-center relative overflow-hidden">
            <div className="w-full flex items-center justify-between text-xs font-mono text-[#888888] mb-2 px-1">
              <span className="flex items-center space-x-1.5">
                <span className={`w-2 h-2 rounded-full ${vadActive ? 'bg-[#00ffcc] animate-ping' : 'bg-[#444]'}`} />
                <span>{vadActive ? 'VAD SPEECH DETECTED' : 'SILENCE / IDLE'}</span>
              </span>
              <span className="text-white font-bold">{formatDuration(recordDuration)}</span>
            </div>

            <canvas
              ref={canvasRef}
              width={480}
              height={100}
              className="w-full h-24 rounded-lg bg-[#0d0d0d]"
            />
          </div>

          {/* Record / Stop Button */}
          <div className="flex items-center space-x-4">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="w-16 h-16 rounded-full bg-gradient-to-r from-[#ff0055] to-[#ff55aa] hover:brightness-110 active:scale-95 text-white flex items-center justify-center shadow-[0_0_20px_rgba(255,0,85,0.4)] transition cursor-pointer"
                title="Start Recording"
              >
                <Mic className="w-7 h-7" />
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="w-16 h-16 rounded-full bg-[#ff0000] hover:bg-[#ff2222] active:scale-95 text-white flex items-center justify-center shadow-[0_0_20px_rgba(255,0,0,0.6)] animate-pulse transition cursor-pointer"
                title="Stop Recording"
              >
                <Square className="w-7 h-7 fill-white" />
              </button>
            )}
          </div>

          <div className="text-xs font-mono text-center text-[#aaaaaa]">
            {isRecording ? 'Capturing live microphone stream...' : recordedBlob ? 'Recording captured.' : 'Click to start recording microphone.'}
          </div>

          {/* Live Transcript Stream */}
          {liveTranscript.length > 0 && (
            <div className="w-full bg-[#181818] border border-[#2a2a2a] rounded-lg p-3 space-y-1 text-xs font-mono">
              <div className="text-[10px] uppercase text-[#00ffcc] font-bold tracking-wider">Live Speech Stream:</div>
              <div className="text-[#dddddd] italic">{liveTranscript[liveTranscript.length - 1]}</div>
            </div>
          )}

          {/* Audio Preview & Send to Pipeline */}
          {audioUrl && !isRecording && (
            <div className="w-full bg-[#181818] border border-[#2a2a2a] rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3">
              <audio src={audioUrl} controls className="w-full sm:w-64 h-8" />
              <button
                onClick={handleSendToTranscribe}
                className="w-full sm:w-auto bg-gradient-to-r from-[#00ffcc] to-[#0099ff] hover:brightness-110 text-black font-extrabold text-xs px-4 py-2.5 rounded-lg flex items-center justify-center space-x-2 transition shadow cursor-pointer"
              >
                <span>Send to Transcriber</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
