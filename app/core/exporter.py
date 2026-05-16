import os
from pathlib import Path
from datetime import datetime

class MarkdownExporter:
    @staticmethod
    def export(segments, input_filename, output_dir="app/output", fingerprinter=None):
        """
        Exports transcription segments to a Markdown file.
        
        Args:
            segments: List of segment dicts with 'start', 'text', 'speaker_id'.
            input_filename: Path to the original input file.
            output_dir: Directory to write the output file.
            fingerprinter: Optional VoiceFingerprint instance to resolve speaker names.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{Path(input_filename).stem}_{timestamp}.md"
        file_path = output_path / filename

        # Ensure segments are sorted chronologically
        segments.sort(key=lambda x: x['start'])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {input_filename}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Segments: {len(segments)}\n\n")
            
            f.write("---\n\n")
            
            for seg in segments:
                start_time = MarkdownExporter._format_time(seg['start'])
                speaker_id = seg.get('speaker_id', 'unknown')
                
                # Try to get a friendly name from the fingerprinter
                speaker_label = MarkdownExporter._resolve_speaker_name(speaker_id, fingerprinter)
                
                f.write(f"**{start_time}** ({speaker_label}): {seg['text']}\n\n")
        
        return str(file_path)

    @staticmethod
    def _resolve_speaker_name(speaker_id, fingerprinter):
        """
        Returns a human-readable speaker label.
        Prefers fingerprinter metadata (e.g., 'Male Speaker 001'),
        falls back to speaker_id, then 'Unknown'.
        """
        if speaker_id == "unknown":
            return "Unknown Speaker"
        
        if fingerprinter and hasattr(fingerprinter, 'speakers'):
            speaker_data = fingerprinter.speakers.get(speaker_id)
            if speaker_data:
                metadata = speaker_data.get("metadata", {})
                name = metadata.get("name", "").strip()
                if name:
                    return name
        
        # Fallback: format raw ID nicely e.g. speaker_001 -> Speaker 001
        if speaker_id.startswith("speaker_"):
            idx = speaker_id.split("_")[-1]
            return f"Speaker {idx}"
        
        return speaker_id

    @staticmethod
    def _format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
