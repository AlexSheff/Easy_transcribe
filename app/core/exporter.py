import os
from pathlib import Path
from datetime import datetime

class MarkdownExporter:
    @staticmethod
    def export(segments, input_filename, output_dir="app/output"):
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
                f.write(f"**{start_time}** ({seg.get('speaker_id', 'Unknown')}): {seg['text']}\n\n")
        
        return str(file_path)

    @staticmethod
    def _format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
