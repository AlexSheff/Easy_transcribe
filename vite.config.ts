import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// The dev server is reachable from this machine only. Saving transcripts and all
// heavy processing are handled by server_faster_whisper.py on 127.0.0.1:8000.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: 'localhost',
    port: 3000,
    strictPort: true, // the Python server only allows the http://localhost:3000 origin
  },
  preview: {
    host: 'localhost',
    port: 3000,
    strictPort: true,
  },
});
