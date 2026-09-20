import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs';
import path from 'node:path';

function autoSaveMarkdownPlugin(): Plugin {
  return {
    name: 'auto-save-markdown-plugin',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url || '';
        
        // POST /api/save-markdown
        if (req.method === 'POST' && (url === '/api/save-markdown' || url.startsWith('/api/save-markdown?'))) {
          let body = '';
          req.on('data', (chunk) => {
            body += chunk;
          });
          req.on('end', () => {
            try {
              const data = JSON.parse(body);
              const filename = data.filename || `transcript_${Date.now()}.md`;
              const safeName = filename.endsWith('.md') ? filename : `${filename}.md`;
              const sanitizedName = safeName.replace(/[^a-zA-Z0-9._\- ]/g, '_');
              
              const outputDir = path.resolve(process.cwd(), 'transcripts');
              if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
              }

              const targetPath = path.join(outputDir, sanitizedName);
              fs.writeFileSync(targetPath, data.content || '', 'utf8');

              res.statusCode = 200;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({
                success: true,
                filename: sanitizedName,
                relativePath: `transcripts/${sanitizedName}`,
                fullPath: targetPath,
                size: (data.content || '').length,
                savedAt: new Date().toISOString()
              }));
            } catch (err: any) {
              res.statusCode = 500;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: false, error: err?.message || 'Failed to save markdown' }));
            }
          });
          return;
        }

        // GET /api/transcripts
        if (req.method === 'GET' && (url === '/api/transcripts' || url.startsWith('/api/transcripts?'))) {
          try {
            const outputDir = path.resolve(process.cwd(), 'transcripts');
            if (!fs.existsSync(outputDir)) {
              res.statusCode = 200;
              res.setHeader('Content-Type', 'application/json');
              return res.end(JSON.stringify({ files: [] }));
            }

            const files = fs.readdirSync(outputDir)
              .filter(f => f.endsWith('.md'))
              .map(name => {
                const stat = fs.statSync(path.join(outputDir, name));
                return {
                  name,
                  size: stat.size,
                  modifiedAt: stat.mtime.toISOString(),
                  relativePath: `transcripts/${name}`
                };
              })
              .sort((a, b) => new Date(b.modifiedAt).getTime() - new Date(a.modifiedAt).getTime());

            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ files }));
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: err?.message }));
          }
          return;
        }

        next();
      });
    }
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), autoSaveMarkdownPlugin()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: 'all'
  }
});
