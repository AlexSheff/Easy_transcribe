import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

console.log('===================================================');
console.log('  EASY TSCRIBE - SETUP');
console.log('===================================================\n');

console.log('Installing dependencies via npm...');
const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';

const res = spawnSync(npmCmd, ['install'], { stdio: 'inherit', shell: true });

if (res.status !== 0 || !existsSync('./node_modules/vite')) {
  console.log('\nRetrying with --legacy-peer-deps...');
  spawnSync(npmCmd, ['install', '--legacy-peer-deps'], { stdio: 'inherit', shell: true });
}

if (existsSync('./node_modules/vite')) {
  console.log('\n===================================================');
  console.log('[SUCCESS] Dependencies installed successfully!');
  console.log('Run "npm run dev" or double-click "run.bat" to start.');
  console.log('===================================================\n');
} else {
  console.error('\n[ERROR] Failed to install dependencies. Please run "npm install" manually.');
  process.exit(1);
}
