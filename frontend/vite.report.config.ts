import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

// レポート HTML のテンプレート: JS・CSS をすべてインライン化した 1 ファイルを作る
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: 'dist-report',
    emptyOutDir: true,
    rollupOptions: { input: 'report.html' },
  },
});
