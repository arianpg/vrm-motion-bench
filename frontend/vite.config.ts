import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 開発時は /api をオーケストレーター（既定 :8800）へ転送する
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: { '/api': { target: process.env.VMB_BACKEND ?? 'http://127.0.0.1:8800', changeOrigin: true } },
  },
});
