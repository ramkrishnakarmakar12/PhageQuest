import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true, rewrite: p => p.replace(/^\/api/, '') } },
  },
  // Pyodide is loaded from a CDN at runtime inside a worker, so it is never
  // bundled. Keeping it out of the build is what makes the first paint fast on
  // a shared school PC; the 6 MB runtime then loads once and is cached.
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
})
