import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/simulate': 'http://localhost:8000',
      '/biomarkers': 'http://localhost:8000',
      '/agents': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
