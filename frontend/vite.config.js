import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      // Prevent MIME-type sniffing
      'X-Content-Type-Options': 'nosniff',
      // Prevent embedding in iframes
      'X-Frame-Options': 'DENY',
      // Don't send referrer info
      'Referrer-Policy': 'no-referrer',
      // Allow scripts/styles from same origin; inline styles needed for Vite HMR
      'Content-Security-Policy':
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
      // Disable browser features the app doesn't use
      'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
    },
    proxy: {
      '/config':  'http://localhost:5000',
      '/predict': 'http://localhost:5000',
      '/health':  'http://localhost:5000',
    },
  },
})
