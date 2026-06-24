import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      base: '/studio/',
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify("AIzaSyAuqFL8VJ1pQS0ZvVwQEwY4RIwH-wJ7qa4"),
        'process.env.GEMINI_API_KEY': JSON.stringify("AIzaSyAuqFL8VJ1pQS0ZvVwQEwY4RIwH-wJ7qa4")
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
