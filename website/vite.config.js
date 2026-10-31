import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  appType: 'mpa',
  build: {
    rolldownOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        replay: fileURLToPath(new URL('./replay.html', import.meta.url)),
      },
    },
  },
});
