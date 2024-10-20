import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import { configDefaults, defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [vue()],

  test: {
    globals: true,
    environment: 'happy-dom',
    exclude: [...configDefaults.exclude, 'e2e/**', '.vercel/**'],
    root: fileURLToPath(new URL('./', import.meta.url))
  }
})
