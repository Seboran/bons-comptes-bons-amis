import { fileURLToPath, URL } from 'node:url'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from '@farmfe/core'

// https://vitejs.dev/config/
export default defineConfig({
  vitePlugins: [vue()],
  compilation: {
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    }
  }
})
