import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// DEPLOYMENT CONFIGURATION
import cloudflare from '@astrojs/cloudflare';
import tailwind from "@astrojs/tailwind";
export default defineConfig({
  adapter: cloudflare(),
  output: 'static',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  },
  integrations: [tailwind()]
});