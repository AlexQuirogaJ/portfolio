import netlify from '@astrojs/netlify';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// 
// https://astro.build/config
//export default defineConfig({});

// DEPLOYMENT CONFIGURATION
import tailwind from "@astrojs/tailwind";
export default defineConfig({
  adapter: netlify(),
  output: 'hybrid',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  },
  integrations: [tailwind()]
});