import { defineConfig } from 'astro/config';
import netlify from '@astrojs/netlify'
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// // https://astro.build/config
// export default defineConfig({});

// DEPLOYMENT CONFIGURATION
export default defineConfig({
    adapter: netlify(),
    output: 'hybrid',
    markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex]
	}
  })
