# Alex Quiroga - Portfolio Web Page

## Page

This site is available at [Portfolio Web Page](https://alex-quiroga.com/).

> [!NOTE]
> The development of this page is still in progress.

## Development

### Astro Framework
- [Astro Framework](https://astro.build/)
- [Astro Docs](https://docs.astro.build/)

#### Start New Project
```bash
npm create astro@latest
```

#### Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
/
├── public/
│   ├── favicon.svg
│   ├── files/
│   └── images/
├── src/
│   ├── components/
│   ├── layouts/
│   └── pages/
│       ├── index.astro
│       └── projects/
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

#### Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

### Requirements


> [!TIP]
> - [How To Render LaTeX In Markdown With Astro.js](https://blog.alexafazio.dev/blog/render-latex-in-astro/)

```bash
npm install @astrojs/cloudflare
npm install @astrojs/tailwind
npm install remark-math
npm install rehype-katex
```

### Update packages

```bash
npm update
```

### Run locally and check build for deployment
```bash
npm run dev # Run locally
npm run build # Build for production
```
