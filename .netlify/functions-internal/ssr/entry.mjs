import { renderers } from './renderers.mjs';
import { manifest } from './manifest_CMlkkKP-.mjs';
import * as serverEntrypointModule from '@astrojs/netlify/ssr-function.js';
import { onRequest } from './_noop-middleware.mjs';

const _page0 = () => import('./chunks/generic_BSNU38C9.mjs');
const _page1 = () => import('./chunks/pid_ball_balance_D4qoVxO1.mjs');
const _page2 = () => import('./chunks/portfolio_BPSEmQqR.mjs');
const _page3 = () => import('./chunks/index_DigmUh-Q.mjs');
const pageMap = new Map([
    ["node_modules/astro/dist/assets/endpoint/generic.js", _page0],
    ["src/pages/projects/pid_ball_balance.md", _page1],
    ["src/pages/projects/portfolio.md", _page2],
    ["src/pages/index.astro", _page3]
]);

const _manifest = Object.assign(manifest, {
    pageMap,
    renderers,
    middleware: onRequest
});
const _args = undefined;
const _exports = serverEntrypointModule.createExports(_manifest, _args);
const __astrojsSsrVirtualEntry = _exports.default;
const _start = 'start';
if (_start in serverEntrypointModule) {
	serverEntrypointModule[_start](_manifest, _args);
}

export { __astrojsSsrVirtualEntry as default, pageMap };
