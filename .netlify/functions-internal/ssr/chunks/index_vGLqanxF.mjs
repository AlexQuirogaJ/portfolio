export { renderers } from '../renderers.mjs';
export { onRequest } from '../_empty-middleware.mjs';

const page = () => import('./prerender_QURC0Rlw.mjs').then(n => n.i);

export { page };
