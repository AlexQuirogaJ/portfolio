import '@astrojs/internal-helpers/path';
import 'cookie';
import 'kleur/colors';
import 'es-module-lexer';
import 'devalue';
import { g as decodeKey } from './chunks/astro/server_PnbVYqT_.mjs';
import 'clsx';
import 'html-escaper';
import { compile } from 'path-to-regexp';

const codeToStatusMap = {
  // Implemented from tRPC error code table
  // https://trpc.io/docs/server/error-handling#error-codes
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  TIMEOUT: 405,
  CONFLICT: 409,
  PRECONDITION_FAILED: 412,
  PAYLOAD_TOO_LARGE: 413,
  UNSUPPORTED_MEDIA_TYPE: 415,
  UNPROCESSABLE_CONTENT: 422,
  TOO_MANY_REQUESTS: 429,
  CLIENT_CLOSED_REQUEST: 499,
  INTERNAL_SERVER_ERROR: 500
};
Object.entries(codeToStatusMap).reduce(
  // reverse the key-value pairs
  (acc, [key, value]) => ({ ...acc, [value]: key }),
  {}
);

function sanitizeParams(params) {
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => {
      if (typeof value === "string") {
        return [key, value.normalize().replace(/#/g, "%23").replace(/\?/g, "%3F")];
      }
      return [key, value];
    })
  );
}
function getRouteGenerator(segments, addTrailingSlash) {
  const template = segments.map((segment) => {
    return "/" + segment.map((part) => {
      if (part.spread) {
        return `:${part.content.slice(3)}(.*)?`;
      } else if (part.dynamic) {
        return `:${part.content}`;
      } else {
        return part.content.normalize().replace(/\?/g, "%3F").replace(/#/g, "%23").replace(/%5B/g, "[").replace(/%5D/g, "]").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      }
    }).join("");
  }).join("");
  let trailing = "";
  if (addTrailingSlash === "always" && segments.length) {
    trailing = "/";
  }
  const toPath = compile(template + trailing);
  return (params) => {
    const sanitizedParams = sanitizeParams(params);
    const path = toPath(sanitizedParams);
    return path || "/";
  };
}

function deserializeRouteData(rawRouteData) {
  return {
    route: rawRouteData.route,
    type: rawRouteData.type,
    pattern: new RegExp(rawRouteData.pattern),
    params: rawRouteData.params,
    component: rawRouteData.component,
    generate: getRouteGenerator(rawRouteData.segments, rawRouteData._meta.trailingSlash),
    pathname: rawRouteData.pathname || void 0,
    segments: rawRouteData.segments,
    prerender: rawRouteData.prerender,
    redirect: rawRouteData.redirect,
    redirectRoute: rawRouteData.redirectRoute ? deserializeRouteData(rawRouteData.redirectRoute) : void 0,
    fallbackRoutes: rawRouteData.fallbackRoutes.map((fallback) => {
      return deserializeRouteData(fallback);
    }),
    isIndex: rawRouteData.isIndex
  };
}

function deserializeManifest(serializedManifest) {
  const routes = [];
  for (const serializedRoute of serializedManifest.routes) {
    routes.push({
      ...serializedRoute,
      routeData: deserializeRouteData(serializedRoute.routeData)
    });
    const route = serializedRoute;
    route.routeData = deserializeRouteData(serializedRoute.routeData);
  }
  const assets = new Set(serializedManifest.assets);
  const componentMetadata = new Map(serializedManifest.componentMetadata);
  const inlinedScripts = new Map(serializedManifest.inlinedScripts);
  const clientDirectives = new Map(serializedManifest.clientDirectives);
  const serverIslandNameMap = new Map(serializedManifest.serverIslandNameMap);
  const key = decodeKey(serializedManifest.key);
  return {
    // in case user middleware exists, this no-op middleware will be reassigned (see plugin-ssr.ts)
    middleware(_, next) {
      return next();
    },
    ...serializedManifest,
    assets,
    componentMetadata,
    inlinedScripts,
    clientDirectives,
    routes,
    serverIslandNameMap,
    key
  };
}

const manifest = deserializeManifest({"hrefRoot":"file:///home/alexquiroga/Storage/GitHub/WORKING/portfolio/","adapterName":"@astrojs/netlify","routes":[{"file":"projects/pid_ball_balance/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/projects/pid_ball_balance","isIndex":false,"type":"page","pattern":"^\\/projects\\/pid_ball_balance\\/?$","segments":[[{"content":"projects","dynamic":false,"spread":false}],[{"content":"pid_ball_balance","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/projects/pid_ball_balance.md","pathname":"/projects/pid_ball_balance","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"projects/portfolio/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/projects/portfolio","isIndex":false,"type":"page","pattern":"^\\/projects\\/portfolio\\/?$","segments":[[{"content":"projects","dynamic":false,"spread":false}],[{"content":"portfolio","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/projects/portfolio.md","pathname":"/projects/portfolio","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/","isIndex":true,"type":"page","pattern":"^\\/$","segments":[],"params":[],"component":"src/pages/index.astro","pathname":"/","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"","links":[],"scripts":[],"styles":[],"routeData":{"type":"endpoint","isIndex":false,"route":"/_image","pattern":"^\\/_image$","segments":[[{"content":"_image","dynamic":false,"spread":false}]],"params":[],"component":"node_modules/astro/dist/assets/endpoint/generic.js","pathname":"/_image","prerender":false,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}}],"base":"/","trailingSlash":"ignore","compressHTML":true,"componentMetadata":[["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/pages/index.astro",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/pages/projects/pid_ball_balance.md",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/pages/projects/portfolio.md",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/layouts/BaseProjectLayout.astro",{"propagation":"in-tree","containsHead":false}],["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/layouts/ProjectLayout.astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/projects/pid_ball_balance@_@md",{"propagation":"in-tree","containsHead":false}],["\u0000@astrojs-ssr-virtual-entry",{"propagation":"in-tree","containsHead":false}],["/home/alexquiroga/Storage/GitHub/WORKING/portfolio/src/components/Projects.astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/index@_@astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/projects/portfolio@_@md",{"propagation":"in-tree","containsHead":false}]],"renderers":[],"clientDirectives":[["idle","(()=>{var l=(o,t)=>{let i=async()=>{await(await o())()},e=typeof t.value==\"object\"?t.value:void 0,s={timeout:e==null?void 0:e.timeout};\"requestIdleCallback\"in window?window.requestIdleCallback(i,s):setTimeout(i,s.timeout||200)};(self.Astro||(self.Astro={})).idle=l;window.dispatchEvent(new Event(\"astro:idle\"));})();"],["load","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).load=e;window.dispatchEvent(new Event(\"astro:load\"));})();"],["media","(()=>{var s=(i,t)=>{let a=async()=>{await(await i())()};if(t.value){let e=matchMedia(t.value);e.matches?a():e.addEventListener(\"change\",a,{once:!0})}};(self.Astro||(self.Astro={})).media=s;window.dispatchEvent(new Event(\"astro:media\"));})();"],["only","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).only=e;window.dispatchEvent(new Event(\"astro:only\"));})();"],["visible","(()=>{var l=(s,i,o)=>{let r=async()=>{await(await s())()},t=typeof i.value==\"object\"?i.value:void 0,c={rootMargin:t==null?void 0:t.rootMargin},n=new IntersectionObserver(e=>{for(let a of e)if(a.isIntersecting){n.disconnect(),r();break}},c);for(let e of o.children)n.observe(e)};(self.Astro||(self.Astro={})).visible=l;window.dispatchEvent(new Event(\"astro:visible\"));})();"]],"entryModules":{"\u0000@astro-page:node_modules/astro/dist/assets/endpoint/generic@_@js":"pages/_image.astro.mjs","\u0000@astro-page:src/pages/projects/pid_ball_balance@_@md":"pages/projects/pid_ball_balance.astro.mjs","\u0000@astro-page:src/pages/projects/portfolio@_@md":"pages/projects/portfolio.astro.mjs","\u0000@astro-page:src/pages/index@_@astro":"pages/index.astro.mjs","\u0000@astrojs-ssr-virtual-entry":"entry.mjs","\u0000noop-middleware":"_noop-middleware.mjs","\u0000@astro-renderers":"renderers.mjs","\u0000@astrojs-ssr-adapter":"_@astrojs-ssr-adapter.mjs","\u0000@astrojs-manifest":"manifest_ZKB3Q1mi.mjs","/astro/hoisted.js?q=0":"_astro/hoisted.CmkSwYHY.js","astro:scripts/before-hydration.js":""},"inlinedScripts":[],"assets":["/_astro/index.BxbmtrgH.css","/_astro/index.CgYe1ijp.css","/favicon.svg","/favicon_old.svg","/profile.jpg","/_astro/hoisted.CmkSwYHY.js","/files/CV.pdf","/files/post_img.webp","/images/fdm.png","/images/portfolio-old.png","/images/portfolio.png","/logos/UPM_logo.svg","/logos/UPM_logo_complete.png","/logos/srm.png","/logos/upm.png","/images/projects/pid_ball_balance/catia_design.png","/images/projects/pid_ball_balance/circuit_assembled.jpg","/images/projects/pid_ball_balance/circuit_schematic.png","/images/projects/pid_ball_balance/front_photo.jpg","/images/projects/pid_ball_balance/imagej_data_extraction.png","/images/projects/pid_ball_balance/ir_sensor_curve_fit.png","/images/projects/pid_ball_balance/ir_sensor_graph.png","/projects/pid_ball_balance/index.html","/projects/portfolio/index.html","/index.html"],"buildFormat":"directory","checkOrigin":false,"serverIslandNameMap":[],"key":"qq2l29iCxdf3IQLUzI/tBiS25tCAaAy6FNzQUF6t2vc=","experimentalEnvGetSecretEnabled":false});

export { manifest };
