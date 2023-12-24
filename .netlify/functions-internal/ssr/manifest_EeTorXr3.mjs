import '@astrojs/internal-helpers/path';
import 'cookie';
import 'kleur/colors';
import 'string-width';
import './chunks/astro_YYLxitrD.mjs';
import 'clsx';
import { compile } from 'path-to-regexp';

if (typeof process !== "undefined") {
  let proc = process;
  if ("argv" in proc && Array.isArray(proc.argv)) {
    if (proc.argv.includes("--verbose")) ; else if (proc.argv.includes("--silent")) ; else ;
  }
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
  return toPath;
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
    })
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
  const clientDirectives = new Map(serializedManifest.clientDirectives);
  return {
    ...serializedManifest,
    assets,
    componentMetadata,
    clientDirectives,
    routes
  };
}

const manifest = deserializeManifest({"adapterName":"@astrojs/netlify","routes":[{"file":"index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/","type":"page","pattern":"^\\/$","segments":[],"params":[],"component":"src/pages/index.astro","pathname":"/","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"projects/portfolio/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/projects/portfolio","type":"page","pattern":"^\\/projects\\/portfolio\\/?$","segments":[[{"content":"projects","dynamic":false,"spread":false}],[{"content":"portfolio","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/projects/portfolio.md","pathname":"/projects/portfolio","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"projects/fdm/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/projects/fdm","type":"page","pattern":"^\\/projects\\/fdm\\/?$","segments":[[{"content":"projects","dynamic":false,"spread":false}],[{"content":"fdm","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/projects/fdm.md","pathname":"/projects/fdm","prerender":true,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}},{"file":"","links":[],"scripts":[],"styles":[],"routeData":{"type":"endpoint","route":"/_image","pattern":"^\\/_image$","segments":[[{"content":"_image","dynamic":false,"spread":false}]],"params":[],"component":"node_modules/astro/dist/assets/endpoint/generic.js","pathname":"/_image","prerender":false,"fallbackRoutes":[],"_meta":{"trailingSlash":"ignore"}}}],"base":"/","trailingSlash":"ignore","compressHTML":true,"componentMetadata":[["/home/alexquiroga/Storage/GitHub/portfolio/src/pages/index.astro",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/portfolio/src/pages/projects/fdm.md",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/portfolio/src/pages/projects/portfolio.md",{"propagation":"in-tree","containsHead":true}],["/home/alexquiroga/Storage/GitHub/portfolio/src/layouts/Layout.astro",{"propagation":"in-tree","containsHead":false}],["/home/alexquiroga/Storage/GitHub/portfolio/src/layouts/ProjectLayout.astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/projects/fdm@_@md",{"propagation":"in-tree","containsHead":false}],["\u0000@astrojs-ssr-virtual-entry",{"propagation":"in-tree","containsHead":false}],["/home/alexquiroga/Storage/GitHub/portfolio/src/components/Projects.astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/index@_@astro",{"propagation":"in-tree","containsHead":false}],["\u0000@astro-page:src/pages/projects/portfolio@_@md",{"propagation":"in-tree","containsHead":false}]],"renderers":[],"clientDirectives":[["idle","(()=>{var i=t=>{let e=async()=>{await(await t())()};\"requestIdleCallback\"in window?window.requestIdleCallback(e):setTimeout(e,200)};(self.Astro||(self.Astro={})).idle=i;window.dispatchEvent(new Event(\"astro:idle\"));})();"],["load","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).load=e;window.dispatchEvent(new Event(\"astro:load\"));})();"],["media","(()=>{var s=(i,t)=>{let a=async()=>{await(await i())()};if(t.value){let e=matchMedia(t.value);e.matches?a():e.addEventListener(\"change\",a,{once:!0})}};(self.Astro||(self.Astro={})).media=s;window.dispatchEvent(new Event(\"astro:media\"));})();"],["only","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).only=e;window.dispatchEvent(new Event(\"astro:only\"));})();"],["visible","(()=>{var r=(i,c,s)=>{let n=async()=>{await(await i())()},t=new IntersectionObserver(e=>{for(let o of e)if(o.isIntersecting){t.disconnect(),n();break}});for(let e of s.children)t.observe(e)};(self.Astro||(self.Astro={})).visible=r;window.dispatchEvent(new Event(\"astro:visible\"));})();"]],"entryModules":{"\u0000@astrojs-ssr-virtual-entry":"entry.mjs","\u0000@astro-renderers":"renderers.mjs","\u0000empty-middleware":"_empty-middleware.mjs","\u0000@astrojs-manifest":"manifest_EeTorXr3.mjs","\u0000@astro-page:node_modules/astro/dist/assets/endpoint/generic@_@js":"chunks/generic_LBimRP9R.mjs","\u0000@astro-page:src/pages/index@_@astro":"chunks/index_vGLqanxF.mjs","\u0000@astro-page:src/pages/projects/portfolio@_@md":"chunks/portfolio_6A8MhuoZ.mjs","\u0000@astro-page:src/pages/projects/fdm@_@md":"chunks/fdm_H0c8AWL5.mjs","/astro/hoisted.js?q=0":"_astro/hoisted.4PK_pqbL.js","astro:scripts/before-hydration.js":""},"assets":["/_astro/index.nZTEqXt6.css","/favicon.svg","/favicon_old.svg","/profile (copy).png","/profile.png","/profile_2.png","/profile_3.png","/_astro/hoisted.4PK_pqbL.js","/files/CV.pdf","/files/post_img.webp","/images/fdm.png","/images/portfolio.png","/index.html","/projects/portfolio/index.html","/projects/fdm/index.html"]});

export { manifest };
