// Pure client-side app: every page talks to the JSON API from the browser,
// so there is nothing to render on a server. Required by adapter-static's
// SPA fallback (vite.config.ts).
export const ssr = false;
