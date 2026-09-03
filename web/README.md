# web/ — Horse frontend

SvelteKit (Svelte 5 runes) + `@tanstack/svelte-query`. Talks to the Flask
API in `server/` over CORS.

## Run

```bash
cd web
npm install
cp .env.example .env          # PUBLIC_API_BASE=http://localhost:8000
npm run dev                   # http://localhost:5173
```

The API (`python -m server.wsgi`) must be running too.

## Layout

| path | role |
|---|---|
| `src/app.css` | **the whole stylesheet** — one hand-written global sheet, ported from `prototypes/`. Edit the look here. |
| `src/lib/api.ts` | fetch wrapper; JWT from localStorage; throws `ApiError` |
| `src/lib/auth.svelte.ts` | global `$state` auth (`auth.user`, `auth.ready`) |
| `src/lib/query.ts` | shared `QueryClient` |
| `src/lib/types.ts` | API response shapes |
| `src/lib/format.ts` | `timeAgo()` for bylines |
| `src/lib/components/RateWidget.svelte` | "rate this horse" 1–5 buttons + average |
| `src/routes/+layout.svelte` | `QueryClientProvider`, masthead + nav + footer, token check, the "website is a horse" footer flip |
| `src/routes/+page.svelte` | feed — `createInfiniteQuery` + `IntersectionObserver`, `<article class="post">` per item |
| `src/routes/login/+page.svelte` | two stacked forms (log in / make an account) |
| `src/routes/upload/+page.svelte` | file pick → `POST /api/images` → "your horse is up" / "not horse enough" |
| `prototypes/*.html` | standalone static mockups the routes were ported from — reference only, will drift |

## Notes

- The look is deliberately plain / old-web and light-only. All styling is
  global in `src/app.css`; route components use the classes, not scoped
  `<style>`.
- `adapter-auto` is still in place — pick a real adapter when a deploy
  target exists.
- No route guards / SSR auth: `auth.svelte.ts` is client-only. Fine for
  now; revisit if pages need server-side protection.
- `/me`, `/about`, `/rules` are linked in the chrome but not built yet.
