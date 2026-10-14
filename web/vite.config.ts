import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},

			// Static single-page app: `npm run build` writes plain files to build/,
			// which any web server can serve (deploy/Caddyfile). Every unknown
			// path falls back to index.html and the client-side router takes it
			// from there -- see ssr = false in src/routes/+layout.ts.
			adapter: adapter({ fallback: 'index.html' })
		})
	]
});
