import { QueryClient } from '@tanstack/svelte-query';

/** One client for the app, handed to <QueryClientProvider> in +layout.svelte. */
export const queryClient = new QueryClient({
	defaultOptions: {
		queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false }
	}
});
