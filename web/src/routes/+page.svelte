<script lang="ts">
	import { createInfiniteQuery } from '@tanstack/svelte-query';
	import { api, apiBase } from '$lib/api';
	import { timeAgo } from '$lib/format';
	import type { FeedPage } from '$lib/types';
	import RateWidget from '$lib/components/RateWidget.svelte';

	const feed = createInfiniteQuery<FeedPage, Error, { pages: FeedPage[] }, string[], number>(
		() => ({
			queryKey: ['feed'],
			queryFn: ({ pageParam }) => api<FeedPage>(`/api/feed?limit=12&offset=${pageParam}`),
			initialPageParam: 0,
			getNextPageParam: (last) => last.next_offset ?? undefined
		})
	);

	const items = $derived(feed.data?.pages.flatMap((p) => p.items) ?? []);

	// Infinite scroll: watch a sentinel at the end of the list.
	let sentinel = $state<HTMLDivElement | null>(null);
	$effect(() => {
		if (!sentinel) return;
		const io = new IntersectionObserver((entries) => {
			if (entries[0].isIntersecting && feed.hasNextPage && !feed.isFetchingNextPage) {
				feed.fetchNextPage();
			}
		});
		io.observe(sentinel);
		return () => io.disconnect();
	});
</script>

{#if feed.isPending}
	<p class="feed-msg">loading&hellip;</p>
{:else if feed.isError}
	<p class="feed-msg err">{feed.error.message}</p>
{:else if items.length === 0}
	<p class="feed-msg">nothing here yet. <a href="/upload">post a horse.</a></p>
{:else}
	{#each items as item (item.id)}
		<article class="post">
			<img class="photo" src={`${apiBase}${item.image_url}`} alt={item.caption ?? 'a horse'} loading="lazy" />
			{#if item.caption}<p class="caption">{item.caption}</p>{/if}
			<p class="byline"><b>{item.owner}</b> &nbsp;&middot;&nbsp; {timeAgo(item.created_at)}</p>
			<RateWidget imageId={item.id} mean={item.rating_mean} count={item.rating_count} />
		</article>
	{/each}

	<div bind:this={sentinel} class="sentinel">
		{#if feed.isFetchingNextPage}
			loading more&hellip;
		{:else if !feed.hasNextPage}
			&mdash; end &mdash;
		{/if}
	</div>
{/if}
