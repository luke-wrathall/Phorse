<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { goto } from '$app/navigation';
	import { api, apiBase } from '$lib/api';
	import { timeAgo } from '$lib/format';
	import { auth } from '$lib/auth.svelte';
	import type { MyImage } from '$lib/types';

	// Client-only guard — send guests to the login page.
	$effect(() => {
		if (auth.ready && !auth.user) goto('/login');
	});

	const mine = createQuery<{ items: MyImage[] }>(() => ({
		queryKey: ['me', 'images'],
		queryFn: () => api('/api/images/mine'),
		enabled: !!auth.user
	}));

	const posts = $derived(mine.data?.items ?? []);
	const up = $derived(posts.filter((p) => p.passed));
	const rejected = $derived(posts.filter((p) => !p.passed));
</script>

{#if auth.user}
	<h2>your page</h2>
	<p class="byline"><b>{auth.user.username}</b></p>

	{#if mine.isPending}
		<p class="feed-msg">loading&hellip;</p>
	{:else if mine.isError}
		<p class="feed-msg err">{mine.error.message}</p>
	{:else if posts.length === 0}
		<p class="feed-msg">you haven&rsquo;t posted a horse yet. <a href="/upload">post one.</a></p>
	{:else}
		{#each up as item (item.id)}
			<article class="post">
				<img
					class="photo"
					src={`${apiBase}${item.image_url}`}
					alt={item.caption ?? 'a horse'}
					loading="lazy"
				/>
				{#if item.caption}<p class="caption">{item.caption}</p>{/if}
				<p class="byline">
					{timeAgo(item.created_at)}
					{#if item.rating_count > 0 && item.rating_mean !== null}
						&nbsp;&middot;&nbsp; {item.rating_mean.toFixed(1)} out of 5, from {item.rating_count}
						{item.rating_count === 1 ? 'person' : 'people'}
					{:else}
						&nbsp;&middot;&nbsp; no votes yet
					{/if}
				</p>
			</article>
		{/each}

		{#if rejected.length > 0}
			<hr />
			<p class="note">these didn&rsquo;t make it through:</p>
			<ul class="plain-list">
				{#each rejected as item (item.id)}
					<li>a picture that wasn&rsquo;t horse enough &nbsp;&middot;&nbsp; {timeAgo(item.created_at)}</li>
				{/each}
			</ul>
		{/if}
	{/if}
{/if}
