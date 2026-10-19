<script lang="ts">
	// Six big 0-5 buttons, one tap to vote (web/prototypes/rating.html, "B").
	// 0-5 matches the model's scale: 0 is the not-horse anchors' target.
	// Your vote stays pressed in; tap another number to change it.
	import { createMutation, useQueryClient } from '@tanstack/svelte-query';
	import { goto } from '$app/navigation';
	import { api } from '$lib/api';
	import { auth } from '$lib/auth.svelte';

	let {
		imageId,
		mean,
		count,
		mine
	}: { imageId: string; mean: number | null; count: number; mine: number | null } = $props();

	const VALUES = [0, 1, 2, 3, 4, 5];

	// Shown pressed straight away on tap; the feed refetch then brings the
	// server's copy in `mine`.
	let picked = $state<number | null>(null);
	const current = $derived(picked ?? mine);

	const client = useQueryClient();
	const rate = createMutation(() => ({
		mutationFn: (value: number) =>
			api(`/api/images/${imageId}/ratings`, { method: 'POST', json: { rating: value } }),
		onSuccess: () => client.invalidateQueries({ queryKey: ['feed'] }),
		onError: () => (picked = null)
	}));

	function vote(n: number) {
		if (!auth.user) {
			goto('/login');
			return;
		}
		picked = n;
		rate.mutate(n);
	}

	const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
</script>

<div class="rating">
	<p class="label">how horse is this?</p>
	<div class="big">
		{#each VALUES as n (n)}
			<button
				type="button"
				aria-pressed={current === n}
				disabled={rate.isPending}
				onclick={() => vote(n)}>{n}</button
			>
		{/each}
	</div>
	<p class="ends"><span>&larr; not a horse</span><span>super horse &rarr;</span></p>
	<p class="average">
		{#if current !== null}you said <b>{fmt(current)}</b> &nbsp;&middot;&nbsp;{/if}
		{#if count > 0 && mean !== null}
			{mean.toFixed(1)} out of 5, from {count} {count === 1 ? 'person' : 'people'}
		{:else}
			no votes yet
		{/if}
	</p>
	{#if rate.isError}<p class="note err">{rate.error.message}</p>{/if}
</div>
