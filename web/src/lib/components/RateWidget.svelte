<script lang="ts">
	import { createMutation, useQueryClient } from '@tanstack/svelte-query';
	import { goto } from '$app/navigation';
	import { api } from '$lib/api';
	import { auth } from '$lib/auth.svelte';

	let { imageId, mean, count }: { imageId: string; mean: number | null; count: number } = $props();

	const client = useQueryClient();
	const rate = createMutation(() => ({
		mutationFn: (value: number) =>
			api(`/api/images/${imageId}/ratings`, { method: 'POST', json: { rating: value } }),
		onSuccess: () => client.invalidateQueries({ queryKey: ['feed'] })
	}));

	function vote(n: number) {
		if (!auth.user) {
			goto('/login');
			return;
		}
		rate.mutate(n);
	}
</script>

<p class="rating">
	<span class="label">rate this horse:</span>
	<span class="stars">
		{#each [1, 2, 3, 4, 5] as n (n)}
			<button type="button" disabled={rate.isPending} onclick={() => vote(n)}>{n}</button>
		{/each}
	</span>
	&nbsp;
	{#if count > 0 && mean !== null}
		<span class="average"
			>{mean.toFixed(1)} out of 5, from {count} {count === 1 ? 'person' : 'people'}</span
		>
	{:else}
		<span class="average">no votes yet</span>
	{/if}
</p>
