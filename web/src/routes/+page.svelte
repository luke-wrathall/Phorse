<script lang="ts">
	// Front page: horses you haven't rated yet (logged out: every horse).
	import Feed from '$lib/components/Feed.svelte';
	import { auth } from '$lib/auth.svelte';
</script>

<Feed kind="new">
	{#snippet empty()}
		{#if auth.user}
			you've rated every horse. <a href="/upload">post one</a>, or see
			<a href="/ranked">the most horse</a>.
		{:else}
			nothing here yet. <a href="/upload">post a horse.</a>
		{/if}
	{/snippet}
</Feed>
