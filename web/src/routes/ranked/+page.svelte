<script lang="ts">
	// Every horse, most horse first.
	import Feed from '$lib/components/Feed.svelte';
</script>

<p class="note">every horse on phorse, from most horse to least.</p>

<Feed kind="ranked">
	{#snippet empty()}
		nothing here yet. <a href="/upload">post a horse.</a>
	{/snippet}
</Feed>
