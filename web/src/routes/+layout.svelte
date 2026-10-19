<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { QueryClientProvider } from '@tanstack/svelte-query';
	import { queryClient } from '$lib/query';
	import { auth, refresh, logout } from '$lib/auth.svelte';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();

	// The footer usually reads "horse is a website"; now and then it doesn't.
	let whatis = $state('horse is a website');

	onMount(() => {
		refresh(); // validate any stored token
		if (Math.random() < 0.15) whatis = 'website is a horse';
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>phorse</title>
</svelte:head>

<QueryClientProvider client={queryClient}>
	<header class="masthead">
		<h1><a href="/">phorse</a></h1>
		<p class="tagline">pictures of horses, decide how horsey they really are</p>
		<nav>
			<!-- /me, /about, /rules aren't built yet — they 404 for now -->
			<a href="/">feed</a>
			&nbsp;|&nbsp; <a href="/ranked">most horse</a>
			{#if auth.ready && auth.user}
				&nbsp;|&nbsp; <a href="/upload">post a horse</a>
				&nbsp;|&nbsp; <a href="/me">your page</a>
				&nbsp;|&nbsp; <button type="button" onclick={logout}>log out</button>
			{:else if auth.ready}
				&nbsp;|&nbsp; <a href="/login">log in</a>
			{/if}
		</nav>
	</header>

	<hr />

	<main>
		{@render children()}
	</main>

	<hr />

	<footer class="note">
		<p><span>{whatis}</span>. <a href="/about">about</a> &nbsp;&middot;&nbsp; <a href="/rules">rules</a></p>
	</footer>
</QueryClientProvider>
