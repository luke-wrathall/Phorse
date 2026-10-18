<script lang="ts">
	import { goto } from '$app/navigation';
	import { auth, authenticate } from '$lib/auth.svelte';
	import { ApiError } from '$lib/api';

	let li = $state({ username: '', password: '', err: '', busy: false });

	// Already signed in? Nothing to do here.
	$effect(() => {
		if (auth.ready && auth.user) goto('/');
	});

	async function logIn() {
		li.err = '';
		li.busy = true;
		try {
			await authenticate('login', li.username, li.password);
			goto('/');
		} catch (err) {
			li.err = err instanceof ApiError ? err.message : 'something went wrong';
		} finally {
			li.busy = false;
		}
	}
</script>

<p class="note">
	phorse is a website for pictures of horses. anyone can post one; everyone decides whether it
	counts. you need an account to join in.
</p>

<form
	onsubmit={(e) => {
		e.preventDefault();
		logIn();
	}}
>
	<fieldset>
		<legend>log in</legend>
		<label for="li-u">username</label>
		<input id="li-u" type="text" autocomplete="username" bind:value={li.username} />
		<label for="li-p">password</label>
		<input id="li-p" type="password" autocomplete="current-password" bind:value={li.password} />
		{#if li.err}<p class="note err">{li.err}</p>{/if}
		<button type="submit" disabled={li.busy}>{li.busy ? '…' : 'log in'}</button>
	</fieldset>
</form>

<p class="note">no account? phorse is invite-only for now: you need an invite link to make one.</p>
