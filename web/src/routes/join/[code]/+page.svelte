<script lang="ts">
	// Sign-up page, reached only through an invite link: /join/<code>
	// (made by `python -m server.invites create`). The code comes from the
	// URL, so invitees never type it.
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { auth, authenticate } from '$lib/auth.svelte';
	import { ApiError } from '$lib/api';

	let reg = $state({ username: '', password: '', err: '', busy: false });

	// Already signed in? Nothing to do here.
	$effect(() => {
		if (auth.ready && auth.user) goto('/');
	});

	async function signUp() {
		reg.err = '';
		reg.busy = true;
		try {
			await authenticate('register', reg.username, reg.password, page.params.code);
			goto('/');
		} catch (err) {
			reg.err = err instanceof ApiError ? err.message : 'something went wrong';
		} finally {
			reg.busy = false;
		}
	}
</script>

<p class="note">
	phorse is a website for pictures of horses. anyone can post one; everyone decides whether it
	counts. you've been invited, so pick a name and a password and you're in.
</p>

<form
	onsubmit={(e) => {
		e.preventDefault();
		signUp();
	}}
>
	<fieldset>
		<legend>make an account</legend>
		<label for="reg-u">pick a username</label>
		<input id="reg-u" type="text" autocomplete="username" bind:value={reg.username} />
		<label for="reg-p">pick a password</label>
		<input id="reg-p" type="password" autocomplete="new-password" bind:value={reg.password} />
		<p class="note">three or more letters for the name, eight or more for the password.</p>
		{#if reg.err}<p class="note err">{reg.err}</p>{/if}
		<button type="submit" disabled={reg.busy}>{reg.busy ? '…' : 'make account'}</button>
	</fieldset>
</form>

<p class="note">already have an account? <a href="/login">log in</a>.</p>
