<script lang="ts">
	import { goto } from '$app/navigation';
	import { auth, authenticate } from '$lib/auth.svelte';
	import { ApiError } from '$lib/api';

	// One state box per form.
	let li = $state({ username: '', password: '', err: '', busy: false });
	let reg = $state({ username: '', password: '', err: '', busy: false });

	// Already signed in? Nothing to do here.
	$effect(() => {
		if (auth.ready && auth.user) goto('/');
	});

	async function run(kind: 'login' | 'register', box: typeof li) {
		box.err = '';
		box.busy = true;
		try {
			await authenticate(kind, box.username, box.password);
			goto('/');
		} catch (err) {
			box.err = err instanceof ApiError ? err.message : 'something went wrong';
		} finally {
			box.busy = false;
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
		run('login', li);
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

<hr />

<form
	onsubmit={(e) => {
		e.preventDefault();
		run('register', reg);
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
