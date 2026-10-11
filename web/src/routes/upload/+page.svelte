<script lang="ts">
	import { createMutation, useQueryClient } from '@tanstack/svelte-query';
	import { api, ApiError } from '$lib/api';
	import { auth } from '$lib/auth.svelte';
	import type { UploadResult } from '$lib/types';

	const client = useQueryClient();
	let file = $state<File | null>(null);
	let caption = $state('');

	const upload = createMutation<UploadResult, Error, void>(() => ({
		mutationFn: () => {
			const form = new FormData();
			form.append('image', file!);
			if (caption.trim()) form.append('caption', caption.trim());
			return api<UploadResult>('/api/images', { method: 'POST', body: form });
		},
		onSuccess: () => {
			client.invalidateQueries({ queryKey: ['feed'] });
			client.invalidateQueries({ queryKey: ['me', 'images'] });
		}
	}));

	function pick(event: Event) {
		file = (event.target as HTMLInputElement).files?.[0] ?? null;
	}
	function submit(event: SubmitEvent) {
		event.preventDefault();
		if (file) upload.mutate();
	}
	function again() {
		upload.reset();
		file = null;
		caption = '';
	}
</script>

<h2>post a horse</h2>

{#if !auth.ready}
	<p class="note">…</p>
{:else if !auth.user}
	<p class="note">you need to <a href="/login">log in</a> to post.</p>
{:else if upload.isSuccess}
	<div class="result">
		{#if upload.data.passed}
			<p><b>your horse is up.</b> <a href="/">see it in the feed &rarr;</a></p>
		{:else}
			<p><b>not horse enough!</b></p>
			<p class="note">
				you were {Math.max(0.01, upload.data.threshold - upload.data.score).toFixed(2)} horse short
				of the line.
			</p>
		{/if}
	</div>
	<button class="btn" type="button" onclick={again}>post another</button>
{:else}
	<form onsubmit={submit}>
		<fieldset>
			<legend>your picture</legend>
			<label for="f">choose a file</label>
			<input id="f" type="file" accept="image/*" onchange={pick} />
			<label for="c">say something about it (optional)</label>
			<input id="c" type="text" maxlength="280" bind:value={caption} />
			<button type="submit" disabled={!file || upload.isPending}>
				{upload.isPending ? 'checking…' : 'post it'}
			</button>
		</fieldset>
	</form>

	{#if upload.isError}
		<p class="note err">
			{upload.error instanceof ApiError ? upload.error.message : 'upload failed'}
		</p>
	{/if}

	<p class="note">every picture has to pass the horse test</p>
{/if}
