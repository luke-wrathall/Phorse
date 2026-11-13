<script lang="ts">
	// Report a post (/report?post=<id>, from the link on each post) or
	// complain about anything else (/report, from the footer). No login
	// needed. Plain form controls on their own page, so it works by keyboard
	// and screen reader. The reasons must match services/reports.py REASONS.
	import { page } from '$app/state';
	import { api, ApiError } from '$lib/api';

	const REASONS = [
		['child_safety', 'sexual content involving a child'],
		['intimate_image', 'an intimate image of someone, shared without their consent'],
		['illegal', 'something else illegal (terrorism, threats, extreme violence...)'],
		['harmful', 'porn, gore or self-harm'],
		['hate', 'hate or harassment'],
		['copyright', "it's my copyright and i didn't post it"],
		['appeal', 'you took my post down and i think that was wrong'],
		['other', 'something else']
	];

	const post = page.url.searchParams.get('post');
	let reason = $state(page.url.searchParams.get('appeal') ? 'appeal' : '');
	let details = $state('');
	let contact = $state('');
	let err = $state('');
	let busy = $state(false);
	let sent = $state(false);

	const sensitive = $derived(reason === 'child_safety' || reason === 'intimate_image');

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		err = '';
		busy = true;
		try {
			await api('/api/reports', {
				method: 'POST',
				json: { reason, image_id: post, details, contact }
			});
			sent = true;
		} catch (e) {
			err = e instanceof ApiError ? e.message : 'that didn’t send, try again';
		} finally {
			busy = false;
		}
	}
</script>

<h2>{post ? 'report this post' : 'report a problem'}</h2>

{#if sent}
	<div class="result" role="status">
		<p><b>thanks, we've got it.</b> a person will look at it.</p>
		{#if contact.trim()}<p class="note">we'll use the address you gave if we need to reply.</p>{/if}
	</div>
	<p><a href="/">back to the feed</a></p>
{:else}
	<p class="note">just not much of a horse? that's what the vote buttons are for.</p>

	<form onsubmit={submit}>
		<fieldset>
			<legend>what's wrong?</legend>
			<label for="reason">reason</label>
			<select id="reason" bind:value={reason} required>
				<option value="" disabled>choose one</option>
				{#each REASONS as [value, label] (value)}
					<option {value}>{label}</option>
				{/each}
			</select>

			{#if sensitive}
				<div class="result" role="note">
					<p>
						<b>if someone is in immediate danger, call 999.</b>
						please don't download, screenshot or share the image, even to show us.
						the post is hidden as soon as you send this.
					</p>
					{#if reason === 'child_safety'}
						<p>
							you can also report child sexual abuse images directly to the
							<a href="https://report.iwf.org.uk/">internet watch foundation</a>.
						</p>
					{/if}
				</div>
			{/if}

			<label for="details">anything we should know{post ? ' (optional)' : ''}</label>
			<textarea id="details" rows="4" maxlength="2000" required={!post} bind:value={details}></textarea>

			<label for="contact">your email, if you'd like a reply (optional)</label>
			<input id="contact" type="email" maxlength="200" autocomplete="email" bind:value={contact} />

			<button type="submit" disabled={busy || !reason}>{busy ? 'sending…' : 'send'}</button>
		</fieldset>
	</form>

	{#if err}<p class="note err" role="alert">{err}</p>{/if}
{/if}
