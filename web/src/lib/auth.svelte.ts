/**
 * Global auth state (Svelte 5 runes). One `$state` object, mutated in
 * place -- imported anywhere as `auth.user`.
 *
 *   auth.ready  -> false until the initial /me check resolves
 *   auth.user   -> {id, username} or null
 */
import { api, setToken, getToken, ApiError } from './api';

type User = { id: number; username: string };

export const auth = $state<{ user: User | null; ready: boolean }>({
	user: null,
	ready: false
});

/** Called once on app mount: validate any stored token. */
export async function refresh(): Promise<void> {
	if (!getToken()) {
		auth.user = null;
		auth.ready = true;
		return;
	}
	try {
		const { user } = await api<{ user: User }>('/api/auth/me');
		auth.user = user;
	} catch (err) {
		if (err instanceof ApiError && err.status === 401) setToken(null);
		auth.user = null;
	}
	auth.ready = true;
}

export async function authenticate(
	kind: 'login' | 'register',
	username: string,
	password: string
): Promise<void> {
	const { token, user } = await api<{ token: string; user: User }>(`/api/auth/${kind}`, {
		method: 'POST',
		json: { username, password }
	});
	setToken(token);
	auth.user = user;
}

export function logout(): void {
	setToken(null);
	auth.user = null;
}
