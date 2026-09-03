/**
 * Thin fetch wrapper around the Flask API.
 *
 * Auth is a JWT kept in localStorage (CLAUDE.md: token-based, cross-origin
 * SPA). Every call attaches `Authorization: Bearer <token>` when we have
 * one. Non-2xx responses become a thrown `ApiError` carrying the server's
 * `{error}` message, which svelte-query surfaces as `query.error`.
 */
import { PUBLIC_API_BASE } from '$env/static/public';

const TOKEN_KEY = 'horse_token';

export function getToken(): string | null {
	if (typeof localStorage === 'undefined') return null;
	return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
	if (typeof localStorage === 'undefined') return;
	if (token) localStorage.setItem(TOKEN_KEY, token);
	else localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
	status: number;
	constructor(message: string, status: number) {
		super(message);
		this.status = status;
	}
}

type ApiOptions = Omit<RequestInit, 'body'> & { json?: unknown; body?: BodyInit };

export async function api<T = unknown>(path: string, opts: ApiOptions = {}): Promise<T> {
	const headers = new Headers(opts.headers);
	const token = getToken();
	if (token) headers.set('Authorization', `Bearer ${token}`);

	let body = opts.body;
	if (opts.json !== undefined) {
		headers.set('Content-Type', 'application/json');
		body = JSON.stringify(opts.json);
	}

	const res = await fetch(`${PUBLIC_API_BASE}${path}`, { ...opts, headers, body });
	const data = res.status === 204 ? null : await res.json().catch(() => null);
	if (!res.ok) {
		throw new ApiError((data && data.error) || res.statusText, res.status);
	}
	return data as T;
}

export const apiBase = PUBLIC_API_BASE;
