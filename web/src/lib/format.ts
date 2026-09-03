/** "2 hours ago" from a unix timestamp in seconds (what the API returns). */
export function timeAgo(unixSeconds: number): string {
	const secs = Math.max(0, Date.now() / 1000 - unixSeconds);
	const mins = Math.floor(secs / 60);
	if (mins < 1) return 'just now';
	if (mins < 60) return `${mins} minute${mins === 1 ? '' : 's'} ago`;
	const hrs = Math.floor(mins / 60);
	if (hrs < 24) return `${hrs} hour${hrs === 1 ? '' : 's'} ago`;
	const days = Math.floor(hrs / 24);
	return `${days} day${days === 1 ? '' : 's'} ago`;
}
