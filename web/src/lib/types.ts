export type FeedItem = {
	id: string;
	owner: string;
	score: number;
	caption: string | null;
	created_at: number;
	rating_count: number;
	rating_mean: number | null;
	/** The viewer's own rating of this post; null if they haven't, or logged out. */
	my_rating: number | null;
	image_url: string;
};

export type FeedPage = { items: FeedItem[]; next_offset: number | null };

/** One of the caller's own uploads (GET /api/images/mine) — includes rejected and removed ones. */
export type MyImage = {
	id: string;
	caption: string | null;
	passed: boolean;
	removed: boolean;
	created_at: number;
	rating_count: number;
	rating_mean: number | null;
	/** Null when there's nothing to show: rejected at the gate, or taken down. */
	image_url: string | null;
};

/** GET /api/images/limits — the server's upload caps, for a pre-check before sending. */
export type UploadLimits = { max_bytes: number; max_pixels: number };

export type UploadResult = {
	id: string;
	score: number;
	passed: boolean;
	threshold: number;
	head_loaded: boolean;
};
