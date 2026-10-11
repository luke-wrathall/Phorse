export type FeedItem = {
	id: string;
	owner: string;
	score: number;
	caption: string | null;
	created_at: number;
	rating_count: number;
	rating_mean: number | null;
	image_url: string;
};

export type FeedPage = { items: FeedItem[]; next_offset: number | null };

/** One of the caller's own uploads (GET /api/images/mine) — includes rejected ones. */
export type MyImage = {
	id: string;
	caption: string | null;
	passed: boolean;
	created_at: number;
	rating_count: number;
	rating_mean: number | null;
	image_url: string;
};

export type UploadResult = {
	id: string;
	score: number;
	passed: boolean;
	threshold: number;
	head_loaded: boolean;
};
