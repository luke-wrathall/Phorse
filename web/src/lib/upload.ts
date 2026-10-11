/**
 * Pre-upload size check against the server's caps (GET /api/images/limits),
 * so a file that is bound to be refused is never sent. The server enforces
 * the same caps regardless — this only saves the wait.
 */
import type { UploadLimits } from './types';

const MB = 1024 * 1024;

/** Trim float noise: 20 -> "20", 0.5 -> "0.5". */
const num = (n: number) => String(+n.toFixed(2));

/** Pixel dimensions without uploading, or null if the browser can't read the format. */
function imageSize(file: File): Promise<{ width: number; height: number } | null> {
	return new Promise((resolve) => {
		const url = URL.createObjectURL(file);
		const img = new Image();
		img.onload = () => {
			URL.revokeObjectURL(url);
			resolve({ width: img.naturalWidth, height: img.naturalHeight });
		};
		img.onerror = () => {
			URL.revokeObjectURL(url);
			resolve(null); // e.g. HEIC in some browsers — let the server decide
		};
		img.src = url;
	});
}

/** Why `file` would be refused for size, or null if it's fine (or limits are unknown). */
export async function uploadSizeProblem(
	file: File,
	limits: UploadLimits | undefined
): Promise<string | null> {
	if (!limits) return null;
	if (file.size > limits.max_bytes) {
		return `file too big -- the limit is ${num(limits.max_bytes / MB)} MB`;
	}
	const size = await imageSize(file);
	if (size && size.width * size.height > limits.max_pixels) {
		return (
			`image too big -- ${size.width}x${size.height} is over the ` +
			`${num(limits.max_pixels / 1_000_000)} megapixel limit`
		);
	}
	return null;
}
