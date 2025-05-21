/**
 * Calculates the entropy for a list of class labels.
 * @param y - An array of class labels (numbers or strings).
 * @returns The entropy.
 */
export function calculateEntropy(y: (number | string)[]): number {
	const numSamples = y.length;
	if (numSamples === 0) {
		return 0;
	}

	const counts: Record<string | number, number> = {};
	for (const label of y) {
		counts[label] = (counts[label] || 0) + 1;
	}

	let entropyValue = 0;
	for (const label in counts) {
		const probability = counts[label] / numSamples;
		if (probability > 0) {
			entropyValue -= probability * Math.log2(probability);
		}
	}
	return entropyValue;
}
