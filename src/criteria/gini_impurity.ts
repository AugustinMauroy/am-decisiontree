/**
 * Calculates the Gini impurity for a list of class labels.
 * @param y - An array of class labels (numbers or strings).
 * @returns The Gini impurity.
 */
export function calculateGiniImpurity(y: (number | string)[]): number {
	const numSamples = y.length;
	if (numSamples === 0) {
		return 0;
	}

	const counts: Record<string | number, number> = {};
	for (const label of y) {
		counts[label] = (counts[label] || 0) + 1;
	}

	let impurity = 1;
	for (const label in counts) {
		const probability = counts[label] / numSamples;
		impurity -= probability * probability;
	}
	return impurity;
}
