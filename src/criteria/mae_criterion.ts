/**
 * Calculates the Mean Absolute Error (MAE) for a list of target values.
 * @param y - An array of numerical target values.
 * @returns The Mean Absolute Error.
 */
export function calculateMAE(y: number[]): number {
	const numSamples = y.length;
	if (numSamples === 0) {
		return 0;
	}

	const mean = y.reduce((sum, val) => sum + val, 0) / numSamples;
	const mae =
		y.reduce((sum, val) => sum + Math.abs(val - mean), 0) / numSamples;
	return mae;
}
