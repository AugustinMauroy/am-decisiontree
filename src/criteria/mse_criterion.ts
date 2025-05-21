/**
 * Calculates the Mean Squared Error (MSE) for a list of target values.
 * @param y - An array of numerical target values.
 * @returns The Mean Squared Error.
 */
export function calculateMSE(y: number[]): number {
	const numSamples = y.length;
	if (numSamples === 0) {
		return 0;
	}

	const mean = y.reduce((sum, val) => sum + val, 0) / numSamples;
	const mse = y.reduce((sum, val) => sum + (val - mean) ** 2, 0) / numSamples;
	return mse;
}
