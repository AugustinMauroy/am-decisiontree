import type { YInputRegression } from "../decision_tree.ts";

/**
 * Calculates the Mean Absolute Error (MAE) between true and predicted values.
 * Note: This is different from the MAE criterion used for splitting,
 * which is calculated on a single set of values (residuals from the mean).
 * This MAE is between two sets of values.
 * @param yTrue - Array of true numerical values.
 * @param yPred - Array of predicted numerical values.
 * @returns The Mean Absolute Error.
 */
export function meanAbsoluteError(
	yTrue: YInputRegression,
	yPred: YInputRegression,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return 0;
	}
	let sumAbsoluteError = 0;
	for (let i = 0; i < yTrue.length; i++) {
		sumAbsoluteError += Math.abs(yTrue[i] - yPred[i]);
	}
	return sumAbsoluteError / yTrue.length;
}

/**
 * Calculates the Mean Squared Error (MSE) between true and predicted values.
 * @param yTrue - Array of true numerical values.
 * @param yPred - Array of predicted numerical values.
 * @returns The Mean Squared Error.
 */
export function meanSquaredError(
	yTrue: YInputRegression,
	yPred: YInputRegression,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return 0;
	}
	let sumSquaredError = 0;
	for (let i = 0; i < yTrue.length; i++) {
		sumSquaredError += (yTrue[i] - yPred[i]) ** 2;
	}
	return sumSquaredError / yTrue.length;
}

/**
 * Calculates the R-squared (coefficient of determination) regression score.
 * @param yTrue - Array of true numerical values.
 * @param yPred - Array of predicted numerical values.
 * @returns The R-squared score.
 */
export function rSquared(
	yTrue: YInputRegression,
	yPred: YInputRegression,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length < 2) {
		// R-squared is not well-defined for less than 2 samples.
		// Can return NaN, 0, or throw error based on desired behavior.
		// Returning NaN is common in some libraries.
		return Number.NaN;
	}

	const meanYTrue = yTrue.reduce((sum, val) => sum + val, 0) / yTrue.length;

	const totalSumOfSquares = yTrue.reduce(
		(sum, val) => sum + (val - meanYTrue) ** 2,
		0,
	);
	const residualSumOfSquares = yTrue.reduce(
		(sum, val, i) => sum + (val - yPred[i]) ** 2,
		0,
	);

	if (totalSumOfSquares === 0) {
		// This means all yTrue values are the same.
		// If yPred also perfectly predicts these values, R^2 is 1.
		// Otherwise, it can be considered 0 or undefined.
		return residualSumOfSquares === 0 ? 1 : 0;
	}

	return 1 - residualSumOfSquares / totalSumOfSquares;
}
