import type { YInputClassification } from "../decision_tree.ts";

/**
 * Calculates the accuracy of classification predictions.
 * @param yTrue - Array of true class labels.
 * @param yPred - Array of predicted class labels.
 * @returns The accuracy score.
 */
export function accuracyScore(
	yTrue: YInputClassification,
	yPred: YInputClassification,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return 0; // Or 1, depending on convention for empty inputs
	}
	let correct = 0;
	for (let i = 0; i < yTrue.length; i++) {
		if (yTrue[i] === yPred[i]) {
			correct++;
		}
	}
	return correct / yTrue.length;
}

/**
 * Calculates the precision for a given class in classification predictions.
 * Precision = True Positives / (True Positives + False Positives)
 * @param yTrue - Array of true class labels.
 * @param yPred - Array of predicted class labels.
 * @param positiveLabel - The label to consider as the positive class.
 * @returns The precision score for the specified positive class. Returns 0 if (TP + FP) is 0.
 */
export function precisionScore(
	yTrue: YInputClassification,
	yPred: YInputClassification,
	positiveLabel: string | number,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return 0;
	}

	let truePositives = 0;
	let falsePositives = 0;

	for (let i = 0; i < yTrue.length; i++) {
		if (yPred[i] === positiveLabel) {
			if (yTrue[i] === positiveLabel) {
				truePositives++;
			} else {
				falsePositives++;
			}
		}
	}

	if (truePositives + falsePositives === 0) {
		// No predictions made for the positive class, or no instances predicted as positive.
		// Conventionally, precision is 0 in this case.
		return 0;
	}

	return truePositives / (truePositives + falsePositives);
}

/**
 * Calculates the recall for a given class in classification predictions.
 * Recall = True Positives / (True Positives + False Negatives)
 * @param yTrue - Array of true class labels.
 * @param yPred - Array of predicted class labels.
 * @param positiveLabel - The label to consider as the positive class.
 * @returns The recall score for the specified positive class. Returns 0 if (TP + FN) is 0.
 */
export function recallScore(
	yTrue: YInputClassification,
	yPred: YInputClassification,
	positiveLabel: string | number,
): number {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return 0;
	}

	let truePositives = 0;
	let falseNegatives = 0;

	for (let i = 0; i < yTrue.length; i++) {
		if (yTrue[i] === positiveLabel) {
			if (yPred[i] === positiveLabel) {
				truePositives++;
			} else {
				falseNegatives++;
			}
		}
	}

	if (truePositives + falseNegatives === 0) {
		// No true instances of the positive class.
		return 0;
	}

	return truePositives / (truePositives + falseNegatives);
}

/**
 * Calculates the F1 score for a given class in classification predictions.
 * F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
 * @param yTrue - Array of true class labels.
 * @param yPred - Array of predicted class labels.
 * @param positiveLabel - The label to consider as the positive class.
 * @returns The F1 score for the specified positive class. Returns 0 if (Precision + Recall) is 0.
 */
export function f1Score(
	yTrue: YInputClassification,
	yPred: YInputClassification,
	positiveLabel: string | number,
): number {
	const precision = precisionScore(yTrue, yPred, positiveLabel);
	const recall = recallScore(yTrue, yPred, positiveLabel);

	if (precision + recall === 0) {
		return 0;
	}

	return (2 * (precision * recall)) / (precision + recall);
}

/**
 * Generates the confusion matrix for classification predictions.
 * @param yTrue - Array of true class labels.
 * @param yPred - Array of predicted class labels.
 * @returns A 2D array representing the confusion matrix.
 */
export function confusionMatrix(
	yTrue: YInputClassification,
	yPred: YInputClassification,
): number[][] {
	if (yTrue.length !== yPred.length) {
		throw new Error("yTrue and yPred must have the same length.");
	}
	if (yTrue.length === 0) {
		return [];
	}

	const uniqueLabels = Array.from(new Set([...yTrue, ...yPred]));
	const labelIndexMap = new Map(
		uniqueLabels.map((label, index) => [label, index]),
	);
	const matrix = Array.from({ length: uniqueLabels.length }, () =>
		Array(uniqueLabels.length).fill(0),
	);

	for (let i = 0; i < yTrue.length; i++) {
		const trueIndex = labelIndexMap.get(yTrue[i])!;
		const predIndex = labelIndexMap.get(yPred[i])!;
		matrix[trueIndex][predIndex]++;
	}

	return matrix;
}
