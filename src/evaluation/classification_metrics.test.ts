import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { meanAbsoluteError, meanSquaredError } from "./regression_metrics.ts";
import {
	accuracyScore,
	precisionScore,
	recallScore,
	f1Score,
	confusionMatrix,
} from "./classification_metrics.ts";

describe("Regression Metrics", () => {
	describe("Mean Absolute Error", () => {
		it("should calculate MAE for a list of true and predicted values", () => {
			const yTrue = [3, -0.5, 2, 7];
			const yPred = [2.5, 0.0, 2, 8];

			const result = meanAbsoluteError(yTrue, yPred);

			assert.strictEqual(result, 0.5);
		});

		it("should return 0 for empty lists", () => {
			const yTrue: number[] = [];
			const yPred: number[] = [];

			const result = meanAbsoluteError(yTrue, yPred);

			assert.strictEqual(result, 0);
		});

		it("should handle a single value", () => {
			const yTrue = [3];
			const yPred = [2];

			const result = meanAbsoluteError(yTrue, yPred);

			assert.strictEqual(result, 1);
		});
	});

	describe("Mean Squared Error", () => {
		it("should calculate MSE for a list of true and predicted values", () => {
			const yTrue = [3, -0.5, 2, 7];
			const yPred = [2.5, 0.0, 2, 8];

			const result = meanSquaredError(yTrue, yPred);

			assert.strictEqual(result, 0.375);
		});

		it("should return 0 for empty lists", () => {
			const yTrue: number[] = [];
			const yPred: number[] = [];

			const result = meanSquaredError(yTrue, yPred);

			assert.strictEqual(result, 0);
		});

		it("should handle a single value", () => {
			const yTrue = [3];
			const yPred = [2];

			const result = meanSquaredError(yTrue, yPred);

			assert.strictEqual(result, 1);
		});
	});
});

describe("Classification Metrics", () => {
	describe("Accuracy Score", () => {
		it("should calculate accuracy for a list of true and predicted labels", () => {
			const yTrue = ["A", "B", "A", "C", "B"];
			const yPred = ["A", "C", "A", "C", "B"];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 0.8); // 4 out of 5 correct
		});

		it("should return 1.0 for perfect predictions", () => {
			const yTrue = ["A", "B", "C"];
			const yPred = ["A", "B", "C"];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 1.0);
		});

		it("should return 0.0 for completely incorrect predictions", () => {
			const yTrue = ["A", "B", "C"];
			const yPred = ["X", "Y", "Z"];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 0.0);
		});

		it("should return 0 for empty lists (as per current implementation)", () => {
			const yTrue: string[] = [];
			const yPred: string[] = [];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 0);
		});

		it("should handle a single value correctly (correct)", () => {
			const yTrue = ["A"];
			const yPred = ["A"];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 1.0);
		});

		it("should handle a single value correctly (incorrect)", () => {
			const yTrue = ["A"];
			const yPred = ["B"];

			const result = accuracyScore(yTrue, yPred);

			assert.strictEqual(result, 0.0);
		});

		it("should throw error for mismatched lengths", () => {
			const yTrue = ["A", "B"];
			const yPred = ["A"];

			assert.throws(
				() => accuracyScore(yTrue, yPred),
				Error,
				"yTrue and yPred must have the same length.",
			);
		});
	});

	describe("Precision Score", () => {
		it("should calculate precision for a specific class", () => {
			const yTrue = ["A", "B", "A", "C", "B"];
			const yPred = ["A", "C", "A", "C", "B"];

			const result = precisionScore(yTrue, yPred, "A");

			// For yTrue = ["A", "B", "A", "C", "B"], yPred = ["A", "C", "A", "C", "B"], positiveLabel = "A":
			// TP = 2 (yPred[0]='A', yTrue[0]='A'; yPred[2]='A', yTrue[2]='A')
			// FP = 0 (no cases where yPred='A' and yTrue!='A')
			// Precision = TP / (TP + FP) = 2 / (2 + 0) = 1
			assert.strictEqual(result, 1);
		});

		it("should return 0 for no positive predictions", () => {
			const yTrue = ["A", "B", "A"];
			const yPred = ["C", "C", "C"];

			const result = precisionScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should return 0 for empty lists", () => {
			const yTrue: string[] = [];
			const yPred: string[] = [];

			const result = precisionScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should handle a single value correctly (TP)", () => {
			const yTrue = ["A"];
			const yPred = ["A"];

			const result = precisionScore(yTrue, yPred, "A");

			assert.strictEqual(result, 1.0);
		});

		it("should handle a single value correctly (FP)", () => {
			const yTrue = ["A"];
			const yPred = ["B"];

			const result = precisionScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should throw error for mismatched lengths", () => {
			const yTrue = ["A", "B"];
			const yPred = ["A"];

			assert.throws(
				() => precisionScore(yTrue, yPred, "A"),
				Error,
				"yTrue and yPred must have the same length.",
			);
		});
	});

	describe("Recall Score", () => {
		it("should calculate recall for a specific class", () => {
			const yTrue = ["A", "B", "A", "C", "B"];
			const yPred = ["A", "C", "A", "C", "B"];

			const result = recallScore(yTrue, yPred, "A");

			// For yTrue = ["A", "B", "A", "C", "B"], yPred = ["A", "C", "A", "C", "B"], positiveLabel = "A":
			// TP = 2 (yPred[0]='A', yTrue[0]='A'; yPred[2]='A', yTrue[2]='A')
			// FN = 0 (Actual "A"s are at indices 0 and 2. Both are predicted as "A". No missed "A"s.)
			// Recall = TP / (TP + FN) = 2 / (2 + 0) = 1
			assert.strictEqual(result, 1);
		});

		it("should return 0 for no true positives", () => {
			const yTrue = ["A", "B", "A"];
			const yPred = ["C", "C", "C"];

			const result = recallScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should return 0 for empty lists", () => {
			const yTrue: string[] = [];
			const yPred: string[] = [];

			const result = recallScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should handle a single value correctly (TP)", () => {
			const yTrue = ["A"];
			const yPred = ["A"];

			const result = recallScore(yTrue, yPred, "A");

			assert.strictEqual(result, 1.0);
		});

		it("should handle a single value correctly (FN)", () => {
			const yTrue = ["A"];
			const yPred = ["B"];

			const result = recallScore(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should throw error for mismatched lengths", () => {
			const yTrue = ["A", "B"];
			const yPred = ["A"];

			assert.throws(
				() => recallScore(yTrue, yPred, "A"),
				Error,
				"yTrue and yPred must have the same length.",
			);
		});
	});

	describe("F1 Score", () => {
		it("should calculate F1 score for a specific class", () => {
			const yTrue = ["A", "B", "A", "C", "B"];
			const yPred = ["A", "C", "A", "C", "B"];

			const result = f1Score(yTrue, yPred, "A");

			// Precision = 1, Recall = 1 (as calculated above)
			// F1 = 2 * (Precision * Recall) / (Precision + Recall)
			// F1 = 2 * (1 * 1) / (1 + 1) = 2 / 2 = 1
			assert.strictEqual(result, 1);
		});

		it("should return 0 for no positive predictions or true positives", () => {
			const yTrue = ["A", "B", "A"];
			const yPred = ["C", "C", "C"];

			const result = f1Score(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should return 0 for empty lists", () => {
			const yTrue: string[] = [];
			const yPred: string[] = [];

			const result = f1Score(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should handle a single value correctly (TP)", () => {
			const yTrue = ["A"];
			const yPred = ["A"];

			const result = f1Score(yTrue, yPred, "A");

			assert.strictEqual(result, 1.0);
		});

		it("should handle a single value correctly (FP)", () => {
			const yTrue = ["A"];
			const yPred = ["B"];

			const result = f1Score(yTrue, yPred, "A");

			assert.strictEqual(result, 0);
		});

		it("should throw error for mismatched lengths", () => {
			const yTrue = ["A", "B"];
			const yPred = ["A"];

			assert.throws(
				() => f1Score(yTrue, yPred, "A"),
				Error,
				"yTrue and yPred must have the same length.",
			);
		});
	});

	describe("Confusion Matrix", () => {
		it("should calculate confusion matrix for a list of true and predicted labels", () => {
			const yTrue = ["A", "B", "A", "C", "B"];
			const yPred = ["A", "C", "A", "C", "B"];

			const result = confusionMatrix(yTrue, yPred);

			// Assuming uniqueLabels are sorted: ["A", "B", "C"]
			// Rows: True labels, Columns: Predicted labels
			//       Pred A, Pred B, Pred C
			// True A:  [2,      0,      0]
			// True B:  [0,      1,      1]
			// True C:  [0,      0,      1]
			assert.deepStrictEqual(result, [
				[2, 0, 0],
				[0, 1, 1],
				[0, 0, 1],
			]);
		});

		it("should return an empty matrix for empty lists", () => {
			const yTrue: string[] = [];
			const yPred: string[] = [];

			const result = confusionMatrix(yTrue, yPred);

			assert.deepStrictEqual(result, []);
		});

		it("should handle a single value correctly (TP)", () => {
			const yTrue = ["A"];
			const yPred = ["A"];

			const result = confusionMatrix(yTrue, yPred);

			// uniqueLabels = ["A"]
			//       Pred A
			// True A:  [1]
			assert.deepStrictEqual(result, [[1]]);
		});

		it("should handle a single value correctly (FP)", () => {
			const yTrue = ["A"];
			const yPred = ["B"];

			const result = confusionMatrix(yTrue, yPred);

			// Assuming uniqueLabels are sorted: ["A", "B"]
			//       Pred A, Pred B
			// True A:  [0,      1]
			// True B:  [0,      0]
			assert.deepStrictEqual(result, [
				[0, 1],
				[0, 0],
			]);
		});

		it("should throw error for mismatched lengths", () => {
			const yTrue = ["A", "B"];
			const yPred = ["A"];

			assert.throws(
				() => confusionMatrix(yTrue, yPred),
				Error,
				"yTrue and yPred must have the same length.",
			);
		});
	});

	describe("Confusion Matrix with Numeric Labels", () => {
		it("should calculate confusion matrix for numeric labels", () => {
			const yTrue = [1, 2, 1, 3, 2];
			const yPred = [1, 3, 1, 3, 2];

			const result = confusionMatrix(yTrue, yPred);

			// Assuming uniqueLabels are sorted: [1, 2, 3]
			//       Pred 1, Pred 2, Pred 3
			// True 1:  [2,      0,      0]
			// True 2:  [0,      1,      1]
			// True 3:  [0,      0,      1]
			assert.deepStrictEqual(result, [
				[2, 0, 0],
				[0, 1, 1],
				[0, 0, 1],
			]);
		});
	});
});
