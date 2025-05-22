import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	meanAbsoluteError,
	meanSquaredError,
	rSquared,
} from "./regression_metrics.ts";

describe("Regression Metrics", { concurrency: true}, () => {
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

	describe("R-squared", () => {
		it("should calculate R-squared for a list of true and predicted values", () => {
			const yTrue = [3, -0.5, 2, 7];
			const yPred = [2.5, 0.0, 2, 8];
			// Mean of yTrue = (3 - 0.5 + 2 + 7) / 4 = 11.5 / 4 = 2.875
			// TSS = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2
			// TSS = (0.125)^2 + (-3.375)^2 + (-0.875)^2 + (4.125)^2
			// TSS = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
			// RSS = (3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2
			// RSS = (0.5)^2 + (-0.5)^2 + (0)^2 + (-1)^2
			// RSS = 0.25 + 0.25 + 0 + 1 = 1.5
			// R^2 = 1 - (1.5 / 29.1875) = 1 - 0.0513913043 = 0.9486086957
			const result = rSquared(yTrue, yPred);

			assert.ok(Math.abs(result - 0.9486081370449679) < 1e-9);
		});

		it("should return NaN for lists with less than 2 samples", () => {
			const yTrue = [3];
			const yPred = [2];

			const result = rSquared(yTrue, yPred);

			assert.ok(Number.isNaN(result));
		});

		it("should return 1 for perfect predictions", () => {
			const yTrue = [1, 2, 3];
			const yPred = [1, 2, 3];

			const result = rSquared(yTrue, yPred);

			assert.strictEqual(result, 1);
		});

		it("should return 0 if model predicts mean and TSS is not 0", () => {
			const yTrue = [1, 2, 3, 4]; // Mean = 2.5
			const yPred = [2.5, 2.5, 2.5, 2.5];

			const result = rSquared(yTrue, yPred);

			assert.strictEqual(result, 0);
		});

		it("should handle case where TSS is 0 (all yTrue are same)", () => {
			const yTrue = [5, 5, 5];
			const yPredPerfect = [5, 5, 5];
			const yPredImperfect = [5, 5, 6];

			assert.strictEqual(rSquared(yTrue, yPredPerfect), 1);
			assert.strictEqual(rSquared(yTrue, yPredImperfect), 0); // Or some other value depending on convention
		});
	});
});
