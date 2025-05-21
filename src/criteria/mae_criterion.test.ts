import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { calculateMAE } from "./mae_criterion.ts";

describe("Mean Absolute Error Calculation", () => {
	it("should calculate MAE for a list of numerical target values", () => {
		const y = [1, 2, 3, 4, 5];
		// Mean is 3. Abs differences: |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2. Sum = 6. MAE = 6/5 = 1.2
		const result = calculateMAE(y);

		assert.strictEqual(result, 1.2);
	});

	it("should return 0 for an empty list", () => {
		const y: number[] = [];
		const result = calculateMAE(y);

		assert.strictEqual(result, 0);
	});

	it("should handle a single value", () => {
		const y = [3];
		const result = calculateMAE(y);

		assert.strictEqual(result, 0);
	});

	it("should calculate MAE for another example", () => {
		const y = [10, 12, 15, 11, 13];
		// Mean = (10+12+15+11+13)/5 = 61/5 = 12.2
		// Abs diffs: |10-12.2|=2.2, |12-12.2|=0.2, |15-12.2|=2.8, |11-12.2|=1.2, |13-12.2|=0.8
		// Sum = 2.2 + 0.2 + 2.8 + 1.2 + 0.8 = 7.2
		// MAE = 7.2 / 5 = 1.44
		const result = calculateMAE(y);

		assert.strictEqual(result, 1.44);
	});
});
