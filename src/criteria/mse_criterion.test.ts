import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { calculateMSE } from "./mse_criterion.ts";

describe("Mean Squared Error Calculation", () => {
	it("should calculate MSE for a list of numerical target values", () => {
		const y = [1, 2, 3, 4, 5];

		const result = calculateMSE(y);

		assert.strictEqual(result, 2.0);
	});

	it("should return 0 for an empty list", () => {
		const y: number[] = [];

		const result = calculateMSE(y);

		assert.strictEqual(result, 0);
	});

	it("should handle a single value", () => {
		const y = [3];

		const result = calculateMSE(y);

		assert.strictEqual(result, 0);
	});
});
