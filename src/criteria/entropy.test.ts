import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { calculateEntropy } from "./entropy.ts";

describe("Entropy Calculation", () => {
	it("should calculate entropy for a list of class labels", () => {
		const y = ["A", "A", "B", "B", "B"];

		const result = calculateEntropy(y);

		assert.strictEqual(result, 0.9709505944546686);
	});

	it("should return 0 for an empty list", () => {
		const y: string[] = [];

		const result = calculateEntropy(y);

		assert.strictEqual(result, 0);
	});

	it("should handle a single class label", () => {
		const y = ["A", "A", "A"];

		const result = calculateEntropy(y);

		assert.strictEqual(result, 0);
	});
});
