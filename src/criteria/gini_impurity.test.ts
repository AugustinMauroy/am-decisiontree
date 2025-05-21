import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { calculateGiniImpurity } from "./gini_impurity.ts";

describe("Gini Impurity Calculation", () => {
	it("should calculate Gini impurity for a list of class labels", () => {
		const y = ["A", "A", "B", "B", "B"];

		const result = calculateGiniImpurity(y);

		assert.strictEqual(result, 0.48);
	});

	it("should return 0 for an empty list", () => {
		const y: string[] = [];

		const result = calculateGiniImpurity(y);

		assert.strictEqual(result, 0);
	});

	it("should handle a single class label", () => {
		const y = ["A", "A", "A"];

		const result = calculateGiniImpurity(y);

		assert.strictEqual(result, 0);
	});
});
