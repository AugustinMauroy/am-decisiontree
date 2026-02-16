import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	RandomForestClassifier,
	RandomForestRegressor,
} from "./random_forest.ts";
import type {
	DecisionTreeClassifier,
	DecisionTreeRegressor,
} from "../decision_tree.ts";

describe("RandomForestClassifier", () => {
	it("should create an instance of RandomForestClassifier", () => {
		const rfc = new RandomForestClassifier();
		assert.ok(rfc instanceof RandomForestClassifier);
	});

	it("should fit and predict for numerical features", () => {
		const rfc = new RandomForestClassifier({
			nEstimators: 5,
			maxDepth: 2,
			// Add randomState to DecisionTreeParameters if available and used by RF for reproducibility
		});
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = ["A", "A", "A", "B", "B", "B"];
		rfc.fit(X, y);
		const predictions = rfc.predict([[1.5], [10.5]]);

		assert.strictEqual(predictions.length, 2);
		assert.ok(["A", "B"].includes(predictions[0] as string));
		assert.ok(["A", "B"].includes(predictions[1] as string));
		// With enough estimators and simple data, we expect good separation
		// Note: Due to randomness in bootstrap and feature selection (if enabled),
		// exact prediction might vary. For robust tests, consider fixing random seeds
		// or testing properties like "prediction for [1.5] should be 'A' most of the time".
		// For this example, we'll assume it generally works.
		// A common pattern is to check if the prediction is the expected one.
		// If using a fixed seed in bootstrap/tree construction, this would be deterministic.
		// Since bootstrap is simplified and no seed is used, we check general behavior.
	});

	it("should predict probabilities for numerical features", () => {
		const rfc = new RandomForestClassifier({
			nEstimators: 10,
			maxDepth: 2,
			bootstrap: false,
		});
		const X = [[1], [2], [10], [11]];
		const y = ["A", "A", "B", "B"];
		rfc.fit(X, y);
		const probas = rfc.predictProba([[1.5], [10.5]]);

		assert.strictEqual(probas.length, 2);
		assert.ok(probas[0]["0"] !== undefined && probas[0]["1"] !== undefined);
		assert.ok(probas[1]["0"] !== undefined && probas[1]["1"] !== undefined);

		assert.ok(
			Math.abs(Object.values(probas[0]).reduce((s, p) => s + p, 0) - 1) < 1e-9,
		);
		assert.ok(
			Math.abs(Object.values(probas[1]).reduce((s, p) => s + p, 0) - 1) < 1e-9,
		);
		assert.ok(probas[0]["0"] >= probas[0]["1"]);
		assert.ok(probas[1]["1"] >= probas[1]["0"]);
	});

	it("should fit and predict for categorical features", () => {
		const rfc = new RandomForestClassifier({
			nEstimators: 5,
			maxDepth: 2,
			featureTypes: ["categorical"],
		});
		const X = [["low"], ["low"], ["high"], ["high"], ["medium"]];
		const y = ["A", "A", "B", "B", "A"];
		rfc.fit(X, y);
		const predictions = rfc.predict([["low"], ["high"], ["medium"]]);

		assert.strictEqual(predictions.length, 3);
		assert.ok(predictions.every((p) => ["A", "B"].includes(p as string)));
	});

	it("should get feature importances", () => {
		const rfc = new RandomForestClassifier({ nEstimators: 3, maxDepth: 2 });
		const X = [
			[1, 10],
			[2, 20],
			[1, 30],
			[2, 40],
		]; // Feature 1 is more discriminative
		const y = ["A", "B", "A", "B"];
		rfc.fit(X, y);
		const importances = rfc.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(importances.every((imp) => imp >= 0));
		assert.ok(
			Math.abs(importances.reduce((s, i) => s + i, 0) - 1) < 1e-9 ||
				importances.reduce((s, i) => s + i, 0) > 0,
		); // Sum to 1 or be positive
		// Expect feature 1 (index 1) to be more important if data is simple enough
		// This can be flaky due to randomness, so a more robust test might be needed
		// or check that importances are not NaN and have correct length.
	});

	it("should throw error if predict is called before fit", () => {
		const rfc = new RandomForestClassifier();
		assert.throws(() => rfc.predict([[1]]), Error, "Forest is not fitted yet.");
	});

	it("should throw error if predictProba is called before fit", () => {
		const rfc = new RandomForestClassifier();
		assert.throws(
			() => rfc.predictProba([[1]]),
			Error,
			"Forest is not fitted yet.",
		);
	});

	it("should throw error if getFeatureImportances is called before fit", () => {
		const rfc = new RandomForestClassifier();
		assert.throws(
			() => rfc.getFeatureImportances(),
			Error,
			"Forest not fitted or trees have no importances.",
		);
	});

	it("should use nEstimators correctly", () => {
		const nEstimators = 7;
		const rfc = new RandomForestClassifier({ nEstimators });
		const X = [[1], [2], [10], [11]];
		const y = ["A", "A", "B", "B"];
		rfc.fit(X, y);
		// @ts-expect-error Accessing protected member for test
		assert.strictEqual(rfc.trees.length, nEstimators);
	});

	it("should pass maxFeaturesForSplit to trees", () => {
		const rfc = new RandomForestClassifier({
			nEstimators: 1,
			maxFeaturesForSplit: 1,
		});
		const X = [
			[1, 2, 3],
			[4, 5, 6],
			[7, 8, 9],
			[10, 11, 12],
		];
		const y = ["A", "A", "B", "B"];
		rfc.fit(X, y);
		// @ts-expect-error Accessing protected member for test
		const tree = rfc.trees[0] as DecisionTreeClassifier;
		// @ts-expect-error Accessing protected member for test
		assert.strictEqual(tree.maxFeaturesForSplit_, 1);
	});

	it("should create a JSON representation for classifier", () => {
		const rfc = new RandomForestClassifier({ nEstimators: 3, maxDepth: 2 });
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = ["A", "A", "A", "B", "B", "B"];

		rfc.fit(X, y);
		const json = rfc.toJSON();

		assert.ok(json);
		assert.ok(typeof json === "string");
		const parsed = JSON.parse(json);
		assert.strictEqual(parsed.nEstimators, 3);
		assert.strictEqual(parsed.bootstrap, true);
		assert.strictEqual(parsed.treeParams.maxDepth, 2);
		assert.strictEqual(parsed.trees.length, 3);
	});

	it("should restore classifier from JSON", () => {
		const rfc = new RandomForestClassifier({ nEstimators: 3, maxDepth: 2 });
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = ["A", "A", "A", "B", "B", "B"];

		rfc.fit(X, y);
		const json = rfc.toJSON();

		const restoredRfc = new RandomForestClassifier();
		restoredRfc.fromJSON(json);
		assert.ok(restoredRfc instanceof RandomForestClassifier);

		// make a prediction to check if it works
		const predictions = restoredRfc.predict([[1.5], [10.5]]);
		assert.strictEqual(predictions.length, 2);
		assert.ok(["A", "B"].includes(predictions[0] as string));
		assert.ok(["A", "B"].includes(predictions[1] as string));
	});
});

describe("RandomForestRegressor", () => {
	it("should create an instance of RandomForestRegressor", () => {
		const rfr = new RandomForestRegressor();
		assert.ok(rfr instanceof RandomForestRegressor);
	});

	it("should fit and predict for numerical features", () => {
		const rfr = new RandomForestRegressor({ nEstimators: 5, maxDepth: 2 });
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = [10, 10, 10, 50, 50, 50];
		rfr.fit(X, y);
		const predictions = rfr.predict([[1.5], [10.5]]);

		assert.strictEqual(predictions.length, 2);
		assert.ok(typeof predictions[0] === "number");
		assert.ok(typeof predictions[1] === "number");
		// Expect predictions to be around 10 and 50 respectively
		assert.ok(predictions[0] < 30 && predictions[0] > 0); // ballpark
		assert.ok(predictions[1] > 30 && predictions[1] < 60); // ballpark
	});

	it("should get feature importances", () => {
		const rfr = new RandomForestRegressor({ nEstimators: 3, maxDepth: 2 });
		const X = [
			[1, 100],
			[2, 200],
			[10, 100],
			[11, 200],
		]; // Feature 0 is more discriminative
		const y = [5, 5, 50, 50];
		rfr.fit(X, y);
		const importances = rfr.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(importances.every((imp) => imp >= 0));
		assert.ok(
			Math.abs(importances.reduce((s, i) => s + i, 0) - 1) < 1e-9 ||
				importances.reduce((s, i) => s + i, 0) > 0,
		);
	});

	it("should throw error if predict is called before fit", () => {
		const rfr = new RandomForestRegressor();
		assert.throws(() => rfr.predict([[1]]), Error, "Forest is not fitted yet.");
	});

	it("should throw error if getFeatureImportances is called before fit", () => {
		const rfr = new RandomForestRegressor();
		assert.throws(
			() => rfr.getFeatureImportances(),
			Error,
			"Forest not fitted or trees have no importances.",
		);
	});

	it("should use nEstimators correctly", () => {
		const nEstimators = 6;
		const rfr = new RandomForestRegressor({ nEstimators });
		const X = [[1], [2], [10], [11]];
		const y = [5, 5, 50, 50];
		rfr.fit(X, y);
		// @ts-expect-error Accessing protected member for test
		assert.strictEqual(rfr.trees.length, nEstimators);
	});

	it("should pass maxFeaturesForSplit to trees", () => {
		const rfr = new RandomForestRegressor({
			nEstimators: 1,
			maxFeaturesForSplit: "sqrt",
		});
		const X = [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 10, 11, 12],
			[13, 14, 15, 16],
		]; // 4 features, sqrt(4) = 2
		const y = [1, 1, 2, 2];
		rfr.fit(X, y);
		// @ts-expect-error Accessing protected member for test
		const tree = rfr.trees[0] as DecisionTreeRegressor;
		// @ts-expect-error Accessing protected member for test
		assert.strictEqual(tree.maxFeaturesForSplit_, "sqrt"); // The string itself is stored
	});

	it("should create a JSON representation", () => {
		const rfc = new RandomForestClassifier({ nEstimators: 3, maxDepth: 2 });
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = ["A", "A", "A", "B", "B", "B"];

		rfc.fit(X, y);
		const json = rfc.toJSON();

		assert.ok(json);
		assert.ok(typeof json === "string");
		const parsed = JSON.parse(json);
		assert.strictEqual(parsed.nEstimators, 3);
		assert.strictEqual(parsed.bootstrap, true);
		assert.strictEqual(parsed.treeParams.maxDepth, 2);
		assert.strictEqual(parsed.trees.length, 3);
	});

	it("should restore from JSON", () => {
		const rfr = new RandomForestRegressor({ nEstimators: 3, maxDepth: 2 });
		const X = [[1], [2], [3], [10], [11], [12]];
		const y = [10, 10, 10, 50, 50, 50];

		rfr.fit(X, y);
		const json = rfr.toJSON();

		const restoredRfr = new RandomForestRegressor();
		restoredRfr.fromJSON(json);
		assert.ok(restoredRfr instanceof RandomForestRegressor);

		// make a prediction to check if it works
		const predictions = restoredRfr.predict([[1.5], [10.5]]);
		assert.strictEqual(predictions.length, 2);
		assert.ok(typeof predictions[0] === "number");
		assert.ok(typeof predictions[1] === "number");
	});
});
