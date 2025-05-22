import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	DecisionTreeClassifier,
	DecisionTreeRegressor,
} from "./decision_tree.ts";

describe("DecisionTreeClassifier", () => {
	it("should create an instance of DecisionTreeClassifier", () => {
		const classifier = new DecisionTreeClassifier();

		assert.ok(classifier instanceof DecisionTreeClassifier);
	});

	it("should predict class labels for new data", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [
			[1, 2],
			[3, 4],
			[5, 6],
		];
		const y = ["A", "B", "A"];

		classifier.fit(X, y);

		const predictions = classifier.predict([
			[2, 3],
			[4, 5],
		]);

		assert.deepEqual(predictions, ["A", "B"]);
	});

	it("should predict class probabilities for new data", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [[1], [2], [10], [11]];
		const y = ["A", "A", "B", "B"];

		classifier.fit(X, y);
		const probabilities = classifier.predictProba([[1.5], [10.5]]);

		// Probabilities depend on the split, but should sum to 1 for each sample
		assert.strictEqual(probabilities.length, 2);
		assert.strictEqual(probabilities[0].length, 2); // Two classes: A, B
		assert.ok(probabilities[0][0] > 0.5); // Expect class A for [1.5]
		assert.ok(probabilities[1][1] > 0.5); // Expect class B for [10.5]
		assert.strictEqual(
			Math.round(probabilities[0][0] + probabilities[0][1]),
			1,
		);
		assert.strictEqual(
			Math.round(probabilities[1][0] + probabilities[1][1]),
			1,
		);
	});

	it("should throw an error if predict is called before fit", () => {
		const classifier = new DecisionTreeClassifier();

		assert.throws(
			() => classifier.predict([[1, 2]]),
			Error,
			"Tree is not fitted yet.",
		);
	});

	it("should throw an error if predictProba is called before fit", () => {
		const classifier = new DecisionTreeClassifier();

		assert.throws(
			() => classifier.predictProba([[1, 2]]),
			Error,
			"Tree is not fitted yet.",
		);
	});

	it("should respect maxDepth parameter", () => {
		const classifier = new DecisionTreeClassifier({ maxDepth: 1 });
		const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
		const y = ["A", "A", "A", "A", "B", "B", "B", "B"];

		classifier.fit(X, y);

		// Accessing private/protected 'root' for testing depth is not ideal,
		// but for this specific test, we can infer from predictions or structure.
		// A more direct way would be to expose tree depth or structure for testing.
		// For now, we check if predictions are consistent with a shallow tree.
		// @ts-expect-error
		const rootNode = (classifier as unknown).root;

		assert.ok(rootNode, "Root node should exist");

		if (rootNode && !rootNode.isLeaf) {
			assert.ok(
				rootNode.leftChild?.isLeaf,
				"Left child should be a leaf at depth 1",
			);
			assert.ok(
				rootNode.rightChild?.isLeaf,
				"Right child should be a leaf at depth 1",
			);
		}
	});

	it("should work with entropy criterion", () => {
		const classifier = new DecisionTreeClassifier({ criterion: "entropy" });
		const X = [
			[1, 2],
			[3, 4],
			[5, 6],
		];
		const y = ["A", "B", "A"];

		classifier.fit(X, y);
		const predictions = classifier.predict([
			[2, 3],
			[4, 5],
		]);

		assert.deepEqual(predictions, ["A", "B"]);
	});

	it("should calculate feature importances for classifier", () => {
		const classifier = new DecisionTreeClassifier({ minSamplesLeaf: 1 });
		// Feature 1 is more discriminative
		const X = [
			[1, 10],
			[2, 20],
			[1, 30],
			[2, 40],
		];
		const y = ["A", "B", "A", "B"];
		classifier.fit(X, y);
		const importances = classifier.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(
			importances[0] > importances[1],
			"Feature 0 should be more important",
		);
		assert.strictEqual(Math.round(importances.reduce((a, b) => a + b, 0)), 1);
	});

	it("should serialize and deserialize a classifier", () => {
		const classifier = new DecisionTreeClassifier({ maxDepth: 2 });
		const X = [[1], [2], [10], [11]];
		const y = ["A", "A", "B", "B"];
		classifier.fit(X, y);
		const testSample = [[1.5], [10.5]];
		const originalPredictions = classifier.predict(testSample);

		const json = classifier.toJSON();
		const loadedClassifier = DecisionTreeClassifier.fromJSON(json);
		const loadedPredictions = loadedClassifier.predict(testSample);

		assert.deepStrictEqual(loadedPredictions, originalPredictions);
		assert.deepStrictEqual(
			loadedClassifier.predictProba(testSample),
			classifier.predictProba(testSample),
		);

		assert.strictEqual(
			// @ts-expect-error
			loadedClassifier.uniqueClasses_.length,
			// @ts-expect-error
			classifier.uniqueClasses_.length,
		);
	});

	it("should handle categorical features for classifier", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical", "numerical"],
			minSamplesLeaf: 1,
		});
		const X = [
			["A", 1],
			["B", 2],
			["A", 3],
			["C", 1],
			["B", 5],
			["C", 6],
		];
		const y = ["X", "Y", "X", "Z", "Y", "Z"];

		classifier.fit(X, y);
		const predictions = classifier.predict([
			["A", 2],
			["B", 3],
			["C", 4],
		]);

		// Exact predictions depend on splits, but should be one of the classes
		assert.ok(predictions.every((p) => ["X", "Y", "Z"].includes(p as string)));
		assert.strictEqual(predictions.length, 3);

		const probabilities = classifier.predictProba([
			["A", 2],
			["C", 0],
		]);

		assert.strictEqual(probabilities.length, 2);
		assert.strictEqual(probabilities[0].length, 3); // X, Y, Z
		assert.strictEqual(
			Math.round(probabilities[0].reduce((a, b) => a + b, 0)),
			1,
		);
	});

	it("should handle all categorical features for classifier", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical", "categorical"],
			minSamplesLeaf: 1,
		});
		const X = [
			["A", "Low"],
			["B", "High"],
			["A", "Medium"],
			["C", "Low"],
		];
		const y = [0, 1, 0, 1];

		classifier.fit(X, y);
		const predictions = classifier.predict([
			["A", "High"],
			["C", "Low"],
		]);

		assert.ok(predictions.every((p) => [0, 1].includes(p as number)));
	});

	it("should serialize and deserialize a classifier with featureTypes", () => {
		const classifier = new DecisionTreeClassifier({
			maxDepth: 2,
			featureTypes: ["categorical", "numerical"],
		});
		const X = [
			["A", 1],
			["B", 2],
			["A", 10],
			["B", 11],
		];
		const y = ["X", "X", "Y", "Y"];

		classifier.fit(X, y);

		const testSample: (string | number)[][] = [
			["A", 1.5],
			["B", 10.5],
		];

		const originalPredictions = classifier.predict(testSample);

		const json = classifier.toJSON();
		const loadedClassifier = DecisionTreeClassifier.fromJSON(json);
		const loadedPredictions = loadedClassifier.predict(testSample);

		assert.deepStrictEqual(loadedPredictions, originalPredictions);
		assert.deepStrictEqual(
			loadedClassifier.predictProba(testSample),
			classifier.predictProba(testSample),
		);
		// @ts-expect-error
		assert.deepStrictEqual(loadedClassifier.featureTypes_, [
			"categorical",
			"numerical",
		]);
	});

	it("should calculate feature importances with categorical features", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical", "numerical"],
			minSamplesLeaf: 1,
		});
		// Feature 0 (categorical) is more discriminative
		const X = [
			["A", 10],
			["B", 20],
			["A", 30],
			["B", 40],
		]; // A vs B clearly separates classes
		const y = ["X", "Y", "X", "Y"];

		classifier.fit(X, y);
		const importances = classifier.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(
			importances[0] > importances[1],
			"Categorical Feature 0 should be more important",
		);
		assert.strictEqual(Math.round(importances.reduce((a, b) => a + b, 0)), 1);
	});

	it("should throw error if featureTypes length mismatches nFeatures", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
		});
		const X = [
			[1, 2],
			[3, 4],
		];
		const y = ["A", "B"];

		assert.throws(
			() => classifier.fit(X, y),
			Error,
			"Length of featureTypes must match the number of features in X.",
		);
	});
});

describe("DecisionTreeRegressor", () => {
	it("should create an instance of DecisionTreeRegressor", () => {
		const regressor = new DecisionTreeRegressor();

		assert.ok(regressor instanceof DecisionTreeRegressor);
	});

	it("should predict numerical values for new data", () => {
		const regressor = new DecisionTreeRegressor();
		const X = [
			[1, 2],
			[3, 4],
			[5, 6],
		];
		const y = [10, 20, 30];

		regressor.fit(X, y);

		const predictions = regressor.predict([
			[2, 3],
			[4, 5],
		]);

		assert.deepEqual(predictions, [10, 20]);
	});

	it("should throw an error if predict is called before fit", () => {
		const regressor = new DecisionTreeRegressor();
		assert.throws(
			() => regressor.predict([[1, 2]]),
			Error,
			"Tree is not fitted yet.",
		);
	});

	it("should respect maxDepth parameter for regressor", () => {
		const regressor = new DecisionTreeRegressor({ maxDepth: 1 });
		const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
		const y = [10, 10, 10, 10, 20, 20, 20, 20];

		regressor.fit(X, y);
		// @ts-expect-error
		const rootNode = (regressor as unknown).root;

		assert.ok(rootNode, "Root node should exist");
		if (rootNode && !rootNode.isLeaf) {
			assert.ok(
				rootNode.leftChild?.isLeaf,
				"Left child should be a leaf at depth 1",
			);
			assert.ok(
				rootNode.rightChild?.isLeaf,
				"Right child should be a leaf at depth 1",
			);
		}

		const predictions = regressor.predict([[1], [8]]);

		assert.ok(predictions[0] < 15);
		assert.ok(predictions[1] > 15);
	});

	it("should calculate feature importances for regressor", () => {
		const regressor = new DecisionTreeRegressor({
			criterion: "mse",
			minSamplesLeaf: 1,
		});
		// Feature 0 is more discriminative for regression targets
		const X = [
			[1, 100],
			[2, 100],
			[10, 100],
			[11, 100],
		];
		const y = [5, 5, 50, 50];
		regressor.fit(X, y);
		const importances = regressor.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(
			importances[0] > importances[1],
			"Feature 0 should be more important",
		);
		assert.strictEqual(Math.round(importances.reduce((a, b) => a + b, 0)), 1);
	});

	it("should serialize and deserialize a regressor", () => {
		const regressor = new DecisionTreeRegressor({
			maxDepth: 2,
			criterion: "mse",
		});
		const X = [[1], [2], [10], [11]];
		const y = [5, 5, 50, 50];
		regressor.fit(X, y);
		const testSample = [[1.5], [10.5]];
		const originalPredictions = regressor.predict(testSample);

		const json = regressor.toJSON();
		const loadedRegressor = DecisionTreeRegressor.fromJSON(json);
		const loadedPredictions = loadedRegressor.predict(testSample);
		assert.deepStrictEqual(loadedPredictions, originalPredictions);
	});

	it("should work with MAE criterion", () => {
		const regressor = new DecisionTreeRegressor({
			criterion: "mae",
			maxDepth: 1,
		});
		const X = [[1], [2], [10], [11]];
		const y = [5, 5, 50, 50];
		regressor.fit(X, y);
		const predictions = regressor.predict([[1.5], [10.5]]);
		// Exact values depend on MAE splits, but should be reasonable
		assert.ok(predictions[0] < 25);
		assert.ok(predictions[1] > 25);
	});

	it("should handle categorical features for regressor", () => {
		const regressor = new DecisionTreeRegressor({
			featureTypes: ["categorical", "numerical"],
			minSamplesLeaf: 1,
			criterion: "mse",
		});
		const X = [
			["A", 1],
			["B", 2],
			["A", 3],
			["C", 1],
			["B", 5],
			["C", 6],
		];
		const y = [10, 20, 15, 5, 25, 8];

		regressor.fit(X, y);
		const predictions = regressor.predict([
			["A", 2], // Expect something around 10-15
			["B", 3], // Expect something around 20-25
			["C", 4], // Expect something around 5-8
		]);

		assert.strictEqual(predictions.length, 3);
		assert.ok(predictions[0] > 5 && predictions[0] < 20);
		assert.ok(predictions[1] > 15 && predictions[1] < 30);
		assert.ok(predictions[2] > 0 && predictions[2] < 15);
	});

	it("should handle all categorical features for regressor", () => {
		const regressor = new DecisionTreeRegressor({
			featureTypes: ["categorical", "categorical"],
			minSamplesLeaf: 1,
			criterion: "mse",
		});
		const X = [
			["Type1", "Low"],
			["Type2", "High"],
			["Type1", "Medium"],
			["Type3", "Low"],
		];
		const y = [100, 500, 150, 50];

		regressor.fit(X, y);
		const predictions = regressor.predict([
			["Type1", "High"], // Should be close to avg of Type1
			["Type3", "Low"], // Should be close to 50
		]);

		assert.ok(predictions[0] > 100 && predictions[0] < 200);
		assert.ok(predictions[1] > 40 && predictions[1] < 60);
	});

	it("should serialize and deserialize a regressor with featureTypes", () => {
		const regressor = new DecisionTreeRegressor({
			maxDepth: 2,
			featureTypes: ["categorical", "numerical"],
			criterion: "mse",
		});
		const X = [
			["A", 1],
			["B", 2],
			["A", 10],
			["B", 11],
		];
		const y = [5, 5, 50, 50];

		regressor.fit(X, y);
		const testSample: (string | number)[][] = [
			["A", 1.5],
			["B", 10.5],
		];

		const originalPredictions = regressor.predict(testSample);

		const json = regressor.toJSON();
		const loadedRegressor = DecisionTreeRegressor.fromJSON(json);
		const loadedPredictions = loadedRegressor.predict(testSample);

		assert.deepStrictEqual(loadedPredictions, originalPredictions);
		// @ts-expect-error
		assert.deepStrictEqual(loadedRegressor.featureTypes_, [
			"categorical",
			"numerical",
		]);
	});

	it("should calculate feature importances with categorical features for regressor", () => {
		const regressor = new DecisionTreeRegressor({
			featureTypes: ["numerical", "categorical"],
			minSamplesLeaf: 1,
			criterion: "mse",
		});
		// Feature 1 (categorical) is more discriminative
		const X = [
			[10, "X"],
			[20, "Y"],
			[30, "X"],
			[40, "Y"],
		];
		const y = [100, 500, 110, 520]; // "X" maps to low values, "Y" to high
		regressor.fit(X, y);
		const importances = regressor.getFeatureImportances();

		assert.strictEqual(importances.length, 2);
		assert.ok(
			importances[1] > importances[0],
			"Categorical Feature 1 should be more important",
		);
		assert.strictEqual(Math.round(importances.reduce((a, b) => a + b, 0)), 1);
	});
});
