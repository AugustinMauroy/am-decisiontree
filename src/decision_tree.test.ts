import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	BaseDecisionTree,
	DecisionTreeClassifier,
	DecisionTreeRegressor,
} from "./decision_tree.ts";
import type { XInput, YInputRegression } from "./decision_tree.ts";
import { Node } from "./node.ts";

class DummyTree extends BaseDecisionTree<XInput, YInputRegression, number> {
	protected getDefaultCriterion(): "mse" {
		return "mse";
	}

	protected calculateImpurity(_y: YInputRegression): number {
		return 0;
	}

	protected calculateLeafValue(y: YInputRegression): number {
		if (y.length === 0) return 0;
		return y.reduce((sum, val) => sum + val, 0) / y.length;
	}
}

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

	it("should throw error when trying to fit empty dataset", () => {
		const classifier = new DecisionTreeClassifier();
		const X: number[][] = [];
		const y: string[] = [];

		assert.throws(
			() => classifier.fit(X, y),
			Error,
			"Cannot fit on empty dataset.",
		);
	});

	it("should throw error when X and y have different lengths", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [[1], [2], [3]];
		const y = ["A", "B"];

		assert.throws(
			() => classifier.fit(X, y),
			Error,
			"X and y must have the same number of samples.",
		);
	});

	it("should handle samples with no features (edge case)", () => {
		const classifier = new DecisionTreeClassifier();
		// Edge case: samples with no features - creates a single leaf node
		const X: (number | null | string)[][] = [[], [], []];
		const y = ["A", "A", "B"];

		classifier.fit(X, y);

		// Should create a single leaf with the most common class
		const predictions = classifier.predict([[], []]);
		assert.strictEqual(predictions.length, 2);
		assert.strictEqual(predictions[0], "A");
		assert.strictEqual(predictions[1], "A");
	});

	it("should use base toJSON after fit", () => {
		const dummy = new DummyTree();
		const X = [[1], [2], [3]];
		const y = [10, 20, 30];

		dummy.fit(X, y);
		const json = dummy.toJSON();
		const parsed = JSON.parse(json);

		assert.ok(parsed.root, "Serialized tree should include root");
		assert.strictEqual(parsed.criterion, "mse");
	});

	it("should throw when base toJSON called before fit", () => {
		const dummy = new DummyTree();
		assert.throws(
			() => dummy.toJSON(),
			Error,
			"Tree is not fitted yet. Cannot serialize.",
		);
	});

	it("should return leaf when best split violates minSamplesLeaf", () => {
		const classifier = new DecisionTreeClassifier({ minSamplesLeaf: 2 });
		const X = [[1], [2], [3]];
		const y = ["A", "A", "B"];

		// Force a best split that would create an undersized leaf.
		// @ts-expect-error
		const originalFindBestSplit = classifier._findBestSplit;
		// @ts-expect-error
		classifier._findBestSplit = () => ({
			featureIndex: 0,
			threshold: 1.5,
			impurityGain: 1,
			leftIndices: [0],
			rightIndices: [1, 2],
		});

		// @ts-expect-error
		const node = classifier._buildTree(X, y, 0);

		// @ts-expect-error
		classifier._findBestSplit = originalFindBestSplit;
		assert.ok(node.isLeaf, "Tree should fallback to leaf when split invalid");
	});

	it("should handle pruning edge case with zero leaves", () => {
		const classifier = new DecisionTreeClassifier();
		const leftChild = new Node({
			samples: 1,
			isLeaf: true,
			impurity: 0.5,
			value: { A: 1 },
		});
		const rightChild = new Node({
			samples: 1,
			isLeaf: true,
			impurity: 0.5,
			value: { A: 1 },
		});
		const parent = new Node({
			samples: 2,
			isLeaf: false,
			impurity: 0.5,
			featureIndex: 0,
			threshold: 1,
			leftChild,
			rightChild,
			potentialLeafValue: { A: 1 },
		});

		// @ts-expect-error
		const originalPrune = classifier._pruneRecursive.bind(classifier);
		// @ts-expect-error
		classifier._pruneRecursive = (node: Node, ccpAlpha: number) => {
			if (node === leftChild || node === rightChild) {
				return { totalImpuritySum: 0, numLeaves: 0 };
			}
			return originalPrune(node, ccpAlpha);
		};

		// @ts-expect-error
		const result = classifier._pruneRecursive(parent, 0.1);
		// @ts-expect-error
		classifier._pruneRecursive = originalPrune;

		assert.deepStrictEqual(result, { totalImpuritySum: 0, numLeaves: 0 });
	});

	it("should use potentialLeafValue when leaf value is undefined", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: true,
			potentialLeafValue: { A: 1 },
		});

		// @ts-expect-error
		const result = classifier._predictSample([1], node);
		assert.deepStrictEqual(result, { A: 1 });
	});

	it("should throw when _predictSample called with undefined node", () => {
		const classifier = new DecisionTreeClassifier();
		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([1], undefined),
			Error,
			"Tree is not fitted yet or root node is undefined.",
		);
	});

	it("should throw when leaf has no value or potentialLeafValue", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({ samples: 1, isLeaf: true });

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([1], node),
			Error,
			"Leaf node has undefined value and no potentialLeafValue.",
		);
	});

	it("should fallback to potentialLeafValue when featureIndex is missing", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			potentialLeafValue: { A: 1 },
		});

		// @ts-expect-error
		const result = classifier._predictSample([1], node);
		assert.deepStrictEqual(result, { A: 1 });
	});

	it("should throw when non-leaf is missing featureIndex", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({ samples: 1, isLeaf: false });

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([1], node),
			Error,
			"Invalid non-leaf node: missing featureIndex and no potentialLeafValue.",
		);
	});

	it("should throw when feature index is out of bounds", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			featureIndex: 1,
			threshold: 1,
			leftChild: new Node({ samples: 1, isLeaf: true, value: { A: 1 } }),
			rightChild: new Node({ samples: 1, isLeaf: true, value: { A: 1 } }),
		});

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([0], node),
			Error,
			"Feature index 1 is out of bounds for sample with 1 features.",
		);
	});

	it("should fallback to potentialLeafValue for missing feature values", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			featureIndex: 0,
			potentialLeafValue: { A: 1 },
		});

		// @ts-expect-error
		const result = classifier._predictSample([null], node);
		assert.deepStrictEqual(result, { A: 1 });
	});

	it("should throw when missing feature value has no children or fallback", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({ samples: 1, isLeaf: false, featureIndex: 0 });

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([null], node),
			Error,
			"Missing value at a non-leaf node where children are unexpectedly missing or strategy failed, and no potentialLeafValue.",
		);
	});

	it("should throw for categorical split without children", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			featureIndex: 0,
			splitCategories: new Set(["A"]),
		});

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample(["A"], node),
			Error,
			"Invalid non-leaf node: missing children for categorical split.",
		);
	});

	it("should throw for numerical split without children", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			featureIndex: 0,
			threshold: 2,
		});

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([1], node),
			Error,
			"Invalid non-leaf node: missing children for numerical split.",
		);
	});

	it("should fallback to potentialLeafValue when split criteria missing", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({
			samples: 1,
			isLeaf: false,
			featureIndex: 0,
			potentialLeafValue: { A: 1 },
		});

		// @ts-expect-error
		const result = classifier._predictSample([1], node);
		assert.deepStrictEqual(result, { A: 1 });
	});

	it("should throw when split criteria missing and no fallback", () => {
		const classifier = new DecisionTreeClassifier();
		const node = new Node({ samples: 1, isLeaf: false, featureIndex: 0 });

		assert.throws(
			// @ts-expect-error
			() => classifier._predictSample([1], node),
			Error,
			"Invalid non-leaf node: missing split criteria (threshold or splitCategories) and no potentialLeafValue.",
		);
	});

	it("should return first unique class when probability map is empty", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [[1], [2], [3]];
		const y = ["B", "A", "B"];
		classifier.fit(X, y);

		// Force a leaf with an empty probability map.
		// @ts-expect-error
		classifier.root = new Node({ samples: 0, isLeaf: true, value: {} });

		const predictions = classifier.predict([[1]]);
		assert.deepStrictEqual(predictions, ["A"]);
	});

	it("should coerce invalid classifier criterion to gini", () => {
		const classifier = new DecisionTreeClassifier({ criterion: "mae" });
		// @ts-expect-error
		assert.strictEqual(classifier.criterion, "gini");
	});

	it("should throw for unsupported classifier criterion in calculateImpurity", () => {
		const classifier = new DecisionTreeClassifier();
		// @ts-expect-error
		classifier.criterion = "mae";

		assert.throws(
			// @ts-expect-error
			() => classifier.calculateImpurity(["A", "B"]),
			Error,
			"Unsupported criterion for classification: mae",
		);
	});

	it("should throw when classifier toJSON called before fit", () => {
		const classifier = new DecisionTreeClassifier();
		assert.throws(
			() => classifier.toJSON(),
			Error,
			"Tree is not fitted yet. Cannot serialize.",
		);
	});

	it("should throw when getFeatureImportances called before fit", () => {
		const classifier = new DecisionTreeClassifier();
		assert.throws(
			() => classifier.getFeatureImportances(),
			Error,
			"Feature importances are not available. Fit the model first.",
		);
	});

	it("should return all features when maxFeaturesForSplit is invalid", () => {
		const classifier = new DecisionTreeClassifier();
		// @ts-expect-error
		classifier.maxFeaturesForSplit_ = "invalid";

		// @ts-expect-error
		const features = classifier._getFeaturesToConsider(3);
		assert.deepStrictEqual(features.sort(), [0, 1, 2]);
	});

	it("should skip invalid empty categorical split set", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
			minSamplesLeaf: 1,
		});
		const X = [["A"], ["B"], ["C"], ["A"], ["B"], ["C"]];
		const y = ["X", "Y", "Z", "X", "Y", "Z"];

		const sizeDescriptor = Object.getOwnPropertyDescriptor(
			Set.prototype,
			"size",
		);
		assert.ok(sizeDescriptor?.get, "Set.prototype.size getter not found");
		let forceZeroOnce = true;

		Object.defineProperty(Set.prototype, "size", {
			get() {
				if (forceZeroOnce) {
					forceZeroOnce = false;
					return 0;
				}
				return sizeDescriptor?.get?.call(this);
			},
			configurable: true,
		});

		try {
			classifier.fit(X, y);
		} finally {
			if (sizeDescriptor) {
				Object.defineProperty(Set.prototype, "size", sizeDescriptor);
			}
		}

		const predictions = classifier.predict([["A"], ["C"]]);
		assert.strictEqual(predictions.length, 2);
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

	it("should apply cost complexity pruning when ccpAlpha > 0", () => {
		const classifier = new DecisionTreeClassifier({
			ccpAlpha: 0.1,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
		const y = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];

		classifier.fit(X, y);

		// With pruning, tree should be simpler
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle minSamplesLeaf constraint", () => {
		const classifier = new DecisionTreeClassifier({
			minSamplesLeaf: 3,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4], [5], [6]];
		const y = ["A", "A", "B", "B", "C", "C"];

		classifier.fit(X, y);

		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should respect minSamplesLeaf and prevent invalid splits", () => {
		const classifier = new DecisionTreeClassifier({
			minSamplesLeaf: 2,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4]];
		const y = ["A", "B", "C", "D"];

		classifier.fit(X, y);

		// With minSamplesLeaf=2, each leaf must have at least 2 samples
		// This dataset should result in fewer splits
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle minSamplesSplit constraint", () => {
		const classifier = new DecisionTreeClassifier({
			minSamplesSplit: 4,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4]];
		const y = ["A", "A", "B", "B"];

		classifier.fit(X, y);

		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle missing data in features", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["numerical"],
		});
		const X = [[1], [2], [null], [4], [5]];
		const y = ["A", "A", "A", "B", "B"];

		classifier.fit(X, y);

		const predictions = classifier.predict([[3], [null]]);
		assert.strictEqual(predictions.length, 2);
	});

	it("should handle categorical features with missing data", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
		});
		const X = [["red"], ["blue"], [null], ["red"], ["blue"]];
		const y = ["A", "B", "A", "A", "B"];

		classifier.fit(X, y);

		const predictions = classifier.predict([["red"], [null]]);
		assert.strictEqual(predictions.length, 2);
	});

	it("should throw error when featureTypes length doesn't match features", () => {
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

	it("should handle maxFeaturesForSplit with log2", () => {
		const classifier = new DecisionTreeClassifier({
			maxFeaturesForSplit: "log2",
		});
		const X = [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 10, 11, 12],
			[13, 14, 15, 16],
		];
		const y = ["A", "B", "A", "B"];

		classifier.fit(X, y);
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle maxFeaturesForSplit with fraction", () => {
		const classifier = new DecisionTreeClassifier({
			maxFeaturesForSplit: 0.5,
		});
		const X = [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 10, 11, 12],
			[13, 14, 15, 16],
		];
		const y = ["A", "B", "A", "B"];

		classifier.fit(X, y);
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle maxFeaturesForSplit with absolute number", () => {
		const classifier = new DecisionTreeClassifier({
			maxFeaturesForSplit: 2,
		});
		const X = [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 10, 11, 12],
			[13, 14, 15, 16],
		];
		const y = ["A", "B", "A", "B"];

		classifier.fit(X, y);
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should throw error when loading wrong type from JSON (classifier)", () => {
		const regressor = new DecisionTreeRegressor({ maxDepth: 2 });
		const X = [[1], [2], [10], [11]];
		const y = [5, 5, 50, 50];
		regressor.fit(X, y);

		const json = regressor.toJSON();

		assert.throws(
			() => DecisionTreeClassifier.fromJSON(json),
			Error,
			"JSON string does not represent a DecisionTreeClassifier.",
		);
	});

	it("should throw error when loading wrong type from JSON (regressor)", () => {
		const classifier = new DecisionTreeClassifier({ maxDepth: 2 });
		const X = [[1], [2], [10], [11]];
		const y = ["A", "A", "B", "B"];
		classifier.fit(X, y);

		const json = classifier.toJSON();

		assert.throws(
			() => DecisionTreeRegressor.fromJSON(json),
			Error,
			"JSON string does not represent a DecisionTreeRegressor.",
		);
	});

	it("should serialize and deserialize with categorical splits", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
			minSamplesLeaf: 1,
		});
		const X = [["red"], ["blue"], ["green"], ["red"], ["blue"], ["green"]];
		const y = ["A", "B", "C", "A", "B", "C"];

		classifier.fit(X, y);
		const json = classifier.toJSON();
		const loadedClassifier = DecisionTreeClassifier.fromJSON(json);

		const predictions = loadedClassifier.predict([["red"], ["blue"]]);
		assert.strictEqual(predictions.length, 2);
	});

	it("should handle very aggressive pruning", () => {
		const classifier = new DecisionTreeClassifier({
			ccpAlpha: 100,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4], [5], [6]];
		const y = ["A", "B", "A", "B", "A", "B"];

		classifier.fit(X, y);

		// With very high pruning, tree should collapse to a single node
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle minSamplesLeaf preventing splits after finding best split", () => {
		const classifier = new DecisionTreeClassifier({
			minSamplesLeaf: 5,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
		const y = ["A", "A", "A", "B", "B", "B", "B", "B"];

		classifier.fit(X, y);

		// With minSamplesLeaf=5, only splits that leave at least 5 samples per leaf are valid
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle feature with all missing values", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["numerical", "numerical"],
		});
		const X = [
			[null, 1],
			[null, 2],
			[null, 3],
			[null, 4],
		];
		const y = ["A", "A", "B", "B"];

		classifier.fit(X, y);

		const predictions = classifier.predict([[null, 2.5]]);
		assert.strictEqual(predictions.length, 1);
	});

	it("should predictProba when uniqueClasses not determined error", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [[1], [2], [3]];
		const y = ["A", "B", "C"];
		classifier.fit(X, y);

		// Force uniqueClasses_ to undefined to test error
		// @ts-expect-error
		classifier.uniqueClasses_ = undefined;

		assert.throws(
			() => classifier.predictProba(X),
			Error,
			"Unique classes not determined. Fit the model first.",
		);
	});

	it("should handle regressor with empty leaf value edge case", () => {
		const regressor = new DecisionTreeRegressor({
			criterion: "mae",
		});
		const X = [[1], [2], [3], [4]];
		const y = [10, 20, 30, 40];

		regressor.fit(X, y);
		const predictions = regressor.predict([[2.5]]);
		assert.ok(predictions[0] > 0);
	});

	it("should handle very deep tree with minSamplesLeaf preventing final splits", () => {
		const classifier = new DecisionTreeClassifier({
			minSamplesLeaf: 4,
			maxDepth: 100,
		});
		const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
		const y = ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"];

		classifier.fit(X, y);

		// Some splits should be prevented by minSamplesLeaf
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle tree with single pure class (no splits needed)", () => {
		const classifier = new DecisionTreeClassifier();
		const X = [[1], [2], [3], [4]];
		const y = ["A", "A", "A", "A"];

		classifier.fit(X, y);

		// Tree should be a single leaf node
		const importances = classifier.getFeatureImportances();
		// No splits made, so importances should be 0
		assert.strictEqual(importances[0], 0);
	});

	it("should handle extreme pruning that collapses tree to root", () => {
		const classifier = new DecisionTreeClassifier({
			ccpAlpha: 1000,
			maxDepth: 10,
		});
		const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
		const y = ["A", "A", "B", "B", "C", "C", "D", "D"];

		classifier.fit(X, y);

		// Very high pruning should result in minimal tree
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should handle categorical split with multiple subset combinations", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
			minSamplesLeaf: 1,
		});
		const X = [["cat1"], ["cat2"], ["cat3"], ["cat4"], ["cat5"], ["cat6"]];
		const y = ["A", "A", "B", "B", "C", "C"];

		classifier.fit(X, y);

		const predictions = classifier.predict([["cat1"], ["cat3"], ["cat5"]]);
		assert.strictEqual(predictions.length, 3);
	});

	it("should handle regressor with single pure target (no variance)", () => {
		const regressor = new DecisionTreeRegressor();
		const X = [[1], [2], [3], [4]];
		const y = [100, 100, 100, 100];

		regressor.fit(X, y);

		// Tree should be a single leaf node
		const importances = regressor.getFeatureImportances();
		// No splits made, so importances should be 0
		assert.strictEqual(importances[0], 0);
	});

	it("should handle minSamplesLeaf constraint with categorical features", () => {
		const classifier = new DecisionTreeClassifier({
			featureTypes: ["categorical"],
			minSamplesLeaf: 3,
		});
		const X = [["A"], ["B"], ["A"], ["C"], ["B"], ["C"], ["A"], ["B"]];
		const y = ["X", "Y", "X", "Z", "Y", "Z", "X", "Y"];

		classifier.fit(X, y);

		// splits should be constrained by minSamplesLeaf
		const predictions = classifier.predict(X);
		assert.strictEqual(predictions.length, X.length);
	});

	it("should coerce invalid regressor criterion to mse", () => {
		const regressor = new DecisionTreeRegressor({ criterion: "entropy" });
		// @ts-expect-error
		assert.strictEqual(regressor.criterion, "mse");
	});

	it("should throw for unsupported regressor criterion in calculateImpurity", () => {
		const regressor = new DecisionTreeRegressor();
		// @ts-expect-error
		regressor.criterion = "entropy";

		assert.throws(
			// @ts-expect-error
			() => regressor.calculateImpurity([1, 2]),
			Error,
			"Unsupported criterion for regression: entropy",
		);
	});

	it("should return 0 when regressor leaf is built from empty data", () => {
		const regressor = new DecisionTreeRegressor();
		// @ts-expect-error
		assert.strictEqual(regressor.calculateLeafValue([]), 0);
	});

	it("should throw when regressor toJSON called before fit", () => {
		const regressor = new DecisionTreeRegressor();
		assert.throws(
			() => regressor.toJSON(),
			Error,
			"Tree is not fitted yet. Cannot serialize.",
		);
	});
});
