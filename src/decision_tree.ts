import { Node, type NodeValue } from "./node.ts";
import { calculateGiniImpurity } from "./criteria/gini_impurity.ts";
import { calculateEntropy } from "./criteria/entropy.ts";
import { calculateMSE } from "./criteria/mse_criterion.ts";
import { calculateMAE } from "./criteria/mae_criterion.ts";

/**
 * Supported criteria for splitting nodes in the decision tree.
 */
export type Criterion = "gini" | "entropy" | "mse" | "mae";

/**
 * Represents a single feature value, which can be numerical or categorical.
 */
export type FeatureValue = number | string;

/**
 * Input feature matrix where each row represents a sample and each column represents a feature.
 * Features can be numerical or categorical.
 */
export type XInput = FeatureValue[][];

/**
 * Target values for classification tasks. Can be numbers or strings.
 */
export type YInputClassification = (number | string)[];

/**
 * Target values for regression tasks. Must be numbers.
 */
export type YInputRegression = number[];

/**
 * General target values for both classification and regression tasks.
 */
export type YInput = YInputClassification | YInputRegression;

/**
 * Parameters for configuring the decision tree.
 */
export interface DecisionTreeParameters<Y_TYPE extends YInput> {
	/** The criterion used to measure the quality of a split. */
	criterion?: Criterion;

	/** The maximum depth of the tree. Defaults to unlimited depth. */
	maxDepth?: number;

	/** The minimum number of samples required to split an internal node. */
	minSamplesSplit?: number;

	/** The minimum number of samples required to be at a leaf node. */
	minSamplesLeaf?: number;

	/** The minimum impurity decrease required to perform a split. */
	minImpurityDecrease?: number;

	/** Specifies the type of each feature, 'numerical' or 'categorical'. If not provided, all features are assumed to be numerical. */
	featureTypes?: ("numerical" | "categorical")[];
}

/**
 * Abstract base class for decision trees.
 * @template X_IN - The type of the input feature matrix.
 * @template Y_IN - The type of the target values.
 * @template P_OUT - The type of the predicted output values.
 */
abstract class BaseDecisionTree<
	X_IN extends XInput,
	Y_IN extends YInput,
	P_OUT extends NodeValue,
> {
	/** The root node of the decision tree. */
	protected root?: Node<P_OUT>;

	/** The criterion used to measure the quality of a split. */
	protected criterion: Criterion;

	/** The maximum depth of the tree. */
	protected maxDepth: number;

	/** The minimum number of samples required to split an internal node. */
	protected minSamplesSplit: number;

	/** The minimum number of samples required to be at a leaf node. */
	protected minSamplesLeaf: number;

	/** The minimum impurity decrease required to perform a split. */
	protected minImpurityDecrease: number;

	/** The number of features in the input data. */
	protected nFeatures?: number;

	/** The importance of each feature in the decision tree. */
	protected featureImportances_?: number[];

	/** Stores the type of each feature ('numerical' or 'categorical'). */
	protected featureTypes_?: ("numerical" | "categorical")[];

	/**
	 * Creates a new instance of a decision tree.
	 * @param params - The parameters to configure the decision tree.
	 */
	constructor(params: DecisionTreeParameters<Y_IN> = {}) {
		this.criterion = params.criterion || this.getDefaultCriterion();
		this.maxDepth =
			params.maxDepth === undefined
				? Number.POSITIVE_INFINITY
				: params.maxDepth;
		this.minSamplesSplit =
			params.minSamplesSplit === undefined ? 2 : params.minSamplesSplit;
		this.minSamplesLeaf =
			params.minSamplesLeaf === undefined ? 1 : params.minSamplesLeaf;
		this.minImpurityDecrease =
			params.minImpurityDecrease === undefined
				? 0.0
				: params.minImpurityDecrease;
		this.featureTypes_ = params.featureTypes;
	}

	/**
	 * Returns the default criterion for the decision tree.
	 */
	protected abstract getDefaultCriterion(): Criterion;

	/**
	 * Calculates the impurity of the target values.
	 * @param y - The target values.
	 */
	protected abstract calculateImpurity(y: Y_IN): number;

	/**
	 * Calculates the predicted value for a leaf node.
	 * @param y - The target values.
	 */
	protected abstract calculateLeafValue(y: Y_IN): P_OUT;

	/**
	 * Fits the decision tree to the given data.
	 * @param X - The input feature matrix.
	 * @param y - The target values.
	 */
	public fit(X: X_IN, y: Y_IN): void {
		if (X.length !== y.length) {
			throw new Error("X and y must have the same number of samples.");
		}
		if (X.length === 0) {
			throw new Error("Cannot fit on empty dataset.");
		}
		this.nFeatures = X[0]?.length || 0;
		if (this.featureTypes_ && this.featureTypes_.length !== this.nFeatures) {
			throw new Error(
				"Length of featureTypes must match the number of features in X.",
			);
		}
		// If featureTypes_ is not provided, assume all are numerical
		if (!this.featureTypes_ && this.nFeatures > 0) {
			this.featureTypes_ = Array(this.nFeatures).fill("numerical");
		}

		this.featureImportances_ =
			this.nFeatures > 0 ? Array(this.nFeatures).fill(0) : [];
		if (this.nFeatures === 0 && X.length > 0) {
			// Handle case of samples with no features, typically results in a single leaf node.
		}
		this.root = this._buildTree(X, y, 0);
	}

	/**
	 * Recursively builds the decision tree.
	 * @param X - The input feature matrix.
	 * @param y - The target values.
	 * @param depth - The current depth of the tree.
	 */
	private _buildTree(X: X_IN, y: Y_IN, depth: number): Node<P_OUT> {
		const nSamples = X.length;
		const currentImpurity = this.calculateImpurity(y);

		if (
			depth >= this.maxDepth ||
			nSamples < this.minSamplesSplit ||
			currentImpurity === 0 || // Pure node
			nSamples === 0 // Should not happen if minSamplesSplit >= 1
		) {
			return new Node<P_OUT>({
				value: this.calculateLeafValue(y),
				impurity: currentImpurity,
				samples: nSamples,
				isLeaf: true,
			});
		}

		const bestSplit = this._findBestSplit(X, y, currentImpurity);

		if (!bestSplit || bestSplit.impurityGain <= this.minImpurityDecrease) {
			return new Node<P_OUT>({
				value: this.calculateLeafValue(y),
				impurity: currentImpurity,
				samples: nSamples,
				isLeaf: true,
			});
		}

		const {
			featureIndex,
			threshold,
			splitCategories,
			leftIndices,
			rightIndices,
		} = bestSplit;

		const XLeft = leftIndices.map((i) => X[i]) as X_IN;
		const yLeft = leftIndices.map((i) => y[i]) as Y_IN;
		const XRight = rightIndices.map((i) => X[i]) as X_IN;
		const yRight = rightIndices.map((i) => y[i]) as Y_IN;

		if (
			XLeft.length < this.minSamplesLeaf ||
			XRight.length < this.minSamplesLeaf
		) {
			return new Node<P_OUT>({
				value: this.calculateLeafValue(y),
				impurity: currentImpurity,
				samples: nSamples,
				isLeaf: true,
			});
		}

		const leftChild = this._buildTree(XLeft, yLeft, depth + 1);
		const rightChild = this._buildTree(XRight, yRight, depth + 1);

		return new Node<P_OUT>({
			featureIndex: featureIndex,
			threshold: threshold,
			splitCategories: splitCategories,
			impurity: currentImpurity,
			leftChild: leftChild,
			rightChild: rightChild,
			samples: nSamples,
			isLeaf: false,
		});
	}

	/**
	 * Finds the best split for the given data.
	 * @param X - The input feature matrix.
	 * @param y - The target values.
	 * @param parentImpurity - The impurity of the parent node.
	 */
	private _findBestSplit(
		X: X_IN,
		y: Y_IN,
		parentImpurity: number,
	): {
		featureIndex: number;
		threshold?: number;
		splitCategories?: Set<string | number>;
		impurityGain: number;
		leftIndices: number[];
		rightIndices: number[];
	} | null {
		let bestGain = Number.NEGATIVE_INFINITY;
		let bestSplitResult: {
			featureIndex: number;
			threshold?: number;
			splitCategories?: Set<string | number>;
			leftIndices: number[];
			rightIndices: number[];
		} | null = null;
		const nSamples = X.length;
		const parentProportion =
			nSamples / (this.root ? this.root.samples : nSamples);

		if (this.nFeatures === undefined || this.nFeatures === 0) return null;

		for (let featureIdx = 0; featureIdx < this.nFeatures; featureIdx++) {
			const featureType = this.featureTypes_?.[featureIdx] || "numerical";
			const featureValues = X.map((sample) => sample[featureIdx]);

			if (featureType === "numerical") {
				const uniqueSortedValues = [
					...new Set(featureValues.map((v) => Number(v))),
				].sort((a, b) => a - b);

				if (uniqueSortedValues.length <= 1) continue;

				for (let i = 0; i < uniqueSortedValues.length - 1; i++) {
					const threshold =
						(uniqueSortedValues[i] + uniqueSortedValues[i + 1]) / 2;
					const leftIndices: number[] = [];
					const rightIndices: number[] = [];

					for (let sampleIdx = 0; sampleIdx < nSamples; sampleIdx++) {
						if (Number(X[sampleIdx][featureIdx]) <= threshold) {
							leftIndices.push(sampleIdx);
						} else {
							rightIndices.push(sampleIdx);
						}
					}

					if (
						leftIndices.length < this.minSamplesLeaf ||
						rightIndices.length < this.minSamplesLeaf
					) {
						continue;
					}

					const yLeft = leftIndices.map((idx) => y[idx]) as Y_IN;
					const yRight = rightIndices.map((idx) => y[idx]) as Y_IN;

					const impurityLeft = this.calculateImpurity(yLeft);
					const impurityRight = this.calculateImpurity(yRight);

					const pLeft = leftIndices.length / nSamples;
					const pRight = rightIndices.length / nSamples;
					const weightedImpurity =
						pLeft * impurityLeft + pRight * impurityRight;
					const impurityGain = parentImpurity - weightedImpurity;

					if (impurityGain > bestGain) {
						bestGain = impurityGain;
						bestSplitResult = {
							featureIndex: featureIdx,
							threshold,
							leftIndices,
							rightIndices,
						};
					}
				}
			} else {
				// Categorical feature
				const uniqueCategories = Array.from(new Set(featureValues));
				if (uniqueCategories.length <= 1) continue;

				const potentialCategorySplits: Set<FeatureValue>[] = [];

				if (uniqueCategories.length === 2) {
					potentialCategorySplits.push(new Set([uniqueCategories[0]]));
				} else if (uniqueCategories.length > 2) {
					const categoryMetrics: { category: FeatureValue; metric: number }[] =
						[];
					if (this instanceof DecisionTreeClassifier) {
						const firstClass = (
							this as DecisionTreeClassifier
						).getFirstUniqueClass();
						if (firstClass === undefined) continue;

						for (const cat of uniqueCategories) {
							const yForCat = y.filter(
								(_val, i) => X[i][featureIdx] === cat,
							) as YInputClassification;
							if (yForCat.length === 0) continue;
							const probFirstClass =
								yForCat.filter((label) => label === firstClass).length /
								yForCat.length;
							categoryMetrics.push({ category: cat, metric: probFirstClass });
						}
					} else if (this instanceof DecisionTreeRegressor) {
						for (const cat of uniqueCategories) {
							const yForCat = y
								.filter((_val, i) => X[i][featureIdx] === cat)
								.map((val) => Number(val)) as YInputRegression;
							if (yForCat.length === 0) continue;
							const meanTarget =
								yForCat.reduce((a, b) => a + b, 0) / yForCat.length;
							categoryMetrics.push({ category: cat, metric: meanTarget });
						}
					}

					categoryMetrics.sort((a, b) => a.metric - b.metric);
					const sortedCategories = categoryMetrics.map((cm) => cm.category);

					for (let i = 0; i < sortedCategories.length - 1; i++) {
						potentialCategorySplits.push(
							new Set(sortedCategories.slice(0, i + 1)),
						);
					}
				}

				for (const leftCategorySet of potentialCategorySplits) {
					if (
						leftCategorySet.size === 0 ||
						leftCategorySet.size === uniqueCategories.length
					) {
						continue; // Avoid empty or full splits that don't partition
					}
					const leftIndices: number[] = [];
					const rightIndices: number[] = [];
					for (let sampleIdx = 0; sampleIdx < nSamples; sampleIdx++) {
						if (leftCategorySet.has(X[sampleIdx][featureIdx])) {
							leftIndices.push(sampleIdx);
						} else {
							rightIndices.push(sampleIdx);
						}
					}

					if (
						leftIndices.length < this.minSamplesLeaf ||
						rightIndices.length < this.minSamplesLeaf ||
						leftIndices.length === 0 ||
						rightIndices.length === 0
					) {
						continue;
					}

					const yLeft = leftIndices.map((idx) => y[idx]) as Y_IN;
					const yRight = rightIndices.map((idx) => y[idx]) as Y_IN;

					const impurityLeft = this.calculateImpurity(yLeft);
					const impurityRight = this.calculateImpurity(yRight);

					const pLeft = leftIndices.length / nSamples;
					const pRight = rightIndices.length / nSamples;
					const weightedImpurity =
						pLeft * impurityLeft + pRight * impurityRight;
					const impurityGain = parentImpurity - weightedImpurity;

					if (impurityGain > bestGain) {
						bestGain = impurityGain;
						bestSplitResult = {
							featureIndex: featureIdx,
							splitCategories: leftCategorySet,
							leftIndices,
							rightIndices,
						};
					}
				}
			}
		}

		if (bestSplitResult && this.featureImportances_ && bestGain > 0) {
			this.featureImportances_[bestSplitResult.featureIndex] +=
				bestGain * parentProportion;
		}

		return bestSplitResult
			? { ...bestSplitResult, impurityGain: bestGain }
			: null;
	}

	/**
	 * Predicts the output for a single sample.
	 * @param sample - The input sample.
	 * @param node - The current node in the tree.
	 */
	protected _predictSample(sample: FeatureValue[], node?: Node<P_OUT>): P_OUT {
		if (!node) {
			throw new Error("Tree is not fitted yet or root node is undefined.");
		}
		if (node.isLeaf || node.value !== undefined) {
			if (node.value === undefined) {
				// This should not happen for a leaf node if calculateLeafValue is correct
				throw new Error("Leaf node has undefined value.");
			}
			return node.value;
		}

		if (node.featureIndex === undefined) {
			throw new Error("Invalid non-leaf node: missing featureIndex.");
		}
		if (node.featureIndex >= sample.length) {
			throw new Error(
				`Feature index ${node.featureIndex} is out of bounds for sample with ${sample.length} features.`,
			);
		}

		const featureValue = sample[node.featureIndex];

		if (node.splitCategories) {
			// Categorical split
			if (node.leftChild && node.rightChild) {
				// Ensure children exist
				if (node.splitCategories.has(featureValue)) {
					return this._predictSample(sample, node.leftChild);
				}
				return this._predictSample(sample, node.rightChild);
			}
			throw new Error(
				"Invalid non-leaf node: missing children for categorical split.",
			);
		}
		if (node.threshold !== undefined) {
			// Numerical split
			if (node.leftChild && node.rightChild) {
				// Ensure children exist
				if (Number(featureValue) <= node.threshold) {
					return this._predictSample(sample, node.leftChild);
				}
				return this._predictSample(sample, node.rightChild);
			}
			throw new Error(
				"Invalid non-leaf node: missing children for numerical split.",
			);
		}
		throw new Error(
			"Invalid non-leaf node: missing split criteria (threshold or splitCategories).",
		);
	}

	/**
	 * Returns the feature importances.
	 * The importance of a feature is computed as the (normalized)
	 * total reduction of the criterion brought by that feature.
	 * It is also known as the Gini importance.
	 */
	public getFeatureImportances(): number[] {
		if (!this.featureImportances_) {
			throw new Error(
				"Feature importances are not available. Fit the model first.",
			);
		}
		const totalImportance = this.featureImportances_.reduce(
			(sum, imp) => sum + imp,
			0,
		);
		if (totalImportance === 0) {
			return Array(this.nFeatures || 0).fill(0);
		}
		return this.featureImportances_.map((imp) => imp / totalImportance);
	}

	/**
	 * Returns the decision tree structure as a JSON string.
	 */
	public toJSON(): string {
		if (!this.root) {
			throw new Error("Tree is not fitted yet. Cannot serialize.");
		}
		// Basic serialization, can be expanded with more metadata
		return JSON.stringify({
			root: this.root, // Assumes Node class can be serialized by JSON.stringify
			criterion: this.criterion,
			maxDepth: this.maxDepth,
			minSamplesSplit: this.minSamplesSplit,
			minSamplesLeaf: this.minSamplesLeaf,
			minImpurityDecrease: this.minImpurityDecrease,
			nFeatures: this.nFeatures,
			// For classifier, uniqueClasses_ might be needed
			// For regressor, no extra specific state beyond base
		});
	}
}

// Helper function to prepare node for JSON serialization
// biome-ignore lint: Any is needed here
function serializeNode<T extends NodeValue>(node: Node<T>): any {
	// biome-ignore lint: Any is needed here
	const serialized: any = { ...node };
	// Convert Set to Array for JSON compatibility
	if (node.splitCategories instanceof Set) {
		serialized.splitCategories = Array.from(node.splitCategories);
	}
	if (node.leftChild) {
		serialized.leftChild = serializeNode(node.leftChild);
	}
	if (node.rightChild) {
		serialized.rightChild = serializeNode(node.rightChild);
	}
	return serialized;
}

// Helper function to deserialize node from JSON object
// biome-ignore lint: Any is needed here
function deserializeNode<T extends NodeValue>(nodeData: any): Node<T> {
	const options: ConstructorParameters<typeof Node>[0] = {
		...nodeData,
		leftChild: nodeData.leftChild
			? deserializeNode(nodeData.leftChild)
			: undefined,
		rightChild: nodeData.rightChild
			? deserializeNode(nodeData.rightChild)
			: undefined,
	};
	// Convert Array back to Set
	if (Array.isArray(nodeData.splitCategories)) {
		options.splitCategories = new Set(nodeData.splitCategories);
	}

	// @ts-ignore - IDK how to fix this
	return new Node<T>(options);
}

/**
 * A decision tree classifier for classification tasks.
 */
export class DecisionTreeClassifier extends BaseDecisionTree<
	XInput,
	YInputClassification,
	Record<string | number, number>
> {
	/** The unique classes in the target values. */
	private uniqueClasses_?: (string | number)[];

	/**
	 * Creates a new instance of a decision tree classifier.
	 * @param params - The parameters to configure the classifier.
	 */
	constructor(params: DecisionTreeParameters<YInputClassification> = {}) {
		super(params);
		if (this.criterion === "mse" || this.criterion === "mae") {
			this.criterion = "gini"; // Default to gini if mse or mae is chosen for classifier
		}
	}

	/**
	 * Returns the default criterion for the classifier.
	 */
	protected getDefaultCriterion(): Criterion {
		return "gini";
	}

	/**
	 * Calculates the impurity of the target values for classification.
	 * @param y - The target values.
	 */
	protected calculateImpurity(y: YInputClassification): number {
		if (this.criterion === "gini") {
			return calculateGiniImpurity(y);
		}
		if (this.criterion === "entropy") {
			return calculateEntropy(y);
		}
		throw new Error(
			`Unsupported criterion for classification: ${this.criterion}`,
		);
	}

	/**
	 * Calculates the predicted value for a leaf node in classification.
	 * @param y - The target values.
	 */
	protected calculateLeafValue(
		y: YInputClassification,
	): Record<string | number, number> {
		const nSamples = y.length;
		const probabilities: Record<string | number, number> = {};
		if (nSamples === 0) return probabilities; // Empty distribution

		const counts: Record<string | number, number> = {};
		for (const label of y) {
			counts[label] = (counts[label] || 0) + 1;
		}

		for (const label in counts) {
			probabilities[label] = counts[label] / nSamples;
		}
		return probabilities;
	}

	/**
	 * Fits the classifier to the given data.
	 * @param X - The input feature matrix.
	 * @param y - The target values.
	 */
	public fit(X: XInput, y: YInputClassification): void {
		this.uniqueClasses_ = [...new Set(y)].sort((a, b) =>
			String(a).localeCompare(String(b)),
		);
		super.fit(X, y);
	}

	/**
	 * Predicts the class labels for the given input data.
	 * @param X - The input feature matrix.
	 */
	public predict(X: XInput): (string | number)[] {
		if (!this.root) throw new Error("Tree is not fitted yet.");
		const probaPredictions = X.map((sample) =>
			this._predictSample(sample, this.root),
		);

		return probaPredictions.map((probMap) => {
			let maxProb = -1;
			// Default to first unique class or handle empty probMap if necessary
			let predictedClass: string | number =
				this.uniqueClasses_ && this.uniqueClasses_.length > 0
					? this.uniqueClasses_[0]
					: Object.keys(probMap)[0];

			if (
				Object.keys(probMap).length === 0 &&
				this.uniqueClasses_ &&
				this.uniqueClasses_.length > 0
			) {
				// If probMap is empty (e.g. leaf from 0 samples), predict the first class or handle as error
				// This case should be rare if minSamplesLeaf > 0
				return predictedClass;
			}

			for (const clsStr in probMap) {
				// Ensure cls is of the correct type (number or string) based on uniqueClasses_
				let cls: string | number = clsStr;
				if (
					this.uniqueClasses_ &&
					this.uniqueClasses_.length > 0 &&
					typeof this.uniqueClasses_[0] === "number"
				) {
					const numVal = Number.parseFloat(clsStr);
					if (!Number.isNaN(numVal)) {
						cls = numVal;
					}
				}

				if (probMap[clsStr] > maxProb) {
					maxProb = probMap[clsStr];
					predictedClass = cls;
				}
			}
			return predictedClass;
		});
	}

	/**
	 * Predicts the class probabilities for the given input data.
	 * @param X - The input feature matrix.
	 */
	public predictProba(X: XInput): number[][] {
		if (!this.root) throw new Error("Tree is not fitted yet.");
		if (!this.uniqueClasses_) {
			throw new Error("Unique classes not determined. Fit the model first.");
		}
		const probaMapPredictions = X.map((sample) =>
			this._predictSample(sample, this.root),
		);

		return probaMapPredictions.map((probMap) => {
			return this.uniqueClasses_?.map((cls) => probMap[String(cls)] || 0);
		});
	}

	/**
	 * Internal helper to get the first unique class, used for categorical split heuristic.
	 */
	public getFirstUniqueClass(): string | number | undefined {
		return this.uniqueClasses_ && this.uniqueClasses_.length > 0
			? this.uniqueClasses_[0]
			: undefined;
	}

	/**
	 * Returns the decision tree structure as a JSON string, including classifier-specific info.
	 */
	public override toJSON(): string {
		if (!this.root) {
			throw new Error("Tree is not fitted yet. Cannot serialize.");
		}
		const serializedRoot = serializeNode(this.root);
		return JSON.stringify({
			type: "classifier",
			root: serializedRoot,
			criterion: this.criterion,
			maxDepth: this.maxDepth,
			minSamplesSplit: this.minSamplesSplit,
			minSamplesLeaf: this.minSamplesLeaf,
			minImpurityDecrease: this.minImpurityDecrease,
			nFeatures: this.nFeatures,
			featureTypes: this.featureTypes_,
			uniqueClasses: this.uniqueClasses_,
		});
	}

	/**
	 * Loads a decision tree classifier from a JSON string.
	 * @param jsonString - The JSON string representing the classifier.
	 */
	public static fromJSON(jsonString: string): DecisionTreeClassifier {
		const obj = JSON.parse(jsonString);
		if (obj.type !== "classifier") {
			throw new Error(
				"JSON string does not represent a DecisionTreeClassifier.",
			);
		}
		const classifier = new DecisionTreeClassifier({
			criterion: obj.criterion,
			maxDepth: obj.maxDepth,
			minSamplesSplit: obj.minSamplesSplit,
			minSamplesLeaf: obj.minSamplesLeaf,
			minImpurityDecrease: obj.minImpurityDecrease,
			featureTypes: obj.featureTypes,
		});
		if (obj.root) {
			classifier.root = deserializeNode(obj.root) as Node<
				Record<string | number, number>
			>;
		}
		classifier.nFeatures = obj.nFeatures;
		classifier.uniqueClasses_ = obj.uniqueClasses;
		return classifier;
	}
}

/**
 * A decision tree regressor for regression tasks.
 */
export class DecisionTreeRegressor extends BaseDecisionTree<
	XInput,
	YInputRegression,
	number
> {
	/**
	 * Creates a new instance of a decision tree regressor.
	 * @param params - The parameters to configure the regressor.
	 */
	constructor(params: DecisionTreeParameters<YInputRegression> = {}) {
		super(params);
		if (this.criterion === "gini" || this.criterion === "entropy") {
			this.criterion = "mse"; // Default to mse if classification criterion chosen for regressor
		}
	}

	/**
	 * Returns the default criterion for the regressor.
	 */
	protected getDefaultCriterion(): Criterion {
		return "mse";
	}

	/**
	 * Calculates the impurity of the target values for regression.
	 * @param y - The target values.
	 */
	protected calculateImpurity(y: YInputRegression): number {
		if (this.criterion === "mse") {
			return calculateMSE(y);
		}
		if (this.criterion === "mae") {
			return calculateMAE(y);
		}
		throw new Error(`Unsupported criterion for regression: ${this.criterion}`);
	}

	/**
	 * Calculates the predicted value for a leaf node in regression.
	 * @param y - The target values.
	 */
	protected calculateLeafValue(y: YInputRegression): number {
		if (y.length === 0) {
			// Or throw error, or return NaN. Depends on desired behavior for empty leaf.
			return 0;
		}
		return y.reduce((sum, val) => sum + val, 0) / y.length;
	}

	/**
	 * Predicts the target values for the given input data.
	 * @param X - The input feature matrix.
	 */
	public predict(X: XInput): number[] {
		if (!this.root) throw new Error("Tree is not fitted yet.");
		return X.map((sample) => this._predictSample(sample, this.root));
	}

	/**
	 * Returns the decision tree structure as a JSON string, including regressor-specific info.
	 */
	public override toJSON(): string {
		if (!this.root) {
			throw new Error("Tree is not fitted yet. Cannot serialize.");
		}
		const serializedRoot = serializeNode(this.root);
		return JSON.stringify({
			type: "regressor",
			root: serializedRoot,
			criterion: this.criterion,
			maxDepth: this.maxDepth,
			minSamplesSplit: this.minSamplesSplit,
			minSamplesLeaf: this.minSamplesLeaf,
			minImpurityDecrease: this.minImpurityDecrease,
			nFeatures: this.nFeatures,
			featureTypes: this.featureTypes_,
		});
	}

	/**
	 * Loads a decision tree regressor from a JSON string.
	 * @param jsonString - The JSON string representing the regressor.
	 */
	public static fromJSON(jsonString: string): DecisionTreeRegressor {
		const obj = JSON.parse(jsonString);
		if (obj.type !== "regressor") {
			throw new Error(
				"JSON string does not represent a DecisionTreeRegressor.",
			);
		}
		const regressor = new DecisionTreeRegressor({
			criterion: obj.criterion,
			maxDepth: obj.maxDepth,
			minSamplesSplit: obj.minSamplesSplit,
			minSamplesLeaf: obj.minSamplesLeaf,
			minImpurityDecrease: obj.minImpurityDecrease,
			featureTypes: obj.featureTypes,
		});
		if (obj.root) {
			regressor.root = deserializeNode(obj.root) as Node<number>;
		}
		regressor.nFeatures = obj.nFeatures;
		return regressor;
	}
}
