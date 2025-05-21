import { Node, type NodeValue } from "./node.ts";
import { calculateGiniImpurity } from "./criteria/gini_impurity.ts";
import { calculateEntropy } from "./criteria/entropy.ts";
import { calculateMSE } from "./criteria/mse_criterion.ts";
import { calculateMAE } from "./criteria/mae_criterion.ts";

export type Criterion = "gini" | "entropy" | "mse" | "mae";
export type XInput = number[][];
export type YInputClassification = (number | string)[];
export type YInputRegression = number[];
export type YInput = YInputClassification | YInputRegression;

export interface DecisionTreeParameters<Y_TYPE extends YInput> {
	criterion?: Criterion;
	maxDepth?: number;
	minSamplesSplit?: number;
	minSamplesLeaf?: number;
	minImpurityDecrease?: number;
}

abstract class BaseDecisionTree<
	X_IN extends XInput,
	Y_IN extends YInput,
	P_OUT extends NodeValue,
> {
	protected root?: Node<P_OUT>;
	protected criterion: Criterion;
	protected maxDepth: number;
	protected minSamplesSplit: number;
	protected minSamplesLeaf: number;
	protected minImpurityDecrease: number;
	protected nFeatures?: number;
	protected featureImportances_?: number[];

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
	}

	protected abstract getDefaultCriterion(): Criterion;
	protected abstract calculateImpurity(y: Y_IN): number;
	protected abstract calculateLeafValue(y: Y_IN): P_OUT;

	public fit(X: X_IN, y: Y_IN): void {
		if (X.length !== y.length) {
			throw new Error("X and y must have the same number of samples.");
		}
		if (X.length === 0) {
			throw new Error("Cannot fit on empty dataset.");
		}
		this.nFeatures = X[0]?.length || 0;
		this.featureImportances_ =
			this.nFeatures > 0 ? Array(this.nFeatures).fill(0) : [];
		if (this.nFeatures === 0 && X.length > 0) {
			// Handle case of samples with no features, typically results in a single leaf node.
		}
		this.root = this._buildTree(X, y, 0);
	}

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

		const { featureIndex, threshold, leftIndices, rightIndices } = bestSplit;

		const XLeft = leftIndices.map((i) => X[i]) as X_IN;
		const yLeft = leftIndices.map((i) => y[i]) as Y_IN;
		const XRight = rightIndices.map((i) => X[i]) as X_IN;
		const yRight = rightIndices.map((i) => y[i]) as Y_IN;

		// This check should ideally use minSamplesLeaf from the parameters
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
			impurity: currentImpurity,
			leftChild: leftChild,
			rightChild: rightChild,
			samples: nSamples,
			isLeaf: false,
		});
	}

	private _findBestSplit(
		X: X_IN,
		y: Y_IN,
		parentImpurity: number,
	): {
		featureIndex: number;
		threshold: number;
		impurityGain: number;
		leftIndices: number[];
		rightIndices: number[];
	} | null {
		let bestGain = Number.NEGATIVE_INFINITY;
		let bestSplitResult: {
			featureIndex: number;
			threshold: number;
			leftIndices: number[];
			rightIndices: number[];
		} | null = null;
		const nSamples = X.length;
		const parentProportion =
			nSamples / (this.root ? this.root.samples : nSamples); // Proportion of total samples

		if (this.nFeatures === undefined || this.nFeatures === 0) return null;

		for (let featureIdx = 0; featureIdx < this.nFeatures; featureIdx++) {
			const featureValues = X.map((sample) => sample[featureIdx]);
			const uniqueSortedValues = [...new Set(featureValues)].sort(
				(a, b) => a - b,
			);

			if (uniqueSortedValues.length <= 1) continue;

			for (let i = 0; i < uniqueSortedValues.length - 1; i++) {
				const threshold =
					(uniqueSortedValues[i] + uniqueSortedValues[i + 1]) / 2;
				const leftIndices: number[] = [];
				const rightIndices: number[] = [];

				for (let sampleIdx = 0; sampleIdx < nSamples; sampleIdx++) {
					if (X[sampleIdx][featureIdx] <= threshold) {
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
				const weightedImpurity = pLeft * impurityLeft + pRight * impurityRight;
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
		}

		if (bestSplitResult && this.featureImportances_) {
			this.featureImportances_[bestSplitResult.featureIndex] +=
				bestGain * parentProportion;
		}

		return bestSplitResult
			? { ...bestSplitResult, impurityGain: bestGain }
			: null;
	}

	protected _predictSample(sample: number[], node?: Node<P_OUT>): P_OUT {
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

		if (node.featureIndex === undefined || node.threshold === undefined) {
			throw new Error(
				"Invalid non-leaf node: missing featureIndex or threshold.",
			);
		}
		if (node.featureIndex >= sample.length) {
			throw new Error(
				`Feature index ${node.featureIndex} is out of bounds for sample with ${sample.length} features.`,
			);
		}

		if (sample[node.featureIndex] <= node.threshold) {
			return this._predictSample(sample, node.leftChild);
		}
		return this._predictSample(sample, node.rightChild);
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

export class DecisionTreeClassifier extends BaseDecisionTree<
	XInput,
	YInputClassification,
	Record<string | number, number>
> {
	private uniqueClasses_?: (string | number)[];

	constructor(params: DecisionTreeParameters<YInputClassification> = {}) {
		super(params);
		if (this.criterion === "mse" || this.criterion === "mae") {
			this.criterion = "gini"; // Default to gini if mse or mae is chosen for classifier
		}
	}

	protected getDefaultCriterion(): Criterion {
		return "gini";
	}

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

	public fit(X: XInput, y: YInputClassification): void {
		this.uniqueClasses_ = [...new Set(y)].sort((a, b) =>
			String(a).localeCompare(String(b)),
		);
		super.fit(X, y);
	}

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
	 * Returns the decision tree structure as a JSON string, including classifier-specific info.
	 */
	public override toJSON(): string {
		if (!this.root) {
			throw new Error("Tree is not fitted yet. Cannot serialize.");
		}
		return JSON.stringify({
			type: "classifier",
			root: this.root,
			criterion: this.criterion,
			maxDepth: this.maxDepth,
			minSamplesSplit: this.minSamplesSplit,
			minSamplesLeaf: this.minSamplesLeaf,
			minImpurityDecrease: this.minImpurityDecrease,
			nFeatures: this.nFeatures,
			uniqueClasses: this.uniqueClasses_,
		});
	}

	// Basic static method to load a model. Error handling and validation should be more robust.
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
		});
		classifier.root = obj.root as Node<Record<string | number, number>>; // Type assertion
		classifier.nFeatures = obj.nFeatures;
		classifier.uniqueClasses_ = obj.uniqueClasses;
		// Reconstruct feature importances if they were part of serialization (not currently)
		// Or mark them as needing recalculation if that's feasible post-load.
		// For simplicity, featureImportances_ are not serialized/deserialized here.
		// They would be re-calculated if fit was called again, or be unavailable.
		return classifier;
	}
}

export class DecisionTreeRegressor extends BaseDecisionTree<
	XInput,
	YInputRegression,
	number
> {
	constructor(params: DecisionTreeParameters<YInputRegression> = {}) {
		super(params);
		if (this.criterion === "gini" || this.criterion === "entropy") {
			this.criterion = "mse"; // Default to mse if classification criterion chosen for regressor
		}
	}

	protected getDefaultCriterion(): Criterion {
		return "mse";
	}

	protected calculateImpurity(y: YInputRegression): number {
		if (this.criterion === "mse") {
			return calculateMSE(y);
		}
		if (this.criterion === "mae") {
			return calculateMAE(y);
		}
		throw new Error(`Unsupported criterion for regression: ${this.criterion}`);
	}

	protected calculateLeafValue(y: YInputRegression): number {
		if (y.length === 0) {
			// Or throw error, or return NaN. Depends on desired behavior for empty leaf.
			return 0;
		}
		return y.reduce((sum, val) => sum + val, 0) / y.length;
	}

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
		return JSON.stringify({
			type: "regressor",
			root: this.root,
			criterion: this.criterion,
			maxDepth: this.maxDepth,
			minSamplesSplit: this.minSamplesSplit,
			minSamplesLeaf: this.minSamplesLeaf,
			minImpurityDecrease: this.minImpurityDecrease,
			nFeatures: this.nFeatures,
		});
	}

	// Basic static method to load a model.
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
		});
		regressor.root = obj.root as Node<number>; // Type assertion
		regressor.nFeatures = obj.nFeatures;
		return regressor;
	}
}
