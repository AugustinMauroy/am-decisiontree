import {
    DecisionTreeClassifier,
    DecisionTreeRegressor,
	type BaseDecisionTree,
	type DecisionTreeParameters,
    type XInput,
    type YInputClassification,
    type YInputRegression,
} from "../decision_tree.ts";
import type { NodeValue } from "../node.ts";

// Helper for bootstrap sampling (simplified)
function bootstrapSample<T, U>(X: T[], y: U[]): { X_sample: T[]; y_sample: U[] } {
    const n = X.length;
    const X_sample: T[] = [];
    const y_sample: U[] = [];
    for (let i = 0; i < n; i++) {
        const randomIndex = Math.floor(Math.random() * n);
        X_sample.push(X[randomIndex]);
        y_sample.push(y[randomIndex]);
    }
    return { X_sample, y_sample };
}

export interface RandomForestParameters<Y_IN extends YInputClassification | YInputRegression>
    extends DecisionTreeParameters<Y_IN> {
    nEstimators?: number;
    bootstrap?: boolean;
    // randomState for reproducibility could be added
}

abstract class BaseRandomForest<
    TREE_TYPE extends BaseDecisionTree<X_IN, Y_IN, P_OUT>,
    X_IN extends XInput,
    Y_IN extends YInputClassification | YInputRegression,
    P_OUT extends NodeValue,
> {
    protected nEstimators: number;
    protected bootstrap: boolean;
    protected trees: TREE_TYPE[];
    protected treeParams: DecisionTreeParameters<Y_IN>;

    constructor(params: RandomForestParameters<Y_IN>) {
        this.nEstimators = params.nEstimators ?? 100;
        this.bootstrap = params.bootstrap ?? true;
        this.trees = [];
        // Pass through DecisionTree specific parameters
        this.treeParams = { ...params };
        delete this.treeParams.nEstimators; // Not a DT param
        delete this.treeParams.bootstrap; // Not a DT param
    }

    protected abstract createTree(params: DecisionTreeParameters<Y_IN>): TREE_TYPE;

    public fit(X: X_IN, y: Y_IN): void {
        this.trees = [];
        for (let i = 0; i < this.nEstimators; i++) {
            const tree = this.createTree(this.treeParams);
            let X_sample = X;
            let y_sample = y;

            if (this.bootstrap) {
                const sample = bootstrapSample(X, y);
                X_sample = sample.X_sample as X_IN; // Type assertion
                y_sample = sample.y_sample as Y_IN; // Type assertion
            }
            tree.fit(X_sample, y_sample);
            this.trees.push(tree);
        }
    }

    public getFeatureImportances(): number[] {
        if (this.trees.length === 0 || !this.trees[0].getFeatureImportances()) {
            throw new Error("Forest not fitted or trees have no importances.");
        }
        const nFeatures = this.trees[0].getFeatureImportances().length;
        const importances = Array(nFeatures).fill(0);

        for (const tree of this.trees) {
            const treeImportances = tree.getFeatureImportances();
            for (let i = 0; i < nFeatures; i++) {
                importances[i] += treeImportances[i];
            }
        }
        return importances.map((imp) => imp / this.nEstimators);
    }

    // toJSON and fromJSON methods would also be needed here
    // to serialize/deserialize the forest (collection of trees and parameters)
}

export class RandomForestClassifier extends BaseRandomForest<
    DecisionTreeClassifier,
    XInput,
    YInputClassification,
    Record<string | number, number>
> {
    constructor(params: RandomForestParameters<YInputClassification> = {}) {
        super(params);
    }

    protected createTree(
        params: DecisionTreeParameters<YInputClassification>,
    ): DecisionTreeClassifier {
        return new DecisionTreeClassifier(params);
    }

    public predict(X: XInput): (string | number)[] {
        if (this.trees.length === 0) {
            throw new Error("Forest is not fitted yet.");
        }
        const predictionsMatrix: (string | number)[][] = [];
        for (const tree of this.trees) {
            predictionsMatrix.push(tree.predict(X));
        }

        // Transpose and find majority vote for each sample
        const finalPredictions: (string | number)[] = [];
        for (let i = 0; i < X.length; i++) {
            const samplePredictions: (string | number)[] = predictionsMatrix.map(
                (treePreds) => treePreds[i],
            );
            const votes: Record<string | number, number> = {};
            for (const pred of samplePredictions) {
                votes[pred] = (votes[pred] || 0) + 1;
            }
            let majorityVote: string | number = "";
            let maxCount = 0;
            for (const [label, count] of Object.entries(votes)) {
                if (count > maxCount) {
                    maxCount = count;
                    majorityVote = label;
                    // Convert back to number if original labels were numbers
                    if (!isNaN(Number(label))) majorityVote = Number(label);
                }
            }
            finalPredictions.push(majorityVote);
        }
        return finalPredictions;
    }

    public predictProba(X: XInput): Record<string | number, number>[] {
        if (this.trees.length === 0) {
            throw new Error("Forest is not fitted yet.");
        }
        const probasSum: Record<string | number, number>[] = Array(X.length)
            .fill(null)
            .map(() => ({}));

        for (const tree of this.trees) {
            const treeProbas = tree.predictProba(X); // Array of probability objects
            for (let i = 0; i < X.length; i++) {
                for (const [label, prob] of Object.entries(treeProbas[i])) {
                    probasSum[i][label] = (probasSum[i][label] || 0) + prob;
                }
            }
        }

        // Average probabilities
        return probasSum.map((sampleProbas) => {
            const averaged: Record<string | number, number> = {};
            for (const [label, sumProb] of Object.entries(sampleProbas)) {
                averaged[label] = sumProb / this.nEstimators;
            }
            return averaged;
        });
    }
}

export class RandomForestRegressor extends BaseRandomForest<
    DecisionTreeRegressor,
    XInput,
    YInputRegression,
    number
> {
    constructor(params: RandomForestParameters<YInputRegression> = {}) {
        super(params);
    }

    protected createTree(
        params: DecisionTreeParameters<YInputRegression>,
    ): DecisionTreeRegressor {
        return new DecisionTreeRegressor(params);
    }

    public predict(X: XInput): number[] {
        if (this.trees.length === 0) {
            throw new Error("Forest is not fitted yet.");
        }
        const predictionsMatrix: number[][] = [];
        for (const tree of this.trees) {
            predictionsMatrix.push(tree.predict(X));
        }

        // Transpose and average for each sample
        const finalPredictions: number[] = [];
        for (let i = 0; i < X.length; i++) {
            const samplePredictions = predictionsMatrix.map(
                (treePreds) => treePreds[i],
            );
            const sum = samplePredictions.reduce((acc, val) => acc + val, 0);
            finalPredictions.push(sum / this.nEstimators);
        }
        return finalPredictions;
    }
}
