import { RandomForestClassifier, accuracyScore } from "../src/mod.ts";

// 1. Define sample data
const X_train: number[][] = [
	[10],
	[12],
	[15],
	[11], // Class A
	[18],
	[20],
	[22],
	[19], // Class B
	[8],
	[23],
	[13],
	[17], // Mixed
];
const y_train: string[] = [
	"A",
	"A",
	"A",
	"A",
	"B",
	"B",
	"B",
	"B",
	"A",
	"B",
	"A",
	"B",
];

console.log("Training Data:");
console.log("X_train:", X_train);
console.log("y_train:", y_train);
console.log("---");

// 2. Create a RandomForestClassifier instance
const forestClassifier = new RandomForestClassifier({
	nEstimators: 10, // Number of trees in the forest
	maxDepth: 3, // Maximum depth of individual trees
	minSamplesLeaf: 1,
	bootstrap: true, // Whether to use bootstrap samples when building trees
});

// 3. Fit the model
console.log("Fitting the Random Forest model...");
forestClassifier.fit(X_train, y_train);
console.log("Model fitting complete.");
console.log("---");

// 4. Define test data
const X_test: number[][] = [[9], [13], [16], [21]];
const y_test_true: string[] = ["A", "A", "B", "B"]; // True labels for test data

console.log("Test Data:");
console.log("X_test:", X_test);
console.log("y_test_true (true labels):", y_test_true);
console.log("---");

// 5. Make predictions
const predictions = forestClassifier.predict(X_test);
console.log("Predictions for X_test:", predictions);
console.log("---");

// 6. Make probability predictions (optional)
// Note: RandomForestClassifier predictProba returns an array of Records
// where keys are class labels and values are their probabilities.
const probabilities = forestClassifier.predictProba(X_test);
console.log("Probabilities for X_test:");
probabilities.forEach((probRecord, i) => {
	// Assuming you know the possible class labels or can infer them
	// For simplicity, let's assume classes are 'A' and 'B'
	const probA = probRecord.A !== undefined ? probRecord.A.toFixed(2) : "N/A";
	const probB = probRecord.B !== undefined ? probRecord.B.toFixed(2) : "N/A";
	console.log(`Sample ${X_test[i]}: Class A: ${probA}, Class B: ${probB}`);
});
console.log("---");

// 7. Evaluate the model (optional)
const acc = accuracyScore(y_test_true, predictions);
console.log(`Accuracy on X_test: ${acc.toFixed(2)}`);
console.log("---");

// 8. Get Feature Importances (optional)
const featureImportances = forestClassifier.getFeatureImportances();
console.log("Feature Importances:", featureImportances);
