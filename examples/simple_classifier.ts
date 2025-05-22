import { DecisionTreeClassifier, accuracyScore } from "../src/mod.ts";

// 1. Define sample data
const X_train: number[][] = [
	[10], // feature 1
	[12],
	[15],
	[18],
	[20],
	[22],
];
const y_train: string[] = ["A", "A", "A", "B", "B", "B"];

console.log("Training Data:");
console.log("X_train:", X_train);
console.log("y_train:", y_train);
console.log("---");

// 2. Create a DecisionTreeClassifier instance
// We can specify parameters, or use defaults
const classifier = new DecisionTreeClassifier({
	criterion: "gini",
	maxDepth: 3,
	minSamplesSplit: 2,
	minSamplesLeaf: 1,
});

// 3. Fit the model
console.log("Fitting the model...");
classifier.fit(X_train, y_train);
console.log("Model fitting complete.");
console.log("---");

// 4. Define test data
const X_test: number[][] = [[9], [13], [16], [21]];

console.log("Test Data:");
console.log("X_test:", X_test);
console.log("---");

// 5. Make predictions
const predictions = classifier.predict(X_test);
console.log("Predictions for X_test:", predictions); // Expected: ['A', 'A', 'B', 'B'] or similar based on split

// 6. Make probability predictions
const probabilities = classifier.predictProba(X_test);
console.log("Probabilities for X_test (Classes: A, B):");
probabilities.forEach((prob, i) => {
	console.log(
		`Sample ${i} (${X_test[i]}): Class A: ${prob[0].toFixed(2)}, Class B: ${prob[1].toFixed(2)}`,
	);
});
console.log("---");

// Test with a value clearly in one class
const X_single_A = [[5]];
const pred_A = classifier.predict(X_single_A);
const prob_A = classifier.predictProba(X_single_A);
console.log(`Prediction for ${X_single_A}: ${pred_A[0]}`);
console.log(
	`Probabilities for ${X_single_A}: Class A: ${prob_A[0][0].toFixed(2)}, Class B: ${prob_A[0][1].toFixed(2)}`,
);

const X_single_B = [[25]];
const pred_B = classifier.predict(X_single_B);
const prob_B = classifier.predictProba(X_single_B);
console.log(`Prediction for ${X_single_B}: ${pred_B[0]}`);
console.log(
	`Probabilities for ${X_single_B}: Class A: ${prob_B[0][0].toFixed(2)}, Class B: ${prob_B[0][1].toFixed(2)}`,
);

// 7. Evaluate the model
const y_test: string[] = ["A", "A", "B", "B"];
const accuracy = accuracyScore(y_test, predictions);
console.log("Accuracy of the model:", accuracy.toFixed(2)); // Expected: 1.00 (or close to it)
console.log("---");
