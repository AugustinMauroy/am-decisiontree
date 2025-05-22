import { DecisionTreeRegressor, meanSquaredError } from "../src/mod.ts";

// 1. Define sample data for regression
const X_train: number[][] = [
	[1],
	[2],
	[3],
	[4],
	[5], // Single feature
	[6],
	[7],
	[8],
	[9],
	[10],
];
const y_train: number[] = [
	10,
	20,
	30,
	40,
	50, // Target values
	60,
	70,
	80,
	90,
	100,
];

console.log("Training Data:");
console.log("X_train:", X_train);
console.log("y_train:", y_train);
console.log("---");

// 2. Create a DecisionTreeRegressor instance
const regressor = new DecisionTreeRegressor({
	criterion: "mse", // Use Mean Squared Error for splitting
	maxDepth: 3, // Limit the depth of the tree
	minSamplesLeaf: 1,
});

// 3. Fit the model
console.log("Fitting the model...");
regressor.fit(X_train, y_train);
console.log("Model fitting complete.");
console.log("---");

// 4. Define test data
const X_test: number[][] = [[2.5], [4.5], [7.5], [11]];
const y_test: number[] = [25, 45, 75, 110]; // True values for test data

console.log("Test Data:");
console.log("X_test:", X_test);
console.log("y_test (true values):", y_test);
console.log("---");

// 5. Make predictions
const predictions = regressor.predict(X_test);
console.log("Predictions for X_test:", predictions);

// 6. Evaluate the model (optional)
const mse = meanSquaredError(y_test, predictions);
console.log(`Mean Squared Error on X_test: ${mse.toFixed(2)}`);
console.log("---");

// 7. Get Feature Importances
const featureImportances = regressor.getFeatureImportances();
console.log("Feature Importances:", featureImportances);
