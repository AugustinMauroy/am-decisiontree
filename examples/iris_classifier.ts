import { readFile } from "node:fs/promises";
import { DecisionTreeClassifier, accuracyScore } from "../src/mod.ts";

async function loadIrisDataFromCsv(
	filePath: string,
): Promise<{ X: number[][]; y: string[] }> {
	const fileContent = await readFile(filePath, { encoding: "utf-8" });
	const lines = fileContent.trim().split("\n");

	lines.shift(); // Remove header

	const X_data: number[][] = [];
	const y_data: string[] = [];

	for (const line of lines) {
		const values = line.split(",");
		if (values.length === 5) {
			// Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
			const features = [
				Number.parseFloat(values[0]), // SepalLengthCm
				Number.parseFloat(values[1]), // SepalWidthCm
				Number.parseFloat(values[2]), // PetalLengthCm
				Number.parseFloat(values[3]), // PetalWidthCm
			];
			const target = values[4].trim().replace(/^"|"$/g, ""); // Species
			X_data.push(features);
			y_data.push(target);
		} else {
			console.warn(`Skipping line due to unexpected format: \n${line}`);
		}
	}
	return { X: X_data, y: y_data };
}

// 1. Load and prepare Iris data from CSV
console.log("Loading data from iris.csv...");
const data = await loadIrisDataFromCsv("examples/iris.csv");

// suffle data before splitting beause the CSV is ordered by species
for (let i = data.X.length - 1; i > 0; i--) {
	const j = Math.floor(Math.random() * (i + 1));
	[data.X[i], data.X[j]] = [data.X[j], data.X[i]];
	[data.y[i], data.y[j]] = [data.y[j], data.y[i]];
}

// split data into 80/20 train/test
const splitIndex = Math.floor(data.X.length * 0.8);
const X_train = data.X.slice(0, splitIndex);
const y_train = data.y.slice(0, splitIndex);
const X_test = data.X.slice(splitIndex);
const y_test_true = data.y.slice(splitIndex);

console.log("Training Data (from CSV):");
console.log("X_train:", X_train.length, "samples");
console.log("y_train:", y_train.length, "samples");
console.log("---");

// 2. Create a DecisionTreeClassifier instance
const classifier = new DecisionTreeClassifier({
	criterion: "gini", // or "entropy"
	maxDepth: 4, // Example depth
	minSamplesLeaf: 1,
});

// 3. Fit the model
console.log("Fitting the model...");
classifier.fit(X_train, y_train);
console.log("Model fitting complete.");
console.log("---");

// 5. Make predictions
const predictions = classifier.predict(X_test);
console.log("Predictions for X_test:", predictions);
console.log("---");

// 6. Make probability predictions
const probabilities = classifier.predictProba(X_test);
console.log("Probabilities for X_test (Classes sorted alphabetically):");

const uniqueClasses =
	// @ts-ignore
	classifier.uniqueClasses_ ||
	[...new Set(y_train)].sort((a, b) => String(a).localeCompare(String(b)));
console.log("Class order for probabilities:", uniqueClasses);

probabilities.forEach((prob, i) => {
	const classProbs = uniqueClasses
		.map((cls, j) => `${cls}: ${prob[j]?.toFixed(2) ?? "N/A"}`)
		.join(", ");
	const expectedClass = y_test_true[i];
	const predictedClass = predictions[i];
	const isCorrect = expectedClass === predictedClass;
	console.log(`Sample ${i + 1}:`, {
		expected: expectedClass,
		predicted: predictedClass,
		isCorrect: isCorrect,
		probabilities: classProbs,
	});
});
console.log("---");

// 7. Evaluate the model (optional)
if (y_test_true.length > 0) {
	const acc = accuracyScore(y_test_true, predictions);
	console.log(`Accuracy on X_test: ${acc.toFixed(2)}`);
	console.log("---");
}

// 8. Get Feature Importances
const featureImportances = classifier.getFeatureImportances();
console.log(
	"Feature Importances (Sepal Length, Sepal Width, Petal Length, Petal Width):",
	featureImportances,
);
