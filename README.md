# Decision Tree

[![JSR](https://jsr.io/badges/@am/decisiontree)](https://jsr.io/@am/decisiontree)

`@am/decisiontree` is a TypeScript library for creating and using decision tree models for classification and regression tasks. It aims to be straightforward to use while providing core functionalities for building effective tree-based models.

The library is designed to be cross-runtime compatible, allowing usage in Node.js, Deno, Bun, and modern browsers.

## How to construct and use a Decision Tree

Using `@am/decisiontree` involves these main steps:

1.  **Import necessary classes**:
    ```typescript
    import { DecisionTreeClassifier, DecisionTreeRegressor } from "@am/decisiontree";
    // Or from "./src/mod.ts" if using locally
    // import { DecisionTreeClassifier } from "./src/mod.ts";
    ```
2.  **Prepare your data**:
    *   `X`: An array of arrays, where each inner array represents a sample's features (numeric).
    *   `y`: An array of target values. For classification, these are class labels (string or number). For regression, these are continuous numeric values.
    ```typescript
    // Example for Classification
    const X_train_clf: number[][] = [[10], [12], [15], [18], [20], [22]];
    const y_train_clf: string[] = ["A", "A", "A", "B", "B", "B"];

    // Example for Regression
    const X_train_reg: number[][] = [[1], [2], [3], [4], [5]];
    const y_train_reg: number[] = [10, 20, 30, 40, 50];
    ```
3.  **Instantiate the Model**: Create an instance of `DecisionTreeClassifier` or `DecisionTreeRegressor`. You can specify parameters like `criterion`, `maxDepth`, `minSamplesSplit`, etc.
    ```typescript
    // For Classification
    const classifier = new DecisionTreeClassifier({
        criterion: "gini", // or "entropy"
        maxDepth: 3,
        minSamplesLeaf: 1,
    });

    // For Regression
    const regressor = new DecisionTreeRegressor({
        criterion: "mse", // or "mae"
        maxDepth: 4,
    });
    ```
4.  **Train the Model**: Use the `fit` method with your training data.
    ```typescript
    classifier.fit(X_train_clf, y_train_clf);
    regressor.fit(X_train_reg, y_train_reg);
    ```
5.  **Make Predictions**:
    *   Use the `predict` method with new input data (`X_test`) to get predictions.
    *   For `DecisionTreeClassifier`, you can also use `predictProba` to get class probabilities.
    ```typescript
    const X_test_clf: number[][] = [[9], [17]];
    const predictions_clf = classifier.predict(X_test_clf);
    const probabilities_clf = classifier.predictProba(X_test_clf);

    const X_test_reg: number[][] = [[2.5], [4.5]];
    const predictions_reg = regressor.predict(X_test_reg);
    ```
6.  **Evaluate the Model (Optional)**: Use metrics functions from the library (e.g., `accuracyScore`, `meanSquaredError`) to evaluate performance.
    ```typescript
    import { accuracyScore, meanSquaredError } from "@am/decisiontree";
    // import { accuracyScore, meanSquaredError } from "./src/mod.ts"; // if local

    // Assuming y_test_clf and y_test_reg are available
    // const acc = accuracyScore(y_test_clf, predictions_clf);
    // const mse = meanSquaredError(y_test_reg, predictions_reg);
    ```
7.  **Inspect Feature Importances**:
    ```typescript
    const importances_clf = classifier.getFeatureImportances();
    console.log("Classifier Feature Importances:", importances_clf);

    const importances_reg = regressor.getFeatureImportances();
    console.log("Regressor Feature Importances:", importances_reg);
    ```

## Simple Classifier Example

Here's a basic example of using `DecisionTreeClassifier` to classify data:

```typescript
// filepath: examples/simple_classifier_readme.ts
import { DecisionTreeClassifier, accuracyScore } from "@am/decisiontree";
// Or if running directly from the repository:
// import { DecisionTreeClassifier, accuracyScore } from "../src/mod.ts";

// 1. Define sample data
const X_train: number[][] = [
    [10], [12], [15], // Class A
    [18], [20], [22], // Class B
];
const y_train: string[] = ["A", "A", "A", "B", "B", "B"];

console.log("Training Data:");
console.log("X_train:", X_train);
console.log("y_train:", y_train);
console.log("---");

// 2. Create a DecisionTreeClassifier instance
const classifier = new DecisionTreeClassifier({
    criterion: "gini",
    maxDepth: 2, // Keep it simple for the example
    minSamplesLeaf: 1,
});

// 3. Fit the model
console.log("Fitting the model...");
classifier.fit(X_train, y_train);
console.log("Model fitting complete.");
console.log("---");

// 4. Define test data
const X_test: number[][] = [[9], [13], [16], [21]];
const y_test: string[] = ["A", "A", "B", "B"]; // True labels for test data

console.log("Test Data:");
console.log("X_test:", X_test);
console.log("y_test (true labels):", y_test);
console.log("---");

// 5. Make predictions
const predictions = classifier.predict(X_test);
console.log("Predictions for X_test:", predictions);

// 6. Make probability predictions
const probabilities = classifier.predictProba(X_test);
console.log("Probabilities for X_test (Classes sorted alphabetically, e.g., A, B):");
probabilities.forEach((prob, i) => {
    // Assuming unique classes are sorted, e.g., ['A', 'B']
    // You can get sorted unique classes from classifier.uniqueClasses_ (internal, for inspection)
    console.log(
        `Sample ${X_test[i]}: Class A: ${prob[0]?.toFixed(2) ?? "N/A"}, Class B: ${prob[1]?.toFixed(2) ?? "N/A"}`,
    );
});
console.log("---");

// 7. Evaluate the model (optional)
const acc = accuracyScore(y_test, predictions);
console.log(`Accuracy on X_test: ${acc.toFixed(2)}`);
console.log("---");

// 8. Get Feature Importances
const featureImportances = classifier.getFeatureImportances();
console.log("Feature Importances:", featureImportances);

console.log(
    "\nNote: The exact split point and thus predictions/probabilities might vary if multiple splits yield similar impurity reductions.",
);
```

For more detailed examples, including regression and usage of different parameters, please check out the [examples/](examples/) folder in the repository.
