# Specifications: Decision Tree Library in TypeScript

## 1. General Objective

Develop a robust, modular, high-performance, and easy-to-use TypeScript library for creating, training, and using decision tree models for classification and regression tasks. The library should offer a clear API and features comparable in completeness to a modern neural network library.

# 2. Key Features

2.1. Types of Decision Trees
	* Classification Tree: ✅
	* Predict discrete class labels. ✅
	* Calculate class membership probabilities. ✅
	* Regression Tree: ✅
	* Predict continuous values. ✅

2.2. Tree Construction (Learning / fit)
	* Construction Algorithms: ✅
		* Implementation of a basic algorithm (e.g., CART variant). ✅ * Split Criteria: ✅
		* For classification: Gini Index, Entropy (Information Gain). ✅
		* Gini Index is implemented in calculateGiniImpurity.
		* Entropy is implemented in calculateEntropy.
		* For regression: Mean Squared Error (MSE), Mean Absolute Error (MAE). ✅
		* MSE is implemented in calculateMSE.
		* MAE is implemented in calculateMAE.
		* Feature Type Management: ✅
		* Numerical features (continuous or discrete). ✅
		* Categorical features (with appropriate management, e.g., implicit one-hot encoding or native handling if possible). ✅
		* Growth Control Parameters: ✅
			* max_depth: Maximum tree depth. ✅ (Implemented as maxDepth in DecisionTreeParameters) * min_samples_split: Minimum number of samples required to split an internal node. ✅
			* min_samples_leaf: Minimum number of samples required in a leaf node. ✅
			* min_impurity_decrease: Minimum impurity decrease threshold to perform a split. ✅
		* Missing Value Management (Optional, for advanced completeness): ✅
		* Basic strategies for handling NaNs (e.g., simple imputation, sending samples to both branches with weighting). ✅

2.3. Prediction (predict, predict_proba) ✅
	* Efficient tree traversal for new instances. ✅
	* predict(X): Returns predictions (classes or values). ✅
	* predict_proba(X) (for classification): Returns class probabilities. ✅

2.4. Pruning ✅
	* Mechanisms to reduce overfitting. ✅
	* Example: Minimal Cost-Complexity Pruning. ✅
	* Configurable pruning parameters. ✅

2.5. Model Evaluation ✅
	* Provide functions or integrate with utilities to calculate common performance metrics: ✅
	* Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix. ✅
		* Accuracy: accuracyScore ✅
		* Precision: precisionScore ✅
		* Recall: recallScore ✅
		* F1 Score: f1Score ✅
		* Confusion Matrix: confusionMatrix ✅
		* Regression: MSE, MAE, R². ✅
		* MSE: meanSquaredError ✅
		* MAE: meanAbsoluteError ✅
		* R²: rSquared ✅

2.6. Feature Importance ✅
	* Calculate and expose the importance of each feature in the trained model (e.g., based on mean impurity reduction). ✅

2.7. Serialization and Deserialization ✅
	* Save the structure and parameters of a trained tree (e.g., in JSON format). ✅
	* Load a saved model for reuse. ✅

2.8. (Advanced) Tree Ensembles - For Maximum Completeness ✅
	* Random Forest:
		* For classification and regression. ✅
		* Construction of multiple trees on subsets of samples (bagging) and features. ✅
		* Aggregation of predictions (majority vote for classification, average for regression). ✅
		* Specific parameters: n_estimators (number of trees), max_features (number of features to consider for each split). ✅

## 3. Architecture and Design

3.1. Modularity:
	* Clear separation of responsibilities:
	* Node Structure (Node). ✅
	* Tree Structure.✅
	* Split criteria logic. ✅
	* Learning algorithms. ✅
	* Prediction functions. ✅

3.2. API:
	* Intuitive and consistent user interface, potentially inspired by popular APIs (e.g., scikit-learn). ✅
	* Main methods: fit(X, y), predict(X), predict_proba(X). ✅
	* Model configuration via constructor parameters or dedicated methods. ✅ (Constructor parameters)

3.3. Typing:
	* Rigorous use of TypeScript for code robustness and clarity. ✅
	* Clear type definitions for input data, parameters, and outputs. ✅ (e.g. XInput, YInputClassification, YInputRegression, DecisionTreeParameters)

3.4. Performance:
	* Optimization of construction and prediction algorithms for good performance, especially with large datasets. 🏗️ (Basic optimizations are in place, further profiling and optimization could be a continuous effort)
	* Use of efficient data structures. 🏗️ (Standard arrays and objects are used; more specialized structures could be considered for extreme performance needs)

3.5. Extensibility:
	* Design allowing easy addition of new split criteria, pruning strategies, or even new types of trees/ensembles in the future. ✅

## 4. Input and Output Data

Input (X): Accept 2D arrays (or similar structures) of numbers for features. For categorical features, define a convention (e.g., pre-encoded numerically or internal handling). ✅ (XInput and featureTypes parameter) Targets (y): Accept 1D arrays of numbers (for regression) or labels (numeric or strings for classification). ✅ (YInputClassification, YInputRegression) Output of predict: 1D array of predictions. ✅ Output of predict_proba: 2D array of class probabilities. ✅

## 5. Documentation and Tests

5.1. Documentation:
	* Complete API documentation (each class, method, parameter). ✅ (JSDoc comments are present but could be more comprehensive for a full API documentation)
	* Tutorials and usage examples for classification and regression. ✅
	* Explanation of key concepts and implemented algorithms. 🏗️

5.2. Tests:
	* Exhaustive unit test coverage for all modules. ✅
	* Integration tests to validate the complete flow (training, prediction, evaluation). ✅
	* Non-regression tests. ✅

# 6. Proposed Directory Structure (matches current structure)

```
am-decisiontree
├── .gitattributes
├── .gitignore
├── .nvmrc
├── biome.json
├── jsr.json
├── LICENSE
├── package.json
├── README.md
├── tsconfig.json
├── .github/
│   └── workflows/  // Optional (Implemented)
│       ├── ci.yml
│       └── ...
├── examples/
│   ├── iris_classification.ts // Not present, but similar examples exist
│   ├── simple_classifier.ts
│   └── regression_tree_example.ts // Renamed to simple_regresion.ts
└── src/
    ├── core/ // Merged into src/ directly
    │   ├── mod.ts
    │   ├── decision_tree.ts
    │   └── node.ts
    └── criteria/
        ├── mod.ts // Not present, exports are in src/mod.ts
        ├── gini_impurity.ts
        ├── entropy.ts
        └── mse_criterion.ts
````
