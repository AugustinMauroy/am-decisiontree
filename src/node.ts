/**
 * Represents the value of a node in the decision tree.
 * It can be a number, a string, or a distribution of class probabilities.
 */
export type NodeValue = number | string | Record<string | number, number>;

/**
 * Represents a node in the decision tree.
 * Each node can either be a leaf node or an internal node with children.
 */
export class Node<T extends NodeValue = NodeValue> {
	/** The index of the feature used for splitting at this node (if not a leaf). */
	public featureIndex?: number;

	/** The threshold value for numerical feature splits (if not a leaf). */
	public threshold?: number;

	/** The set of categories that go to the left child for categorical feature splits (if not a leaf). */
	public splitCategories?: Set<string | number>;

	/** The predicted value or class distribution (if this is a leaf node). */
	public value?: T;

	/** The impurity value at this node (e.g., Gini impurity, entropy, etc.). */
	public impurity?: number;

	/** The left child node (if this is not a leaf node). */
	public leftChild?: Node<T>;

	/** The right child node (if this is not a leaf node). */
	public rightChild?: Node<T>;

	/** The number of samples that reached this node. */
	public samples: number;

	/** Indicates whether this node is a leaf node. */
	public isLeaf: boolean;

	/** The value this node would have if it were a leaf (used for pruning). */
	public potentialLeafValue?: T;

	/**
	 * Creates a new instance of a decision tree node.
	 * @param options - The options to initialize the node.
	 * @param options.featureIndex - The index of the feature used for splitting (if not a leaf).
	 * @param options.threshold - The threshold value for numerical feature splits (if not a leaf).
	 * @param options.splitCategories - The set of categories that go to the left child for categorical feature splits (if not a leaf).
	 * @param options.value - The predicted value or class distribution (if this is a leaf node).
	 * @param options.impurity - The impurity value at this node.
	 * @param options.leftChild - The left child node (if this is not a leaf node).
	 * @param options.rightChild - The right child node (if this is not a leaf node).
	 * @param options.samples - The number of samples that reached this node.
	 * @param options.isLeaf - Indicates whether this node is a leaf node.
	 * @param options.potentialLeafValue - The value this node would have if it were a leaf.
	 */
	constructor(options: {
		featureIndex?: number;
		threshold?: number;
		splitCategories?: Set<string | number>;
		value?: T;
		impurity?: number;
		leftChild?: Node<T>;
		rightChild?: Node<T>;
		samples: number;
		isLeaf?: boolean;
		potentialLeafValue?: T;
	}) {
		this.featureIndex = options.featureIndex;
		this.threshold = options.threshold;
		this.splitCategories = options.splitCategories;
		this.value = options.value;
		this.impurity = options.impurity;
		this.leftChild = options.leftChild;
		this.rightChild = options.rightChild;
		this.samples = options.samples;
		this.isLeaf = options.isLeaf ?? false;
		this.potentialLeafValue = options.potentialLeafValue;
	}
}
