export type NodeValue = number | string | Record<string | number, number>;

export class Node<T extends NodeValue = NodeValue> {
	public featureIndex?: number;
	public threshold?: number;
	public value?: T; // Predicted value/distribution if leaf
	public impurity?: number;
	public leftChild?: Node<T>;
	public rightChild?: Node<T>;
	public samples: number;
	public isLeaf: boolean;

	constructor(options: {
		featureIndex?: number;
		threshold?: number;
		value?: T;
		impurity?: number;
		leftChild?: Node<T>;
		rightChild?: Node<T>;
		samples: number;
		isLeaf?: boolean;
	}) {
		this.featureIndex = options.featureIndex;
		this.threshold = options.threshold;
		this.value = options.value;
		this.impurity = options.impurity;
		this.leftChild = options.leftChild;
		this.rightChild = options.rightChild;
		this.samples = options.samples;
		this.isLeaf = options.isLeaf ?? false;
	}
}
