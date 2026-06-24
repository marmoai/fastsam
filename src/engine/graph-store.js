import { NODE_REGISTRY } from './node-registry.js';

/**
 * Graph Store
 * Manages the state of the Node Graph (nodes, edges, and their execution status).
 */

export class GraphStore {
    constructor() {
        this.nodes = new Map(); // id -> node object
        this.edges = []; // array of { from: { nodeId, outputId }, to: { nodeId, inputId } }
        this.results = new Map(); // nodeId -> { outputId: value }
    }

    addNode(node) {
        this.nodes.set(node.id, {
            ...node,
            status: 'idle', // 'idle', 'running', 'completed', 'failed'
            error: null
        });
    }

    removeNode(nodeId) {
        this.nodes.delete(nodeId);
        this.edges = this.edges.filter(e => e.from.nodeId !== nodeId && e.to.nodeId !== nodeId);
        this.results.delete(nodeId);
    }

    addEdge(from, to) {
        // from: { nodeId, outputId }, to: { nodeId, inputId }
        this.edges.push({ from, to });
    }

    removeEdge(fromNodeId, fromOutputId, toNodeId, toInputId) {
        this.edges = this.edges.filter(e => 
            !(e.from.nodeId === fromNodeId && e.from.outputId === fromOutputId && 
              e.to.nodeId === toNodeId && e.to.inputId === toInputId)
        );
    }

    setNodeStatus(nodeId, status, error = null) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.status = status;
            node.error = error;
        }
    }

    setNodeResult(nodeId, result) {
        this.results.set(nodeId, result);
    }

    getNodeResult(nodeId) {
        return this.results.get(nodeId);
    }

    getInputsForNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return {};

        // Deep copy array inputs to avoid mutating the original node.inputs
        const inputs = {};
        for (const [key, value] of Object.entries(node.inputs || {})) {
            if (Array.isArray(value)) {
                inputs[key] = [...value];
            } else {
                inputs[key] = value;
            }
        }

        const nodeDef = NODE_REGISTRY[node.type];

        // Resolve dynamic inputs from edges
        const incomingEdges = this.edges.filter(e => e.to.nodeId === nodeId);
        for (const edge of incomingEdges) {
            const sourceResult = this.results.get(edge.from.nodeId);
            if (sourceResult && sourceResult[edge.from.outputId] !== undefined) {
                const inputDef = nodeDef?.inputs.find(i => i.id === edge.to.inputId);
                if (inputDef && inputDef.type === 'array') {
                    if (!inputs[edge.to.inputId]) inputs[edge.to.inputId] = [];
                    if (!Array.isArray(inputs[edge.to.inputId])) inputs[edge.to.inputId] = [inputs[edge.to.inputId]];
                    inputs[edge.to.inputId].push(sourceResult[edge.from.outputId]);
                } else {
                    inputs[edge.to.inputId] = sourceResult[edge.from.outputId];
                }
            }
        }

        return inputs;
    }

    // Serialize the graph for saving
    toJSON() {
        return {
            nodes: Array.from(this.nodes.values()),
            edges: this.edges
        };
    }

    // Load a graph from JSON
    fromJSON(data) {
        this.nodes.clear();
        this.edges = data.edges || [];
        this.results.clear();
        if (data.nodes) {
            data.nodes.forEach(n => this.addNode(n));
        }
    }
}

export const graphStore = new GraphStore();
