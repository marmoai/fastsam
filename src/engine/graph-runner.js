import { graphStore } from './graph-store.js';
import { executeNode } from './node-executor.js';

/**
 * Graph Runner
 * Responsible for executing the graph in the correct order.
 */

export class GraphRunner {
    constructor(store) {
        this.store = store || graphStore;
        this.running = false;
    }

    async runGraph() {
        if (this.running) return;
        this.running = true;

        console.log("[GraphRunner] Starting graph execution...");

        try {
            let changed = true;
            while (changed) {
                changed = false;
                const nodesToRun = this.getExecutableNodes();

                if (nodesToRun.length === 0) break;

                // Execute nodes in parallel (or sequentially if needed)
                const promises = nodesToRun.map(nodeId => this.runNode(nodeId));
                await Promise.all(promises);
                changed = true;
            }

            // Check if all nodes completed
            const uncompletedNodes = Array.from(this.store.nodes.values()).filter(n => n.status !== 'completed');
            if (uncompletedNodes.length > 0) {
                const failedNodes = uncompletedNodes.filter(n => n.status === 'failed');
                if (failedNodes.length > 0) {
                    throw new Error(`Graph execution failed. Nodes failed: ${failedNodes.map(n => n.id).join(', ')}`);
                } else {
                    throw new Error(`Graph execution deadlocked. Uncompleted nodes: ${uncompletedNodes.map(n => n.id).join(', ')}`);
                }
            }
        } finally {
            this.running = false;
            console.log("[GraphRunner] Graph execution finished.");
        }
    }

    getExecutableNodes() {
        const executable = [];
        for (const [nodeId, node] of this.store.nodes) {
            if (node.status !== 'idle') continue;

            // Check if all inputs from edges are available
            const incomingEdges = this.store.edges.filter(e => e.to.nodeId === nodeId);
            const allDependenciesMet = incomingEdges.every(edge => {
                const sourceNode = this.store.nodes.get(edge.from.nodeId);
                return sourceNode && sourceNode.status === 'completed';
            });

            if (allDependenciesMet) {
                executable.push(nodeId);
            }
        }
        return executable;
    }

    async runNode(nodeId) {
        const node = this.store.nodes.get(nodeId);
        if (!node) return;

        this.store.setNodeStatus(nodeId, 'running');

        const MAX_RETRIES = 3;
        let attempt = 0;
        let lastError = null;

        while (attempt < MAX_RETRIES) {
            try {
                attempt++;
                if (attempt > 1) {
                    console.log(`[GraphRunner] Node ${nodeId} (${node.type}) retry attempt ${attempt}/${MAX_RETRIES}...`);
                }
                
                const inputs = this.store.getInputsForNode(nodeId);
                const outputs = await executeNode(node, inputs);
                
                this.store.setNodeResult(nodeId, outputs);
                this.store.setNodeStatus(nodeId, 'completed');
                
                console.log(`[GraphRunner] Node ${nodeId} (${node.type}) completed successfully.`);
                return; // Success, exit the retry loop
            } catch (error) {
                lastError = error;
                console.warn(`[GraphRunner] Node ${nodeId} (${node.type}) failed on attempt ${attempt}:`, error.message);
                
                if (attempt < MAX_RETRIES) {
                    console.log(`[GraphRunner] Waiting 2 seconds before retrying node ${nodeId}...`);
                    // Wait 2 seconds before retrying to avoid hitting rate limits or immediate repeated failures
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }

        // If we reach here, all retries failed
        console.error(`[GraphRunner] Node ${nodeId} (${node.type}) ultimately failed after ${MAX_RETRIES} attempts:`, lastError);
        this.store.setNodeStatus(nodeId, 'failed', lastError.message);
        // Stop the entire graph execution if a node fails completely
        throw lastError; 
    }

    // Reset all nodes to 'idle' to re-run the graph
    resetGraph() {
        for (const [nodeId] of this.store.nodes) {
            this.store.setNodeStatus(nodeId, 'idle');
            this.store.results.delete(nodeId);
        }
    }
}

export const graphRunner = new GraphRunner();
