import { graphStore } from './graph-store.js';
import { graphRunner } from './graph-runner.js';
import { NODE_REGISTRY } from './node-registry.js';

/**
 * Node Engine Demo
 * Demonstrates how to build and run a simple graph using the new engine.
 */

export async function runDemoGraph(prompt, baseImageBase64) {
    console.log("[Demo] Building graph for prompt:", prompt);

    // 1. Create Nodes
    const genNode = {
        id: "node_gen_1",
        type: "generate-image",
        inputs: { prompt: prompt, aspectRatio: "1:1" }
    };

    const relightNode = {
        id: "node_relight_1",
        type: "relight",
        inputs: { prompt: "warm sunset lighting from the left" }
    };

    const analyzeNode = {
        id: "node_analyze_1",
        type: "analyze",
        inputs: { agentType: "describe" }
    };

    // 2. Add to Store
    graphStore.nodes.clear();
    graphStore.edges = [];
    graphStore.results.clear();

    graphStore.addNode(genNode);
    graphStore.addNode(relightNode);
    graphStore.addNode(analyzeNode);

    // 3. Create Edges (Connections)
    // Connect genNode output 'image' to relightNode input 'image'
    graphStore.addEdge(
        { nodeId: "node_gen_1", outputId: "image" },
        { nodeId: "node_relight_1", inputId: "image" }
    );

    // Connect relightNode output 'image' to analyzeNode input 'image'
    graphStore.addEdge(
        { nodeId: "node_relight_1", outputId: "image" },
        { nodeId: "node_analyze_1", inputId: "image" }
    );

    // 4. Run the Graph
    try {
        await graphRunner.runGraph();
        
        const finalResult = graphStore.getNodeResult("node_analyze_1");
        const finalImage = graphStore.getNodeResult("node_relight_1")?.image;

        console.log("[Demo] Graph execution complete!");
        console.log("[Demo] Final Analysis:", finalResult);
        
        return {
            analysis: finalResult,
            image: finalImage
        };
    } catch (error) {
        console.error("[Demo] Graph execution failed:", error);
        throw error;
    }
}

// Expose to window for testing from console
window.runDemoGraph = runDemoGraph;
