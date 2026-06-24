export interface EvolutionNode {
    id: string;
    assetUid: string;
    stateProps: any; // The state that differentiates this node (e.g., fusionProperties)
    intent: string;  // The semantic label representing this branch
    timestamp: number;
}

class EvolutionSystem {
    // Maps an asset ID to its evolution branches
    private branches: Map<string, EvolutionNode[]> = new Map();

    recordVariant(assetUid: string, stateProps: any, intent: string) {
        if (!this.branches.has(assetUid)) {
            this.branches.set(assetUid, []);
            // Initialize with an origin node if it doesn't exist
            this.branches.get(assetUid)!.push({
                id: 'origin-' + Date.now(),
                assetUid,
                stateProps: {}, // base implicit state
                intent: '原始资产 (Origin)',
                timestamp: Date.now() - 1000
            });
        }
        
        // Push the new genetic variant into the evolution tree
        this.branches.get(assetUid)!.push({
            id: 'variant-' + Date.now() + '-' + Math.floor(Math.random()*1000),
            assetUid,
            stateProps: JSON.parse(JSON.stringify(stateProps)),
            intent,
            timestamp: Date.now()
        });
        
        console.log(`[EvolutionSystem] Recorded new genetic variant for ${assetUid}: ${intent}`);
    }

    getVariants(assetUid: string): EvolutionNode[] {
        return this.branches.get(assetUid) || [];
    }
}

export const evolutionEngine = new EvolutionSystem();
