/**
 * Visual Asset OS - Minimal Viable Graph Engine (Stage 1)
 * 核心目标：满足最基本的画布对象关系，不做复杂的物理/光影拓扑
 */

export type BasicRelationType = 'parent_of' | 'above' | 'below' | 'mask_dependency';

export interface ObjectRelation {
    id: string;
    sourceId: string;
    targetId: string;
    relationType: BasicRelationType;
    properties: Record<string, any>;
}

import { AssetRegistry, AssetTransform } from '../runtime/AssetRuntime';

export class SceneGraph {
    private nodes: Set<string> = new Set(); // Asset UIDs
    private edges: Map<string, ObjectRelation> = new Map(); // Edge ID -> Relation
    private assetRegistry: AssetRegistry | null = null;
    
    setAssetRegistry(registry: AssetRegistry) {
        this.assetRegistry = registry;
    }

    updateNode(assetUid: string, transform: AssetTransform) {
        if (!this.assetRegistry) {
            console.error('SceneGraph: AssetRegistry not set!');
            return;
        }
        const asset = this.assetRegistry.get(assetUid);
        if (asset) {
            asset.transform = transform;
        }
    }

    addNode(assetUid: string) {
        this.nodes.add(assetUid);
    }

    getNodes(): string[] {
        return Array.from(this.nodes);
    }

    removeNode(assetUid: string) {
        this.nodes.delete(assetUid);
        for (const [edgeId, edge] of this.edges.entries()) {
            if (edge.sourceId === assetUid || edge.targetId === assetUid) {
                this.edges.delete(edgeId);
            }
        }
    }

    addEdge(relation: ObjectRelation) {
        if (!this.nodes.has(relation.sourceId) || !this.nodes.has(relation.targetId)) {
            throw new Error('Both source and target must exist in the SceneGraph');
        }
        this.edges.set(relation.id, relation);
    }

    getEdge(edgeId: string): ObjectRelation | undefined {
        return this.edges.get(edgeId);
    }

    removeEdge(edgeId: string) {
        this.edges.delete(edgeId);
    }

    getEdgesForNode(assetUid: string): ObjectRelation[] {
        const result: ObjectRelation[] = [];
        for (const edge of this.edges.values()) {
            if (edge.sourceId === assetUid || edge.targetId === assetUid) {
                result.push(edge);
            }
        }
        return result;
    }

    getAllEdges(): ObjectRelation[] {
        return Array.from(this.edges.values());
    }

    clear(): void {
        this.nodes.clear();
        this.edges.clear();
    }
}
