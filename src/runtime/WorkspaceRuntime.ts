/**
 * Visual Asset OS - Minimal Viable Workspace Runtime (Stage 1)
 * 核心目标：持续编辑，告别阅后即焚，不涉及复杂的Git-like快照和变体目前
 */

import { AssetRegistry, GlobalAssetCatalog } from './AssetRuntime';
import { SceneGraph } from './GraphEngine';
import { DecisionGraph } from './DecisionRuntime';
import { Dispatcher } from './Dispatcher';

export interface CanvasState {
    width: number;
    height: number;
    backgroundColor: string; // 基础背景
}

export class WorldState {
    public stateId: string;
    public assetRegistry: AssetRegistry;
    public sceneGraph: SceneGraph; // 早期版本仅仅处理层级关系
    public canvasState: CanvasState;
    private listeners: Array<() => void> = [];

    constructor(stateId: string) {
        this.stateId = stateId;
        this.assetRegistry = new AssetRegistry();
        this.sceneGraph = new SceneGraph();
        this.sceneGraph.setAssetRegistry(this.assetRegistry);
        this.canvasState = {
            width: 1024,
            height: 1024,
            backgroundColor: '#ffffff'
        };
    }

    subscribe(listener: () => void) {
        if (!this.listeners.includes(listener)) {
            this.listeners.push(listener);
        }
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    notify() {
        this.listeners.forEach(l => l());
    }

    serialize() {
        return {
            stateId: this.stateId,
            assets: JSON.parse(JSON.stringify(this.assetRegistry.getAll())),
            sceneNodes: this.sceneGraph.getNodes(),
            sceneEdges: this.sceneGraph.getAllEdges(),
            canvasState: { ...this.canvasState }
        };
    }

    hydrate(snap: any) {
        this.stateId = snap.stateId;
        this.canvasState = { ...snap.canvasState };
        this.assetRegistry.clear();
        this.sceneGraph.clear(); // Need to implement clear() in SceneGraph

        if (snap.assets) {
            for (const asset of snap.assets) {
                this.assetRegistry.register(asset);
                this.sceneGraph.addNode(asset.uid);
            }
        }
        if (snap.sceneNodes) {
            for (const node of snap.sceneNodes) {
                this.sceneGraph.addNode(node);
            }
        }
        if (snap.sceneEdges) {
            for (const edge of snap.sceneEdges) {
                const sourceExists = this.assetRegistry.get(edge.sourceId) || snap.sceneNodes?.includes(edge.sourceId);
                const targetExists = this.assetRegistry.get(edge.targetId) || snap.sceneNodes?.includes(edge.targetId);
                if (sourceExists && targetExists) {
                    this.sceneGraph.addEdge(edge);
                }
            }
        }
        this.notify();
    }
}

export class ProjectWorkspace {
    public projectId: string;
    public name: string;
    
    public currentState: WorldState;
    public decisionGraph: DecisionGraph;
    public dispatcher: Dispatcher;

    public historyIndex: number = -1;
    public history: any[] = []; // Immutable Snapshots array

    constructor(projectId: string, name: string = 'Untitled Asset Universe') {
        this.projectId = projectId;
        this.name = name;
        this.currentState = new WorldState(`state_${Date.now()}`);
        this.decisionGraph = new DecisionGraph();
        this.dispatcher = new Dispatcher(this);
    }

    snapshot() {
        const snap = this.currentState.serialize();
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        this.history.push(snap);
        this.historyIndex++;
        console.log(`[WorkspaceRuntime] Snapshot taken. History size: ${this.history.length}, Index: ${this.historyIndex}`);
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.currentState.hydrate(this.history[this.historyIndex]);
            console.log(`[WorkspaceRuntime] Undo. Index: ${this.historyIndex}`);
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.currentState.hydrate(this.history[this.historyIndex]);
            console.log(`[WorkspaceRuntime] Redo. Index: ${this.historyIndex}`);
        }
    }

    async save(fullData?: any) {
        // Push assets to Global Catalog
        const assets = this.currentState.assetRegistry.getAll();
        for (const asset of assets) {
            await GlobalAssetCatalog.publish(asset);
        }
        console.log(`[WorkspaceRuntime] Project ${this.projectId} saved to local runtime cache and GlobalAssetCatalog.`);
    }

    async load() {
        return;
    }
}
