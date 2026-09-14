/**
 * Visual Asset OS - Minimal Viable Workspace Runtime (Stage 1)
 * 核心目标：持续编辑，告别阅后即焚，不涉及复杂的Git-like快照和变体目前
 */

import { AssetRegistry, GlobalAssetCatalog } from './AssetRuntime';
import { SceneGraph } from './GraphEngine';
import { DecisionGraph } from './DecisionRuntime';
import { Dispatcher } from './Dispatcher';
import { resultFeedbackRuntime } from './ResultFeedbackRuntime';
import { interactionAttributionRuntime } from './InteractionAttributionRuntime';

export interface CanvasState {
    width: number;
    height: number;
    backgroundColor: string; // 基础背景
}

export interface WorkspaceCommitResult {
    commitId: string;
    jobId: string;
    projectId: string;
    historyIndex: number;
    committedAt: number;
}

export interface WorkspaceRoundTripCheck {
    id: string;
    passed: boolean;
    message: string;
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
        // runtimeDisplayUrl is a page-local object URL and cannot survive a
        // reload. Only the durable sourceImage belongs in snapshots.
        const assets = this.assetRegistry.getAll().map(asset => {
            const serializableAsset = { ...asset } as any;
            delete serializableAsset.runtimeDisplayUrl;
            return serializableAsset;
        });
        return {
            stateId: this.stateId,
            assets: JSON.parse(JSON.stringify(assets)),
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
    // Full snapshots include durable image payloads. Keep the undo history
    // bounded so repeated extraction cannot retain every prior PNG forever.
    private static readonly MAX_HISTORY_SNAPSHOTS = 12;

    public projectId: string;
    public name: string;
    
    public currentState: WorldState;
    public decisionGraph: DecisionGraph;
    public dispatcher: Dispatcher;

    public historyIndex: number = -1;
    public history: any[] = []; // Immutable Snapshots array
    private commits = new Map<string, WorkspaceCommitResult>();
    private commitInFlight = new Map<string, Promise<WorkspaceCommitResult>>();

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
        if (this.history.length > ProjectWorkspace.MAX_HISTORY_SNAPSHOTS) {
            this.history.splice(0, this.history.length - ProjectWorkspace.MAX_HISTORY_SNAPSHOTS);
        }
        this.historyIndex = this.history.length - 1;
        console.log(`[WorkspaceRuntime] Snapshot taken. History size: ${this.history.length}, Index: ${this.historyIndex}`);
    }

    undo() {
        if (this.historyIndex > 0) {
            const currentSnapshot = this.history[this.historyIndex];
            this.historyIndex--;
            this.currentState.hydrate(this.history[this.historyIndex]);
            try {
                const revertedAssetCount = Array.isArray(currentSnapshot?.assets) ? currentSnapshot.assets.length : 0;
                const source = interactionAttributionRuntime.resolveSource({
                    fallbackSourceType: 'manual'
                });
                resultFeedbackRuntime.recordEvent({
                    type: 'result_reverted',
                    sessionId: this.projectId || undefined,
                    projectId: this.projectId || undefined,
                    sourceType: source.sourceType,
                    sourceId: source.sourceId,
                    context: {
                        workflowStage: 'workspace_undo'
                    },
                    metadata: {
                        historyIndexAfterUndo: this.historyIndex,
                        revertedAssetCount,
                        attributionMetadata: source.metadata || null
                    }
                });
            } catch (error) {
                console.error('[ResultFeedback] Failed to record result_reverted:', error);
            }
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

    /**
     * The only snapshot boundary used by Agent Jobs. Commands mutate through
     * Dispatcher with snapshotting disabled; this method creates one durable
     * history point after verification and publishes the asset versions once.
     */
    async commit(jobId: string, persist: () => Promise<void> = async () => {}): Promise<WorkspaceCommitResult> {
        const existing = this.commits.get(jobId);
        if (existing) return existing;

        const inFlight = this.commitInFlight.get(jobId);
        if (inFlight) return inFlight;

        const promise = (async () => {
            await this.save();
            const historyBeforeCommit = this.history.slice();
            const historyIndexBeforeCommit = this.historyIndex;
            this.snapshot();
            try {
                await persist();
            } catch (error) {
                // Keep the established Runtime -> Workspace -> History ->
                // persistence ordering, but never leave a failed Commit's
                // snapshot behind as if it were durable.
                this.history = historyBeforeCommit;
                this.historyIndex = historyIndexBeforeCommit;
                throw error;
            }
            const result: WorkspaceCommitResult = {
                commitId: `commit_${jobId}`,
                jobId,
                projectId: this.projectId,
                historyIndex: this.historyIndex,
                committedAt: Date.now()
            };
            this.commits.set(jobId, result);
            return result;
        })();

        this.commitInFlight.set(jobId, promise);
        try {
            return await promise;
        } finally {
            this.commitInFlight.delete(jobId);
        }
    }

    verifyRoundTrip(assetIds: string[]): { passed: boolean; checks: WorkspaceRoundTripCheck[] } {
        const snapshot = this.currentState.serialize();
        const restored = new WorldState(`verify_${Date.now()}`);
        restored.hydrate(snapshot);
        const checks: WorkspaceRoundTripCheck[] = [];
        const assetsRestored = assetIds.every(uid => Boolean(restored.assetRegistry.get(uid)));
        checks.push({
            id: 'workspace_round_trip_assets',
            passed: assetsRestored,
            message: assetsRestored ? 'Target assets survive Workspace serialization.' : 'Target assets were lost during Workspace serialization.'
        });
        const transformsRestored = assetIds.every(uid => {
            const before = this.currentState.assetRegistry.get(uid)?.transform;
            const after = restored.assetRegistry.get(uid)?.transform;
            return JSON.stringify(before) === JSON.stringify(after);
        });
        checks.push({
            id: 'workspace_round_trip_transforms',
            passed: transformsRestored,
            message: transformsRestored ? 'Target transforms survive Workspace serialization.' : 'Target transforms changed during Workspace serialization.'
        });
        const graphRestored = assetIds.every(uid => restored.sceneGraph.getNodes().includes(uid));
        checks.push({
            id: 'workspace_round_trip_graph',
            passed: graphRestored,
            message: graphRestored ? 'Target scene nodes survive Workspace serialization.' : 'Target scene nodes were lost during Workspace serialization.'
        });
        return { passed: checks.every(check => check.passed), checks };
    }

    async load() {
        return;
    }
}
