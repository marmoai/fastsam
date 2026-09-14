/**
 * Visual Asset OS - Core Runtime
 * 系统全局运行时单例，管理从 Asset 到 Workspace 的所有依赖。
 */

import { ProjectWorkspace, WorldState, CanvasState } from './WorkspaceRuntime';
import { AssetEntity, AssetVariant } from './AssetRuntime';
import { DecisionLog } from './DecisionRuntime';
import { ObjectRelation } from './GraphEngine';
import localforage from 'localforage';
import { AgentRuntime } from './AgentRuntime';
import type {
    AgentRuntimeHost,
    AgentCommitResult,
    AssetSearchCandidate,
    AgentCommandVerificationCheck,
    AgentLayerExtractionRequest,
    AgentLayerExtractionResult,
    AgentLayerExtractionUndoRequest,
    AgentLayerExtractionCommitRequest,
    AgentLayerExtractionReleaseRequest,
    AgentCapabilityRequest,
    AgentCapabilityResult,
    AgentCapabilityUndoRequest,
    AgentCapabilityCommitRequest,
    AgentCapabilityReleaseRequest
} from './AgentRuntime';
import { assetCatalog } from './AssetCatalog';

export class CoreRuntime implements AgentRuntimeHost {
    private static instance: CoreRuntime;
    
    private currentWorkspace: ProjectWorkspace | null = null;
    private allWorkspaces: Map<string, ProjectWorkspace> = new Map();
    private saveInFlight: Promise<void> | null = null;
    private saveRequested = false;
    private saveTimer: ReturnType<typeof setTimeout> | null = null;
    private agentRuntime: AgentRuntime | null = null;
    private agentAssetResolver: ((ref: { projectId: string; assetId: string }) => Promise<AssetEntity | null>) | null = null;
    private agentVerificationHooks: {
        verify?: (args: { workspace: ProjectWorkspace; assetIds: string[] }) => Promise<AgentCommandVerificationCheck[]>;
        repair?: (args: { workspace: ProjectWorkspace; assetIds: string[] }) => Promise<void>;
    } = {};
    private agentLayerExtractionHooks: {
        execute?: (args: AgentLayerExtractionRequest) => Promise<AgentLayerExtractionResult>;
        undo?: (args: AgentLayerExtractionUndoRequest) => Promise<void>;
        commit?: (args: AgentLayerExtractionCommitRequest) => Promise<void>;
        release?: (args: AgentLayerExtractionReleaseRequest) => Promise<void>;
    } = {};
    private agentCapabilityHooks: {
        execute?: (args: AgentCapabilityRequest) => Promise<AgentCapabilityResult>;
        undo?: (args: AgentCapabilityUndoRequest) => Promise<void>;
        commit?: (args: AgentCapabilityCommitRequest) => Promise<void>;
        release?: (args: AgentCapabilityReleaseRequest) => Promise<void>;
    } = {};
    private agentCommitPersistence: (() => Promise<void>) | null = null;

    private readonly MVR_DB_KEY = "mvr_core_workspace";

    private constructor() {}

    static getInstance(): CoreRuntime {
        if (!CoreRuntime.instance) {
            CoreRuntime.instance = new CoreRuntime();
        }
        return CoreRuntime.instance;
    }

    createWorkspace(projectId: string, name: string): ProjectWorkspace {
        const workspace = new ProjectWorkspace(projectId, name);
        this.allWorkspaces.set(projectId, workspace);
        this.currentWorkspace = workspace;
        return workspace;
    }

    loadWorkspace(projectId: string): ProjectWorkspace {
        const workspace = this.allWorkspaces.get(projectId);
        if (!workspace) throw new Error(`Workspace ${projectId} not found`);
        this.currentWorkspace = workspace;
        return workspace;
    }

    getCurrentWorkspace(): ProjectWorkspace | null {
        return this.currentWorkspace;
    }

    listWorkspaces(): ProjectWorkspace[] {
        return Array.from(this.allWorkspaces.values());
    }

    getAgentRuntime(): AgentRuntime {
        if (!this.agentRuntime) {
            this.agentRuntime = new AgentRuntime(this);
        }
        return this.agentRuntime;
    }

    setAgentAssetResolver(resolver: ((ref: { projectId: string; assetId: string }) => Promise<AssetEntity | null>) | null): void {
        this.agentAssetResolver = resolver;
    }

    setAgentVerificationHooks(hooks: typeof this.agentVerificationHooks = {}): void {
        this.agentVerificationHooks = hooks;
    }

    setAgentLayerExtractionHooks(hooks: typeof this.agentLayerExtractionHooks = {}): void {
        this.agentLayerExtractionHooks = hooks;
    }

    setAgentCapabilityHooks(hooks: typeof this.agentCapabilityHooks = {}): void {
        this.agentCapabilityHooks = hooks;
    }

    setAgentCommitPersistence(persist: (() => Promise<void>) | null): void {
        this.agentCommitPersistence = persist;
    }

    async extractAgentLayers(args: AgentLayerExtractionRequest): Promise<AgentLayerExtractionResult> {
        if (!this.agentLayerExtractionHooks.execute) {
            throw new Error('Magic Layers Agent adapter is unavailable.');
        }
        return this.agentLayerExtractionHooks.execute(args);
    }

    async undoAgentLayers(args: AgentLayerExtractionUndoRequest): Promise<void> {
        if (this.agentLayerExtractionHooks.undo) await this.agentLayerExtractionHooks.undo(args);
    }

    async commitAgentLayers(args: AgentLayerExtractionCommitRequest): Promise<void> {
        if (this.agentLayerExtractionHooks.commit) await this.agentLayerExtractionHooks.commit(args);
    }

    async releaseAgentLayers(args: AgentLayerExtractionReleaseRequest): Promise<void> {
        if (this.agentLayerExtractionHooks.release) await this.agentLayerExtractionHooks.release(args);
    }

    async executeAgentCapability(args: AgentCapabilityRequest): Promise<AgentCapabilityResult> {
        if (!this.agentCapabilityHooks.execute) throw new Error('Agent capability adapter is unavailable.');
        return this.agentCapabilityHooks.execute(args);
    }

    async undoAgentCapability(args: AgentCapabilityUndoRequest): Promise<void> {
        if (this.agentCapabilityHooks.undo) await this.agentCapabilityHooks.undo(args);
    }

    async commitAgentCapabilities(args: AgentCapabilityCommitRequest): Promise<void> {
        if (this.agentCapabilityHooks.commit) await this.agentCapabilityHooks.commit(args);
    }

    async releaseAgentCapabilities(args: AgentCapabilityReleaseRequest): Promise<void> {
        if (this.agentCapabilityHooks.release) await this.agentCapabilityHooks.release(args);
    }

    async verifyAgentWorkspace(args: { workspace: ProjectWorkspace; assetIds: string[] }): Promise<AgentCommandVerificationCheck[]> {
        return this.agentVerificationHooks.verify ? this.agentVerificationHooks.verify(args) : [];
    }

    async repairAgentWorkspace(args: { workspace: ProjectWorkspace; assetIds: string[] }): Promise<void> {
        if (this.agentVerificationHooks.repair) await this.agentVerificationHooks.repair(args);
    }

    async listAgentAssets(): Promise<AssetSearchCandidate[]> {
        const candidates = new Map<string, AssetSearchCandidate>();
        this.listWorkspaces().forEach(workspace => {
            workspace.currentState.assetRegistry.getAll().forEach(asset => {
                candidates.set(`${workspace.projectId}:${asset.uid}`, {
                    asset,
                    assetId: asset.uid,
                    projectId: workspace.projectId,
                    version: Number(asset.version) || 1,
                    name: String((asset as any).name || asset.layerName || asset.uid),
                    type: String(asset.type || 'unknown'),
                    tags: Array.isArray(asset.metadata?.tags) ? asset.metadata.tags.map(String) : []
                });
            });
        });

        const indexed = await assetCatalog.list();
        indexed.forEach(entry => {
            const key = `${entry.projectId}:${entry.assetId}`;
            if (!candidates.has(key)) {
                candidates.set(key, {
                    assetId: entry.assetId,
                    projectId: entry.projectId,
                    version: entry.version,
                    name: entry.name,
                    type: entry.type,
                    tags: entry.tags
                });
            }
        });
        return [...candidates.values()];
    }

    async resolveAgentAsset(ref: { projectId: string; assetId: string }): Promise<AssetEntity | null> {
        const loaded = this.allWorkspaces.get(ref.projectId)?.currentState.assetRegistry.get(ref.assetId);
        if (loaded) return loaded;
        return this.agentAssetResolver ? this.agentAssetResolver(ref) : null;
    }

    async commitWorkspace(workspace: ProjectWorkspace, jobId: string): Promise<AgentCommitResult> {
        if (this.currentWorkspace !== workspace) {
            throw new Error('Agent Jobs can only commit the current workspace.');
        }
        return workspace.commit(jobId, async () => {
            await this.saveCurrentWorkspace();
            await assetCatalog.upsertWorkspace(workspace);
            const persisted = await localforage.getItem<any>(this.MVR_DB_KEY);
            const persistedAssets = persisted?.currentState?.assets;
            const expectedIds = workspace.currentState.assetRegistry.getAll().map(asset => asset.uid).sort();
            const actualIds = Array.isArray(persistedAssets) ? persistedAssets.map(asset => asset.uid).sort() : [];
            if (JSON.stringify(expectedIds) !== JSON.stringify(actualIds)) {
                throw new Error('Persisted Workspace verification failed: asset IDs are inconsistent.');
            }
            if (this.agentCommitPersistence) await this.agentCommitPersistence();
        });
    }

    async saveCurrentWorkspace(options: { defer?: boolean } = {}): Promise<void> {
        if (!this.currentWorkspace) return;

        this.saveRequested = true;
        if (options.defer) {
            if (this.saveTimer) clearTimeout(this.saveTimer);
            this.saveTimer = setTimeout(() => {
                this.saveTimer = null;
                this.startWorkspaceSave();
            }, 350);
            return;
        }

        if (this.saveTimer) {
            clearTimeout(this.saveTimer);
            this.saveTimer = null;
        }
        await this.startWorkspaceSave();
    }

    private startWorkspaceSave(): Promise<void> {
        if (!this.saveInFlight) {
            this.saveInFlight = this.flushWorkspaceSaves().finally(() => {
                this.saveInFlight = null;
                // Do not lose a save requested after the last loop condition
                // was evaluated while this write was completing.
                if (this.saveRequested && !this.saveTimer) {
                    this.startWorkspaceSave();
                }
            });
        }
        return this.saveInFlight;
    }

    private async flushWorkspaceSaves(): Promise<void> {
        while (this.saveRequested) {
            this.saveRequested = false;
            const workspace = this.currentWorkspace;
            if (!workspace) return;

            try {
                const assets = workspace.currentState.assetRegistry.getAll().map(asset => {
                    const serializableAsset = { ...asset } as any;
                    delete serializableAsset.runtimeDisplayUrl;
                    return serializableAsset;
                });
                const data = {
                    projectId: workspace.projectId,
                    name: workspace.name,
                    currentState: {
                        stateId: workspace.currentState.stateId,
                        canvasState: workspace.currentState.canvasState,
                        assets,
                        nodes: Array.from(workspace.currentState.sceneGraph['nodes']),
                        edges: Array.from(workspace.currentState.sceneGraph['edges'].values())
                    },
                    decisionGraph: workspace.decisionGraph.getHistory()
                };

                await localforage.setItem(this.MVR_DB_KEY, data);
                console.log("[MVR] Workspace saved to IndexedDB.");
                void assetCatalog.upsertWorkspace(workspace);
            } catch (e) {
                console.error("[MVR] Failed to save workspace:", e);
            }
        }
    }

    async restoreWorkspace(): Promise<ProjectWorkspace | null> {
        try {
            const data = await localforage.getItem<any>(this.MVR_DB_KEY);

            if (!data) return null;

            const workspace = this.createWorkspace(data.projectId, data.name);
            workspace.currentState.stateId = data.currentState.stateId;
            workspace.currentState.canvasState = data.currentState.canvasState;
            
            // Restore assets
            if (data.currentState.assets) {
                for (const asset of data.currentState.assets) {
                    // Global Reference Check: 同步全局资产池的最新版本
                    const globalRef = await import('./AssetRuntime.js').then(m => m.GlobalAssetCatalog.getLatestRef(asset.uid));
                    if (globalRef && globalRef.version > (asset.version || 0)) {
                        console.log(`[MVR] Syncing ${asset.uid} to latest global version ${globalRef.version}`);
                        asset.sourceImage = globalRef.url;
                        asset.version = globalRef.version;
                    }
                    workspace.currentState.assetRegistry.register(asset);
                    workspace.currentState.sceneGraph.addNode(asset.uid);
                }
            }

            // Restore Graph
            if (data.currentState.nodes) {
                for (const node of data.currentState.nodes) {
                    workspace.currentState.sceneGraph.addNode(node);
                }
            }
            if (data.currentState.edges) {
                for (const edge of data.currentState.edges) {
                    const sourceExists = workspace.currentState.assetRegistry.get(edge.sourceId) || data.currentState.nodes?.includes(edge.sourceId);
                    const targetExists = workspace.currentState.assetRegistry.get(edge.targetId) || data.currentState.nodes?.includes(edge.targetId);
                    if (sourceExists && targetExists) {
                        workspace.currentState.sceneGraph.addEdge(edge);
                    }
                }
            }

            // Restore Decisions
            if (data.decisionGraph) {
                for (const log of data.decisionGraph) {
                    workspace.decisionGraph.addLog(log);
                }
            }

            // Write initial snapshot to history timeline
            workspace.snapshot();

            console.log("[MVR] Workspace restored from IndexedDB.");
            return workspace;
        } catch (e) {
            console.error("[MVR] Failed to restore workspace:", e);
            return null;
        }
    }
}

export const runtime = CoreRuntime.getInstance();
