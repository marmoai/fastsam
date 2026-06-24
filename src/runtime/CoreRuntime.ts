/**
 * Visual Asset OS - Core Runtime
 * 系统全局运行时单例，管理从 Asset 到 Workspace 的所有依赖。
 */

import { ProjectWorkspace, WorldState, CanvasState } from './WorkspaceRuntime';
import { AssetEntity, AssetVariant } from './AssetRuntime';
import { DecisionLog } from './DecisionRuntime';
import { ObjectRelation } from './GraphEngine';
import localforage from 'localforage';

export class CoreRuntime {
    private static instance: CoreRuntime;
    
    private currentWorkspace: ProjectWorkspace | null = null;
    private allWorkspaces: Map<string, ProjectWorkspace> = new Map();

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

    async saveCurrentWorkspace(): Promise<void> {
        if (!this.currentWorkspace) return;
        
        try {
            const data = {
                projectId: this.currentWorkspace.projectId,
                name: this.currentWorkspace.name,
                currentState: {
                    stateId: this.currentWorkspace.currentState.stateId,
                    canvasState: this.currentWorkspace.currentState.canvasState,
                    assets: this.currentWorkspace.currentState.assetRegistry.getAll(),
                    nodes: Array.from(this.currentWorkspace.currentState.sceneGraph['nodes']),
                    edges: Array.from(this.currentWorkspace.currentState.sceneGraph['edges'].values())
                },
                decisionGraph: this.currentWorkspace.decisionGraph.getHistory()
            };
            
            // 1. 同步保存到本地 IndexedDB
            await localforage.setItem(this.MVR_DB_KEY, data);
            console.log("[MVR] Workspace saved to IndexedDB.");
        } catch (e) {
            console.error("[MVR] Failed to save workspace:", e);
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
