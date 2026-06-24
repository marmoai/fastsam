/**
 * Visual Asset OS - Runtime State Dispatcher
 * 核心任务：接收交互Intent，统一分发至State更新，并携带中间件逻辑与Redux级快照
 */
import type { ProjectWorkspace } from './WorkspaceRuntime';
import { FusionProperties, AssetTransform } from './AssetRuntime';
import { memoryLayer } from './CreativeMemoryLayer';
import { IntentEngine } from './IntentEngine';
import { evolutionEngine } from './EvolutionEngine';

export type ActionType = 'UPDATE_FUSION' | 'UPDATE_TRANSFORM' | 'ADD_ASSET' | 'ADD_ASSET_WITH_RELATIONS' | 'REMOVE_ASSET' | 'REMOVE_ASSETS' | 'BATCH_UPDATE_TRANSFORMS' | 'CASCADE_TRANSFORM' | 'SYNC_ASSET_VERSION' | 'UPDATE_ASSET_METADATA' | 'CO_CREATE_ISOLATED' | 'CO_CREATE_SYNC' | 'CLEAR_WORKSPACE' | 'ADD_RELATION' | 'ADD_RELATIONS' | 'REMOVE_RELATION' | 'REMOVE_RELATIONS';

export interface Action {
    type: ActionType;
    payload: any;
    meta?: { silent?: boolean, intent?: string }; // If silent=true, we don't snapshot
    
    // Creative Memory Fields
    intent?: string; // e.g. "make_premium"
    reason?: string; 
    context?: string; 
}

class GraphRunnerRules {
    static evaluate(action: Action, workspace: ProjectWorkspace, deltaX: number = 0, deltaY: number = 0): Action[] {
        const { sceneGraph, assetRegistry } = workspace.currentState;
        const subActions: Action[] = [];

        // 拦截特定事件过一遍规则引擎，顺便 Dispatch 附带生成的子 Action，建立联动骨架
        if (action.type === 'UPDATE_TRANSFORM' || action.type === 'CASCADE_TRANSFORM') {
            const { uid } = action.payload;
            
            if (deltaX !== 0 || deltaY !== 0) {
                const edges = sceneGraph.getEdgesForNode(uid);
                edges.forEach(edge => {
                    // 若发生位移的是父节点，所有的子节点都跟着联动位移
                    if (edge.sourceId === uid && edge.relationType === 'parent_of') {
                        const childAsset = assetRegistry.get(edge.targetId);
                        if (childAsset && childAsset.transform) {
                            subActions.push({
                                type: 'CASCADE_TRANSFORM',
                                payload: {
                                    uid: edge.targetId,
                                    transform: {
                                        x: (childAsset.transform.x || 0) + deltaX,
                                        y: (childAsset.transform.y || 0) + deltaY
                                    }
                                },
                                meta: { silent: true }
                            });
                        }
                    }
                });
            }
        }

        return subActions;
    }
}

export class Dispatcher {
    private workspace: ProjectWorkspace;

    constructor(workspace: ProjectWorkspace) {
        this.workspace = workspace;
    }

    dispatch(action: Action) {
        console.log('[Dispatcher] Processing:', action);
        
        // --- Middleware 0: Intent & Semantic Inference ---
        // Infer intent before recording to memory
        const inferredIntent = IntentEngine.infer(action);
        if (inferredIntent) {
            action.intent = inferredIntent;
            if (!action.meta) action.meta = {};
            action.meta.intent = inferredIntent;
        }

        // --- Extract Before State for Delta ---
        let beforeState: any = null;
        if (action.type === 'ADD_ASSET_WITH_RELATIONS') {
            beforeState = {
                asset: null,
                relations: []
            };
        } else if (action.type === 'ADD_RELATIONS') {
            beforeState = {
                relations: []
            };
        } else if (action.type === 'REMOVE_RELATION') {
            const edgeId = action.payload?.id;
            const edgeBefore = edgeId ? this.workspace.currentState.sceneGraph.getEdge(edgeId) : null;
            beforeState = {
                relation: edgeBefore ? JSON.parse(JSON.stringify(edgeBefore)) : null
            };
        } else if (action.type === 'REMOVE_RELATIONS') {
            const relationIds = Array.isArray(action.payload?.ids) ? action.payload.ids : [];
            beforeState = {
                relations: relationIds
                    .map((id: string) => this.workspace.currentState.sceneGraph.getEdge(id))
                    .filter(Boolean)
                    .map((edge: any) => JSON.parse(JSON.stringify(edge)))
            };
        } else if (action.type === 'ADD_RELATION') {
            beforeState = {
                relation: null
            };
        } else if (action.payload?.uid) {
            const assetBefore = this.workspace.currentState.assetRegistry.get(action.payload.uid);
            if (assetBefore) beforeState = JSON.parse(JSON.stringify(assetBefore));
        } else if (action.payload?.uids) {
            const list: any[] = [];
            action.payload.uids.forEach((id: string) => {
                const asset = this.workspace.currentState.assetRegistry.get(id);
                if (asset) list.push(JSON.parse(JSON.stringify(asset)));
            });
            beforeState = list;
        } else if (action.payload?.transforms) {
            const list: any[] = [];
            action.payload.transforms.forEach((item: any) => {
                const asset = this.workspace.currentState.assetRegistry.get(item.uid);
                if (asset) list.push(JSON.parse(JSON.stringify(asset)));
            });
            beforeState = list;
        } else if (action.type === 'CLEAR_WORKSPACE') {
            beforeState = JSON.parse(JSON.stringify(this.workspace.currentState.assetRegistry.getAll()));
        }

        // --- Middleware 2: Snapshot Check ---
        if (!action.meta?.silent) {
            this.workspace.snapshot();
        }

        const queue: Action[] = [action];

        while (queue.length > 0) {
            const currentAction = queue.shift()!;

            if (currentAction.type === 'BATCH_UPDATE_TRANSFORMS') {
                const { transforms } = currentAction.payload;
                if (Array.isArray(transforms)) {
                    transforms.forEach(({ uid, transform }) => {
                        let dX = 0;
                        let dY = 0;
                        const originalAsset = this.workspace.currentState.assetRegistry.get(uid);
                        if (originalAsset && originalAsset.transform) {
                            dX = transform.x !== undefined ? (transform.x - (originalAsset.transform.x || 0)) : 0;
                            dY = transform.y !== undefined ? (transform.y - (originalAsset.transform.y || 0)) : 0;
                        }

                        const individualAction: Action = {
                            type: 'UPDATE_TRANSFORM',
                            payload: { uid, transform },
                            meta: { silent: true }
                        };
                        this._applyAction(individualAction);

                        const cascades = GraphRunnerRules.evaluate(individualAction, this.workspace, dX, dY);
                        queue.push(...cascades);
                    });
                }
            } else {
                let dX = 0;
                let dY = 0;
                if (currentAction.type === 'UPDATE_TRANSFORM' || currentAction.type === 'CASCADE_TRANSFORM') {
                    const { uid, transform } = currentAction.payload;
                    const originalAsset = this.workspace.currentState.assetRegistry.get(uid);
                    if (originalAsset && originalAsset.transform) {
                        dX = transform.x !== undefined ? (transform.x - (originalAsset.transform.x || 0)) : 0;
                        dY = transform.y !== undefined ? (transform.y - (originalAsset.transform.y || 0)) : 0;
                    }
                }

                this._applyAction(currentAction);

                // --- Middleware 2: Graph Runner Rules (Cascade) ---
                const cascades = GraphRunnerRules.evaluate(currentAction, this.workspace, dX, dY);
                queue.push(...cascades);
            }
        }

        if (!action.meta?.silent) {
            this.workspace.currentState.notify();
            if (typeof (window as any).mvrRuntime?.saveCurrentWorkspace === 'function') {
                (window as any).mvrRuntime.saveCurrentWorkspace().catch((e: any) => {
                    console.error("[MVR] Dispatcher automatic workspace save failed:", e);
                });
            }
        }

        // --- Extract After State for Delta ---
        let afterState: any = null;
        if (action.type === 'ADD_ASSET_WITH_RELATIONS') {
            const { asset, relations = [] } = action.payload;
            afterState = {
                asset: asset ? JSON.parse(JSON.stringify(this.workspace.currentState.assetRegistry.get(asset.uid) || asset)) : null,
                relations: Array.isArray(relations) ? relations.map((edge: any) => JSON.parse(JSON.stringify(edge))) : []
            };
        } else if (action.type === 'ADD_RELATION') {
            const { edge } = action.payload;
            afterState = {
                relation: edge ? JSON.parse(JSON.stringify(this.workspace.currentState.sceneGraph.getEdge(edge.id) || edge)) : null
            };
        } else if (action.type === 'ADD_RELATIONS') {
            const { edges = [] } = action.payload;
            afterState = {
                relations: Array.isArray(edges)
                    ? edges.map((edge: any) => JSON.parse(JSON.stringify(this.workspace.currentState.sceneGraph.getEdge(edge.id) || edge)))
                    : []
            };
        } else if (action.type === 'REMOVE_RELATION') {
            afterState = {
                relation: null
            };
        } else if (action.type === 'REMOVE_RELATIONS') {
            afterState = {
                relations: []
            };
        } else if (action.payload?.uid) {
            const assetAfter = this.workspace.currentState.assetRegistry.get(action.payload.uid);
            if (assetAfter) afterState = JSON.parse(JSON.stringify(assetAfter));
        } else if (action.payload?.uids) {
            const list: any[] = [];
            action.payload.uids.forEach((id: string) => {
                const asset = this.workspace.currentState.assetRegistry.get(id);
                if (asset) list.push(JSON.parse(JSON.stringify(asset)));
            });
            afterState = list;
        } else if (action.payload?.transforms) {
            const list: any[] = [];
            action.payload.transforms.forEach((item: any) => {
                const asset = this.workspace.currentState.assetRegistry.get(item.uid);
                if (asset) list.push(JSON.parse(JSON.stringify(asset)));
            });
            afterState = list;
        } else if (action.type === 'CLEAR_WORKSPACE') {
            afterState = [];
        }

        const stateDelta = { before: beforeState, after: afterState };

        // --- Middleware 1: Creative Memory Logging ---
        // Log to memory only if we have an intent or it's a significant action
        if (
            action.intent ||
            [
                'UPDATE_FUSION',
                'ADD_ASSET',
                'ADD_ASSET_WITH_RELATIONS',
                'BATCH_UPDATE_TRANSFORMS',
                'REMOVE_ASSETS',
                'CLEAR_WORKSPACE',
                'ADD_RELATION',
                'ADD_RELATIONS',
                'REMOVE_RELATION',
                'REMOVE_RELATIONS'
            ].includes(action.type)
        ) {
            memoryLayer.logDecision(action, stateDelta);
            
            // --- Hook into Evolution Engine: Branch off semantic variants ---
            if (action.type === 'UPDATE_FUSION' && action.intent && action.payload?.uid) {
                evolutionEngine.recordVariant(action.payload.uid, action.payload.fusionProperties, action.intent);
            }
        }
    }

    private _applyAction(action: Action) {
        const { assetRegistry, sceneGraph } = this.workspace.currentState;
        
        switch (action.type) {
            case 'ADD_ASSET': {
                const { asset } = action.payload;
                assetRegistry.register(asset);
                sceneGraph.addNode(asset.uid);
                break;
            }
            case 'ADD_ASSET_WITH_RELATIONS': {
                const { asset, relations = [] } = action.payload;
                if (asset) {
                    assetRegistry.register(asset);
                    sceneGraph.addNode(asset.uid);
                }
                if (Array.isArray(relations)) {
                    relations.forEach((edge) => {
                        sceneGraph.addEdge(edge);
                    });
                }
                break;
            }
            case 'REMOVE_ASSET': {
                const { uid } = action.payload;
                assetRegistry.delete(uid);
                sceneGraph.removeNode(uid);
                break;
            }
            case 'REMOVE_ASSETS': {
                const { uids } = action.payload;
                if (Array.isArray(uids)) {
                    uids.forEach((uid) => {
                        assetRegistry.delete(uid);
                        sceneGraph.removeNode(uid);
                    });
                }
                break;
            }
            case 'CLEAR_WORKSPACE': {
                assetRegistry.clear();
                sceneGraph.clear();
                break;
            }
            case 'ADD_RELATION': {
                const { edge } = action.payload;
                sceneGraph.addEdge(edge);
                break;
            }
            case 'ADD_RELATIONS': {
                const { edges } = action.payload;
                if (Array.isArray(edges)) {
                    edges.forEach((edge) => {
                        sceneGraph.addEdge(edge);
                    });
                }
                break;
            }
            case 'REMOVE_RELATION': {
                const { id } = action.payload;
                sceneGraph.removeEdge(id);
                break;
            }
            case 'REMOVE_RELATIONS': {
                const { ids } = action.payload;
                if (Array.isArray(ids)) {
                    ids.forEach((id) => {
                        sceneGraph.removeEdge(id);
                    });
                }
                break;
            }
            case 'BATCH_UPDATE_TRANSFORMS': {
                // Already decomposed and handled in queue processing loop
                break;
            }
            case 'UPDATE_FUSION': {
                const { uid, fusionProperties } = action.payload;
                assetRegistry.updateFusionProperties(uid, fusionProperties);
                break;
            }
            case 'UPDATE_TRANSFORM':
            case 'CASCADE_TRANSFORM': {
                const { uid, transform } = action.payload;
                assetRegistry.updateAssetTransform(uid, transform);
                break;
            }
            case 'SYNC_ASSET_VERSION': {
                const { uid, version, sourceImage } = action.payload;
                const asset = assetRegistry.get(uid);
                if (asset) {
                    asset.version = version;
                    asset.sourceImage = sourceImage;
                }
                break;
            }
            case 'UPDATE_ASSET_METADATA': {
                const { uid, version, ...rest } = action.payload;
                const asset = assetRegistry.get(uid);
                if (asset) {
                    if (version !== undefined) {
                        asset.version = version;
                    }
                    Object.assign(asset, rest);
                }
                break;
            }
            case 'CO_CREATE_ISOLATED': {
                const { uid, prompt } = action.payload;
                if (typeof (window as any).handleIsolatedAssetEdit === 'function') {
                    (window as any).handleIsolatedAssetEdit(uid, prompt).catch(console.error);
                }
                break;
            }
            case 'CO_CREATE_SYNC': {
                const { uid, prompt } = action.payload;
                if (typeof (window as any).handleQuickFusionSync === 'function') {
                    (window as any).handleQuickFusionSync(uid, prompt).catch(console.error);
                }
                break;
            }
            default:
                console.warn('[Dispatcher] Unknown action type:', action.type);
        }
    }
}
