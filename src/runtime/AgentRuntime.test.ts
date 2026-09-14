import { describe, expect, it } from 'vitest';
import { AgentRuntime, type AgentRuntimeHost } from './AgentRuntime';
import { ProjectWorkspace } from './WorkspaceRuntime';
import type { AssetEntity } from './AssetRuntime';

function makeAsset(uid = 'sofa_001'): AssetEntity {
    return {
        uid,
        version: 3,
        type: 'product',
        sourceImage: 'blob:sofa-image',
        masks: [],
        variants: [],
        metadata: {
            createdAt: 1,
            updatedAt: 1,
            creatorId: 'test',
            usageCount: 2,
            tags: ['沙发', '米白', '客厅']
        },
        fusionProperties: {
            brightness: 0,
            contrast: 0,
            saturation: 0,
            blur: 0,
            hueRotate: 0,
            grayscale: 0,
            sepia: 0
        },
        transform: {
            x: 100,
            y: 120,
            width: 400,
            height: 200,
            rotation: 0,
            zIndex: 1
        },
        layerName: '米白沙发'
    };
}

function makeHost(workspace: ProjectWorkspace, searchableWorkspaces: ProjectWorkspace[] = [workspace]): AgentRuntimeHost {
    return {
        getCurrentWorkspace: () => workspace,
        listWorkspaces: () => searchableWorkspaces,
        listAgentAssets: async () => searchableWorkspaces.flatMap(sourceWorkspace =>
            sourceWorkspace.currentState.assetRegistry.getAll().map(asset => ({
                asset,
                assetId: asset.uid,
                projectId: sourceWorkspace.projectId,
                version: asset.version,
                name: asset.layerName || asset.uid,
                type: asset.type,
                tags: asset.metadata.tags
            }))
        ),
        resolveAgentAsset: async ({ projectId, assetId }) =>
            searchableWorkspaces.find(sourceWorkspace => sourceWorkspace.projectId === projectId)
                ?.currentState.assetRegistry.get(assetId) || null,
        commitWorkspace: (target, jobId) => target.commit(jobId)
    };
}

describe('AgentRuntime asset reuse flow', () => {
    it('executes find, duplicate, place, transform and commits once', async () => {
        const workspace = new ProjectWorkspace('current-project');
        const previousProject = new ProjectWorkspace('previous-project');
        previousProject.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset() } });
        workspace.history = [];
        workspace.historyIndex = -1;
        previousProject.history = [];
        previousProject.historyIndex = -1;

        const agent = new AgentRuntime(makeHost(workspace, [workspace, previousProject]));
        const created = agent.createAssetReuseJob(
            '把上次项目中的沙发放进当前房间，缩小 15%。',
            '沙发',
            { placement: { x: 240, y: 300, zIndex: 4 }, scale: 0.85 }
        );

        const waiting = await agent.executeJob(created.id);

        expect(waiting.status).toBe('waiting_confirmation');
        expect(waiting.verification?.passed).toBe(true);
        expect(waiting.targetAssets).toHaveLength(1);
        expect(workspace.history).toHaveLength(0);
        expect(workspace.currentState.assetRegistry.getAll()).toHaveLength(1);

        const newAsset = workspace.currentState.assetRegistry.get(waiting.targetAssets[0]);
        expect(newAsset?.transform).toMatchObject({ x: 240, y: 300, width: 340, height: 170, zIndex: 4 });
        expect(JSON.stringify(waiting)).not.toContain('blob:sofa-image');

        const completed = await agent.confirmJob(created.id);
        const repeated = await agent.commitJob(created.id);

        expect(completed.status).toBe('completed');
        expect(completed.result?.commit?.commitId).toBe(`commit_${created.id}`);
        expect(repeated.result?.commit).toEqual(completed.result?.commit);
        expect(workspace.history).toHaveLength(1);
    });

    it('rolls back mutations when a command fails', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset() } });
        workspace.history = [];
        workspace.historyIndex = -1;

        const agent = new AgentRuntime(makeHost(workspace));
        const job = agent.createAssetReuseJob('找一个不存在的台灯', '台灯');
        const result = await agent.executeJob(job.id);

        expect(result.status).toBe('failed');
        expect(result.error?.message).toContain('No matching asset');
        expect(workspace.currentState.assetRegistry.getAll()).toHaveLength(1);
        expect(workspace.history).toHaveLength(0);
    });

    it('rolls back the staged scene when the user cancels before commit', async () => {
        const workspace = new ProjectWorkspace('current-project');
        const previousProject = new ProjectWorkspace('previous-project');
        previousProject.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset() } });
        workspace.history = [];
        workspace.historyIndex = -1;

        const agent = new AgentRuntime(makeHost(workspace, [workspace, previousProject]));
        const job = agent.createAssetReuseJob('复用上次项目中的沙发', '沙发');
        const waiting = await agent.executeJob(job.id);
        expect(waiting.status).toBe('waiting_confirmation');
        expect(workspace.currentState.assetRegistry.getAll()).toHaveLength(1);

        const cancelled = await agent.cancelJob(job.id);

        expect(cancelled.status).toBe('cancelled');
        expect(cancelled.targetAssets).toHaveLength(0);
        expect(workspace.currentState.assetRegistry.getAll()).toHaveLength(0);
        expect(workspace.history).toHaveLength(0);
    });

    it('resolves an indexed asset from an unloaded project only when duplication needs it', async () => {
        const workspace = new ProjectWorkspace('current-project');
        const source = makeAsset();
        const agent = new AgentRuntime({
            getCurrentWorkspace: () => workspace,
            listWorkspaces: () => [workspace],
            listAgentAssets: async () => [{
                assetId: source.uid,
                projectId: 'previous-project',
                version: source.version,
                name: source.layerName || source.uid,
                type: source.type,
                tags: source.metadata.tags
            }],
            resolveAgentAsset: async ({ projectId, assetId }) =>
                projectId === 'previous-project' && assetId === source.uid ? source : null,
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createAssetReuseJob('从历史项目复用沙发', '沙发');
        const result = await agent.executeJob(job.id);

        expect(result.status).toBe('waiting_confirmation');
        expect(workspace.currentState.assetRegistry.getAll()).toHaveLength(1);
        expect(workspace.currentState.assetRegistry.get(result.targetAssets[0])?.parentId).toBe(source.uid);
    });

    it('repairs a failed external workspace check once before waiting for confirmation', async () => {
        const workspace = new ProjectWorkspace('current-project');
        const source = makeAsset();
        let repaired = false;
        const agent = new AgentRuntime({
            getCurrentWorkspace: () => workspace,
            listWorkspaces: () => [workspace],
            listAgentAssets: async () => [{
                assetId: source.uid,
                projectId: 'previous-project',
                version: source.version,
                name: source.layerName || source.uid,
                type: source.type,
                tags: source.metadata.tags
            }],
            resolveAgentAsset: async () => source,
            verifyAgentWorkspace: async () => [{
                id: 'dom_asset_check',
                passed: repaired,
                message: repaired ? 'DOM repaired.' : 'DOM missing.'
            }],
            repairAgentWorkspace: async () => {
                repaired = true;
            },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createAssetReuseJob('复用沙发并验证工作区', '沙发');
        const result = await agent.executeJob(job.id);

        expect(result.status).toBe('waiting_confirmation');
        expect(result.verification?.passed).toBe(true);
        expect(result.verification?.checks.some(check => check.id === 'agent_auto_repair' && check.passed)).toBe(true);
    });
});

describe('AgentRuntime Magic Layers command boundary', () => {
    it('stages extraction by reference and commits it once', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset('scene_001') } });
        workspace.history = [];
        workspace.historyIndex = -1;
        let undone = false;

        const agent = new AgentRuntime({
            getCurrentWorkspace: () => workspace,
            listWorkspaces: () => [workspace],
            listAgentAssets: async () => [],
            resolveAgentAsset: async () => null,
            extractAgentLayers: async ({ workspace: target, assetId }) => {
                const extracted = makeAsset('layer_001');
                extracted.parentId = assetId;
                target.dispatcher.dispatch({
                    type: 'ADD_ASSET',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { asset: extracted }
                });
                return {
                    operationId: 'magic-op-001',
                    sourceAssetId: assetId,
                    extractedAssetIds: [extracted.uid],
                    extractedLayerIds: ['semantic-layer-001'],
                    extractedCount: 1,
                    heldCount: 0
                };
            },
            undoAgentLayers: async ({ workspace: target }) => {
                undone = true;
                target.dispatcher.dispatch({
                    type: 'REMOVE_ASSETS',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { uids: ['layer_001'] }
                });
            },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const created = agent.createLayerExtractionJob('提取场景中的可编辑图层', 'scene_001');
        const waiting = await agent.executeJob(created.id);

        expect(waiting.status).toBe('waiting_confirmation');
        expect(waiting.targetAssets).toEqual(['layer_001']);
        expect(waiting.result?.commandResults[0].outputRefs).toEqual(expect.objectContaining({
            operationId: 'magic-op-001',
            extractedCount: 1
        }));
        expect(JSON.stringify(waiting)).not.toContain('blob:sofa-image');
        expect(workspace.history).toHaveLength(0);

        const completed = await agent.confirmJob(created.id);
        expect(completed.status).toBe('completed');
        expect(workspace.history).toHaveLength(1);

        const repeated = await agent.commitJob(created.id);
        expect(repeated.result?.commit).toEqual(completed.result?.commit);
        expect(workspace.history).toHaveLength(1);
        expect(undone).toBe(false);
    });

    it('undoes an extracted layer when the staged job is cancelled', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset('scene_001') } });
        workspace.history = [];
        workspace.historyIndex = -1;
        let undone = false;

        const agent = new AgentRuntime({
            getCurrentWorkspace: () => workspace,
            listWorkspaces: () => [workspace],
            listAgentAssets: async () => [],
            resolveAgentAsset: async () => null,
            extractAgentLayers: async ({ workspace: target, assetId }) => {
                const extracted = makeAsset('layer_001');
                extracted.parentId = assetId;
                target.dispatcher.dispatch({ type: 'ADD_ASSET', meta: { skipSnapshot: true }, payload: { asset: extracted } });
                return {
                    operationId: 'magic-op-002',
                    sourceAssetId: assetId,
                    extractedAssetIds: [extracted.uid],
                    extractedLayerIds: ['semantic-layer-002'],
                    extractedCount: 1,
                    heldCount: 0
                };
            },
            undoAgentLayers: async ({ workspace: target }) => {
                undone = true;
                target.dispatcher.dispatch({ type: 'REMOVE_ASSETS', meta: { skipSnapshot: true }, payload: { uids: ['layer_001'] } });
            },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createLayerExtractionJob('提取图层', 'scene_001');
        const waiting = await agent.executeJob(job.id);
        expect(waiting.status).toBe('waiting_confirmation');
        expect(workspace.currentState.assetRegistry.get('layer_001')).toBeTruthy();

        const cancelled = await agent.cancelJob(job.id);
        expect(cancelled.status).toBe('cancelled');
        expect(undone).toBe(true);
        expect(workspace.currentState.assetRegistry.get('layer_001')).toBeUndefined();
        expect(workspace.history).toHaveLength(0);
    });

    it('rolls back staged extraction when Commit persistence fails', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: makeAsset('scene_001') } });
        workspace.history = [];
        workspace.historyIndex = -1;
        let undone = false;

        const agent = new AgentRuntime({
            getCurrentWorkspace: () => workspace,
            listWorkspaces: () => [workspace],
            listAgentAssets: async () => [],
            resolveAgentAsset: async () => null,
            extractAgentLayers: async ({ workspace: target, assetId }) => {
                const extracted = makeAsset('layer_001');
                extracted.parentId = assetId;
                target.dispatcher.dispatch({ type: 'ADD_ASSET', meta: { skipSnapshot: true }, payload: { asset: extracted } });
                return {
                    operationId: 'magic-op-003',
                    sourceAssetId: assetId,
                    extractedAssetIds: [extracted.uid],
                    extractedLayerIds: ['semantic-layer-003'],
                    extractedCount: 1,
                    heldCount: 0
                };
            },
            undoAgentLayers: async ({ workspace: target }) => {
                undone = true;
                target.dispatcher.dispatch({ type: 'REMOVE_ASSETS', meta: { skipSnapshot: true }, payload: { uids: ['layer_001'] } });
            },
            commitWorkspace: (target, jobId) => target.commit(jobId, async () => {
                throw new Error('persistence unavailable');
            })
        });

        const job = agent.createLayerExtractionJob('提交失败的图层提取', 'scene_001', { requiresConfirmation: false });
        const failed = await agent.executeJob(job.id);

        expect(failed.status).toBe('failed');
        expect(failed.error?.message).toContain('persistence unavailable');
        expect(undone).toBe(true);
        expect(workspace.currentState.assetRegistry.get('layer_001')).toBeUndefined();
        expect(workspace.history).toHaveLength(0);
    });

    it('stages a capability result, verifies it by asset reference, and commits once', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.history = [];
        workspace.historyIndex = -1;
        let capabilityCommits = 0;
        const agent = new AgentRuntime({
            ...makeHost(workspace),
            executeAgentCapability: async ({ workspace: target }) => {
                const result = makeAsset('completed-layer-001');
                target.dispatcher.dispatch({
                    type: 'ADD_ASSET',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { asset: result }
                });
                return {
                    operationId: 'capability-op-001',
                    targetAssetIds: [result.uid],
                    outputRefs: { capabilityType: 'object_completion', versionId: 'completion-v1', quality: { status: 'passed' } }
                };
            },
            undoAgentCapability: async ({ workspace: target }) => {
                target.dispatcher.dispatch({
                    type: 'REMOVE_ASSETS',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { uids: ['completed-layer-001'] }
                });
            },
            commitAgentCapabilities: async () => { capabilityCommits += 1; },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createObjectCompletionJob('补全被遮挡的沙发', 'scene_001', 'completion_001');
        const waiting = await agent.executeJob(job.id);
        expect(waiting.status).toBe('waiting_confirmation');
        expect(waiting.verification?.passed).toBe(true);
        expect(JSON.stringify(waiting)).not.toContain('data:');
        expect(JSON.stringify(waiting)).not.toContain('blob:');

        const completed = await agent.confirmJob(job.id);
        expect(completed.status).toBe('completed');
        expect(capabilityCommits).toBe(1);
        expect(workspace.history).toHaveLength(1);
    });

    it('creates an object edit job with an explicit edit mode', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.history = [];
        workspace.historyIndex = -1;
        const agent = new AgentRuntime({
            ...makeHost(workspace),
            executeAgentCapability: async ({ workspace: target, params }) => {
                const result = makeAsset('object-variant-001');
                target.dispatcher.dispatch({
                    type: 'ADD_ASSET',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { asset: result }
                });
                return {
                    operationId: 'object-edit-op-001',
                    targetAssetIds: [result.uid],
                    outputRefs: { capabilityType: 'edit_asset', mode: params.mode, assetId: result.uid }
                };
            },
            commitAgentCapabilities: async () => {},
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createObjectEditJob('生成沙发新版本', 'sofa_001', '换成米白材质', 'variant');
        expect(job.plan?.steps[0].commandType).toBe('edit_asset');
        expect(job.plan?.steps[0].params.mode).toBe('variant');
        const waiting = await agent.executeJob(job.id);
        expect(waiting.status).toBe('waiting_confirmation');
        expect(waiting.verification?.passed).toBe(true);
        expect(JSON.stringify(waiting)).not.toContain('data:');
        await agent.confirmJob(job.id);
        expect(agent.getJob(job.id)?.status).toBe('completed');
    });

    it('rolls back a staged capability when the user cancels', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.history = [];
        workspace.historyIndex = -1;
        let undone = false;
        const agent = new AgentRuntime({
            ...makeHost(workspace),
            executeAgentCapability: async ({ workspace: target }) => {
                const result = makeAsset('material-layer-001');
                target.dispatcher.dispatch({ type: 'ADD_ASSET', meta: { skipSnapshot: true }, payload: { asset: result } });
                return { operationId: 'capability-op-002', targetAssetIds: [result.uid], outputRefs: { capabilityType: 'replace_material' } };
            },
            undoAgentCapability: async ({ workspace: target }) => {
                undone = true;
                target.dispatcher.dispatch({ type: 'REMOVE_ASSETS', meta: { skipSnapshot: true }, payload: { uids: ['material-layer-001'] } });
            },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createMaterialReplacementJob('替换材质', 'source-layer-001', '换成深色木材');
        const waiting = await agent.executeJob(job.id);
        expect(waiting.status).toBe('waiting_confirmation');
        const cancelled = await agent.cancelJob(job.id);
        expect(cancelled.status).toBe('cancelled');
        expect(undone).toBe(true);
        expect(workspace.currentState.assetRegistry.get('material-layer-001')).toBeUndefined();
        expect(workspace.history).toHaveLength(0);
    });

    it('rolls back a capability when the final Commit fails', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.history = [];
        workspace.historyIndex = -1;
        let undone = false;
        const agent = new AgentRuntime({
            ...makeHost(workspace),
            executeAgentCapability: async ({ workspace: target }) => {
                const result = makeAsset('failed-capability-layer');
                target.dispatcher.dispatch({ type: 'ADD_ASSET', meta: { skipSnapshot: true }, payload: { asset: result } });
                return { operationId: 'capability-op-003', targetAssetIds: [result.uid], outputRefs: { capabilityType: 'object_completion' } };
            },
            undoAgentCapability: async ({ workspace: target }) => {
                undone = true;
                target.dispatcher.dispatch({ type: 'REMOVE_ASSETS', meta: { skipSnapshot: true }, payload: { uids: ['failed-capability-layer'] } });
            },
            commitAgentCapabilities: async () => {},
            commitWorkspace: (target, jobId) => target.commit(jobId, async () => { throw new Error('capability persistence unavailable'); })
        });

        const job = agent.createObjectCompletionJob('提交补全结果', 'scene_001', 'completion_003', { requiresConfirmation: false });
        const failed = await agent.executeJob(job.id);
        expect(failed.status).toBe('failed');
        expect(failed.error?.message).toContain('capability persistence unavailable');
        expect(undone).toBe(true);
        expect(workspace.currentState.assetRegistry.get('failed-capability-layer')).toBeUndefined();
        expect(workspace.history).toHaveLength(0);
    });

    it('keeps a structurally invalid result for inspection and retries after repair', async () => {
        const workspace = new ProjectWorkspace('current-project');
        workspace.history = [];
        workspace.historyIndex = -1;
        let attempt = 0;
        const agent = new AgentRuntime({
            ...makeHost(workspace),
            executeAgentCapability: async ({ workspace: target }) => {
                attempt += 1;
                const result = makeAsset('verification-layer-001');
                if (attempt === 1) result.transform.width = Number.NaN;
                target.dispatcher.dispatch({ type: 'ADD_ASSET', meta: { skipSnapshot: true }, payload: { asset: result } });
                return { operationId: `verification-op-${attempt}`, targetAssetIds: [result.uid], outputRefs: { capabilityType: 'object_completion' } };
            },
            undoAgentCapability: async ({ workspace: target }) => {
                target.dispatcher.dispatch({ type: 'REMOVE_ASSETS', meta: { skipSnapshot: true }, payload: { uids: ['verification-layer-001'] } });
            },
            commitWorkspace: (target, jobId) => target.commit(jobId)
        });

        const job = agent.createObjectCompletionJob('验证对象补全', 'scene_001', 'completion-verify');
        const held = await agent.executeJob(job.id);
        expect(held.status).toBe('waiting_confirmation');
        expect(held.verification?.passed).toBe(false);
        expect(held.error?.message).toContain('transforms');
        await expect(agent.confirmJob(job.id)).rejects.toThrow('cannot be committed');

        const repaired = await agent.retryJob(job.id);
        expect(repaired.status).toBe('waiting_confirmation');
        expect(repaired.verification?.passed).toBe(true);
        expect(attempt).toBe(2);
    });
});
