import type { AssetEntity, AssetTransform } from './AssetRuntime';
import type { ProjectWorkspace } from './WorkspaceRuntime';
import type {
    AgentCommand,
    AgentCommandContext,
    AgentCommandExecution,
    AgentCommandType,
    AgentCapabilityType,
    AgentCommandVerification,
    AssetSearchQuery,
    AssetSearchCandidate
} from './AgentRuntime';

function createId(prefix: string): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return `${prefix}_${crypto.randomUUID()}`;
    }
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function clone<T>(value: T): T {
    return JSON.parse(JSON.stringify(value));
}

function assetName(asset: AssetEntity): string {
    return String((asset as any).name || asset.layerName || asset.uid);
}

function candidateName(candidate: AssetSearchCandidate): string {
    return candidate.asset ? assetName(candidate.asset) : candidate.name;
}

function searchText(candidate: AssetSearchCandidate): string {
    const metadata = candidate.asset?.metadata || ({} as any);
    return [
        candidate.assetId,
        candidateName(candidate),
        candidate.type,
        ...(candidate.tags || metadata.tags || [])
    ].filter(Boolean).join(' ').toLowerCase();
}

function scoreAsset(candidate: AssetSearchCandidate, query: AssetSearchQuery): number {
    const text = searchText(candidate);
    const tokens = query.text
        .toLowerCase()
        .split(/\s+/)
        .map(token => token.trim())
        .filter(Boolean);
    if (tokens.length === 0) return 0;

    return tokens.reduce((score, token) => {
        if (candidate.assetId.toLowerCase() === token) return score + 100;
        if (candidateName(candidate).toLowerCase() === token) return score + 80;
        if (text.includes(token)) return score + 10;
        return score;
    }, 0);
}

function getAsset(context: AgentCommandContext, uid: string): AssetEntity {
    const asset = context.workspace.currentState.assetRegistry.get(uid);
    if (!asset) throw new Error(`Asset ${uid} does not exist in the target workspace.`);
    return asset;
}

async function getSourceAsset(context: AgentCommandContext, uid: string, projectId?: string): Promise<AssetEntity> {
    const current = context.workspace.currentState.assetRegistry.get(uid);
    if (current) return current;
    const candidates = await context.assetSource.listAssets();
    const external = candidates.find(candidate => candidate.assetId === uid && (!projectId || candidate.projectId === projectId));
    if (external) {
        const resolved = await context.assetSource.resolveAsset({ projectId: external.projectId, assetId: external.assetId });
        if (resolved) return resolved;
    }
    throw new Error(`Source asset ${uid} does not exist in the available asset catalog.`);
}

function commandBase(
    type: AgentCommandType,
    jobId: string,
    targetAssetIds: string[] = []
): Pick<AgentCommand, 'commandId' | 'jobId' | 'type' | 'targetAssetIds'> {
    return {
        commandId: createId('cmd'),
        jobId,
        type,
        targetAssetIds
    };
}

function silentRuntimeAction(context: AgentCommandContext, action: any): void {
    context.workspace.dispatcher.dispatch({
        ...action,
        meta: {
            ...(action.meta || {}),
            silent: true,
            skipSnapshot: true,
            skipNotify: true
        }
    });
}

export class FindAssetCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type = 'find_asset' as const;
    readonly targetAssetIds: string[] = [];
    readonly preconditions = ['asset search scope is available'];
    readonly query: AssetSearchQuery;

    constructor(jobId: string, query: AssetSearchQuery) {
        Object.assign(this, commandBase(this.type, jobId), { query });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        const candidates = await context.assetSource.listAssets();
        const matches = candidates
            .filter(({ projectId }) => !this.query.projectId || projectId === this.query.projectId)
            .map((candidate) => ({
                candidate,
                projectId: candidate.projectId,
                score: scoreAsset(candidate, this.query)
            }))
            .filter(entry => entry.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, this.query.limit || 10);

        return {
            outputRefs: {
                matches: matches.map(({ candidate, projectId, score }) => ({
                    assetId: candidate.assetId,
                    projectId,
                    version: candidate.version,
                    name: candidateName(candidate),
                    type: candidate.type,
                    score
                }))
            }
        };
    }

    async verify(_context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const matches = execution.outputRefs?.matches;
        return {
            passed: Array.isArray(matches) && matches.length > 0,
            checks: [{
                id: 'asset_found',
                passed: Array.isArray(matches) && matches.length > 0,
                message: Array.isArray(matches) && matches.length > 0
                    ? `Found ${matches.length} matching asset(s).`
                    : 'No matching asset was found.'
            }]
        };
    }

    async undo(): Promise<void> {
        // Read-only command.
    }
}

export class DuplicateAssetCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type = 'duplicate_asset' as const;
    readonly targetAssetIds: string[];
    readonly preconditions = ['source asset exists'];
    readonly sourceAssetId?: string;
    private createdAssetId: string | null = null;

    constructor(jobId: string, sourceAssetId?: string) {
        Object.assign(this, commandBase(this.type, jobId, sourceAssetId ? [sourceAssetId] : []), { sourceAssetId });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        const sourceAssetId = this.sourceAssetId || context.requireOutput('find_asset').matches?.[0]?.assetId;
        if (!sourceAssetId) throw new Error('duplicate_asset requires a source asset.');
        const sourceRef = context.requireOutput('find_asset').matches?.[0];
        const source = await getSourceAsset(context, sourceAssetId, sourceRef?.projectId);
        const now = Date.now();
        const duplicate = clone(source) as AssetEntity;
        const newAssetId = createId('asset');
        duplicate.uid = newAssetId;
        duplicate.version = 1;
        duplicate.runtimeDisplayUrl = undefined;
        duplicate.parentId = source.uid;
        duplicate.genealogy = {
            ...(duplicate.genealogy || {}),
            parents: [source.uid],
            action: 'duplicate_asset'
        };
        duplicate.metadata = {
            ...duplicate.metadata,
            createdAt: now,
            updatedAt: now,
            usageCount: 0,
            tags: [...(duplicate.metadata?.tags || [])]
        };
        (duplicate as any).name = `${assetName(source)} copy`;

        silentRuntimeAction(context, {
            type: 'ADD_ASSET',
            payload: { asset: duplicate }
        });
        this.createdAssetId = newAssetId;

        return {
            targetAssetIds: [newAssetId],
            outputRefs: {
                sourceAssetId: source.uid,
                newAssetId,
                projectId: context.workspace.projectId
            }
        };
    }

    async verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const newAssetId = execution.outputRefs?.newAssetId;
        const exists = Boolean(newAssetId && context.workspace.currentState.assetRegistry.get(newAssetId));
        const isInScene = Boolean(newAssetId && context.workspace.currentState.sceneGraph.getNodes().includes(newAssetId));
        return {
            passed: exists && isInScene,
            checks: [
                { id: 'duplicate_registered', passed: exists, message: exists ? 'Duplicate asset is registered.' : 'Duplicate asset is missing.' },
                { id: 'duplicate_in_scene', passed: isInScene, message: isInScene ? 'Duplicate asset is present in the scene graph.' : 'Duplicate asset is not in the scene graph.' }
            ]
        };
    }

    async undo(context: AgentCommandContext): Promise<void> {
        if (this.createdAssetId && context.workspace.currentState.assetRegistry.get(this.createdAssetId)) {
            silentRuntimeAction(context, {
                type: 'REMOVE_ASSET',
                payload: { uid: this.createdAssetId }
            });
        }
    }
}

export class PlaceAssetCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type = 'place_asset' as const;
    readonly targetAssetIds: string[];
    readonly preconditions = ['target asset exists'];
    readonly assetId?: string;
    readonly transform: Partial<AssetTransform>;
    private beforeTransform: AssetTransform | null = null;

    constructor(jobId: string, assetId: string | undefined, transform: Partial<AssetTransform>) {
        Object.assign(this, commandBase(this.type, jobId, assetId ? [assetId] : []), { assetId, transform });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        const assetId = this.assetId || context.requireOutput('duplicate_asset').newAssetId;
        if (!assetId) throw new Error('place_asset requires a target asset.');
        const asset = getAsset(context, assetId);
        this.beforeTransform = clone(asset.transform);
        silentRuntimeAction(context, {
            type: 'UPDATE_TRANSFORM',
            payload: { uid: assetId, transform: this.transform }
        });
        return {
            targetAssetIds: [assetId],
            outputRefs: { assetId, transform: clone(context.workspace.currentState.assetRegistry.get(assetId)?.transform) }
        };
    }

    async verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const assetId = execution.outputRefs?.assetId;
        const asset = assetId ? context.workspace.currentState.assetRegistry.get(assetId) : undefined;
        const passed = Boolean(asset && this.transformMatches(asset.transform, this.transform));
        return {
            passed,
            checks: [{ id: 'asset_placed', passed, message: passed ? 'Asset placement matches the requested transform.' : 'Asset placement does not match the requested transform.' }]
        };
    }

    async undo(context: AgentCommandContext): Promise<void> {
        if (this.assetId || this.beforeTransform) {
            const assetId = this.assetId || context.requireOutput('duplicate_asset').newAssetId;
            if (assetId && this.beforeTransform) {
                silentRuntimeAction(context, { type: 'UPDATE_TRANSFORM', payload: { uid: assetId, transform: this.beforeTransform } });
            }
        }
    }

    private transformMatches(actual: AssetTransform | undefined, expected: Partial<AssetTransform>): boolean {
        if (!actual) return false;
        return Object.entries(expected).every(([key, value]) => (actual as any)[key] === value);
    }
}

export class TransformAssetCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type = 'transform_asset' as const;
    readonly targetAssetIds: string[];
    readonly preconditions = ['target asset exists'];
    readonly assetId?: string;
    readonly transform?: Partial<AssetTransform>;
    readonly scale?: number;
    private beforeTransform: AssetTransform | null = null;

    constructor(jobId: string, assetId: string | undefined, options: { transform?: Partial<AssetTransform>; scale?: number }) {
        Object.assign(this, commandBase(this.type, jobId, assetId ? [assetId] : []), {
            assetId,
            transform: options.transform,
            scale: options.scale
        });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        const assetId = this.assetId || context.requireOutput('place_asset').assetId || context.requireOutput('duplicate_asset').newAssetId;
        if (!assetId) throw new Error('transform_asset requires a target asset.');
        const asset = getAsset(context, assetId);
        this.beforeTransform = clone(asset.transform);
        const nextTransform: Partial<AssetTransform> = { ...(this.transform || {}) };
        if (this.scale !== undefined) {
            if (!Number.isFinite(this.scale) || this.scale <= 0) throw new Error('transform_asset scale must be greater than zero.');
            nextTransform.width = asset.transform.width * this.scale;
            nextTransform.height = asset.transform.height * this.scale;
        }
        silentRuntimeAction(context, {
            type: 'UPDATE_TRANSFORM',
            payload: { uid: assetId, transform: nextTransform }
        });
        return {
            targetAssetIds: [assetId],
            outputRefs: { assetId, transform: clone(context.workspace.currentState.assetRegistry.get(assetId)?.transform) }
        };
    }

    async verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const assetId = execution.outputRefs?.assetId;
        const asset = assetId ? context.workspace.currentState.assetRegistry.get(assetId) : undefined;
        const expected = execution.outputRefs?.transform;
        const passed = Boolean(asset && expected && this.transformMatches(asset.transform, expected));
        return {
            passed,
            checks: [{ id: 'asset_transformed', passed, message: passed ? 'Asset transform was applied.' : 'Asset transform was not applied.' }]
        };
    }

    async undo(context: AgentCommandContext): Promise<void> {
        const assetId = this.assetId || context.requireOutput('place_asset').assetId || context.requireOutput('duplicate_asset').newAssetId;
        if (assetId && this.beforeTransform) {
            silentRuntimeAction(context, { type: 'UPDATE_TRANSFORM', payload: { uid: assetId, transform: this.beforeTransform } });
        }
    }

    private transformMatches(actual: AssetTransform | undefined, expected: AssetTransform): boolean {
        if (!actual) return false;
        return ['x', 'y', 'width', 'height', 'rotation', 'zIndex'].every(key => actual[key as keyof AssetTransform] === expected[key as keyof AssetTransform]);
    }
}

export class ExtractLayersCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type = 'extract_layers' as const;
    readonly targetAssetIds: string[];
    readonly preconditions = ['source asset exists', 'Magic Layers adapter is available'];
    readonly assetId: string;
    readonly layerIds: string[];

    constructor(jobId: string, assetId: string, layerIds: string[] = []) {
        Object.assign(this, commandBase(this.type, jobId, [assetId]), {
            assetId,
            layerIds: Array.isArray(layerIds) ? [...layerIds] : []
        });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        if (!getAsset(context, this.assetId)) {
            throw new Error(`Source asset ${this.assetId} does not exist in the target workspace.`);
        }
        const extraction = context.agentLayerExtraction;
        if (!extraction) throw new Error('Magic Layers Agent adapter is unavailable.');

        const result = await extraction.execute({
            assetId: this.assetId,
            layerIds: this.layerIds
        });
        if (!result?.operationId) throw new Error('Magic Layers extraction returned no operation reference.');

        return {
            targetAssetIds: result.extractedAssetIds || [],
            outputRefs: {
                operationId: result.operationId,
                sourceAssetId: result.sourceAssetId || this.assetId,
                extractedAssetIds: result.extractedAssetIds || [],
                extractedLayerIds: result.extractedLayerIds || [],
                extractedCount: Number(result.extractedCount) || 0,
                heldCount: Number(result.heldCount) || 0
            }
        };
    }

    async verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const output = execution.outputRefs || {};
        const ids = Array.isArray(output.extractedAssetIds) ? output.extractedAssetIds : [];
        const assetsExist = ids.every(uid => Boolean(context.workspace.currentState.assetRegistry.get(uid)));
        const hasExtractedAsset = ids.length > 0;
        return {
            passed: hasExtractedAsset && assetsExist,
            checks: [
                {
                    id: 'layers_extracted',
                    passed: hasExtractedAsset,
                    message: hasExtractedAsset ? `Extracted ${ids.length} layer asset(s).` : 'Magic Layers produced no standalone asset.'
                },
                {
                    id: 'extracted_assets_registered',
                    passed: assetsExist,
                    message: assetsExist ? 'Extracted assets are registered in Runtime.' : 'One or more extracted assets are missing from Runtime.'
                }
            ]
        };
    }

    async undo(context: AgentCommandContext, execution?: AgentCommandExecution): Promise<void> {
        const operationId = execution?.outputRefs?.operationId;
        if (!operationId || !context.agentLayerExtraction) return;
        await context.agentLayerExtraction.undo({ operationId });
    }
}

export class CapabilityCommand implements AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type: AgentCapabilityType;
    readonly targetAssetIds: string[];
    readonly preconditions = ['capability adapter is available'];
    readonly params: Record<string, any>;

    constructor(jobId: string, type: AgentCapabilityType, params: Record<string, any> = {}) {
        const targetAssetIds = Array.isArray(params.targetAssetIds)
            ? params.targetAssetIds.filter(Boolean)
            : [params.childId, params.itemId].filter(Boolean);
        Object.assign(this, commandBase(type, jobId, targetAssetIds), { type, params });
    }

    async execute(context: AgentCommandContext): Promise<AgentCommandExecution> {
        if (!context.agentCapability) throw new Error(`${this.type} Agent adapter is unavailable.`);
        const result = await context.agentCapability.execute({
            capabilityType: this.type,
            targetAssetIds: this.targetAssetIds,
            params: this.params
        });
        if (!result?.operationId) throw new Error(`${this.type} returned no operation reference.`);
        return {
            targetAssetIds: result.targetAssetIds || [],
            outputRefs: {
                operationId: result.operationId,
                ...result.outputRefs
            }
        };
    }

    async verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification> {
        const output = execution.outputRefs || {};
        const targetIds = Array.isArray(execution.targetAssetIds) ? execution.targetAssetIds : [];
        const assetsExist = targetIds.length > 0 && targetIds.every(uid => Boolean(context.workspace.currentState.assetRegistry.get(uid)));
        const inScene = targetIds.length > 0 && targetIds.every(uid => context.workspace.currentState.sceneGraph.getNodes().includes(uid));
        const qualityStatus = String(output.quality?.status || '').toLowerCase();
        const qualityPassed = !output.quality || !['failed', 'held', 'manual_review'].includes(qualityStatus);
        const passed = Boolean(output.operationId) && assetsExist && inScene && qualityPassed;
        return {
            passed,
            checks: [
                { id: 'capability_operation_created', passed: Boolean(output.operationId), message: output.operationId ? 'Capability operation is staged.' : 'Capability operation reference is missing.' },
                { id: 'capability_assets_exist', passed: assetsExist, message: assetsExist ? 'Capability target assets exist.' : 'Capability target assets are missing.' },
                { id: 'capability_assets_in_scene', passed: inScene, message: inScene ? 'Capability target assets are in the scene graph.' : 'Capability target assets are not in the scene graph.' },
                { id: 'capability_quality', passed: qualityPassed, message: qualityPassed ? 'Capability quality gate passed.' : 'Capability quality gate rejected the result.' }
            ]
        };
    }

    async undo(context: AgentCommandContext, execution?: AgentCommandExecution): Promise<void> {
        const operationId = execution?.outputRefs?.operationId;
        if (operationId && context.agentCapability) {
            await context.agentCapability.undo({ capabilityType: this.type, operationId });
        }
    }
}

export function createAgentCommand(
    type: AgentCommandType,
    jobId: string,
    params: Record<string, any> = {}
): AgentCommand {
    switch (type) {
        case 'find_asset': return new FindAssetCommand(jobId, params.query);
        case 'duplicate_asset': return new DuplicateAssetCommand(jobId, params.assetId);
        case 'place_asset': return new PlaceAssetCommand(jobId, params.assetId, params.transform || {});
        case 'transform_asset': return new TransformAssetCommand(jobId, params.assetId, params);
        case 'extract_layers': return new ExtractLayersCommand(jobId, params.assetId, params.layerIds);
        case 'object_completion': return new CapabilityCommand(jobId, type, params);
        case 'replace_material': return new CapabilityCommand(jobId, type, params);
        case 'edit_asset': return new CapabilityCommand(jobId, type, params);
        default: throw new Error(`Unsupported agent command: ${type}`);
    }
}

export function notifyWorkspace(context: AgentCommandContext): void {
    context.workspace.currentState.notify();
}
