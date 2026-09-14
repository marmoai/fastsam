import type { AssetEntity, AssetTransform } from './AssetRuntime';
import type { ProjectWorkspace } from './WorkspaceRuntime';
import { createAgentCommand, notifyWorkspace } from './AgentCommands';
import { VerificationEngine } from './VerificationEngine';

export type AgentJobStatus =
    | 'created'
    | 'planned'
    | 'running'
    | 'verifying'
    | 'waiting_confirmation'
    | 'committing'
    | 'completed'
    | 'failed'
    | 'cancelled';

export type AgentCapabilityType = 'object_completion' | 'replace_material' | 'edit_asset';
export type AgentCommandType = 'find_asset' | 'duplicate_asset' | 'place_asset' | 'transform_asset' | 'extract_layers' | AgentCapabilityType;

export interface AssetSearchQuery {
    text: string;
    projectId?: string;
    limit?: number;
}

export interface AssetSearchCandidate {
    asset?: AssetEntity;
    projectId: string;
    assetId: string;
    version: number;
    name: string;
    type: string;
    tags: string[];
}

/**
 * The Agent layer only keeps references to the extraction transaction. Pixel
 * payloads remain owned by the existing Magic Layers/asset cache pipeline.
 */
export interface AgentLayerExtractionResult {
    operationId: string;
    sourceAssetId: string;
    extractedAssetIds: string[];
    extractedLayerIds: string[];
    extractedCount: number;
    heldCount: number;
}

export interface AgentLayerExtractionRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    assetId: string;
    layerIds?: string[];
}

export interface AgentLayerExtractionUndoRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    operationId: string;
}

export interface AgentLayerExtractionCommitRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    operationIds: string[];
}

export interface AgentLayerExtractionReleaseRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    operationIds: string[];
}

export interface AgentCapabilityRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    capabilityType: AgentCapabilityType;
    targetAssetIds: string[];
    params: Record<string, any>;
}

export interface AgentCapabilityResult {
    operationId: string;
    targetAssetIds: string[];
    outputRefs: Record<string, any>;
}

export interface AgentCapabilityUndoRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    capabilityType: AgentCapabilityType;
    operationId: string;
}

export interface AgentCapabilityCommitRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    operationIds: string[];
}

export interface AgentCapabilityReleaseRequest {
    workspace: ProjectWorkspace;
    jobId: string;
    operationIds: string[];
}

export interface AgentRuntimeHost {
    getCurrentWorkspace(): ProjectWorkspace | null;
    listWorkspaces(): ProjectWorkspace[];
    listAgentAssets(): Promise<AssetSearchCandidate[]>;
    resolveAgentAsset(ref: { projectId: string; assetId: string }): Promise<AssetEntity | null>;
    verifyAgentWorkspace?(args: { workspace: ProjectWorkspace; assetIds: string[] }): Promise<AgentCommandVerificationCheck[]>;
    repairAgentWorkspace?(args: { workspace: ProjectWorkspace; assetIds: string[] }): Promise<void>;
    repairAgentJob?(args: { workspace: ProjectWorkspace; assetIds: string[]; checks: AgentCommandVerificationCheck[] }): Promise<void>;
    extractAgentLayers?(args: AgentLayerExtractionRequest): Promise<AgentLayerExtractionResult>;
    undoAgentLayers?(args: AgentLayerExtractionUndoRequest): Promise<void>;
    commitAgentLayers?(args: AgentLayerExtractionCommitRequest): Promise<void>;
    releaseAgentLayers?(args: AgentLayerExtractionReleaseRequest): Promise<void>;
    executeAgentCapability?(args: AgentCapabilityRequest): Promise<AgentCapabilityResult>;
    undoAgentCapability?(args: AgentCapabilityUndoRequest): Promise<void>;
    commitAgentCapabilities?(args: AgentCapabilityCommitRequest): Promise<void>;
    releaseAgentCapabilities?(args: AgentCapabilityReleaseRequest): Promise<void>;
    commitWorkspace(workspace: ProjectWorkspace, jobId: string): Promise<AgentCommitResult>;
}

export interface AgentCommandContext {
    jobId: string;
    workspace: ProjectWorkspace;
    assetSource: {
        listAssets(): Promise<AssetSearchCandidate[]>;
        resolveAsset(ref: { projectId: string; assetId: string }): Promise<AssetEntity | null>;
    };
    commandResults: Map<string, AgentCommandExecution>;
    requireOutput(commandType: AgentCommandType): Record<string, any>;
    agentLayerExtraction?: {
        execute(args: { assetId: string; layerIds?: string[] }): Promise<AgentLayerExtractionResult>;
        undo(args: { operationId: string }): Promise<void>;
    };
    agentCapability?: {
        execute(args: { capabilityType: AgentCapabilityType; targetAssetIds: string[]; params: Record<string, any> }): Promise<AgentCapabilityResult>;
        undo(args: { capabilityType: AgentCapabilityType; operationId: string }): Promise<void>;
    };
}

export interface AgentCommandExecution {
    targetAssetIds?: string[];
    outputRefs: Record<string, any>;
}

export interface AgentCommandVerificationCheck {
    id: string;
    passed: boolean;
    message: string;
}

export interface AgentCommandVerification {
    passed: boolean;
    checks: AgentCommandVerificationCheck[];
}

export interface AgentCommand {
    readonly commandId: string;
    readonly jobId: string;
    readonly type: AgentCommandType;
    readonly targetAssetIds: string[];
    readonly preconditions: string[];
    execute(context: AgentCommandContext): Promise<AgentCommandExecution>;
    verify(context: AgentCommandContext, execution: AgentCommandExecution): Promise<AgentCommandVerification>;
    undo(context: AgentCommandContext, execution?: AgentCommandExecution): Promise<void>;
}

export interface AgentPlanStep {
    stepId: string;
    commandType: AgentCommandType;
    label: string;
    params: Record<string, any>;
}

export interface AgentPlan {
    type: 'asset_reuse' | 'layer_extraction' | 'capability';
    steps: AgentPlanStep[];
    requiresConfirmation: boolean;
}

export interface AgentVerification {
    passed: boolean;
    checks: AgentCommandVerificationCheck[];
    checkedAt: number;
}

export interface AgentJobResult {
    assetIds: string[];
    commandResults: Array<{
        commandId: string;
        commandType: AgentCommandType;
        targetAssetIds: string[];
        outputRefs: Record<string, any>;
    }>;
    commit?: AgentCommitResult;
}

export interface AgentCommitResult {
    commitId: string;
    jobId: string;
    projectId: string;
    historyIndex: number;
    committedAt: number;
}

export interface AgentJob {
    id: string;
    goal: string;
    status: AgentJobStatus;
    plan: AgentPlan | null;
    currentStep: number;
    targetAssets: string[];
    result: AgentJobResult | null;
    verification: AgentVerification | null;
    error: { message: string; step?: number } | null;
    createdAt: number;
    updatedAt: number;
}

export interface AgentJobEvent {
    type: 'job_created' | 'job_updated';
    jobId: string;
    status: AgentJobStatus;
    currentStep: number;
    targetAssetCount: number;
    at: number;
}

interface InternalJob {
    job: AgentJob;
    workspace: ProjectWorkspace;
    commands: AgentCommand[];
    commandResults: Map<string, AgentCommandExecution>;
    executedCommands: Array<{ command: AgentCommand; execution: AgentCommandExecution }>;
    context?: AgentCommandContext;
}

function createId(prefix: string): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return `${prefix}_${crypto.randomUUID()}`;
    }
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function clone<T>(value: T): T {
    return JSON.parse(JSON.stringify(value));
}

function now(): number {
    return Date.now();
}

export class AgentRuntime {
    private readonly host: AgentRuntimeHost;
    private readonly jobs = new Map<string, InternalJob>();
    private readonly listeners = new Set<(event: AgentJobEvent) => void>();

    constructor(host: AgentRuntimeHost) {
        this.host = host;
    }

    subscribe(listener: (event: AgentJobEvent) => void): () => void {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    getDiagnostics(): { total: number; byStatus: Record<string, number> } {
        const byStatus: Record<string, number> = {};
        this.jobs.forEach(({ job }) => {
            byStatus[job.status] = (byStatus[job.status] || 0) + 1;
        });
        return { total: this.jobs.size, byStatus };
    }

    createAssetReuseJob(
        goal: string,
        query: string | AssetSearchQuery,
        options: {
            placement?: Partial<AssetTransform>;
            scale?: number;
            requiresConfirmation?: boolean;
        } = {}
    ): AgentJob {
        const workspace = this.host.getCurrentWorkspace();
        if (!workspace) throw new Error('Cannot create an Agent Job without a current workspace.');

        const jobId = createId('job');
        const searchQuery: AssetSearchQuery = typeof query === 'string' ? { text: query } : query;
        const plan: AgentPlan = {
            type: 'asset_reuse',
            requiresConfirmation: options.requiresConfirmation !== false,
            steps: [
                { stepId: createId('step'), commandType: 'find_asset', label: '查找资产', params: { query: searchQuery } },
                { stepId: createId('step'), commandType: 'duplicate_asset', label: '复制资产', params: {} },
                { stepId: createId('step'), commandType: 'place_asset', label: '放入当前 Workspace', params: { transform: options.placement || {} } },
                ...(options.scale !== undefined
                    ? [{ stepId: createId('step'), commandType: 'transform_asset' as const, label: '调整资产尺寸', params: { scale: options.scale } }]
                    : [])
            ]
        };
        const job: AgentJob = {
            id: jobId,
            goal,
            status: 'planned',
            plan,
            currentStep: -1,
            targetAssets: [],
            result: null,
            verification: null,
            error: null,
            createdAt: now(),
            updatedAt: now()
        };
        this.jobs.set(jobId, {
            job,
            workspace,
            commands: [],
            commandResults: new Map(),
            executedCommands: []
        });
        this.emit({ type: 'job_created', jobId, status: job.status, currentStep: job.currentStep, targetAssetCount: 0, at: now() });
        return this.getJob(jobId)!;
    }

    createLayerExtractionJob(
        goal: string,
        assetId: string,
        options: {
            layerIds?: string[];
            requiresConfirmation?: boolean;
        } = {}
    ): AgentJob {
        const workspace = this.host.getCurrentWorkspace();
        if (!workspace) throw new Error('Cannot create an Agent Job without a current workspace.');
        if (!assetId) throw new Error('Layer extraction requires a source asset.');

        const jobId = createId('job');
        const plan: AgentPlan = {
            type: 'layer_extraction',
            requiresConfirmation: options.requiresConfirmation !== false,
            steps: [{
                stepId: createId('step'),
                commandType: 'extract_layers',
                label: '提取 Magic Layers 图层',
                params: { assetId, layerIds: options.layerIds || [] }
            }]
        };
        const job: AgentJob = {
            id: jobId,
            goal,
            status: 'planned',
            plan,
            currentStep: -1,
            targetAssets: [],
            result: null,
            verification: null,
            error: null,
            createdAt: now(),
            updatedAt: now()
        };
        this.jobs.set(jobId, {
            job,
            workspace,
            commands: [],
            commandResults: new Map(),
            executedCommands: []
        });
        this.emit({ type: 'job_created', jobId, status: job.status, currentStep: job.currentStep, targetAssetCount: 0, at: now() });
        return this.getJob(jobId)!;
    }

    createCapabilityJob(
        goal: string,
        capabilityType: AgentCapabilityType,
        params: Record<string, any> = {},
        options: { targetAssetIds?: string[]; requiresConfirmation?: boolean } = {}
    ): AgentJob {
        const workspace = this.host.getCurrentWorkspace();
        if (!workspace) throw new Error('Cannot create an Agent Job without a current workspace.');
        const jobId = createId('job');
        const targetAssetIds = Array.isArray(options.targetAssetIds)
            ? options.targetAssetIds.filter(Boolean)
            : [params.childId, params.itemId].filter(Boolean);
        const labels: Record<AgentCapabilityType, string> = {
            object_completion: '执行对象补全',
            replace_material: '替换资产材质',
            edit_asset: params.mode === 'current_layer' ? '更新当前图层' : '生成对象新版本'
        };
        const plan: AgentPlan = {
            type: 'capability',
            requiresConfirmation: options.requiresConfirmation !== false,
            steps: [{
                stepId: createId('step'),
                commandType: capabilityType,
                label: labels[capabilityType],
                params: { ...params, targetAssetIds }
            }]
        };
        const job: AgentJob = {
            id: jobId,
            goal,
            status: 'planned',
            plan,
            currentStep: -1,
            targetAssets: [],
            result: null,
            verification: null,
            error: null,
            createdAt: now(),
            updatedAt: now()
        };
        this.jobs.set(jobId, {
            job,
            workspace,
            commands: [],
            commandResults: new Map(),
            executedCommands: []
        });
        this.emit({ type: 'job_created', jobId, status: job.status, currentStep: job.currentStep, targetAssetCount: 0, at: now() });
        return this.getJob(jobId)!;
    }

    createObjectCompletionJob(
        goal: string,
        itemId: string,
        completionAssetId: string,
        options: { requiresConfirmation?: boolean } = {}
    ): AgentJob {
        return this.createCapabilityJob(goal, 'object_completion', { itemId, completionAssetId }, options);
    }

    createMaterialReplacementJob(
        goal: string,
        childId: string,
        prompt: string,
        options: { requiresConfirmation?: boolean } = {}
    ): AgentJob {
        return this.createCapabilityJob(goal, 'replace_material', { childId, prompt }, {
            ...options,
            targetAssetIds: [childId]
        });
    }

    createObjectEditJob(
        goal: string,
        itemId: string,
        prompt: string,
        mode: 'variant' | 'current_layer',
        options: { requiresConfirmation?: boolean } = {}
    ): AgentJob {
        return this.createCapabilityJob(goal, 'edit_asset', { itemId, prompt, mode }, {
            ...options,
            targetAssetIds: [itemId]
        });
    }

    getJob(jobId: string): AgentJob | null {
        const internal = this.jobs.get(jobId);
        return internal ? clone(internal.job) : null;
    }

    listJobs(): AgentJob[] {
        return [...this.jobs.values()]
            .map(({ job }) => clone(job))
            .sort((a, b) => b.createdAt - a.createdAt);
    }

    async executeJob(jobId: string): Promise<AgentJob> {
        const internal = this.getInternalJob(jobId);
        const { job } = internal;
        if (job.status === 'completed') return this.getJob(jobId)!;
        if (job.status === 'cancelled') throw new Error(`Job ${jobId} is cancelled.`);
        if (!job.plan) throw new Error(`Job ${jobId} has no plan.`);

        if (job.status === 'waiting_confirmation') return this.getJob(jobId)!;
        job.status = 'running';
        job.error = null;
        this.touch(job);
        internal.commands = job.plan.steps.map(step => createAgentCommand(step.commandType, job.id, step.params));

        const context: AgentCommandContext = {
            jobId: job.id,
            workspace: internal.workspace,
            assetSource: {
                listAssets: () => this.host.listAgentAssets(),
                resolveAsset: (ref) => this.host.resolveAgentAsset(ref)
            },
            commandResults: internal.commandResults,
            requireOutput: (commandType) => {
                const command = internal.commands.find(candidate => candidate.type === commandType);
                const result = command ? internal.commandResults.get(command.commandId) : undefined;
                if (!result) throw new Error(`Required command output is missing: ${commandType}`);
                return result.outputRefs;
            }
        };
        if (this.host.extractAgentLayers) {
            context.agentLayerExtraction = {
                execute: ({ assetId, layerIds }) => this.host.extractAgentLayers!({
                    workspace: internal.workspace,
                    jobId: job.id,
                    assetId,
                    layerIds
                }),
                undo: ({ operationId }) => this.host.undoAgentLayers
                    ? this.host.undoAgentLayers({ workspace: internal.workspace, jobId: job.id, operationId })
                    : Promise.resolve()
            };
        }
        if (this.host.executeAgentCapability) {
            context.agentCapability = {
                execute: ({ capabilityType, targetAssetIds, params }) => this.host.executeAgentCapability!({
                    workspace: internal.workspace,
                    jobId: job.id,
                    capabilityType,
                    targetAssetIds,
                    params
                }),
                undo: ({ capabilityType, operationId }) => this.host.undoAgentCapability
                    ? this.host.undoAgentCapability({
                        workspace: internal.workspace,
                        jobId: job.id,
                        capabilityType,
                        operationId
                    })
                    : Promise.resolve()
            };
        }
        internal.context = context;

        try {
            for (let index = 0; index < internal.commands.length; index += 1) {
                const command = internal.commands[index];
                job.currentStep = index;
                this.touch(job);
                let execution: AgentCommandExecution | undefined;
                try {
                    execution = await command.execute(context);
                    const verification = await command.verify(context, execution);
                    if (!verification.passed) {
                        await command.undo(context, execution);
                        execution = undefined;
                        throw new Error(verification.checks.filter(check => !check.passed).map(check => check.message).join(' ') || `${command.type} verification failed.`);
                    }
                } catch (error) {
                    if (execution) {
                        try {
                            await command.undo(context, execution);
                        } catch (undoError) {
                            console.error(`[AgentRuntime] Failed to undo current ${command.type}:`, undoError);
                        }
                    }
                    throw error;
                }
                internal.commandResults.set(command.commandId, execution);
                internal.executedCommands.push({ command, execution });
                if (execution.targetAssetIds) {
                    job.targetAssets = [...new Set([...job.targetAssets, ...execution.targetAssetIds])];
                }
            }

            notifyWorkspace(context);
            job.status = 'verifying';
            this.touch(job);
            job.verification = await this.verifyJob(internal);
            if (!job.verification.passed) {
                job.result = this.buildJobResult(internal);
                job.status = 'waiting_confirmation';
                job.error = {
                    message: job.verification.checks.filter(check => !check.passed).map(check => check.message).join(' ') || 'Job verification failed.',
                    step: job.currentStep
                };
                this.touch(job);
                return this.getJob(jobId)!;
            }
            job.result = this.buildJobResult(internal);
            job.status = job.plan.requiresConfirmation ? 'waiting_confirmation' : 'committing';
            this.touch(job);
            if (!job.plan.requiresConfirmation) await this.commitJob(jobId);
            return this.getJob(jobId)!;
        } catch (error: any) {
            await this.rollback(internal, context);
            notifyWorkspace(context);
            job.status = 'failed';
            job.error = { message: error?.message || String(error), step: job.currentStep };
            this.touch(job);
            return this.getJob(jobId)!;
        }
    }

    async confirmJob(jobId: string): Promise<AgentJob> {
        const internal = this.getInternalJob(jobId);
        if (internal.job.status === 'completed') return this.getJob(jobId)!;
        if (internal.job.status !== 'waiting_confirmation') {
            throw new Error(`Job ${jobId} is not waiting for confirmation.`);
        }
        if (!internal.job.verification?.passed) {
            throw new Error(`Job ${jobId} cannot be committed before verification passes.`);
        }
        internal.job.status = 'committing';
        this.touch(internal.job);
        return this.commitJob(jobId);
    }

    async retryJob(jobId: string): Promise<AgentJob> {
        const internal = this.getInternalJob(jobId);
        if (internal.job.status === 'completed') return this.getJob(jobId)!;
        if (!['failed', 'waiting_confirmation'].includes(internal.job.status)) {
            throw new Error(`Job ${jobId} is not ready to retry.`);
        }
        if (internal.job.status === 'waiting_confirmation' && internal.job.verification?.passed) {
            throw new Error(`Job ${jobId} is waiting for confirmation, not retry.`);
        }
        if (internal.context && internal.executedCommands.length > 0) {
            await this.rollback(internal, internal.context);
            notifyWorkspace(internal.context);
        }
        internal.job.status = 'planned';
        internal.job.currentStep = -1;
        internal.job.result = null;
        internal.job.verification = null;
        internal.job.error = null;
        this.touch(internal.job);
        return this.executeJob(jobId);
    }

    async commitJob(jobId: string): Promise<AgentJob> {
        const internal = this.getInternalJob(jobId);
        const { job } = internal;
        if (job.status === 'completed') return this.getJob(jobId)!;
        if (job.status !== 'committing') throw new Error(`Job ${jobId} is not ready to commit.`);
        if (!job.verification?.passed || !job.result) throw new Error(`Job ${jobId} cannot commit before verification passes.`);

        try {
            const operationIds = internal.commands
                .map(command => internal.commandResults.get(command.commandId)?.outputRefs?.operationId)
                .filter(Boolean);
            if (operationIds.length > 0 && this.host.commitAgentLayers) {
                await this.host.commitAgentLayers({
                    workspace: internal.workspace,
                    jobId: job.id,
                    operationIds
                });
            }
            const capabilityOperationIds = internal.commands
                .filter(command => command.type === 'object_completion' || command.type === 'replace_material' || command.type === 'edit_asset')
                .map(command => internal.commandResults.get(command.commandId)?.outputRefs?.operationId)
                .filter(Boolean);
            if (capabilityOperationIds.length > 0 && this.host.commitAgentCapabilities) {
                await this.host.commitAgentCapabilities({
                    workspace: internal.workspace,
                    jobId: job.id,
                    operationIds: capabilityOperationIds
                });
            }
            job.result.commit = await this.host.commitWorkspace(internal.workspace, job.id);
            job.status = 'completed';
            this.touch(job);
            if (operationIds.length > 0 && this.host.releaseAgentLayers) {
                try {
                    await this.host.releaseAgentLayers({
                        workspace: internal.workspace,
                        jobId: job.id,
                        operationIds
                    });
                } catch (releaseError) {
                    // Cleanup failure must not turn an already durable Commit
                    // into a rollback attempt.
                    console.error('[AgentRuntime] Failed to release extraction transaction:', releaseError);
                }
            }
            if (capabilityOperationIds.length > 0 && this.host.releaseAgentCapabilities) {
                try {
                    await this.host.releaseAgentCapabilities({
                        workspace: internal.workspace,
                        jobId: job.id,
                        operationIds: capabilityOperationIds
                    });
                } catch (releaseError) {
                    console.error('[AgentRuntime] Failed to release capability transaction:', releaseError);
                }
            }
            return this.getJob(jobId)!;
        } catch (error: any) {
            if (internal.context && internal.executedCommands.length > 0) {
                await this.rollback(internal, internal.context);
                notifyWorkspace(internal.context);
            }
            job.status = 'failed';
            job.error = { message: error?.message || String(error), step: job.currentStep };
            this.touch(job);
            return this.getJob(jobId)!;
        }
    }

    async cancelJob(jobId: string): Promise<AgentJob> {
        const internal = this.getInternalJob(jobId);
        if (internal.job.status === 'completed') throw new Error(`Completed job ${jobId} cannot be cancelled.`);
        if (internal.context && internal.executedCommands.length > 0) {
            await this.rollback(internal, internal.context);
            notifyWorkspace(internal.context);
        }
        internal.job.status = 'cancelled';
        this.touch(internal.job);
        return this.getJob(jobId)!;
    }

    private async verifyJob(internal: InternalJob): Promise<AgentVerification> {
        const initial = VerificationEngine.verifyWorkspace(internal.workspace, internal.job.targetAssets);
        let structuralChecks = initial.checks;
        let externalChecks = internal.context && this.host.verifyAgentWorkspace
            ? await this.host.verifyAgentWorkspace({ workspace: internal.workspace, assetIds: internal.job.targetAssets })
            : [];
        const needsRepair = [...structuralChecks, ...externalChecks].some(check => !check.passed);
        if (needsRepair && internal.context && (this.host.repairAgentWorkspace || this.host.repairAgentJob)) {
            if (this.host.repairAgentWorkspace) {
                await this.host.repairAgentWorkspace({ workspace: internal.workspace, assetIds: internal.job.targetAssets });
            }
            if (this.host.repairAgentJob) {
                await this.host.repairAgentJob({
                    workspace: internal.workspace,
                    assetIds: internal.job.targetAssets,
                    checks: [...structuralChecks, ...externalChecks]
                });
            }
            const repaired = VerificationEngine.verifyWorkspace(internal.workspace, internal.job.targetAssets);
            structuralChecks = repaired.checks;
            externalChecks = this.host.verifyAgentWorkspace
                ? await this.host.verifyAgentWorkspace({ workspace: internal.workspace, assetIds: internal.job.targetAssets })
                : externalChecks;
            const repairPassed = [...structuralChecks, ...externalChecks].every(check => check.passed);
            return {
                passed: repairPassed,
                checks: [
                    ...structuralChecks,
                    ...externalChecks,
                    {
                        id: 'agent_auto_repair',
                        passed: repairPassed,
                        message: repairPassed ? 'Verification passed after one automatic repair.' : 'Verification still fails after one automatic repair.'
                    }
                ],
                checkedAt: now()
            };
        }
        const checks = [...structuralChecks, ...externalChecks];
        return { passed: checks.every(check => check.passed), checks, checkedAt: now() };
    }

    private buildJobResult(internal: InternalJob): AgentJobResult {
        return {
            assetIds: [...internal.job.targetAssets],
            commandResults: internal.commands.map(command => {
                const execution = internal.commandResults.get(command.commandId)!;
                return {
                    commandId: command.commandId,
                    commandType: command.type,
                    targetAssetIds: execution.targetAssetIds || command.targetAssetIds,
                    outputRefs: clone(execution.outputRefs)
                };
            })
        };
    }

    private async rollback(internal: InternalJob, context: AgentCommandContext): Promise<void> {
        for (const { command, execution } of [...internal.executedCommands].reverse()) {
            try {
                await command.undo(context, execution);
            } catch (error) {
                console.error(`[AgentRuntime] Failed to undo ${command.type}:`, error);
            }
        }
        internal.executedCommands = [];
        internal.commandResults.clear();
        internal.job.targetAssets = [];
    }

    private getInternalJob(jobId: string): InternalJob {
        const internal = this.jobs.get(jobId);
        if (!internal) throw new Error(`Agent Job ${jobId} not found.`);
        return internal;
    }

    private touch(job: AgentJob): void {
        job.updatedAt = now();
        this.emit({
            type: 'job_updated',
            jobId: job.id,
            status: job.status,
            currentStep: job.currentStep,
            targetAssetCount: job.targetAssets.length,
            at: job.updatedAt
        });
    }

    private emit(event: AgentJobEvent): void {
        this.listeners.forEach(listener => {
            try {
                listener(event);
            } catch (error) {
                console.error('[AgentRuntime] Job event listener failed:', error);
            }
        });
    }
}
