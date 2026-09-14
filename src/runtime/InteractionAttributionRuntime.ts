export type AttributionSourceType = 'manual' | 'recommendation' | 'prompt_completion' | 'magic_layers' | 'workflow_action';

export interface AttributionSource {
    sourceType: AttributionSourceType;
    sourceId?: string;
    timestamp: number;
    assetUid?: string;
    sessionId?: string;
    projectId?: string;
    metadata?: Record<string, any>;
}

function buildSourceId(prefix: string) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

class InteractionAttributionRuntime {
    private readonly assetSourceMap = new Map<string, AttributionSource>();
    private latestGlobalSource: AttributionSource | null = null;
    private readonly sourceTtlMs = 15 * 60 * 1000;

    private isExpired(source?: AttributionSource | null) {
        if (!source) return true;
        return Date.now() - source.timestamp > this.sourceTtlMs;
    }

    private cloneSource(source: AttributionSource | null) {
        if (!source) return null;
        return {
            ...source,
            metadata: source.metadata ? { ...source.metadata } : undefined
        };
    }

    private setAssetSource(assetUid: string | undefined, source: AttributionSource) {
        if (!assetUid) return;
        this.assetSourceMap.set(assetUid, source);
    }

    registerPromptCompletionShown(payload: {
        matchId?: string;
        initialPrompt?: string;
        finalPrompt?: string;
        typedInput?: string;
        presetTags?: string[];
        sessionId?: string;
        projectId?: string;
    }) {
        return {
            sourceType: 'prompt_completion' as const,
            sourceId: buildSourceId('completion'),
            timestamp: Date.now(),
            sessionId: payload.sessionId,
            projectId: payload.projectId,
            metadata: {
                matchId: payload.matchId || null,
                initialPrompt: payload.initialPrompt || null,
                finalPrompt: payload.finalPrompt || null,
                typedInput: payload.typedInput || null,
                presetTags: payload.presetTags || []
            }
        };
    }

    acceptPromptCompletion(source: AttributionSource | null) {
        if (!source) return null;
        const accepted = {
            ...source,
            timestamp: Date.now()
        };
        this.latestGlobalSource = accepted;
        return this.cloneSource(accepted);
    }

    registerRecommendationApplied(payload: {
        assetUid?: string;
        patternId?: string;
        patternName?: string;
        semanticType?: string;
        sessionId?: string;
        projectId?: string;
    }) {
        const source: AttributionSource = {
            sourceType: 'recommendation',
            sourceId: buildSourceId('recommendation'),
            timestamp: Date.now(),
            assetUid: payload.assetUid,
            sessionId: payload.sessionId,
            projectId: payload.projectId,
            metadata: {
                patternId: payload.patternId || null,
                patternName: payload.patternName || null,
                semanticType: payload.semanticType || null
            }
        };
        this.setAssetSource(payload.assetUid, source);
        this.latestGlobalSource = source;
        return this.cloneSource(source);
    }

    bindSourceToAsset(assetUid: string | undefined, source: AttributionSource | null) {
        if (!assetUid || !source) return null;
        const bound = {
            ...source,
            assetUid,
            timestamp: Date.now()
        };
        this.setAssetSource(assetUid, bound);
        return this.cloneSource(bound);
    }

    clearAssetSource(assetUid: string | undefined) {
        if (!assetUid) return;
        this.assetSourceMap.delete(assetUid);
    }

    resolveSource(options: {
        assetUid?: string;
        fallbackSourceType?: AttributionSourceType;
        preferGlobal?: boolean;
    } = {}) {
        const fallbackSourceType = options.fallbackSourceType || 'manual';
        const assetSource = options.assetUid ? this.assetSourceMap.get(options.assetUid) : null;

        if (assetSource && !this.isExpired(assetSource)) {
            return this.cloneSource(assetSource);
        }

        if (assetSource && this.isExpired(assetSource)) {
            this.assetSourceMap.delete(options.assetUid as string);
        }

        if (options.preferGlobal !== false && this.latestGlobalSource && !this.isExpired(this.latestGlobalSource)) {
            return this.cloneSource(this.latestGlobalSource);
        }

        if (this.latestGlobalSource && this.isExpired(this.latestGlobalSource)) {
            this.latestGlobalSource = null;
        }

        return {
            sourceType: fallbackSourceType,
            sourceId: undefined,
            timestamp: Date.now(),
            metadata: undefined
        };
    }
}

export const interactionAttributionRuntime = new InteractionAttributionRuntime();

if (typeof window !== 'undefined') {
    // @ts-ignore
    window.interactionAttributionRuntime = interactionAttributionRuntime;
}
