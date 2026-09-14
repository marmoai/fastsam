import localforage from 'localforage';
import { normalizeTaskType } from './taskBuckets';

export type ResultEventType =
    | 'recommendation_shown'
    | 'recommendation_accepted'
    | 'recommendation_executed'
    | 'recommendation_failed'
    | 'result_selected'
    | 'result_exported'
    | 'result_saved_to_asset_library'
    | 'result_reused'
    | 'result_deleted'
    | 'result_reverted';

export interface ResultEventContext {
    taskType?: string;
    semanticType?: string;
    designRole?: string;
    workflowStage?: string;
    layerName?: string;
}

export interface ResultEvent {
    id: string;
    timestamp: number;
    type: ResultEventType;
    assetUid?: string;
    layerId?: string;
    schemeId?: string;
    sessionId?: string;
    projectId?: string;
    sourceType?: 'manual' | 'recommendation' | 'prompt_completion' | 'magic_layers' | 'workflow_action';
    sourceId?: string;
    context?: ResultEventContext;
    metadata?: Record<string, any>;
}

export interface SourceFeedbackInsights {
    score: number;
    totalEvents: number;
    positiveEvents: number;
    negativeEvents: number;
    selectedCount: number;
    exportedCount: number;
    savedCount: number;
    reusedCount: number;
    deletedCount: number;
    revertedCount: number;
    recommendationShownCount: number;
    recommendationAcceptedCount: number;
    recommendationExecutedCount: number;
    recommendationFailedCount: number;
    acceptanceRate: number | null;
    topWorkflowStage?: string;
    recentPositiveEvents: number;
    recentNegativeEvents: number;
    shortLabel: string;
    detail: string;
}

const EVENT_WEIGHTS: Record<ResultEventType, number> = {
    recommendation_shown: 0,
    recommendation_accepted: 1,
    recommendation_executed: 8,
    recommendation_failed: -6,
    result_selected: 4,
    result_exported: 6,
    result_saved_to_asset_library: 5,
    result_reused: 5,
    result_deleted: -4,
    result_reverted: -3
};

class ResultFeedbackRuntime {
    private readonly storageKey = 'mvr_result_feedback_events';
    private events: ResultEvent[] = [];
    private readonly maxEvents = 400;
    private loaded = false;

    constructor() {
        this.loadFromStorage();
    }

    private async loadFromStorage() {
        try {
            const stored = await localforage.getItem<ResultEvent[]>(this.storageKey);
            if (Array.isArray(stored)) {
                this.events = stored.slice(-this.maxEvents);
            }
            this.loaded = true;
            console.log(`[ResultFeedback] Restored ${this.events.length} result events.`);
        } catch (error) {
            console.error('[ResultFeedback] Failed to load result events:', error);
        }
    }

    private async saveToStorage() {
        try {
            this.events = this.events.slice(-this.maxEvents);
            await localforage.setItem(this.storageKey, this.events);
        } catch (error) {
            console.error('[ResultFeedback] Failed to save result events:', error);
        }
    }

    recordEvent(input: Omit<ResultEvent, 'id' | 'timestamp'>) {
        const event: ResultEvent = {
            id: `result_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
            timestamp: Date.now(),
            ...input
        };

        this.events.push(event);
        void this.saveToStorage();
        console.log('[ResultFeedback] Recorded event:', event);
        return event;
    }

    getEvents(): ResultEvent[] {
        return [...this.events];
    }

    getEventsByType(type: ResultEventType): ResultEvent[] {
        return this.events.filter(event => event.type === type);
    }

    private getEventWeight(event: ResultEvent) {
        return EVENT_WEIGHTS[event.type] || 0;
    }

    private getAttributionMetadata(event: ResultEvent) {
        return (event.metadata?.attributionMetadata || null) as Record<string, any> | null;
    }

    private matchesContext(event: ResultEvent, filters?: Partial<ResultEventContext>) {
        if (!filters) return true;
        return Object.entries(filters).every(([key, value]) => {
            if (!value) return true;
            if (key === 'taskType') {
                return normalizeTaskType(event.context?.taskType) === normalizeTaskType(String(value));
            }
            return event.context?.[key as keyof ResultEventContext] === value;
        });
    }

    private buildInsights(
        matcher: (event: ResultEvent, attribution: Record<string, any> | null) => boolean,
        filters?: Partial<ResultEventContext>
    ): SourceFeedbackInsights {
        const matchedEvents = this.events.filter((event) => {
            const attribution = this.getAttributionMetadata(event);
            return matcher(event, attribution) && this.matchesContext(event, filters);
        });

        const workflowStageCounts: Record<string, number> = {};
        const now = Date.now();
        const recentWindowMs = 14 * 24 * 60 * 60 * 1000;

        const summary = matchedEvents.reduce((acc, event) => {
            const weight = this.getEventWeight(event);
            const isPositive = weight > 0;
            const isNegative = weight < 0;
            const stage = event.context?.workflowStage;

            acc.score += weight;
            acc.totalEvents += 1;
            if (isPositive) acc.positiveEvents += 1;
            if (isNegative) acc.negativeEvents += 1;
            if (stage) {
                workflowStageCounts[stage] = (workflowStageCounts[stage] || 0) + 1;
            }
            if (now - event.timestamp <= recentWindowMs) {
                if (isPositive) acc.recentPositiveEvents += 1;
                if (isNegative) acc.recentNegativeEvents += 1;
            }

            switch (event.type) {
                case 'recommendation_shown':
                    acc.recommendationShownCount += 1;
                    break;
                case 'recommendation_accepted':
                    acc.recommendationAcceptedCount += 1;
                    break;
                case 'recommendation_executed':
                    acc.recommendationExecutedCount += 1;
                    break;
                case 'recommendation_failed':
                    acc.recommendationFailedCount += 1;
                    break;
                case 'result_selected':
                    acc.selectedCount += 1;
                    break;
                case 'result_exported':
                    acc.exportedCount += 1;
                    break;
                case 'result_saved_to_asset_library':
                    acc.savedCount += 1;
                    break;
                case 'result_reused':
                    acc.reusedCount += 1;
                    break;
                case 'result_deleted':
                    acc.deletedCount += 1;
                    break;
                case 'result_reverted':
                    acc.revertedCount += 1;
                    break;
            }

            return acc;
        }, {
            score: 0,
            totalEvents: 0,
            positiveEvents: 0,
            negativeEvents: 0,
            selectedCount: 0,
            exportedCount: 0,
            savedCount: 0,
            reusedCount: 0,
            deletedCount: 0,
            revertedCount: 0,
            recommendationShownCount: 0,
            recommendationAcceptedCount: 0,
            recommendationExecutedCount: 0,
            recommendationFailedCount: 0,
            recentPositiveEvents: 0,
            recentNegativeEvents: 0
        });

        const decisionCount = summary.positiveEvents + summary.negativeEvents;
        const acceptanceRate = decisionCount > 0 ? summary.positiveEvents / decisionCount : null;
        const topWorkflowStage = Object.entries(workflowStageCounts).sort((a, b) => b[1] - a[1])[0]?.[0];

        let shortLabel = '新推荐';
        if (summary.exportedCount > 0 && (acceptanceRate ?? 0) >= 0.7) {
            shortLabel = '导出表现强';
        } else if (summary.savedCount > 0 && (acceptanceRate ?? 0) >= 0.7) {
            shortLabel = '常入资产库';
        } else if ((acceptanceRate ?? 0) >= 0.75 && summary.totalEvents >= 2) {
            shortLabel = '高采纳';
        } else if (summary.recentPositiveEvents >= 2 && summary.recentPositiveEvents > summary.recentNegativeEvents) {
            shortLabel = '最近常用';
        } else if (summary.negativeEvents > summary.positiveEvents && summary.totalEvents >= 2) {
            shortLabel = '近期回退多';
        }

        const detailParts = [];
        if (summary.exportedCount > 0) detailParts.push(`导出 ${summary.exportedCount}`);
        if (summary.savedCount > 0) detailParts.push(`入库 ${summary.savedCount}`);
        if (summary.selectedCount > 0) detailParts.push(`选用 ${summary.selectedCount}`);
        if (summary.recommendationExecutedCount > 0) detailParts.push(`执行 ${summary.recommendationExecutedCount}`);
        if (summary.recommendationAcceptedCount > 0) detailParts.push(`采纳 ${summary.recommendationAcceptedCount}`);
        if (summary.deletedCount > 0 || summary.revertedCount > 0) {
            detailParts.push(`回退 ${summary.deletedCount + summary.revertedCount}`);
        }

        return {
            ...summary,
            acceptanceRate,
            topWorkflowStage,
            shortLabel,
            detail: detailParts.length > 0 ? detailParts.join(' · ') : '暂无结果反馈'
        };
    }

    getScoreForPromptRule(matchId?: string): number {
        if (!matchId) return 0;
        return this.buildInsights((event, attribution) => {
            return event.sourceType === 'prompt_completion' && attribution?.matchId === matchId;
        }).score;
    }

    getScoreForRecommendationPattern(patternId?: string): number {
        if (!patternId) return 0;
        return this.buildInsights((event, attribution) => {
            return event.sourceType === 'recommendation' && attribution?.patternId === patternId;
        }).score;
    }

    getPromptRuleInsights(matchId?: string, filters?: Partial<ResultEventContext>): SourceFeedbackInsights {
        if (!matchId) {
            return this.buildInsights(() => false, filters);
        }
        return this.buildInsights((event, attribution) => {
            return event.sourceType === 'prompt_completion' && attribution?.matchId === matchId;
        }, filters);
    }

    getRecommendationInsights(patternId?: string, filters?: Partial<ResultEventContext>): SourceFeedbackInsights {
        if (!patternId) {
            return this.buildInsights(() => false, filters);
        }
        return this.buildInsights((event, attribution) => {
            return event.sourceType === 'recommendation' && attribution?.patternId === patternId;
        }, filters);
    }

    isLoaded() {
        return this.loaded;
    }
}

export const resultFeedbackRuntime = new ResultFeedbackRuntime();

if (typeof window !== 'undefined') {
    // @ts-ignore
    window.resultFeedbackRuntime = resultFeedbackRuntime;
}
