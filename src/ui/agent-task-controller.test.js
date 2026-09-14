import { afterEach, describe, expect, it } from 'vitest';
import { isAgentRuntimeFeatureEnabled } from '../core/config.js';
import {
    isAssetReuseRequest,
    parseAssetReuseRequest,
    isLayerExtractionRequest,
    resolveLayerExtractionRequest
} from './agent-task-controller.js';

afterEach(() => {
    delete globalThis.__marmoFeatureFlags;
});

describe('Chat Agent task bridge', () => {
    it('recognizes asset reuse requests and extracts the asset plus scale', () => {
        const text = '把上次项目中的沙发放进当前房间，缩小 15%。';

        expect(isAssetReuseRequest(text)).toBe(true);
        expect(parseAssetReuseRequest(text)).toMatchObject({
            query: '沙发',
            scale: 0.85
        });
    });

    it('accepts the shorter visual-target phrasing used by the workspace hint', () => {
        expect(parseAssetReuseRequest('把上次项目的沙发放进当前场景')).toMatchObject({
            query: '沙发'
        });
    });

    it('does not classify ordinary image prompts as asset reuse', () => {
        expect(isAssetReuseRequest('生成一张有沙发的客厅效果图')).toBe(false);
        expect(isAssetReuseRequest('把这张图改成夜景')).toBe(false);
    });

    it('keeps the feature disabled unless explicitly overridden', () => {
        expect(isAgentRuntimeFeatureEnabled('assetReuseFromChat')).toBe(false);
        globalThis.__marmoFeatureFlags = { assetReuseFromChat: true };
        expect(isAgentRuntimeFeatureEnabled('assetReuseFromChat')).toBe(true);
    });

    it('resolves a natural-language layer extraction request to semantic layer IDs', () => {
        const appState = {
            currentActiveWorkbenchItemId: 'scene_001',
            workspaceContext: {},
            selectedWorkbenchItems: new Set(),
            workbenchItems: new Map([['scene_001', {
                scene: {
                    layers: [
                        { id: 'sofa_layer', name: '米白沙发', semanticType: 'product' },
                        { id: 'lamp_layer', name: '落地灯', semanticType: 'product' },
                        { id: 'background_layer', name: '背景墙', category: 'background' }
                    ]
                },
                layerStates: new Map()
            }]])
        };

        expect(isLayerExtractionRequest('帮我把沙发和灯具提取出来')).toBe(true);
        expect(resolveLayerExtractionRequest('帮我把沙发和灯具提取出来', appState)).toMatchObject({
            ok: true,
            itemId: 'scene_001',
            layerIds: ['sofa_layer', 'lamp_layer']
        });
    });
});
