import { describe, expect, it, vi } from 'vitest';
import {
    executeLensSearch,
    looksLikeLensSearchRequest,
    resolveLensSearchTarget
} from './capability-orchestrator.js';

function createState() {
    const item = {
        name: '客厅参考图',
        scene: {
            layers: [
                { id: 'sofa', name: '红色沙发', semanticType: 'furniture', bbox: [100, 100, 600, 700] },
                { id: 'lamp', name: '吊灯', semanticType: 'lighting', bbox: [50, 700, 350, 950] },
                { id: 'background', name: '背景', category: 'background', bbox: [0, 0, 1000, 1000] }
            ]
        },
        layerStates: new Map([[0, { selected: true }], [1, { selected: false }]])
    };

    return {
        currentActiveWorkbenchItemId: 'image-1',
        selectedWorkbenchItems: new Set(['image-1']),
        workbenchItems: new Map([['image-1', item]])
    };
}

describe('Lens capability orchestration', () => {
    it('recognizes same-product search requests', () => {
        expect(looksLikeLensSearchRequest('搜索沙发同款')).toBe(true);
        expect(looksLikeLensSearchRequest('把沙发改成蓝色')).toBe(false);
    });

    it('prefers the selected semantic layer as the Lens target', () => {
        const target = resolveLensSearchTarget(createState(), '搜索沙发同款');

        expect(target.ok).toBe(true);
        expect(target.source).toBe('selected_layer');
        expect(target.layer.id).toBe('sofa');
        expect(target.layer.bbox).toEqual([100, 100, 600, 700]);
    });

    it('keeps selection indexes aligned when semantic views omit background layers', () => {
        const state = createState();
        const item = state.workbenchItems.get('image-1');
        item.semanticViews = {
            editableSceneLayers: [item.scene.layers[0], item.scene.layers[1]]
        };

        const target = resolveLensSearchTarget(state, '搜索沙发同款');

        expect(target.ok).toBe(true);
        expect(target.source).toBe('selected_layer');
        expect(target.layer.id).toBe('sofa');
    });

    it('uses the explicit active layer id even when layer selection state is unavailable', () => {
        const state = createState();
        const context = {
            activeItemId: 'image-1',
            activeLayerId: 'lamp',
            selectedItemIds: ['image-1']
        };

        const target = resolveLensSearchTarget(state, '搜索同款', context);

        expect(target.ok).toBe(true);
        expect(target.source).toBe('active_layer_id');
        expect(target.layer.id).toBe('lamp');
    });

    it('dispatches the resolved bbox to the existing Lens implementation', async () => {
        const state = createState();
        const performVisualSearch = vi.fn().mockResolvedValue({ success: true });
        globalThis.window = { marmoLens: { performVisualSearch } };

        const result = await executeLensSearch(state, '搜索沙发同款');

        expect(result.success).toBe(true);
        expect(performVisualSearch).toHaveBeenCalledWith(
            state.workbenchItems.get('image-1'),
            expect.objectContaining({ box: [100, 100, 600, 700], label: '红色沙发' })
        );
    });

    it('uses a selected extracted asset as a full-image Lens target', async () => {
        const extractedAsset = {
            name: '拆解-红色沙发',
            layerName: '红色沙发',
            type: 'layer-explode',
            parentId: 'image-1',
            dataUrl: 'data:image/png;base64,transparent-cutout'
        };
        const state = {
            currentActiveWorkbenchItemId: 'asset-1',
            selectedWorkbenchItems: new Set(['asset-1']),
            workbenchItems: new Map([['asset-1', extractedAsset]])
        };
        const target = resolveLensSearchTarget(state, '搜索同款');

        expect(target.ok).toBe(true);
        expect(target.source).toBe('standalone_asset');
        expect(target.layer.name).toBe('红色沙发');
        expect(target.layer.bbox).toEqual([0, 0, 1000, 1000]);

        const performVisualSearch = vi.fn().mockResolvedValue({ success: true });
        globalThis.window = { marmoLens: { performVisualSearch } };
        const result = await executeLensSearch(state, '搜索同款');

        expect(result.success).toBe(true);
        expect(performVisualSearch).toHaveBeenCalledWith(
            extractedAsset,
            expect.objectContaining({
                box: [0, 0, 1000, 1000],
                label: '红色沙发',
                layerId: 'asset-asset-1'
            })
        );
    });
});
