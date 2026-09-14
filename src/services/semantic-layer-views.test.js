import { describe, expect, it } from 'vitest';
import { buildLayerGraph, reconcileCompletionAssetContracts } from './semantic-layer-views.js';
import { buildObjectCompletionPreflight, evaluateCanonicalAssetQuality, prepareCompletionAssetPreflight } from './object-completion-planner.js';

function layer(overrides = {}) {
    return {
        id: overrides.id || 'layer',
        name: overrides.name || '对象',
        semanticType: overrides.semanticType || 'unknown',
        bbox: overrides.bbox || [100, 100, 500, 500],
        zIndex: overrides.zIndex || 0,
        ...overrides
    };
}

describe('completion candidate prescreen', () => {
    it('excludes background, text, panels, soft edges, and semantic groups', () => {
        const graph = buildLayerGraph([
            layer({ id: 'background', category: 'background', bbox: [0, 0, 1000, 1000] }),
            layer({ id: 'text', semanticType: 'element_text', bbox: [200, 200, 400, 600], zIndex: 5 }),
            layer({ id: 'panel', semanticType: 'shape_panel', renderMode: 'vector_shape', bbox: [150, 150, 500, 700], zIndex: 4 }),
            layer({ id: 'veil', name: '婚纱头纱', semanticType: 'wedding_veil', bbox: [150, 150, 550, 650], zIndex: 3 }),
            layer({ id: 'group', runtimeType: 'semantic_group', compositeRole: 'composite_group', bbox: [100, 100, 600, 700], zIndex: 2 })
        ]);

        expect(graph.completionTasks).toEqual([]);
    });

    it('creates one merged candidate when a table occludes a sofa', () => {
        const graph = buildLayerGraph([
            layer({ id: 'sofa', name: '红色沙发', semanticType: 'furniture_sofa', bbox: [100, 100, 700, 800], zIndex: 1 }),
            layer({ id: 'table', name: '茶几', semanticType: 'furniture_table', bbox: [450, 250, 850, 750], zIndex: 3 }),
            layer({ id: 'lamp', name: '落地灯', semanticType: 'lighting_floor_lamp', bbox: [350, 600, 900, 900], zIndex: 4 })
        ]);

        const task = graph.completionTasks.find(candidate => candidate.targetLayerId === 'sofa');
        expect(task).toMatchObject({
            eligibility: 'auto',
            status: 'pending',
            maskValidation: { status: 'pending' }
        });
        expect(task.occluderLayerIds).toEqual(['table', 'lamp']);
        expect(task.missingRegionBbox).toEqual([350, 250, 700, 800]);
        expect(graph.version).toBe('layer-graph-v2');
    });

    it('does not create a candidate from a negligible bbox overlap', () => {
        const graph = buildLayerGraph([
            layer({ id: 'sofa', name: '沙发', semanticType: 'furniture_sofa', bbox: [100, 100, 500, 500], zIndex: 1 }),
            layer({ id: 'table', name: '边几', semanticType: 'furniture_table', bbox: [490, 490, 800, 800], zIndex: 2 })
        ]);

        expect(graph.completionTasks).toEqual([]);
    });

    it('retains a small but real hard-object overlap for mask validation', () => {
        const graph = buildLayerGraph([
            layer({ id: 'console', name: '异形石材玄关案几', semanticType: 'furniture_table', bbox: [100, 100, 800, 850], zIndex: 1 }),
            layer({ id: 'stool', name: '白色圆凳组合', semanticType: 'furniture_stool', bbox: [740, 700, 940, 930], zIndex: 3 })
        ]);

        expect(graph.completionTasks).toHaveLength(1);
        expect(graph.completionTasks[0]).toMatchObject({
            targetLayerId: 'console',
            occluderLayerIds: ['stool'],
            eligibility: 'auto'
        });
    });

    it('expands a composite occluder into every atomic child', () => {
        const graph = buildLayerGraph([
            layer({ id: 'console', name: '玄关桌', semanticType: 'furniture_table', bbox: [100, 100, 800, 850], zIndex: 1 }),
            layer({
                id: 'stool-group',
                name: '凳子组合',
                runtimeType: 'semantic_group',
                compositeRole: 'composite_group',
                childLayerIds: ['stool-a', 'stool-b'],
                bbox: [650, 600, 960, 950],
                zIndex: 3
            }),
            layer({ id: 'stool-a', name: '白色凳子 1', semanticType: 'furniture_stool', parentLayerId: 'stool-group', bbox: [660, 610, 820, 760], zIndex: 3 }),
            layer({ id: 'stool-b', name: '白色凳子 2', semanticType: 'furniture_stool', parentLayerId: 'stool-group', bbox: [780, 760, 950, 940], zIndex: 3 })
        ]);

        const task = graph.completionTasks.find(candidate => candidate.targetLayerId === 'console');
        expect(task).toBeTruthy();
        expect(task.occluderLayerIds).toEqual(expect.arrayContaining(['stool-a', 'stool-b']));
        expect(task.occluderLayerIds).not.toContain('stool-group');
    });

    it('creates a candidate when visible hard-object boxes are narrowly adjacent', () => {
        const graph = buildLayerGraph([
            layer({ id: 'console', name: '异形石材玄关案几', semanticType: 'furniture_table', bbox: [100, 100, 800, 700], zIndex: 1 }),
            layer({ id: 'stool', name: '白色圆凳组合', semanticType: 'furniture_stool', bbox: [400, 708, 780, 920], zIndex: 3 })
        ]);

        expect(graph.completionTasks).toHaveLength(1);
        expect(graph.completionTasks[0]).toMatchObject({
            targetLayerId: 'console',
            occluderLayerIds: ['stool'],
            bboxOverlapRatio: 0,
            nearContactRelations: [{ occluderLayerId: 'stool', distance: 8, axis: 'horizontal' }]
        });
    });

    it('does not treat a panel or text overlay as a physical occluder', () => {
        const graph = buildLayerGraph([
            layer({ id: 'sofa', name: '沙发', semanticType: 'furniture_sofa', bbox: [100, 100, 600, 800], zIndex: 1 }),
            layer({ id: 'badge', name: '促销价格牌', semanticType: 'price_badge', renderMode: 'vector_shape', bbox: [350, 350, 650, 750], zIndex: 3 }),
            layer({ id: 'copy', name: '文字: 限时优惠', semanticType: 'element_text', bbox: [360, 360, 470, 640], zIndex: 4 })
        ]);

        expect(graph.completionTasks).toEqual([]);
    });

    it('keeps a raster food decoration as a physical foreground occluder', () => {
        const graph = buildLayerGraph([
            layer({
                id: 'character_woman',
                name: '主体人物-尖叫女子',
                semanticType: 'other',
                designRole: 'product_image',
                renderMode: 'raster_cutout',
                bbox: [236, 0, 1000, 1000],
                zIndex: 5
            }),
            layer({
                id: 'bottom_left_lemons',
                name: '左下角柠檬组合',
                semanticType: 'product_food',
                // This is the conflicting poster metadata from the failure:
                // it is decorative in composition but still a raster food.
                designRole: 'decor_shape',
                renderMode: 'raster_cutout',
                bbox: [799, 0, 1000, 354],
                zIndex: 6
            })
        ]);

        expect(graph.completionTasks).toHaveLength(1);
        expect(graph.completionTasks[0]).toMatchObject({
            targetLayerId: 'character_woman',
            occluderLayerIds: ['bottom_left_lemons'],
            eligibility: 'auto'
        });
    });

    it('creates a separate immutable observed cutout contract after mask verification', () => {
        const graph = buildLayerGraph([
            layer({ id: 'sofa', name: '沙发', semanticType: 'furniture_sofa', bbox: [100, 100, 700, 800], zIndex: 1, cutoutUrl: 'https://assets.test/sofa.png' }),
            layer({ id: 'table', name: '茶几', semanticType: 'furniture_table', bbox: [450, 250, 850, 750], zIndex: 3, cutoutUrl: 'https://assets.test/table.png' })
        ]);
        const task = graph.completionTasks[0];
        task.maskValidation = { status: 'verified', recommendation: 'auto' };
        const item = {
            originalDataUrl: 'https://assets.test/original.png',
            semanticViews: { layerGraph: graph, completionAssets: [] }
        };

        const [asset] = reconcileCompletionAssetContracts(item);

        expect(asset).toMatchObject({
            targetLayerId: 'sofa',
            status: 'ready_for_completion',
            observedCutout: { cutoutUrl: 'https://assets.test/sofa.png', immutable: true },
            canonicalAsset: { status: 'not_requested' },
            composition: {
                activeRepresentation: 'observed_cutout',
                preserveOriginalOcclusion: true,
                placementPolicy: 'canonical_asset_not_auto_composited'
            }
        });
        expect(task.assetContractId).toBe(asset.id);
    });

    it('retains a canonical asset only while its occlusion relation is unchanged', () => {
        const graph = buildLayerGraph([
            layer({ id: 'sofa', name: '沙发', semanticType: 'furniture_sofa', bbox: [100, 100, 700, 800], zIndex: 1, cutoutUrl: 'https://assets.test/sofa.png' }),
            layer({ id: 'table', name: '茶几', semanticType: 'furniture_table', bbox: [450, 250, 850, 750], zIndex: 3 })
        ]);
        const task = graph.completionTasks[0];
        task.maskValidation = { status: 'verified', recommendation: 'auto' };
        const item = {
            originalDataUrl: 'https://assets.test/original.png',
            semanticViews: { layerGraph: graph, completionAssets: [] }
        };
        const [created] = reconcileCompletionAssetContracts(item);
        created.canonicalAsset = { ...created.canonicalAsset, status: 'ready', cutoutUrl: 'https://assets.test/full-sofa.png' };

        const [preserved] = reconcileCompletionAssetContracts(item);

        expect(preserved.status).toBe('canonical_ready');
        expect(preserved.canonicalAsset.cutoutUrl).toBe('https://assets.test/full-sofa.png');
        expect(preserved.observedCutout.cutoutUrl).toBe('https://assets.test/sofa.png');

        preserved.generation.state = 'completed';
        prepareCompletionAssetPreflight(item, preserved.id);
        expect(preserved.generation.state).toBe('completed');
    });

    it('prepares an aspect-preserving completion plan without creating image data', () => {
        const graph = buildLayerGraph([
            layer({ id: 'chair', name: '休闲椅', semanticType: 'furniture_chair', bbox: [300, 300, 700, 700], zIndex: 1, cutoutUrl: 'https://assets.test/chair.png' }),
            layer({ id: 'table', name: '茶几', semanticType: 'furniture_table', bbox: [550, 450, 850, 800], zIndex: 3 })
        ]);
        const task = graph.completionTasks[0];
        task.maskValidation = { status: 'verified', recommendation: 'auto' };
        const item = {
            originalDataUrl: 'https://assets.test/original.png',
            semanticViews: { layerGraph: graph, completionAssets: [] }
        };
        const [asset] = reconcileCompletionAssetContracts(item);
        const preflight = buildObjectCompletionPreflight(item, asset, graph);

        expect(preflight.geometry.cropBbox).toEqual([220, 220, 780, 780]);
        expect(preflight.geometry.outputSizePolicy).toBe('preserve_crop_aspect_ratio');
        expect(preflight.references.occluderNames).toEqual(['茶几']);
        expect(preflight.prompt.text).toContain('休闲椅');
        expect(preflight).not.toHaveProperty('referenceCropDataUrl');
    });

    it('routes opaque or unbounded canonical results to manual review', () => {
        const qualityGate = {
            requireTransparentResult: true,
            minOpaqueRatio: 0.025,
            maxOpaqueRatio: 0.82,
            minTransparentBorderRatio: 0.9
        };
        expect(evaluateCanonicalAssetQuality({
            hasTransparency: false,
            opaqueRatio: 0.94,
            transparentBorderRatio: 0.1
        }, qualityGate)).toMatchObject({
            accepted: false,
            status: 'manual_review',
            reasons: ['missing_transparency', 'scene_like_opaque_coverage', 'insufficient_transparent_border']
        });
    });
});
