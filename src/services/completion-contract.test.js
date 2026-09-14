import { describe, expect, it } from 'vitest';
import {
    createCompletionLayerIndex,
    expandEntityLayerIds,
    resolveCompletionTaskEntities,
    resolveEntityParts
} from './completion-contract.js';

const layers = [
    { layerId: 'table', bbox: [100, 100, 800, 850] },
    {
        layerId: 'stool-group',
        runtimeType: 'semantic_group',
        compositeRole: 'composite_group',
        childLayerIds: ['stool-a', 'stool-b'],
        bbox: [600, 550, 960, 950]
    },
    { layerId: 'stool-a', parentLayerId: 'stool-group', bbox: [620, 560, 800, 740] },
    { layerId: 'stool-b', parentLayerId: 'stool-group', bbox: [780, 760, 950, 940] },
    { layerId: 'lamp-a', entityId: 'lamp', bbox: [200, 700, 500, 900] },
    { layerId: 'lamp-b', entityId: 'lamp', bbox: [480, 700, 760, 900] }
];

describe('completion entity ownership contract', () => {
    const index = createCompletionLayerIndex(layers);

    it('expands parent and child links without merging unrelated boxes', () => {
        const result = resolveEntityParts(index, 'stool-group');
        expect(result.layerIds).toEqual(['stool-group', 'stool-a', 'stool-b']);
        expect(result.unionBbox).toEqual([600, 550, 960, 950]);
        expect(resolveEntityParts(index, 'lamp-a').layerIds).toEqual(['lamp-a', 'lamp-b']);
    });

    it('expands a task while excluding target entity parts', () => {
        const result = resolveCompletionTaskEntities({
            targetLayerId: 'table',
            occluderLayerIds: ['stool-group']
        }, index);
        expect(result.occluderLayerIds).toEqual(['stool-group', 'stool-a', 'stool-b']);
        expect(result.target.layerIds).toEqual(['table']);
    });

    it('deduplicates multiple references to one composite entity', () => {
        expect(expandEntityLayerIds(index, ['stool-a', 'stool-b'])).toEqual([
            'stool-group', 'stool-a', 'stool-b'
        ]);
    });
});

