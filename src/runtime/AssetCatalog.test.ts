import { describe, expect, it } from 'vitest';
import { AssetCatalogIndex } from './AssetCatalog';
import type { AssetEntity } from './AssetRuntime';

function makeStore() {
    const data = new Map<string, any>();
    return {
        async getItem<T>(key: string) { return (data.get(key) ?? null) as T | null; },
        async setItem<T>(key: string, value: T) { data.set(key, value); return value; },
        async removeItem(key: string) { data.delete(key); }
    };
}

function makeAsset(): AssetEntity {
    return {
        uid: 'sofa_001',
        version: 3,
        type: 'product',
        sourceImage: 'data:image/png;base64,very-large-payload',
        masks: [],
        variants: [],
        metadata: {
            createdAt: 1,
            updatedAt: 10,
            creatorId: 'test',
            usageCount: 1,
            tags: ['沙发', '米白']
        },
        fusionProperties: {
            brightness: 100,
            contrast: 100,
            saturation: 100,
            blur: 0,
            hueRotate: 0,
            grayscale: 0,
            sepia: 0
        },
        transform: { x: 1, y: 2, width: 300, height: 200, rotation: 0, zIndex: 1 },
        layerName: '米白沙发'
    };
}

describe('AssetCatalogIndex', () => {
    it('persists searchable metadata without copying image payloads', async () => {
        const store = makeStore();
        const first = new AssetCatalogIndex(store);
        await first.upsertWorkspaceSnapshot({
            projectId: 'project-a',
            projectName: '客厅方案 2',
            assets: [makeAsset()]
        });

        const storedEntries = await store.getItem<any[]>('asset-catalog-v1');
        expect(storedEntries).toHaveLength(1);
        expect(JSON.stringify(storedEntries)).not.toContain('data:image');

        const second = new AssetCatalogIndex(store);
        const matches = await second.search('米白沙发');
        expect(matches[0]).toMatchObject({
            assetId: 'sofa_001',
            projectId: 'project-a',
            projectName: '客厅方案 2',
            version: 3
        });
    });

    it('removes deleted project entries without affecting other projects', async () => {
        const store = makeStore();
        const catalog = new AssetCatalogIndex(store);
        await catalog.upsertWorkspaceSnapshot({ projectId: 'project-a', assets: [makeAsset()] });
        await catalog.upsertWorkspaceSnapshot({ projectId: 'project-b', assets: [{ ...makeAsset(), uid: 'chair_001' }] });

        await catalog.removeProject('project-a');

        expect((await catalog.list()).map(entry => entry.projectId)).toEqual(['project-b']);
    });
});

