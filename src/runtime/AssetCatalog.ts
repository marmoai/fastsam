import localforage from 'localforage';
import type { AssetEntity } from './AssetRuntime';
import type { ProjectWorkspace } from './WorkspaceRuntime';

export interface AssetCatalogEntry {
    assetId: string;
    projectId: string;
    projectName: string;
    version: number;
    name: string;
    type: string;
    tags: string[];
    sourceRef: {
        projectId: string;
        assetId: string;
        version: number;
    };
    updatedAt: number;
}

interface CatalogStore {
    getItem<T>(key: string): Promise<T | null>;
    setItem<T>(key: string, value: T): Promise<T>;
    removeItem(key: string): Promise<void>;
}

const CATALOG_KEY = 'asset-catalog-v1';

function assetName(asset: AssetEntity): string {
    return String((asset as any).name || asset.layerName || asset.uid);
}

function toEntry(projectId: string, projectName: string, asset: AssetEntity): AssetCatalogEntry {
    return {
        assetId: asset.uid,
        projectId,
        projectName,
        version: Number(asset.version) || 1,
        name: assetName(asset),
        type: String(asset.type || 'unknown'),
        tags: Array.isArray(asset.metadata?.tags) ? asset.metadata.tags.map(String).slice(0, 50) : [],
        sourceRef: {
            projectId,
            assetId: asset.uid,
            version: Number(asset.version) || 1
        },
        updatedAt: Number(asset.metadata?.updatedAt) || Date.now()
    };
}

function clone<T>(value: T): T {
    return JSON.parse(JSON.stringify(value));
}

export class AssetCatalogIndex {
    private readonly store: CatalogStore;
    private readonly memory = new Map<string, AssetCatalogEntry>();
    private loaded = false;
    private loadPromise: Promise<void> | null = null;
    private writePromise: Promise<void> = Promise.resolve();

    constructor(store: CatalogStore = localforage.createInstance({ name: 'MarmoAid', storeName: 'agentAssetCatalog' })) {
        this.store = store;
    }

    async upsertWorkspace(workspace: ProjectWorkspace): Promise<void> {
        await this.upsertWorkspaceSnapshot({
            projectId: workspace.projectId,
            projectName: workspace.name,
            assets: workspace.currentState.assetRegistry.getAll()
        });
    }

    async upsertWorkspaceSnapshot(snapshot: {
        projectId: string;
        projectName?: string;
        assets?: AssetEntity[];
    }): Promise<void> {
        await this.ensureLoaded();
        const projectId = String(snapshot.projectId || '');
        if (!projectId) return;
        const projectName = String(snapshot.projectName || projectId);
        const assets = Array.isArray(snapshot.assets) ? snapshot.assets : [];
        const activeIds = new Set<string>();

        assets.forEach(asset => {
            if (!asset?.uid) return;
            const entry = toEntry(projectId, projectName, asset);
            this.memory.set(this.key(entry.projectId, entry.assetId), entry);
            activeIds.add(entry.assetId);
        });

        // Remove deleted assets from this project without touching other projects.
        for (const [key, entry] of this.memory.entries()) {
            if (entry.projectId === projectId && !activeIds.has(entry.assetId)) {
                this.memory.delete(key);
            }
        }
        await this.persist();
    }

    async list(): Promise<AssetCatalogEntry[]> {
        await this.ensureLoaded();
        return clone([...this.memory.values()]);
    }

    async search(text = ''): Promise<AssetCatalogEntry[]> {
        const normalized = String(text).trim().toLowerCase();
        const entries = await this.list();
        if (!normalized) return entries;
        const tokens = normalized.split(/\s+/).filter(Boolean);
        return entries
            .map(entry => ({ entry, score: this.score(entry, tokens) }))
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .map(item => item.entry);
    }

    async get(projectId: string, assetId: string): Promise<AssetCatalogEntry | null> {
        await this.ensureLoaded();
        const entry = this.memory.get(this.key(projectId, assetId));
        return entry ? clone(entry) : null;
    }

    async removeProject(projectId: string): Promise<void> {
        await this.ensureLoaded();
        for (const [key, entry] of this.memory.entries()) {
            if (entry.projectId === projectId) this.memory.delete(key);
        }
        await this.persist();
    }

    async clear(): Promise<void> {
        this.memory.clear();
        this.loaded = true;
        await this.store.removeItem(CATALOG_KEY).catch(() => undefined);
    }

    private score(entry: AssetCatalogEntry, tokens: string[]): number {
        const name = entry.name.toLowerCase();
        const searchable = [entry.assetId, entry.projectId, entry.projectName, entry.name, entry.type, ...entry.tags]
            .join(' ')
            .toLowerCase();
        return tokens.reduce((score, token) => {
            if (entry.assetId.toLowerCase() === token) return score + 100;
            if (name === token) return score + 80;
            if (searchable.includes(token)) return score + 10;
            return score;
        }, 0);
    }

    private key(projectId: string, assetId: string): string {
        return `${projectId}:${assetId}`;
    }

    private async ensureLoaded(): Promise<void> {
        if (this.loaded) return;
        if (!this.loadPromise) {
            this.loadPromise = (async () => {
                try {
                    const stored = await this.store.getItem<AssetCatalogEntry[]>(CATALOG_KEY);
                    if (Array.isArray(stored)) {
                        stored.forEach(entry => {
                            if (entry?.projectId && entry?.assetId) {
                                this.memory.set(this.key(entry.projectId, entry.assetId), entry);
                            }
                        });
                    }
                } catch (error) {
                    // Runtime memory remains usable when IndexedDB is unavailable
                    // (for example, in unit tests or private browsing contexts).
                    console.warn('[AssetCatalog] Failed to load persisted index:', error);
                } finally {
                    this.loaded = true;
                }
            })();
        }
        await this.loadPromise;
    }

    private async persist(): Promise<void> {
        const write = this.writePromise.then(async () => {
            try {
                await this.store.setItem(CATALOG_KEY, [...this.memory.values()]);
            } catch (error) {
                console.warn('[AssetCatalog] Failed to persist index:', error);
            }
        });
        this.writePromise = write;
        await write;
    }
}

export const assetCatalog = new AssetCatalogIndex();
