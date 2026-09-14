export type TaskBucket = 'product' | 'space' | 'poster' | 'general';

const PRODUCT_REGEX = /(product|packaging|food|drink|menu|dish|beverage|商品|产品|电商|菜品|饮料|数码|包装|耳机|静物)/i;
const SPACE_REGEX = /(space|interior|room|living|bedroom|kitchen|furniture|sofa|chair|table|lamp|wall|floor|landscape|室内|空间|家装|客厅|卧室|厨房|家具|沙发|椅子|桌子|灯|墙面|地面)/i;
const POSTER_REGEX = /(poster|banner|flyer|promotion|promo|ad_|ad background|price badge|headline|price|layout|海报|广告|主视觉|促销|价签|价格区|文案区|排版|菜单板)/i;

function collectAssetText(asset: any): string {
    if (!asset) return '';
    const parts: string[] = [];

    const pushText = (value: any) => {
        if (typeof value === 'string' && value.trim()) {
            parts.push(value.trim());
        }
    };

    pushText(asset.type);
    pushText(asset.name);
    pushText(asset.prompt);
    pushText(asset.label);
    pushText(asset.layerName);
    pushText(asset.semanticType);
    pushText(asset.designRole);

    if (Array.isArray(asset?.metadata?.tags)) {
        asset.metadata.tags.forEach(pushText);
    }

    if (Array.isArray(asset?.layers)) {
        asset.layers.forEach((layer: any) => {
            pushText(layer?.name);
            pushText(layer?.semanticType);
            pushText(layer?.designRole);
            pushText(layer?.reasoning);
        });
    }

    if (asset?.semanticViews && typeof asset.semanticViews === 'object') {
        pushText(JSON.stringify(asset.semanticViews));
    }

    return parts.join(' ');
}

export function normalizeTaskType(raw?: string | null): TaskBucket | undefined {
    if (!raw) return undefined;
    const value = String(raw).trim().toLowerCase();
    if (!value) return undefined;

    if (value === 'product' || PRODUCT_REGEX.test(value)) return 'product';
    if (value === 'space' || SPACE_REGEX.test(value)) return 'space';
    if (value === 'poster' || POSTER_REGEX.test(value)) return 'poster';
    if (value === 'general') return 'general';

    return undefined;
}

export function inferTaskBucketFromText(text?: string | null): TaskBucket | undefined {
    if (!text) return undefined;
    const raw = String(text).trim();
    if (!raw) return undefined;

    if (POSTER_REGEX.test(raw)) return 'poster';
    if (PRODUCT_REGEX.test(raw)) return 'product';
    if (SPACE_REGEX.test(raw)) return 'space';
    return undefined;
}

export function inferTaskBucketFromAsset(asset: any): TaskBucket {
    const normalizedType = normalizeTaskType(asset?.type);
    if (normalizedType && normalizedType !== 'general') {
        return normalizedType;
    }

    const text = collectAssetText(asset);
    const posterHits = (text.match(new RegExp(POSTER_REGEX.source, 'ig')) || []).length;
    const productHits = (text.match(new RegExp(PRODUCT_REGEX.source, 'ig')) || []).length;
    const spaceHits = (text.match(new RegExp(SPACE_REGEX.source, 'ig')) || []).length;

    if (posterHits >= productHits && posterHits >= spaceHits && posterHits > 0) return 'poster';
    if (productHits >= spaceHits && productHits > 0) return 'product';
    if (spaceHits > 0) return 'space';

    return normalizedType || inferTaskBucketFromText(text) || 'general';
}

export function inferTaskBucketFromWorkbenchState(
    workbenchItems?: Map<string, any> | null,
    activeItemId?: string | null,
    fallbackText?: string | null
): TaskBucket {
    if (workbenchItems && activeItemId) {
        const activeItem = workbenchItems.get(activeItemId);
        if (activeItem) {
            const bucket = inferTaskBucketFromAsset(activeItem);
            if (bucket !== 'general') return bucket;
        }
    }

    const tally: Record<TaskBucket, number> = {
        product: 0,
        space: 0,
        poster: 0,
        general: 0
    };

    if (workbenchItems) {
        workbenchItems.forEach((item) => {
            if (!item || ['text-note', 'shape', 'group-label', 'atmosphere'].includes(item.type)) return;
            tally[inferTaskBucketFromAsset(item)] += 1;
        });
    }

    const bucket = (Object.entries(tally)
        .sort((a, b) => b[1] - a[1])[0]?.[0] || 'general') as TaskBucket;

    if (bucket !== 'general' && tally[bucket] > 0) return bucket;
    return inferTaskBucketFromText(fallbackText) || 'general';
}

export function describeTaskBucket(bucket?: string | null): string {
    switch (normalizeTaskType(bucket) || bucket) {
        case 'product':
            return '商品图';
        case 'space':
            return '空间图';
        case 'poster':
            return '海报图';
        default:
            return '当前任务';
    }
}
