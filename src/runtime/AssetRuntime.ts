/**
 * Visual Asset OS - Minimal Viable Asset Runtime (Stage 1)
 * 核心目标：让对象拥有生命周期和持久身份（UID）
 * 防爆破提示：当前阶段不引入PBR材质、光影计算和复杂法线系统
 * 
 * ==========================================
 *        非 Runtime 资产对象边界管理规定
 * ==========================================
 * 以下类型的可视节点在交互设计中存在，但故意不作为 Asset Runtime 资产管理：
 * 
 * 1. Text-Note (便签 / 文字批注层):
 *    - 非图元，主要作为协作标注；状态由 DOM/LocalStorage 缓存，不计入核心渲染
 * 2. Shape (形状 / 辅助虚线框):
 *    - 仅作为画布指示物 (Canvas Decal/Indicator)，不具备独立生存周期和图片编辑链
 * 3. Atmosphere (环境控制球 / 光照粒子):
 *    - 作为 Relighting 和图层预处理的控制器，不作为一等公民 (First-class) 资产参与物种起源融合
 * 4. Layer Internals (图层内部深度选区):
 *    - 图层内部临时笔刷选区在活跃编辑态中常驻内存，一旦提交合成即丢弃，不登记核心 ID 树。
 */

export type SemanticType = 'product' | 'human' | 'background' | 'element' | 'unknown';

export interface Metadata {
    createdAt: number;
    updatedAt: number;
    creatorId: string;
    usageCount: number;
    tags: string[];
}

export interface MaskData {
    maskId: string;
    type: 'alpha' | 'semantic' | 'depth';
    sourceImage: string; // URL
    createdAt: number;
}

export interface AssetVariant {
    variantId: string;
    name: string;
    sourceImage: string; // Blob URL
    createdAt: number;
}

export interface AssetTransform {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation: number;
    zIndex: number;
}

export interface FusionProperties {
    brightness: number;
    contrast: number;
    saturation: number;
    blur: number;
    hueRotate: number;
    grayscale: number;
    sepia: number;
    overlay?: {
        color: string;
        opacity: number;
        blendMode: 'overlay' | 'color-dodge' | 'multiply' | 'screen' | 'normal';
    };
}

export interface AssetEntity {
    uid: string; // 灵魂：全局唯一身份标识
    version: number; // 资产版本控制
    type: SemanticType; 

    // MVR基础数据：告别一次性图片
    sourceImage: string; // Blob URL 或远程 URL
    masks: MaskData[]; // 语义遮罩数据
    
    variants: AssetVariant[]; // 对象修改产生的版本
    
    // 业务元数据
    metadata: Metadata; 

    // 视觉融合属性（Fusion Edit）
    fusionProperties: FusionProperties;

    // 空间变换数据
    transform: AssetTransform;

    // 状态机数据
    status?: 'idle' | 'remixing' | 'error';

    // Workbench image metadata owned by runtime to avoid dual truth with session.workbenchState
    genealogy?: {
        parents?: string[];
        prompt?: string;
        action?: string;
    };
    parentId?: string | null;
    layerName?: string;
    originalBbox?: any;
    layers?: any[];
    scene?: any;
    hasFullSemanticAnalysis?: boolean;
    originalDataUrl?: string;
    cleanPlateDataUrl?: string | null;
    cleanPlateStatus?: string;
}

export interface GlobalAssetRef {
    uid: string;
    version: number;
    url: string;
    name: string;
    lastUpdated: number;
    contentHash?: string;
}

/**
 * Global Asset Catalog: 平行的全局资产目录，项目存储Ref指针
 */
export class GlobalAssetCatalog {
    // 模拟OSS/Firebase的全局存储
    // uid -> version -> GlobalAssetRef
    private static catalog: Map<string, Map<number, GlobalAssetRef>> = new Map();

    private static generateHash(asset: any): string {
        // 核心：剥离所有"场景属性"（位置、旋转、所选状态等）
        // 仅对"资产本身属性"（图层、遮罩、调色）做快照树拦截
        const payload = {
            sourceImage: asset.sourceImage,
            masks: asset.masks,
            fusionProperties: asset.fusionProperties,
            variants: asset.variants
        };
        // 简易 Hash 实现（生产环境应用 crypto SHA-256）
        return JSON.stringify(payload);
    }

    static async getLatestRef(uid: string): Promise<GlobalAssetRef | null> {
        const history = this.catalog.get(uid);
        if (!history || history.size === 0) return null;
        
        let latestVersion = 0;
        for (const v of history.keys()) {
            if (v > latestVersion) latestVersion = v;
        }
        return history.get(latestVersion) || null;
    }

    static async getHistory(uid: string): Promise<GlobalAssetRef[]> {
        const history = this.catalog.get(uid);
        if (!history) return [];
        return Array.from(history.values()).sort((a, b) => b.version - a.version);
    }

    static async publish(asset: any): Promise<void> { // Using any to match existing AssetEntity without circular dependency changes
        const contentHash = this.generateHash(asset);
        
        // --- 核心：基于 Hash 拦截无效的垃圾版本 ---
        const latestRef = await this.getLatestRef(asset.uid);
        if (latestRef && latestRef.contentHash === contentHash) {
            // 资产自身的内涵没有任何改变，这只是一次"场景状态（如缩放、平移）"保存，阻止繁衍！
            return;
        }

        asset.version = latestRef ? latestRef.version + 1 : 1;
        
        if (!this.catalog.has(asset.uid)) {
            this.catalog.set(asset.uid, new Map());
        }
        
        const history = this.catalog.get(asset.uid)!;
        history.set(asset.version, {
            uid: asset.uid,
            version: asset.version,
            url: asset.sourceImage,
            name: asset.name || `${asset.type || 'asset'}_${asset.uid.substring(0, 4)} v${asset.version}`,
            lastUpdated: Date.now(),
            contentHash: contentHash
        });
        
        console.log(`[GlobalAssetCatalog] Published ${asset.uid} - v${asset.version} (Hash Tracked)`);
    }
}

export class AssetRegistry {
    private assets: Map<string, AssetEntity> = new Map();

    register(asset: AssetEntity): void {
        this.assets.set(asset.uid, asset);
    }

    get(uid: string): AssetEntity | undefined {
        return this.assets.get(uid);
    }

    delete(uid: string): boolean {
        return this.assets.delete(uid);
    }

    getAll(): AssetEntity[] {
        return Array.from(this.assets.values());
    }

    clear(): void {
        this.assets.clear();
    }

    updateAssetTransform(uid: string, transform: Partial<AssetTransform>) {
        const asset = this.assets.get(uid);
        if (asset) {
            asset.transform = {
                x: transform.x !== undefined ? transform.x : (asset.transform?.x ?? 0),
                y: transform.y !== undefined ? transform.y : (asset.transform?.y ?? 0),
                width: transform.width !== undefined ? transform.width : (asset.transform?.width ?? 100),
                height: transform.height !== undefined ? transform.height : (asset.transform?.height ?? 100),
                rotation: transform.rotation !== undefined ? transform.rotation : (asset.transform?.rotation ?? 0),
                zIndex: transform.zIndex !== undefined ? transform.zIndex : (asset.transform?.zIndex ?? 1)
            };
        }
    }

    updateFusionProperties(uid: string, fusion: FusionProperties) {
        const asset = this.assets.get(uid);
        if (asset) {
            asset.fusionProperties = fusion;
        }
    }
}
