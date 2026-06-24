import localforage from 'localforage';

export interface SatisfactionRule {
    id: string;
    keywords: string[];     // keywords extracted for matching (e.g. ["cyberpunk", "cat"])
    initialPrompt: string;  // starting point of their idea (e.g. "赛博朋克猫")
    finalPrompt: string;    // the perfected prompt (e.g. "赛博朋克猫，背景带霓虹灯，主色调紫粉色，眼睛发蓝光")
    count: number;          // usage count / weight
    lastUsed: number;
    sessionId?: string;
    fusionProperties?: any; // The item's ambient, shadow, edge, perspective fusion properties
    crop?: { top: number; left: number; right: number; bottom: number }; // cropping metrics
    aspectRatio?: string;   // Aspect Ratio of the generated image
    presetTags?: string[];  // Described parameters block tag label
}

class ImplicitMemoryManager {
    private rules: SatisfactionRule[] = [];
    private activeChain: string[] = []; // current active prompt chain in the session
    private currentSessionId: string = '';
    private storageKey = 'mvr_implicit_satisfaction_rules';

    constructor() {
        this.loadDefaultRules();
        this.loadFromStorage();
        if (typeof window !== 'undefined') {
            setInterval(() => {
                try {
                    this.runTelemetrySweep();
                } catch (err) {
                    console.error('[Telemetry] Sweep failed', err);
                }
            }, 5000); // Check every 5 seconds
        }
    }

    // Prepopulate highly aesthetic default rules so the user immediately gets a magic experience!
    private loadDefaultRules() {
        this.rules = [
            {
                id: 'default-cyberpunk',
                keywords: ['赛博', '猫', '朋克', 'cyberpunk', 'cat'],
                initialPrompt: '赛博朋克猫',
                finalPrompt: '赛博朋克猫，背景带霓虹灯，主色调紫粉色，眼睛发黑科技蓝光，细节超高清，蒸汽波艺术风格',
                count: 1,
                lastUsed: Date.now(),
                fusionProperties: { brightness: 105, contrast: 110, saturation: 115, blur: 0 },
                aspectRatio: '1:1',
                presetTags: ['紫粉色霓虹调', '1:1 正投影']
            },
            {
                id: 'default-interior',
                keywords: ['室内', '设计', '客厅', '北欧', 'interior', 'living'],
                initialPrompt: '北欧极简客厅',
                finalPrompt: '北欧原木极简客厅，微曦阳光从百叶窗斜射，原木茶几，白色绒毛地毯，高级景深温煦色系',
                count: 1,
                lastUsed: Date.now(),
                fusionProperties: { brightness: 100, contrast: 95, saturation: 90, blur: 1 },
                aspectRatio: '4:3',
                presetTags: ['高级景深温煦色系', '4:3 比例']
            },
            {
                id: 'default-product',
                keywords: ['电商', '产品', '白色', '数码', 'product', 'white'],
                initialPrompt: '白色数码产品',
                finalPrompt: '白色高端数码产品，极简原宿背景，微风拂气，微弱和自然光影，高级静物水泥展示台，杂志大片级景深',
                count: 1,
                lastUsed: Date.now(),
                fusionProperties: { brightness: 102, contrast: 100, saturation: 95, blur: 0 },
                aspectRatio: '1:1',
                presetTags: ['高端静物水泥展台', '双倍透光度']
            },
            {
                id: 'default-capsule',
                keywords: ['太空舱', '灵感', '设计舱', 'capsule'],
                initialPrompt: '太空灵感舱',
                finalPrompt: '太空底座高能智能灵感舱，半透明微呼吸红色LED面板，3D微晶玻璃科技质感，悬浮多维全息图层光斑',
                count: 1,
                lastUsed: Date.now(),
                fusionProperties: { brightness: 95, contrast: 105, saturation: 100, blur: 2 },
                aspectRatio: '16:9',
                presetTags: ['智能灵感舱暗红色调', '16:9 高能缩放']
            },
            {
                id: 'default-classic-landscape',
                keywords: ['山水', '山水画', '中国风', 'landscape'],
                initialPrompt: '山水意境画',
                finalPrompt: '泼墨留白意境山水画，薄雾破晓破迷霭朦胧，一叶扁舟微荡湖心，古色古香古典东方美学',
                count: 1,
                lastUsed: Date.now(),
                fusionProperties: { brightness: 98, contrast: 90, saturation: 80, blur: 3 },
                aspectRatio: '16:9',
                presetTags: ['古香古典墨白山水调', '景深三级虚化']
            }
        ];
    }

    private async loadFromStorage() {
        try {
            const stored = await localforage.getItem<SatisfactionRule[]>(this.storageKey);
            if (stored && Array.isArray(stored)) {
                // Combine stored rules, ensuring we overwrite or append appropriately
                this.mergeRules(stored);
                console.log(`[ImplicitMemory] Restored ${stored.length} verified rules from Local. Total cache: ${this.rules.length}`);
            }
            // Asynchronously sync with Cloud OSS
            this.syncWithCloud();
        } catch (e) {
            console.error('[ImplicitMemory] Failed to load rules from IndexedDB:', e);
            this.syncWithCloud();
        }
    }

    private mergeRules(newRules: SatisfactionRule[]) {
        newRules.forEach(newRule => {
            const existingIdx = this.rules.findIndex(r => r.id === newRule.id || r.initialPrompt.toLowerCase() === newRule.initialPrompt.toLowerCase());
            if (existingIdx !== -1) {
                const existing = this.rules[existingIdx];
                // Resolve conflicts: keep the one with larger count or newer timestamp
                if (newRule.count > existing.count || newRule.lastUsed > existing.lastUsed) {
                    this.rules[existingIdx] = {
                        ...existing,
                        ...newRule,
                        count: Math.max(existing.count, newRule.count),
                        lastUsed: Math.max(existing.lastUsed, newRule.lastUsed)
                    };
                }
            } else {
                this.rules.push(newRule);
            }
        });
    }

    private async syncWithCloud() {
        try {
            // Dynamically import to keep load time fast and resolve potential module cycles
            const { getSatisfactionRulesFromOSS } = await import('../services/ossService.js');
            const cloudRules = await getSatisfactionRulesFromOSS();
            if (cloudRules && Array.isArray(cloudRules) && cloudRules.length > 0) {
                this.mergeRules(cloudRules);
                console.log(`[ImplicitMemory] Loaded and merged ${cloudRules.length} rules from OSS. Total rules: ${this.rules.length}`);
                await localforage.setItem(this.storageKey, this.rules);
            }
        } catch (e) {
            console.warn('[ImplicitMemory] Cloud loading failed or skipped (may be new user or old server):', e);
        }
    }

    private cloudSyncTimeout: any = null;

    private async saveToStorage() {
        try {
            await localforage.setItem(this.storageKey, this.rules);
            this.syncUpToCloud();
        } catch (e) {
            console.error('[ImplicitMemory] Failed to save rules:', e);
        }
    }

    private syncUpToCloud() {
        if (this.cloudSyncTimeout) clearTimeout(this.cloudSyncTimeout);
        this.cloudSyncTimeout = setTimeout(async () => {
            try {
                const { saveSatisfactionRulesToOSS } = await import('../services/ossService.js');
                // Sync the rules (excluding standard uncustomized default rules if appropriate, but keeping learned ones)
                const rulesToSync = this.rules.filter(r => r.id.startsWith('learned-') || !r.id.startsWith('default-'));
                if (rulesToSync.length > 0) {
                    await saveSatisfactionRulesToOSS(this.rules);
                    console.log(`[ImplicitMemory] Synchronized ${this.rules.length} satisfaction rules to Cloud OSS successfully.`);
                }
            } catch (e) {
                console.warn('[ImplicitMemory] Outbound cloud sync blocked or failed (old server or network issues):', e);
            }
        }, 1500); // 1.5s debounce
    }

    private trackedTelemetry: Map<string, any> = new Map();

    private autoCommitLastLeafNodeOfChain(chain: string[]) {
        if (chain.length === 0) return;
        const lastPrompt = chain[chain.length - 1];

        let bestTelId = null;
        let newestCreatedAt = 0;

        for (const [id, tel] of this.trackedTelemetry.entries()) {
            if (tel.isCommitted) continue;
            if (this.arePromptsRelated(tel.prompt, lastPrompt) || chain.some(p => this.arePromptsRelated(tel.prompt, p))) {
                if (tel.createdAt > newestCreatedAt) {
                    newestCreatedAt = tel.createdAt;
                    bestTelId = id;
                }
            }
        }

        if (bestTelId) {
            const tel = this.trackedTelemetry.get(bestTelId);
            const wItems = (window as any).workbenchItems;
            if (tel && wItems) {
                const item = wItems.get(bestTelId);
                if (item) {
                     console.log(`[Telemetry] Auto-committing leaf node ${bestTelId} of chain for theme shift: "${tel.prompt}"`);
                     tel.isCommitted = true;
                     this.trackedTelemetry.delete(bestTelId);

                     const state = (window as any).state || {};
                     const imgSrc = item.dataUrl || item.originalDataUrl || item.sourceImage;
                     this.registerImplicitSatisfaction(tel.prompt, imgSrc, state).catch(console.error);
                }
            }
        }
    }

    private handleDeletionPenalty(prompt: string) {
        const cleanPrompt = prompt.trim().toLowerCase();
        const idx = this.rules.findIndex(r => r.initialPrompt.toLowerCase() === cleanPrompt || r.finalPrompt.toLowerCase() === cleanPrompt);
        if (idx !== -1) {
            const r = this.rules[idx];
            r.count -= 0.5;
            console.log(`[Telemetry] Rule "${r.initialPrompt}" penalized by deletion. New count: ${r.count}`);
            if (r.count <= 0) {
                this.rules.splice(idx, 1);
                console.log(`[Telemetry] Rule removed since count is non-positive.`);
            }
            this.saveToStorage();
        }
    }

    public recordAssetAdded(id: string, prompt: string, initialProps?: any) {
        if (!prompt || prompt.toLowerCase().includes('pasted image') || prompt.toLowerCase().includes('uploaded')) {
            if (this.activeChain.length > 0) {
                prompt = this.activeChain[this.activeChain.length - 1];
            }
        }
        if (!prompt) return;
        if (this.trackedTelemetry.has(id)) return;

        this.trackedTelemetry.set(id, {
            id,
            prompt: prompt.trim(),
            createdAt: Date.now(),
            spawnLeft: 0,
            spawnTop: 0,
            spawnWidth: 0,
            spawnHeight: 0,
            hasFineTuned: false,
            hasCropped: false,
            hasIsolatedEdit: false,
            hasRedirected: false,
            isCommitted: false
        });
        console.log(`[Telemetry] Tracking added asset: ${id} ("${prompt.trim()}")`);
    }

    public runTelemetrySweep() {
        if (typeof window === 'undefined') return;
        const wItems = (window as any).workbenchItems;
        if (!wItems) return;

        // Populate initial coordinates if still empty
        for (const [id, item] of wItems.entries()) {
            const tel = this.trackedTelemetry.get(id);
            if (tel && tel.spawnLeft === 0 && tel.spawnTop === 0 && item.el) {
                tel.spawnLeft = parseFloat(item.el.style.left) || 0;
                tel.spawnTop = parseFloat(item.el.style.top) || 0;
                tel.spawnWidth = parseFloat(item.el.style.width) || 0;
                tel.spawnHeight = parseFloat(item.el.style.height) || 0;
            }
        }

        const now = Date.now();
        // Use copy of keys to safely delete during loop
        const keys = Array.from(this.trackedTelemetry.keys());

        for (const id of keys) {
            const tel = this.trackedTelemetry.get(id);
            if (!tel || tel.isCommitted) continue;

            const item = wItems.get(id);
            
            // If item has been deleted
            if (!item) {
                console.log(`[Telemetry] Tracked asset ${id} was deleted from canvas. Penalty applied.`);
                this.trackedTelemetry.delete(id);
                this.handleDeletionPenalty(tel.prompt);
                continue;
            }

            // Check if coordinates or sizes modified (Redirected)
            if (item.el) {
                const currentLeft = parseFloat(item.el.style.left) || 0;
                const currentTop = parseFloat(item.el.style.top) || 0;
                const currentWidth = parseFloat(item.el.style.width) || 0;
                const currentHeight = parseFloat(item.el.style.height) || 0;

                const moved = Math.abs(currentLeft - tel.spawnLeft) > 15 || Math.abs(currentTop - tel.spawnTop) > 15;
                const resized = Math.abs(currentWidth - tel.spawnWidth) > 15 || Math.abs(currentHeight - tel.spawnHeight) > 15;

                if (moved || resized) {
                    tel.hasRedirected = true;
                }
            }

            // Check if fusion parameters tuned
            const fProps = item.fusionProperties;
            if (fProps) {
                const isTuned = 
                    (fProps.brightness !== undefined && fProps.brightness !== 100) ||
                    (fProps.contrast !== undefined && fProps.contrast !== 100) ||
                    (fProps.saturation !== undefined && fProps.saturation !== 100) ||
                    (fProps.blur !== undefined && fProps.blur !== 0) ||
                    (fProps.sepia !== undefined && fProps.sepia > 0);
                if (isTuned) {
                    tel.hasFineTuned = true;
                }
            }

            // Check crop / isolation
            const label = item.label || '';
            if (label.includes('裁切') || label.includes('Crop') || item.crop) {
                tel.hasCropped = true;
            }
            if (label.includes('拆解') || label.includes('独立编辑') || label.includes('Isolated')) {
                tel.hasIsolatedEdit = true;
            }

            const stayedLongEnough = (now - tel.createdAt) >= 120000; // 2 minutes retention
            const interacted = tel.hasCropped || tel.hasIsolatedEdit || tel.hasFineTuned || tel.hasRedirected;

            if (stayedLongEnough && interacted) {
                console.log(`[Telemetry] Auto-committing asset ${id} due to 2 minutes retention & active interactions!`);
                tel.isCommitted = true;
                this.trackedTelemetry.delete(id);

                const state = (window as any).state || {};
                const imgSrc = item.dataUrl || item.originalDataUrl || item.sourceImage;
                this.registerImplicitSatisfaction(tel.prompt, imgSrc, state).catch(console.error);
            }
        }
    }

    /**
     * Start/Track a prompt sub-step inside a session
     */
    public recordPromptStep(prompt: string, sessionId: string) {
        if (!prompt || prompt.trim().length === 0) return;
        const cleanPrompt = prompt.trim();

        if (this.currentSessionId !== sessionId) {
            this.currentSessionId = sessionId;
            this.activeChain = [];
        }

        // Check for session/concept topic shift (End of Iteration Lineage)
        if (this.activeChain.length > 0) {
            const lastActivePromptInSession = this.activeChain[this.activeChain.length - 1];
            
            // If the incoming new prompt is completely unrelated to the last step of the current chain,
            // we conclude that the current branch is naturally closing!
            const isTopicShift = !this.arePromptsRelated(lastActivePromptInSession, cleanPrompt);
            if (isTopicShift) {
                console.log(`[Telemetry] Theme shift detected from "${lastActivePromptInSession}" to "${cleanPrompt}". Auto-committing previous leaf node.`);
                this.autoCommitLastLeafNodeOfChain(this.activeChain);
            }
        }

        // Add to outstanding session prompt chain if not identical to the last step
        if (this.activeChain.length === 0 || this.activeChain[this.activeChain.length - 1] !== cleanPrompt) {
            this.activeChain.push(cleanPrompt);
            console.log(`[ImplicitMemory] Chain appended: "${cleanPrompt}". Size: ${this.activeChain.length}`);
        }
    }

    /**
     * Triggered when user selects a generated image (Click "选用此图") / explicitly approves.
     * This establishes the connection: Chain Starting Prompt -> Chain Ending Prompt.
     */
    public async registerImplicitSatisfaction(selectedPromptOfImage?: string, imgSrc?: string, stateObj?: any) {
        let finalPrompt = selectedPromptOfImage?.trim();
        if (!finalPrompt && this.activeChain.length > 0) {
            finalPrompt = this.activeChain[this.activeChain.length - 1];
        }

        if (!finalPrompt) return;

        // Find the earliest prompt in this active chain that is related to finalPrompt
        let initialPrompt = this.activeChain[0] || finalPrompt;
        
        // If the chain had several steps, let's trace back to find the first prompt
        // which shares some core segments with the chosen prompt.
        for (const prompt of this.activeChain) {
            if (this.arePromptsRelated(prompt, finalPrompt)) {
                initialPrompt = prompt;
                break;
            }
        }

        // Try to retrieve matched workbench item for real parameters
        let fusionProps = null;
        let cropMetrics = null;
        let aspectRatio = null;
        let pTags = null;

        if (imgSrc && stateObj && stateObj.workbenchItems) {
            let matchedItem = null;
            for (const [id, item] of stateObj.workbenchItems.entries()) {
                if (item.dataUrl === imgSrc || item.originalDataUrl === imgSrc || item.sourceImage === imgSrc) {
                    matchedItem = item;
                    break;
                }
            }

            if (matchedItem) {
                if (matchedItem.fusionProperties) {
                    fusionProps = JSON.parse(JSON.stringify(matchedItem.fusionProperties));
                }
                if (matchedItem.crop) {
                    cropMetrics = JSON.parse(JSON.stringify(matchedItem.crop));
                }
                if (matchedItem.aspectRatio || (matchedItem.el && matchedItem.el.dataset.aspectRatio)) {
                    aspectRatio = matchedItem.aspectRatio || matchedItem.el.dataset.aspectRatio;
                }
                
                // Automatically generate elegant descriptive tags based on properties to display beautifully in the UI!
                pTags = [];
                if (fusionProps) {
                    if (fusionProps.brightness > 102) pTags.push('高清增亮');
                    else if (fusionProps.brightness < 98) pTags.push('极简暗色调');
                    
                    if (fusionProps.contrast > 102) pTags.push('高动态对比');
                    if (fusionProps.saturation > 105) pTags.push('极佳色彩饱和');
                    else if (fusionProps.saturation < 95) pTags.push('古典复古调');
                    
                    if (fusionProps.blur > 1) pTags.push('背景深虚化');
                }
                if (cropMetrics) {
                    pTags.push('比例拆解剪裁');
                }
                if (aspectRatio) {
                    pTags.push(`${aspectRatio} 比例`);
                }
                if (pTags.length === 0) {
                    pTags = ['双通道融合', '高保真参数'];
                }
            }
        }

        await this.addOrUpdateRule(initialPrompt, finalPrompt, fusionProps, cropMetrics, aspectRatio, pTags);
        console.log(`[ImplicitMemory] Implicit Satisfaction Recorded! Learned flow: "${initialPrompt}" ===> "${finalPrompt}"`);
    }

    /**
     * Helper to check if two prompt texts are semantically related (sharing words or keywords)
     */
    private arePromptsRelated(p1: string, p2: string): boolean {
        const words = this.extractStemKeywords(p1);
        if (words.length === 0) return false;
        // See if p2 contains any of p1's key segments
        return words.some(w => p2.toLowerCase().includes(w.toLowerCase()));
    }

    /**
     * Extract clean searchable words/keywords from a Chinese/English prompt
     */
    public extractStemKeywords(prompt: string): string[] {
        // Strip punctuation
        const clean = prompt.replace(/[，、。！？；：,.!?;:]/g, ' ').toLowerCase();
        // Split by whitespace
        const tokens = clean.split(/\s+/).filter(t => t.length >= 2);
        
        // Filter out super common stopwords
        const stopwords = ['一个', '一张', '这个', '那个', '请帮', '设计', '生成', '制造', 'with', 'and', 'the', 'a', 'of'];
        return tokens.filter(t => !stopwords.includes(t));
    }

    /**
     * Add rule or increment count
     */
    private async addOrUpdateRule(
        initial: string, 
        final: string, 
        fusionProps?: any, 
        cropMetrics?: any, 
        aspectRatio?: string,
        presetTags?: string[]
    ) {
        if (!initial || !final || initial.trim() === final.trim()) {
            // Also store single prompts if they are substantial (length > 10) to help prompt autocompletion
            if (initial && initial.trim().length > 8) {
                const existing = this.rules.find(r => r.initialPrompt.toLowerCase() === initial.trim().toLowerCase());
                if (existing) {
                    existing.count++;
                    existing.lastUsed = Date.now();
                    if (fusionProps) existing.fusionProperties = fusionProps;
                    if (cropMetrics) existing.crop = cropMetrics;
                    if (aspectRatio) existing.aspectRatio = aspectRatio;
                    if (presetTags) existing.presetTags = presetTags;
                } else {
                    const kws = this.extractStemKeywords(initial);
                    this.rules.push({
                        id: 'learned-' + Math.random().toString(36).substring(2, 7),
                        keywords: kws,
                        initialPrompt: initial.trim(),
                        finalPrompt: final.trim(),
                        count: 1,
                        lastUsed: Date.now(),
                        sessionId: this.currentSessionId,
                        fusionProperties: fusionProps,
                        crop: cropMetrics,
                        aspectRatio: aspectRatio,
                        presetTags: presetTags
                    });
                }
                await this.saveToStorage();
            }
            return;
        }

        const cleanInitial = initial.trim();
        const cleanFinal = final.trim();

        const existingIdx = this.rules.findIndex(r => r.initialPrompt.toLowerCase() === cleanInitial.toLowerCase());
        if (existingIdx !== -1) {
            this.rules[existingIdx].finalPrompt = cleanFinal;
            this.rules[existingIdx].count++;
            this.rules[existingIdx].lastUsed = Date.now();
            this.rules[existingIdx].keywords = Array.from(new Set([...this.rules[existingIdx].keywords, ...this.extractStemKeywords(cleanInitial)]));
            if (fusionProps) this.rules[existingIdx].fusionProperties = fusionProps;
            if (cropMetrics) this.rules[existingIdx].crop = cropMetrics;
            if (aspectRatio) this.rules[existingIdx].aspectRatio = aspectRatio;
            if (presetTags) this.rules[existingIdx].presetTags = presetTags;
        } else {
            const kws = this.extractStemKeywords(cleanInitial);
            this.rules.push({
                id: 'learned-' + Math.random().toString(36).substring(2, 7),
                keywords: kws,
                initialPrompt: cleanInitial,
                finalPrompt: cleanFinal,
                count: 1,
                lastUsed: Date.now(),
                sessionId: this.currentSessionId,
                fusionProperties: fusionProps,
                crop: cropMetrics,
                aspectRatio: aspectRatio,
                presetTags: presetTags
            });
        }
        await this.saveToStorage();
    }

    /**
     * Match user's ongoing typing and suggest a perfected final prompt!
     */
    public findMatch(currentInput: string): SatisfactionRule | null {
        if (!currentInput || currentInput.trim().length < 2) return null;
        const clean = currentInput.trim().toLowerCase();

        // 1. Try exact initialPrompt start matches
        let match = this.rules.find(r => r.initialPrompt.toLowerCase().startsWith(clean) && r.finalPrompt.length > clean.length);
        if (match) return match;

        // 2. Try partial substring matches on initialPrompt
        match = this.rules.find(r => r.initialPrompt.toLowerCase().includes(clean) && r.finalPrompt.length > r.initialPrompt.length);
        if (match) return match;

        // 3. Match via keywords
        const keywords = this.extractStemKeywords(currentInput);
        if (keywords.length > 0) {
            // Find rule that shares the most keywords
            let bestRule: SatisfactionRule | null = null;
            let maxOverlap = 0;

            for (const rule of this.rules) {
                const overlap = rule.keywords.filter(kw => keywords.some(userKw => kw.includes(userKw) || userKw.includes(kw))).length;
                if (overlap > maxOverlap && rule.finalPrompt.length > currentInput.length) {
                    maxOverlap = overlap;
                    bestRule = rule;
                }
            }

            if (bestRule && maxOverlap >= 1) {
                return bestRule;
            }
        }

        return null;
    }

    public getRules(): SatisfactionRule[] {
        return this.rules;
    }
}

export const implicitMemoryEngine = new ImplicitMemoryManager();
// @ts-ignore
if (typeof window !== 'undefined') {
    // @ts-ignore
    window.implicitMemoryEngine = implicitMemoryEngine;
}
