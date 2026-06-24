import { DESIGN_PATTERNS, DesignPattern } from './DesignPatternSystem';
import { memoryLayer } from './CreativeMemoryLayer';

export class SemanticRecommender {
    static recommend(asset: any): { recommended: DesignPattern[], others: DesignPattern[] } {
        // Collect context texts from the asset
        const keywords = new Set<string>();
        
        if (asset) {
            if (asset.type) keywords.add(asset.type.toLowerCase());
            if (asset.name) keywords.add(asset.name.toLowerCase());
            if (asset.prompt) keywords.add(asset.prompt.toLowerCase());
            if (asset.metadata && Array.isArray(asset.metadata.tags)) {
                asset.metadata.tags.forEach((t: string) => keywords.add(t.toLowerCase()));
            }
        }
        
        // Convert to a single string for easy regex matching
        const contextStr = Array.from(keywords).join(' ');

        // Graph Intelligence & Data Flywheel
        const allLogs = memoryLayer.getLogs();
        
        const globalIntentCounts: Record<string, number> = {};
        const contextMatchedIntentCounts: Record<string, number> = {};
        
        // Track unique custom manual prompt triggers to feed Point 4 Co-Creation Memory
        const customPromptIntents = new Set<string>();

        // Track the deltas for dynamically generated intents
        const dynamicIntentDeltas: Record<string, any[]> = {};

        allLogs.forEach(log => {
            if (log.intent) {
                globalIntentCounts[log.intent] = (globalIntentCounts[log.intent] || 0) + 1;
                
                let overlap = 0;
                if (log.context) {
                    const historicalKeywords = log.context.toLowerCase();
                    keywords.forEach(kw => {
                        if (kw.length > 2 && historicalKeywords.includes(kw)) {
                            overlap++;
                        }
                    });
                    if (overlap > 0) {
                        contextMatchedIntentCounts[log.intent] = (contextMatchedIntentCounts[log.intent] || 0) + (overlap * 3);
                    }
                }
                
                // If it's a creative manual NL intent prompt, remember it for Point 4 synthesis
                if (
                    (log.actionType === 'CO_CREATE_ISOLATED' || log.actionType === 'CO_CREATE_SYNC') &&
                    log.intent.trim().length > 1 &&
                    log.intent.trim().length < 40 &&
                    !log.intent.includes('[Creative')
                ) {
                    customPromptIntents.add(log.intent.trim());
                }

                // Track stateDelta if it's a structural adjustment (e.g., fusionProperties)
                if (log.stateDelta && log.stateDelta.before && log.stateDelta.after) {
                    const b = log.stateDelta.before.fusionProperties || {};
                    const a = log.stateDelta.after.fusionProperties || {};
                    
                    const delta: any = {};
                    let hasDelta = false;
                    Object.keys(a).forEach(k => {
                        const bVal = b[k] || 100; // default 100 (for old values if missing)
                        const aVal = a[k] || 100;
                        if (typeof aVal === 'number' && typeof bVal === 'number' && aVal !== bVal) {
                            delta[k] = aVal - bVal;
                            hasDelta = true;
                        }
                    });
                    
                    if (hasDelta) {
                        if (!dynamicIntentDeltas[log.intent]) dynamicIntentDeltas[log.intent] = [];
                        dynamicIntentDeltas[log.intent].push(delta);
                    }
                }
            }
        });

        const scoredPatterns: { pattern: DesignPattern, score: number }[] = [];
        
        // 1. Process Baseline / Static patterns (Point 2: Adaptive Data Flywheel Multiplier)
        DESIGN_PATTERNS.forEach(pattern => {
            let score = 0;
            if (pattern.id === 'make_appetizing' && contextStr.match(/food|burger|pizza|cake|meat|salad|食物|汉堡|披萨|餐饮/)) score += 20;
            if (pattern.id === 'cyberpunk_style' && contextStr.match(/city|neon|night|car|street|future|赛博朋克|夜晚|城市|未来/)) score += 20;
            if (pattern.id === 'increase_depth' && contextStr.match(/portrait|face|person|产品|静物|人像|特写/)) score += 15;
            if (pattern.id === 'highlight_subject' && contextStr.match(/product|item|shoe|bag|car|subject|产品|商品|主体/)) score += 15;
            if (pattern.id === 'faded_cinematic' && contextStr.match(/landscape|view|nature|sky|movie|电影感|风景|自然|胶片/)) score += 15;

            // Highly responsive multiplier: if user applies these multiple times, float to best-match area
            const globalCount = globalIntentCounts[pattern.intent] || 0;
            const contextualCount = contextMatchedIntentCounts[pattern.intent] || 0;
            
            score += (globalCount * 12);  // Substantial flywheel boost
            score += (contextualCount * 25);
            if (pattern.id === 'increase_depth' && score === 0) score += 2; 

            scoredPatterns.push({ pattern, score });
        });

        // 2. Point 4 Co-Creation: Synthesize and learn from Custom manual Sliders / Active Adjustments
        if (asset && asset.fusionProperties) {
            const defaults: Record<string, number> = {
                brightness: 100,
                contrast: 100,
                saturation: 100,
                warmth: 0,
                blur: 0,
                grayscale: 0,
                sepia: 0,
                hueRotate: 0
            };

            const activeProps: Record<string, number> = {};
            let isCustomized = false;
            Object.keys(defaults).forEach(k => {
                const val = asset.fusionProperties[k] !== undefined ? asset.fusionProperties[k] : defaults[k];
                if (val !== defaults[k]) {
                    activeProps[k] = val;
                    isCustomized = true;
                }
            });

            if (isCustomized) {
                const descStr = Object.entries(activeProps).map(([k, v]) => `${k}:${v}`).join(', ');
                const habitPattern: DesignPattern = {
                    id: `active_habit_preset_${asset.uid}`,
                    name: `⚡ 复刻当前微调习惯 (Slider Sync)`,
                    intent: `复刻微调习惯 [${descStr}]`,
                    description: `应用当前图层微调: ${descStr}`,
                    emoji: '⚡',
                    apply: (dispatcher, targetUid) => {
                        dispatcher.dispatch({
                            type: 'UPDATE_FUSION',
                            intent: '应用一键提取的个性化配色方案',
                            payload: {
                                uid: targetUid,
                                fusionProperties: { ...activeProps }
                            }
                        });
                    }
                };
                scoredPatterns.push({ pattern: habitPattern, score: 99 }); // Put at top for easy reuse
            }
        }

        // 3. Point 4 Co-Creation: Synthesize and display user the past typed successful NL Prompt Commands
        Array.from(customPromptIntents).forEach((promptText, idx) => {
            let score = 30; // Solid baseline score to appear in suggestions
            // Match with keywords
            keywords.forEach(kw => {
                if (kw.length > 2 && promptText.toLowerCase().includes(kw)) {
                    score += 15;
                }
            });

            const promptWordPattern: DesignPattern = {
                id: `dynamic_nl_intent_${idx}`,
                name: `🪄 自学创意: "${promptText}"`,
                intent: promptText,
                description: `快捷键命令: 同步创意指令到当前图层`,
                emoji: '🔮',
                apply: (dispatcher, targetUid) => {
                    dispatcher.dispatch({
                        type: 'CO_CREATE_SYNC',
                        intent: promptText,
                        payload: {
                            uid: targetUid,
                            prompt: promptText
                        }
                    });
                }
            };
            scoredPatterns.push({ pattern: promptWordPattern, score });
        });

        // 4. Synthesize experience patterns from Data Flywheel records
        Object.keys(dynamicIntentDeltas).forEach(intentName => {
            // Only convert if it's not already in DESIGN_PATTERNS
            if (!DESIGN_PATTERNS.find(p => p.intent === intentName)) {
                const deltas = dynamicIntentDeltas[intentName];
                
                // Average the deltas
                const avgDelta: any = {};
                deltas.forEach(d => {
                    Object.keys(d).forEach(k => {
                        avgDelta[k] = (avgDelta[k] || 0) + d[k];
                    });
                });
                
                Object.keys(avgDelta).forEach(k => {
                    avgDelta[k] = Math.round(avgDelta[k] / deltas.length);
                });

                let score = 0;
                score += (globalIntentCounts[intentName] || 0) * 2;
                score += (contextMatchedIntentCounts[intentName] || 0) * 10; // High confidence if context matches

                // Create a dynamic pattern if score is decent or we have enough data points
                if (score > 0 || deltas.length > 2) {
                    const dynamicPattern: DesignPattern = {
                        id: `dynamic_${intentName.replace(/\s+/g, '_').toLowerCase()}`,
                        name: `💡 经验: ${intentName}`,
                        intent: intentName,
                        description: '基于过往决策提取的数据策略',
                        emoji: '🧠',
                        apply: (dispatcher, targetUid) => {
                            // First, get the current entity to re-base the avgDelta
                            // Actually, simply dispatch a structural update that applies the delta to the current value
                            // We construct a fusionProperties object to merge in
                            
                            // To actually do this, dispatcher needs to handle 'UPDATE_FUSION_DELTA' 
                            // Or we read current state via workspace.
                            // However, we pass dispatch, we can construct the new properties if we have current state.
                            // But here we're inside SemanticRecommender. Let's just dispatch UPDATE_FUSION_DELTA
                            // or fetch current asset from workspace... 
                            // Let's assume dispatcher can handle getting the current value, or we use the asset we got inside SemanticRecommender
                            const currentTarget = dispatcher.workspace?.currentState.assetRegistry.get(targetUid) || { fusionProperties: {} };
                            const currentFusion = currentTarget.fusionProperties || {};
                            
                            const newFusion = { ...currentFusion };
                            Object.keys(avgDelta).forEach(k => {
                                const baseVal = newFusion[k] !== undefined ? newFusion[k] : 100;
                                newFusion[k] = baseVal + avgDelta[k];
                            });

                            dispatcher.dispatch({
                                type: 'UPDATE_FUSION',
                                intent: intentName,
                                payload: {
                                    uid: targetUid,
                                    fusionProperties: newFusion
                                }
                            });
                        }
                    };
                    
                    scoredPatterns.push({ pattern: dynamicPattern, score });
                }
            }
        });

        // Sort descending by score
        scoredPatterns.sort((a, b) => b.score - a.score);

        const recommended: DesignPattern[] = [];
        const others: DesignPattern[] = [];

        scoredPatterns.forEach((item, index) => {
            if (index < 3 && item.score > 0) { // Bumped up top choices to 3
                recommended.push(item.pattern);
            } else {
                others.push(item.pattern);
            }
        });

        if (recommended.length === 0 && others.length > 0) {
            recommended.push(others.shift()!);
            if (others.length > 0) recommended.push(others.shift()!);
        }

        return { recommended, others };
    }
}
