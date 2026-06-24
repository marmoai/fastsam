export class StrategyDispatcher {
    /**
     * 多策略智能调度系统 (Strategy Dispatcher)
     * 分析图像局部特征，决策最优抠图路径，避免大模型滥用，大幅提升系统吞吐量。
     * 
     * @param {ImageData} cropImageData - 待处理裁剪图的原生像素数据
     * @param {string} layerName - 语义标签（如 "人", "云", "椅子"）
     * @returns {Object} { path: 'cv' | 'hybrid' | 'ai', reason: '原因', metrics: {} }
     */
    static analyze(cropImageData, layerName) {
        const { width, height, data } = cropImageData;
        
        let validBgPixels = 0;
        let sumR = 0, sumG = 0, sumB = 0;
        
        // 1. 获取背景方差 (Background Variance)
        // 对外围一圈 (约 5% 边距) 进行采样，建立高斯分布的均值与方差
        const sampleSize = Math.max(5, Math.floor(Math.min(width, height) * 0.05));
        const bgSamples = [];
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (x < sampleSize || x >= width - sampleSize || y < sampleSize || y >= height - sampleSize) {
                    const i = (y * width + x) * 4;
                    // Ignore fully transparent pixels in metric logic
                    if (data[i+3] < 10) continue; 
                    
                    const r = data[i], g = data[i+1], b = data[i+2];
                    bgSamples.push({r, g, b});
                    sumR += r; sumG += g; sumB += b;
                    validBgPixels++;
                }
            }
        }
        
        let bgMean = { r: 0, g: 0, b: 0 };
        let bgVariance = 0;
        
        if (validBgPixels > 0) {
            bgMean = {
                r: sumR / validBgPixels,
                g: sumG / validBgPixels,
                b: sumB / validBgPixels
            };
            
            let totalVariance = 0;
            for (const p of bgSamples) {
                 totalVariance += Math.pow(p.r - bgMean.r, 2) + Math.pow(p.g - bgMean.g, 2) + Math.pow(p.b - bgMean.b, 2);
            }
            bgVariance = totalVariance / validBgPixels;
        }

        // 2. 获取前景对比度 (Foreground-Background Contrast)
        // 获取图中心区域特征均值
        let fgValidPixels = 0;
        let fgSumR = 0, fgSumG = 0, fgSumB = 0;
        const cxMin = Math.floor(width * 0.3);
        const cxMax = Math.floor(width * 0.7);
        const cyMin = Math.floor(height * 0.3);
        const cyMax = Math.floor(height * 0.7);
        
        for (let y = cyMin; y < cyMax; y++) {
            for (let x = cxMin; x < cxMax; x++) {
                const i = (y * width + x) * 4;
                 if (data[i+3] < 10) continue;
                fgSumR += data[i]; fgSumG += data[i+1]; fgSumB += data[i+2];
                fgValidPixels++;
            }
        }
        
        let fgMean = { r: 127, g: 127, b: 127 };
        
        // 💡 [Chroma Load Tracking] 
        // Track the presence of pure colors in the center to protect glowing edges.
        // If an object has a golden rim, it has high R - max(G,B). We must avoid R background!
        let fgChroma = { r: 0, g: 0, b: 0, m: 0 }; 

        if (fgValidPixels > 0) {
            fgMean = {
                r: fgSumR / fgValidPixels,
                g: fgSumG / fgValidPixels,
                b: fgSumB / fgValidPixels
            };
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = (y * width + x) * 4;
                    if (data[i+3] < 10) continue;
                    const r = data[i], g = data[i+1], b = data[i+2];
                    
                    // Track max chroma conflict for each potential backdrop color
                    fgChroma.r = Math.max(fgChroma.r, r - Math.max(g, b));
                    fgChroma.g = Math.max(fgChroma.g, g - Math.max(r, b));
                    fgChroma.b = Math.max(fgChroma.b, b - Math.max(r, g));
                    fgChroma.m = Math.max(fgChroma.m, Math.min(r, b) - g);
                }
            }
        }
        
        // 算出前后景对比视觉差向量长度
        const fgBgDistance = Math.sqrt(
            Math.pow(fgMean.r - bgMean.r, 2) + 
            Math.pow(fgMean.g - bgMean.g, 2) + 
            Math.pow(fgMean.b - bgMean.b, 2)
        );

        // 3. 高能语义冲突拦截器 (Semantic NLP Rule Engine)
        const names = layerName || "";
        const isTranslucentComplex = /云|烟|雾|火|光|冰|发|水|纱|网|毛|尘|影/i.test(names);
        const isBiologicalEntity = /人|脸|手|腿|脚|猫|狗|鸟|动物|毛发/i.test(names);
        const isUIElement = /UI|ui|字|文本|框|图标|按钮/i.test(names);

        const metrics = { fgBgDistance, bgVariance, bgMean, validBgPixels, fgMean, fgChroma };

        // ----------------------------------------------------
        // 路由决策树 (Decision Tree)
        // ----------------------------------------------------

        if (validBgPixels < 10) {
            // Already transparent or cropped too tight.
            return { path: 'cv', reason: `无可测量背景 (已透明或裁剪极小), 直接透传走极简CV`, metrics };
        }

        // [路由 1]: 背景极度干净，且前景与背景差异显著 -> 直接 CV 秒解 (CV Path)
        // 方差小于 80 意味着背景像素波动不到 5个RGB步进，属于绝对纯色环境（如电商白底、UI图层）
        // **关键修复**：排除语义判定为半透明/流体/发丝的实体（如“云”）。例如蓝天白云，蓝天方差极小满足条件，但云是半透明流体，走纯CV会导致边缘惨烈硬切。
        if (!isTranslucentComplex && bgVariance < 80 && fgBgDistance > 30) {
            return { path: 'cv', reason: `低背景方差 (${bgVariance.toFixed(1)}) 且 高前景剥离度 (${fgBgDistance.toFixed(1)}), 启动毫秒级前端 CV 解算`, metrics };
        }
        
        if (isUIElement) {
             return { path: 'cv', reason: `UI素材识别，直接启动 CV 解算以防大模型污染像素`, metrics };
        }

        // ----------------------------------------------------
        // NEVER fallback to AI mask.
        // If scene is complex, it means we MUST rely MORE on hybrid chroma separation,
        // not bypass it.
        // ----------------------------------------------------
        
        // [路由 2]: 默认主力通道 -> 生成式绿幕换底 (Hybrid Path)
        return { path: 'hybrid', reason: `混合场景, 方差 (${bgVariance.toFixed(1)}), 无条件交由 AI 绿幕互补判定后走本地精准 Matte`, metrics };
    }
}
