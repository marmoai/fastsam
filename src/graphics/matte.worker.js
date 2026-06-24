import { AdaptiveMatteEngine } from './matte-engine.js';

self.onmessage = async (e) => {
    const { taskId, imageDataArray, width, height, layerName, deepMaskArray, options } = e.data;

    try {
        self.postMessage({ taskId, status: 'progress', progressMsg: '⚡ [Worker计算层] 背景分布建模与结构分析...' });
        
        // Instantiate the purely mathematical engine (no DOM/canvas)
        const engine = new AdaptiveMatteEngine();
        engine.data = imageDataArray;
        engine.w = width;
        engine.h = height;
        
        const type = options?.type || engine.analyzeScene(layerName);
        
        if (type !== 'solid') {
            engine.sampleBackground();
            engine.normalizeBackground();
        }

        self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 场景分析 (Task: ${type})...` });

        let alphaData;
        if (type === 'green_screen') {
            self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 强迫绿幕穿透算法 (Task: Generative Chroma Cut)...` });
            engine.bgR = options.bgColor[0];
            engine.bgG = options.bgColor[1];
            engine.bgB = options.bgColor[2];
            alphaData = engine.globalChromaKey(); 
            alphaData = engine.refineEdge(alphaData, 'hard'); 
        } else if (type === 'cv_euclidean') {
            self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 前端极速空间色差解算 (Task: Euclidean Cut)...` });
            if (options && options.bgColor) {
                engine.bgR = options.bgColor[0];
                engine.bgG = options.bgColor[1];
                engine.bgB = options.bgColor[2];
            }
            alphaData = engine.globalChromaKey();
            // Do not use hard refine here to keep anti-aliasing safe on simple BGs
        } else if (type === 'solid') {
            self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 纯色背景穿透检测，启动反走样抠像 (Task: AA Chroma Key)...` });
            
            // CRITICAL FIX: DO NOT blindly use the whole image to sample the background.
            // A flat foreground object (like a smooth black headphone case) can trick engine.sampleBackground().
            // Because we added 15% padding to the bounding box during extraction, we KNOW the outer 5-10% 
            // is 100% the generated solid background. So we strictly sample the borders to find the TRUE generated color (which might have slight variations from pure hex due to AI).
            let sumR = 0, sumG = 0, sumB = 0, count = 0;
            const marginX = engine.w * 0.05;
            const marginY = engine.h * 0.05;
            
            for (let y = 0; y < engine.h; y += 3) {
                for (let x = 0; x < engine.w; x += 3) {
                    if (x < marginX || x >= engine.w - marginX || y < marginY || y >= engine.h - marginY) {
                        const idx = ((y * engine.w) + x) * 4;
                        sumR += engine.data[idx];
                        sumG += engine.data[idx+1];
                        sumB += engine.data[idx+2];
                        count++;
                    }
                }
            }
            if (count > 0) {
                engine.bgR = sumR / count;
                engine.bgG = sumG / count;
                engine.bgB = sumB / count;
            } else if (options && options.bgColor) {
                engine.bgR = options.bgColor[0];
                engine.bgG = options.bgColor[1];
                engine.bgB = options.bgColor[2];
            }
            
            alphaData = engine.fastChromaKey(deepMaskArray);
            // DO NOT apply refineEdge('hard') here. It uses Math.min (erosion) 
            // which violently Destroys thin structures like dark chandelier rods!
            // fastChromaKey now natively outputs perfect anti-aliased S-curve masks.
        } else if (type === 'channel' || type === 'channel_hair' || type === 'channel_fabric') {
            const typeName = type === 'channel_fabric' ? '半透明织物' : (type === 'channel' ? '半透高反差材质' : '细腻毛发');
            self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 检测到${typeName}，启动物理通道分离 (Task: PS Channel Matte)...` });
            alphaData = engine.channelMatting(deepMaskArray, type);
            if (type === 'channel') {
                alphaData = engine.refineEdge(alphaData, 'soft'); // Optional feather matching PS smoothing
            } else {
                alphaData = engine.refineEdge(alphaData, 'hair'); // Zero feather for fabric/hair
            }
        } else {
            self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 执行自适应融合并提取强语义边缘 (Task: ${type})...` });
            alphaData = engine.adaptiveFusion(deepMaskArray, type);
            alphaData = engine.refineEdge(alphaData, type);
        }

        self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 剔除孤岛散落像素 (Area Threshold Filtering)...` });
        alphaData = engine.filterIsolatedIslands(alphaData);

        self.postMessage({ taskId, status: 'progress', progressMsg: `⚙️ [Worker计算层] 执行边缘光晕反解与颜色净化 (Decontamination)...` });
        engine.applyAntiHaloAndOutput(alphaData, type); // Mutates engine.data in place

        // Return the final processed ImageData buffer
        // Note: engine.data now contains the final output (RGB intact, Alpha modified limitlessly)
        self.postMessage({ 
            taskId, 
            status: 'done', 
            payload: new Uint8ClampedArray(engine.data)
        }, [engine.data.buffer]);
        
    } catch (err) {
        self.postMessage({ taskId, status: 'error', error: err.stack || err.message });
    }
};
