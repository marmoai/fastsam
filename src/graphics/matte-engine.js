// Pure Math Engine - Zero DOM dependencies. Safe for Web Workers!
export class AdaptiveMatteEngine {
    constructor() {
        // Will be populated dynamically by the controller/worker
        this.w = 0;
        this.h = 0;
        this.data = null; // Uint8ClampedArray
        
        // Dynamic Sampling Results
        this.bgR = 0;
        this.bgG = 0;
        this.bgB = 0;
        this.bgBrightness = 0;
        this.bgMin = [255, 255, 255];
        this.bgMax = [0, 0, 0];
    }
 
    /* =========================================================
       STEP 0: Dynamic Background Sampling (Low-Frequency Distribution Analysis)
       ========================================================= */
    sampleBackground() {
        let sumR = 0, sumG = 0, sumB = 0;
        let count = 0;
        let minR = 255, minG = 255, minB = 255;
        let maxR = 0, maxG = 0, maxB = 0;

        // Skip borders to avoid edge artifacts, sample across the image
        const step = 4; // Sample every 4th pixel to save compute
        for (let y = 2; y < this.h - 2; y += step) {
            for (let x = 2; x < this.w - 2; x += step) {
                const idx = (y * this.w + x) * 4;
                const r = this.data[idx];
                const g = this.data[idx+1];
                const b = this.data[idx+2];
                
                // Calculate spatial gradient (difference between current pixel and right/bottom neighbors)
                const rightIdx = (y * this.w + (x + 2)) * 4;
                const bottomIdx = ((y + 2) * this.w + x) * 4;
                
                const gradR = Math.abs(r - this.data[rightIdx]) + Math.abs(r - this.data[bottomIdx]);
                const gradG = Math.abs(g - this.data[rightIdx+1]) + Math.abs(g - this.data[bottomIdx+1]);
                const gradB = Math.abs(b - this.data[rightIdx+2]) + Math.abs(b - this.data[bottomIdx+2]);
                
                const gradient = gradR + gradG + gradB;

                // Threshold for "low frequency" flat areas -> visually stable background
                if (gradient < 15) {
                    sumR += r; sumG += g; sumB += b;
                    count++;
                }
            }
        }

        // Fallback in case the whole image is highly textured
        if (count < 50) {
            console.log("[MatteEngine] Low-frequency sampling failed, falling back to corners.");
            let corners = [0, this.w-1, (this.h-1)*this.w, (this.h-1)*this.w + this.w-1];
            sumR = 0; sumG = 0; sumB = 0; count = 0;
            for (let c of corners) {
                const r = this.data[c*4]; const g = this.data[c*4+1]; const b = this.data[c*4+2];
                sumR += r; sumG += g; sumB += b;
                count++;
            }
        }

        if (count > 0) {
            this.bgR = sumR / count;
            this.bgG = sumG / count;
            this.bgB = sumB / count;
        } else {
            this.bgR = this.bgG = this.bgB = 0;
        }
        
        this.bgBrightness = (this.bgR + this.bgG + this.bgB) / 3;
        
        // Stabilize distribution: Statistical tolerance box instead of absolute min/max which is highly susceptible to outliers
        const tol = 15;
        this.bgMin = [Math.max(0, this.bgR - tol), Math.max(0, this.bgG - tol), Math.max(0, this.bgB - tol)];
        this.bgMax = [Math.min(255, this.bgR + tol), Math.min(255, this.bgG + tol), Math.min(255, this.bgB + tol)];
        
        console.log(`[MatteEngine] Sampled BG - Mean: RGB(${this.bgR.toFixed(0)},${this.bgG.toFixed(0)},${this.bgB.toFixed(0)}), Range Configured to Tolerance ${tol}`);
    }

    // Note: The `process` wrapper was moved to MatteWorker. The execution order is preserved there.
    /* =========================================================
       STEP 3: Background Normalize
       ========================================================= */
    normalizeBackground() {
        const threshold = 15;
        for (let i = 0; i < this.data.length; i += 4) {
            const r = this.data[i], g = this.data[i+1], b = this.data[i+2];
            if (Math.abs(r - this.bgR) < threshold &&
                Math.abs(g - this.bgG) < threshold &&
                Math.abs(b - this.bgB) < threshold) {
                this.data[i] = this.bgR;
                this.data[i+1] = this.bgG;
                this.data[i+2] = this.bgB;
            }
        }
    }

    /* =========================================================
       STEP 4: Scene / Subject Analyzer with Tri-Track Routing
       ========================================================= */
    isPureBackground() {
        // Sample edges to determine if the background is a generated flat/solid color.
        let sumR = 0, sumG = 0, sumB = 0;
        let count = 0;
        const w = this.w, h = this.h;
        const edgePixels = [];

        // Sample border pixels at intervals
        for(let x=0; x<w; x+=10) {
            edgePixels.push(x); // Top
            edgePixels.push((h-1)*w + x); // Bottom
        }
        for(let y=0; y<h; y+=10) {
            edgePixels.push(y*w); // Left
            edgePixels.push(y*w + w-1); // Right
        }

        for (const idx of edgePixels) {
            const dataIdx = idx * 4;
            sumR += this.data[dataIdx];
            sumG += this.data[dataIdx+1];
            sumB += this.data[dataIdx+2];
            count++;
        }
        
        const avgR = sumR / count, avgG = sumG / count, avgB = sumB / count;
        
        let maxDist = 0;
        let totalDev = 0;
        for (const idx of edgePixels) {
            const dataIdx = idx * 4;
            const r = this.data[dataIdx], g = this.data[dataIdx+1], b = this.data[dataIdx+2];
            const dist = Math.sqrt((r-avgR)**2 + (g-avgG)**2 + (b-avgB)**2);
            if (dist > maxDist) maxDist = dist;
            totalDev += dist;
        }

        const avgDev = totalDev / count;
        console.log(`[MatteEngine] Router Check - BG Pureness -> AvgDev: ${avgDev.toFixed(2)}, MaxDist: ${maxDist.toFixed(2)}`);

        // STRICT TIGHTENING (Adjusted for Generative Green-Screen): 
        // AI-generated solid plates are very uniform, but might have slight JPEG noise or minor dithering.
        // A real world wall with shadows will easily have maxDist > 50 and avgDev > 20.
        // We set thresholds tightly enough to block real walls, but allow pure AI plates.
        return (avgDev < 6.0 && maxDist < 40);
    }

    analyzeScene(layerName) {
        // Track 1: Semantic Intent Check (Physical Channel / Fine Detail Extractions)
        // Highest priority: If the layer is named explicitly as a transparency/particle type, 
        // DO NOT let the pure background detector greedy-capture it.
        if (layerName) {
            const isVolumetric = layerName.includes('云') || layerName.includes('烟') || layerName.includes('雾') || layerName.includes('火') || layerName.includes('光') || layerName.includes('冰') || layerName.includes('透明') || layerName.includes('水') || layerName.includes('玻璃');
            const isFineDetail = (layerName.includes('发') && !layerName.includes('沙发')) || layerName.includes('毛') || layerName.includes('羽') || layerName.includes('树') || layerName.includes('草') || layerName.includes('叶') || layerName.includes('线') || layerName.includes('网');
            const isTranslucentFabric = layerName.includes('纱') || layerName.includes('窗帘') || layerName.includes('蕾丝') || layerName.includes('丝绸') || layerName.includes('轻纱');

            if (isTranslucentFabric) return 'channel_fabric';
            if (isVolumetric) return 'channel';
            if (isFineDetail) return 'channel_hair';
        }

        // Track 2: Solid Background Fast-Path (Product/Hard Edges on pure backgrounds)
        if (this.isPureBackground()) {
            return 'solid';
        }

        // Track 3: Normal / Complex AI Routing
        let softPixels = 0;
        let edgePixels = 0;
        let varianceTotal = 0;
        const total = this.w * this.h;
        
        for (let y = 1; y < this.h - 1; y++) {
            for (let x = 1; x < this.w - 1; x++) {
                const i = (y * this.w + x) * 4;
                const r = this.data[i], g = this.data[i+1], b = this.data[i+2];
                
                // Diff to background
                const dist = Math.sqrt((r-this.bgR)**2 + (g-this.bgG)**2 + (b-this.bgB)**2);
                if (dist > 15 && dist < 120) softPixels++;

                // Simple Edge Complexity (Sobel-like)
                const upR = this.data[((y-1)*this.w + x)*4];
                const leftR = this.data[(y*this.w + (x-1))*4];
                const edgeStr = Math.abs(r - upR) + Math.abs(r - leftR);
                if (edgeStr > 15) edgePixels++;
                
                varianceTotal += Math.abs(r-g) + Math.abs(g-b) + Math.abs(r-b);
            }
        }
        
        const softRatio = softPixels / total;
        const edgeRatio = edgePixels / total;
        const avgVariance = varianceTotal / total;
        
        console.log(`[Analyzer Stats] softRatio: ${softRatio.toFixed(3)}, edgeRatio: ${edgeRatio.toFixed(3)}, avgVariance: ${avgVariance.toFixed(1)}`);
        
        if (softRatio > 0.4) return 'soft';
        if (edgeRatio > 0.08 && softRatio > 0.1) return 'hair';
        if (edgeRatio > 0.15) return 'hard';
        return 'hard'; // Default fallback
    }

    /* =========================================================
       NEW V4: Physical Channel Segmentation (PS Levels Approach)
       ========================================================= */
    chooseBestChannel() {
        let scoreR = 0, scoreG = 0, scoreB = 0;

        for (let i = 0; i < this.data.length; i += 4) {
            const r = this.data[i];
            const g = this.data[i+1];
            const b = this.data[i+2];

            // 背景差异（越大越好）
            scoreR += Math.abs(r - this.bgR);
            scoreG += Math.abs(g - this.bgG);
            scoreB += Math.abs(b - this.bgB);
        }

        if (scoreR > scoreG && scoreR > scoreB) return 'r';
        if (scoreG > scoreB) return 'g';
        return 'b';
    }

    buildChannelAlpha(channel, deepMask) {
        const alpha = new Uint8ClampedArray(this.w * this.h);
        
        let bgAvg = 0, fgAvg = 0, fgCount = 0;

        if (channel === 'r') bgAvg = this.bgR;
        else if (channel === 'g') bgAvg = this.bgG;
        else bgAvg = this.bgB;

        for (let i = 0; i < this.data.length; i += 4) {
            if (deepMask && deepMask[i/4] > 128) {
                if (channel === 'r') fgAvg += this.data[i];
                else if (channel === 'g') fgAvg += this.data[i+1];
                else fgAvg += this.data[i+2];
                fgCount++;
            }
        }
        if (fgCount > 0) fgAvg /= fgCount;

        // Invert if background is generally brighter than the object
        const shouldInvert = bgAvg > fgAvg;

        for (let i = 0; i < this.data.length; i += 4) {
            let v;
            if (channel === 'r') v = this.data[i];
            else if (channel === 'g') v = this.data[i+1];
            else v = this.data[i+2];

            if (shouldInvert) v = 255 - v;
            alpha[i / 4] = v;
        }

        return alpha;
    }

    applyLevels(alpha) {
        // Auto-Levels using 5th and 95th percentiles (Imitates PS Levels adjustment)
        let hist = new Int32Array(256);
        for (let i = 0; i < alpha.length; i++) {
            hist[alpha[i]]++;
        }
        let total = alpha.length;
        let p5 = 0, p95 = 255;
        let sum = 0;
        for (let i = 0; i < 256; i++) {
            sum += hist[i];
            if (sum >= total * 0.05 && p5 === 0) p5 = i;
            if (sum >= total * 0.95) { p95 = i; break; }
        }
        
        p5 = Math.max(0, p5 - 5);    // slight buffer
        p95 = Math.min(255, p95 + 5); 

        if (p95 <= p5) return alpha;

        const out = new Uint8ClampedArray(alpha.length);
        for (let i = 0; i < alpha.length; i++) {
            let v = (alpha[i] - p5) / (p95 - p5);
            v = Math.pow(Math.max(0, Math.min(1, v)), 0.85); // PS midtone curve tweak
            out[i] = v * 255;
        }
        return out;
    }

    crushMask(alpha, deepMask) {
        const out = new Uint8ClampedArray(alpha.length);

        for (let i = 0; i < alpha.length; i++) {
            let v = alpha[i];

            // 🎯 Threshold crush for mid-tone preservation and background cutting
            if (v < 100) v = 0;
            else if (v > 180) v = 255;

            // Restrict by semantic mask to eliminate irrelevant bounding box pixels
            if (deepMask) {
                const conf = deepMask[i];
                if (conf < 5) v = 0;
            }
            out[i] = v;
        }
        return out;
    }

    channelMatting(deepMask, type) {
        const channel = this.chooseBestChannel();
        console.log(`[MatteEngine] Selected physical channel '${channel}' for PS masking.`);
        let alphaData = this.buildChannelAlpha(channel, deepMask);
        alphaData = this.applyLevels(alphaData);
        
        // V5.7: Ultra-Tightened Neural Mask Envelope.
        // The 1.2 curve was still leaving a faint glow. We now use a massive 1.8 compressive curve.
        // This acts like a 'Choke Matte' in After Effects, pulling the semantic AI bounds strictly inside the visual cloud.
         if (deepMask) {
            for (let i = 0; i < alphaData.length; i++) {
                let mask = deepMask[i];
                if (type === 'channel_fabric') {
                    // Transparent fabrics need their fuzzy edges! 
                    // No harsh choke matte. We only cull pure background (mask < 5) to remove garbage.
                    if (mask < 5) {
                        alphaData[i] = 0;
                    } else if (mask < 250) {
                        // Very gentle envelope for mesh fabric edges so they don't look cut with scissors
                        let env = Math.pow(mask / 255.0, 0.4); 
                        alphaData[i] *= env;
                    }
                } else {
                    if (mask < 15) {
                        alphaData[i] = 0; // Tighter absolute noise culling based on AI bounding
                    } else if (mask < 250) {
                        // Compressive envelope fade for the bounding edges (Choke)
                        let env = Math.pow(mask / 255.0, 1.2); 
                        alphaData[i] = alphaData[i] * env;
                    }
                    
                    // Additional physical noise floor cut to kill the faint white glowing ring
                    if (alphaData[i] < 12) {
                        alphaData[i] = 0;
                    }
                }
            }
        }
        
        return alphaData;
    }

    /* =========================================================
       NEW V5.1: Guided Hard Edge Segmenter (Semantic Snap with Despill)
       ========================================================= */
    guidedHardMatting(deepMask) {
        // STEP 1: Compute Color Distance Matte (Like classic magic wand per pixel)
        const softAlpha = new Uint8ClampedArray(this.w * this.h);
        const hardTolerance = 30; // Anything within 30 color distance of background is 0.
        const featherDistance = 80;

        for (let i = 0; i < this.data.length; i += 4) {
            const r = this.data[i], g = this.data[i+1], b = this.data[i+2];
            
            const distance = Math.sqrt((r-this.bgR)**2 + (g-this.bgG)**2 + (b-this.bgB)**2);

            let maskVal = 255;
            if (distance < hardTolerance) {
                maskVal = 0;
            } else if (distance < featherDistance) {
                // Anti-alias ramp
                let blend = (distance - hardTolerance) / (featherDistance - hardTolerance);
                maskVal = blend * 255;
            }
            softAlpha[i/4] = Math.max(0, Math.min(255, maskVal));
        }

        // STEP 2: The "Intersection Strike" (Math.min)
        // This is where we use the AI mask to bound the object, but if the color 
        // matches the background (e.g. wall reflections), we chop it off!
        const mergedAlpha = new Uint8ClampedArray(this.w * this.h);
        for (let i = 0; i < this.w * this.h; i++) {
            if (deepMask) {
                mergedAlpha[i] = Math.min(softAlpha[i], deepMask[i]);
            } else {
                mergedAlpha[i] = softAlpha[i];
            }
        }

        // STEP 3: Flood Hole Patching (The Internal Savior)
        // Since `Math.min` might punch holes in internal white objects (like the 
        // core of a chandelier), we use our flood fill logic to patch everything 
        // that is safely enclosed.
        this.fillInternalHoles(mergedAlpha);

        return mergedAlpha;
    }

    /* =========================================================
       NEW V6: Global Chroma Key (For Generative Green Screens)
       ========================================================= */
    globalChromaKey() {
        const outAlpha = new Uint8ClampedArray(this.w * this.h);
        const hardTolerance = 30;
        const featherDistance = 100;
        
        for (let i = 0; i < this.data.length; i += 4) {
            const r = this.data[i], g = this.data[i+1], b = this.data[i+2];
            const dist = Math.sqrt((r-this.bgR)**2 + (g-this.bgG)**2 + (b-this.bgB)**2);
            
            let alphaParam = (dist - hardTolerance) / (featherDistance - hardTolerance);
            alphaParam = Math.max(0, Math.min(1, alphaParam)); // 0 = bg, 1 = fg
            
            outAlpha[i/4] = alphaParam * 255;
            
            // Basic early de-spill for perfect solid backgrounds
            if (alphaParam > 0 && alphaParam < 1) {
                if (this.bgG > 240 && this.bgR < 10) { // Green screen
                    if (g > r && g > b) this.data[i+1] = Math.max(r, b); 
                } else if (this.bgR > 240 && this.bgB > 240 && this.bgG < 10) { // Magenta
                    if (r > g) this.data[i] = g;
                    if (b > g) this.data[i+2] = g;
                }
            }
        }
        return outAlpha;
    }

    /* =========================================================
       NEW V5: Fast Solid Chroma Keying (Contiguous Magic Wand)
       ========================================================= */
    fastChromaKey(deepMask) {
        const outAlpha = new Uint8ClampedArray(this.w * this.h);
        
        // ==========================================
        // 1. CHROMA TYPE IDENTIFICATION (Vlahos Core)
        // ==========================================
        // Dynamically identify the predominant key color from the mathematically sampled background.
        let keyType = 'B';
        let bgChroma = this.bgB - Math.max(this.bgR, this.bgG);
        
        let cG = this.bgG - Math.max(this.bgR, this.bgB);
        if (cG > bgChroma) { keyType = 'G'; bgChroma = cG; }
        
        let cR = this.bgR - Math.max(this.bgG, this.bgB);
        if (cR > bgChroma) { keyType = 'R'; bgChroma = cR; }
        
        let cM = Math.min(this.bgR, this.bgB) - this.bgG;
        if (cM > bgChroma) { keyType = 'M'; bgChroma = cM; }

        bgChroma = Math.max(20, bgChroma); // Hard floor to prevent math collapse on grays
        
        // ==========================================
        // 2. DYNAMIC TOLERANCE CALCULATION
        // ==========================================
        // maxChroma: Above this value = 100% Background (0 Alpha)
        // minChroma: Below this value = 100% Foreground (255 Alpha) 
        const maxChroma = Math.max(20, bgChroma * 0.55); // 45% variance allowed for shadows/vignette artifacts
        const minChroma = 15; // Tight floor. Any pixel above this enters the semi-transparent transition
        const chromaRange = Math.max(1, maxChroma - minChroma);

        // ==========================================
        // 3. PIXEL-PERFECT MATTING & TRUE DESPILL
        // ==========================================
        for (let i = 0; i < this.data.length; i += 4) {
            let r = this.data[i], g = this.data[i+1], b = this.data[i+2];

            let primary, maxSecondary;
            if (keyType === 'B') {
                primary = b; maxSecondary = Math.max(r, g);
            } else if (keyType === 'G') {
                primary = g; maxSecondary = Math.max(r, b);
            } else if (keyType === 'R') {
                primary = r; maxSecondary = Math.max(g, b);
            } else { // 'M'
                primary = Math.min(r, b); maxSecondary = g;
            }

            const pixelChroma = primary - maxSecondary;
            
            // Core Alpha Generation (No Spatial Blur needed!)
            let alpha = 255;
            if (pixelChroma >= maxChroma) {
                alpha = 0;
            } else if (pixelChroma > minChroma) {
                let t = (maxChroma - pixelChroma) / chromaRange;
                t = Math.max(0, Math.min(1, t));
                t = t * t * (3 - 2 * t); // Smooth S-Curve anti-aliasing!
                alpha = t * 255;
            }
            outAlpha[i/4] = alpha;

            // Deep Color Despill (removes glowing color fringes instantly)
            // If the retained pixel exhibits the strictly opposed background color, we neutralize it!
            if (alpha > 0 && primary > maxSecondary) {
                let neutralized = maxSecondary; // Clamp the invading background color to neutral gray
                
                if (keyType === 'B') {
                    this.data[i+2] = neutralized;
                } else if (keyType === 'G') {
                    this.data[i+1] = neutralized;
                } else if (keyType === 'R') {
                    this.data[i] = neutralized;
                } else { // 'M'
                    this.data[i] = Math.min(r, neutralized);
                    this.data[i+2] = Math.min(b, neutralized);
                }
            }
        }

        // Apply external mask culling
        if (deepMask) {
            for (let i = 0; i < outAlpha.length; i++) {
                if (deepMask[i] < 5) outAlpha[i] = 0;
            }
        }

        return outAlpha;
    }

    /* =========================================================
       STEP 6: Adaptive Fusion Core (Luminance + Deep Sweep + Structure Aware + Continuity)
       ========================================================= */
    adaptiveFusion(deepMask, type) {

        const alphaData = new Uint8ClampedArray(this.w * this.h);
        const gamma = (type === 'soft') ? 1.35 : 1.0;
        const featherDist = (type === 'soft') ? 180 : 100;

        for (let i = 0; i < this.data.length; i += 4) {
            const r = this.data[i], g = this.data[i+1], b = this.data[i+2];
            let finalAlpha = 0;

            const isLeftEdge = (i % (this.w * 4)) === 0;

            if (type === 'soft' || type === 'cloud') {
                // LUMINANCE MATTE + STRUCTURE AWARENESS
                // 1. Calculate local contrast (look at left neighbor)
                let localContrast = 0;
                if (!isLeftEdge) {
                    const nr = this.data[i - 4];
                    const ng = this.data[i - 3];
                    const nb = this.data[i - 2];
                    localContrast = Math.abs(r - nr) + Math.abs(g - ng) + Math.abs(b - nb);
                }

                // 2. Base Brightness
                const brightness = (r+g+b)/3;
                const delta = brightness - this.bgBrightness;
                
                // 3. Brightness Protection Band
                let densityMask;
                if (delta > -10) {
                    // Close to or brighter than background -> ensure minimum opacity (0.6) + ramp up
                    densityMask = 0.6 + (delta / 255);
                } else {
                    // Darker than background -> calculate fade
                    densityMask = 1 - Math.min(Math.abs(delta) / 255, 1);
                }
                
                // 4. Structure as Override (instead of just multiplier multiplier)
                if (localContrast > 12) {
                    densityMask = Math.max(densityMask, 0.65);
                }

                densityMask = Math.pow(Math.max(0, Math.min(1, densityMask)), gamma);
                finalAlpha = densityMask * 255;
                
                // 5. Saturation & Chromatic Shielding
                const saturation = Math.max(r, g, b) - Math.min(r, g, b);
                const bgSat = Math.max(this.bgR, this.bgG, this.bgB) - Math.min(this.bgR, this.bgG, this.bgB);
                if (Math.abs(saturation - bgSat) > 15) {
                    // Object is distinctly more or less saturated than background -> highly likely foreground
                    finalAlpha = Math.max(finalAlpha, 100); 
                }
                
            } else {
                // COLOR DISTANCE MATTE (For hard edges/hair) using BACKGROUND DISTRIBUTION
                const distR = Math.max(0, r - this.bgMax[0], this.bgMin[0] - r);
                const distG = Math.max(0, g - this.bgMax[1], this.bgMin[1] - g);
                const distB = Math.max(0, b - this.bgMax[2], this.bgMin[2] - b);
                const colorDistRange = Math.sqrt(distR**2 + distG**2 + distB**2);
                
                let softMask = Math.min(colorDistRange / featherDist, 1.0);
                softMask = Math.pow(softMask, gamma);
                finalAlpha = softMask * 255;
            }
            
            // THE DEEP MASK SWEEPER (Confidence-Aware Fusion & Refinement)
            if (deepMask) {
                const hardAlpha = deepMask[i / 4]; // ✅ Now it is a single-channel binary mask!

                const distR = Math.max(0, r - this.bgMax[0], this.bgMin[0] - r);
                const distG = Math.max(0, g - this.bgMax[1], this.bgMin[1] - g);
                const distB = Math.max(0, b - this.bgMax[2], this.bgMin[2] - b);
                const colorDistRange = Math.sqrt(distR**2 + distG**2 + distB**2);

                // Blend carefully, honoring the neural mask 
                if (type === 'hard') {
                    if (hardAlpha > 250) finalAlpha = 255;
                    else if (hardAlpha < 10) finalAlpha = 0;
                    else finalAlpha = (finalAlpha * 0.2) + (hardAlpha * 0.8);
                } else if (type === 'hair') {
                    // Hair needs alpha fidelity
                    finalAlpha = (finalAlpha * 0.4) + (hardAlpha * 0.6);
                } else {
                    let confWeight = 0.7; // trust neural soft mask more on soft edges
                    finalAlpha = (finalAlpha * (1 - confWeight)) + (hardAlpha * confWeight);

                    // Ultimate overrides for hyper-confident signals
                    if (hardAlpha > 240 && colorDistRange > 20) {
                        finalAlpha = Math.max(finalAlpha, 240);
                    } else if (hardAlpha < 10) {
                        finalAlpha = Math.min(finalAlpha, 20);
                    }
                }

                // 🚨 FORCE CLEAR BACKGROUND (Crucial cutoff for removing ghost artifacts left behind)
                if (hardAlpha < 5) {
                    finalAlpha = 0;
                }
            }
            
            alphaData[i / 4] = Math.max(0, Math.min(255, finalAlpha));
        }

        // SYMMETRIC CONNECTIVITY PROPAGATION (Decoupled 3x3 Dilation)
        // Solves directional bias completely and builds unified cores intrinsically
        const dilatedAlpha = new Uint8ClampedArray(alphaData);
        // WARNING: Avoid dilating for 'hair', doing so thickens the fine tips artificially
        if (type !== 'hair') {
            for (let y = 1; y < this.h - 1; y++) {
                for (let x = 1; x < this.w - 1; x++) {
                    const idx = y * this.w + x;
                    const a = alphaData[idx];
                    if (a < 255) {
                        let maxN = Math.max(
                            alphaData[idx - 1], alphaData[idx + 1],
                            alphaData[idx - this.w], alphaData[idx + this.w]
                        );
                        if (maxN > 180 && a < maxN) {
                            dilatedAlpha[idx] = Math.min(255, a + 25);
                        }
                    }
                }
            }
        }

        // CONNECTED COMPONENT / FLOOD FILL HOLE REPAIR
        // WARNING: Avoid filling holes for 'hair', doing so makes hair clumps look blocky & chunky
        if (type !== 'hair') {
            this.fillInternalHoles(dilatedAlpha);
        }

        return dilatedAlpha;
    }

    fillInternalHoles(alphaData) {
        const w = this.w; const h = this.h;
        const visited = new Uint8Array(w * h);
        const qX = new Uint16Array(w * h);
        const qY = new Uint16Array(w * h);
        let head = 0, tail = 0;

        const enqueue = (x, y) => {
            const idx = y * w + x;
            if (!visited[idx]) {
                visited[idx] = 1;
                // Treat anything semi-transparent (< 128) as passable for the flood fill
                if (alphaData[idx] < 128) {
                    qX[tail] = x; qY[tail] = y; tail++;
                }
            }
        };

        // 1. Seed borders (Flood fill from the outside in)
        for (let x=0; x<w; x++) { enqueue(x, 0); enqueue(x, h-1); }
        for (let y=0; y<h; y++) { enqueue(0, y); enqueue(w-1, y); }

        // 2. Flood fill external background
        while(head < tail) {
            const cx = qX[head]; const cy = qY[head]; head++;
            if (cx > 0) enqueue(cx-1, cy);
            if (cx < w-1) enqueue(cx+1, cy);
            if (cy > 0) enqueue(cx, cy-1);
            if (cy < h-1) enqueue(cx, cy+1);
        }

        // Compute raw local variance to protect intricate transparent structures (lace, smoke) from being flooded solid
        const getVariance = (idx) => {
            const x = idx % w; const y = Math.floor(idx / w);
            if (x === 0 || x === w-1 || y === 0 || y === h-1) return 100;
            const ctrR = this.data[idx*4], ctrG = this.data[idx*4+1], ctrB = this.data[idx*4+2];
            let diff = 0;
            const neighbors = [idx - 1, idx + 1, idx - w, idx + w];
            for (let n of neighbors) {
                diff += Math.abs(this.data[n*4] - ctrR) + Math.abs(this.data[n*4+1] - ctrG) + Math.abs(this.data[n*4+2] - ctrB);
            }
            return diff / 4;
        };

        // 3. Any pixel < 128 that is NOT visited by the outside flood fill is an enclosed internal hole!
        for (let i=0; i < w*h; i++) {
            if (alphaData[i] < 128 && !visited[i]) {
                const variance = getVariance(i);
                // Only fill if it's relatively continuous/flat (true hole in a solid body). Saves intricate objects.
                if (variance < 25) {
                    alphaData[i] = Math.max(alphaData[i], 220); // Patch the hole!
                }
            }
        }
    }

    /* =========================================================
       STEP 7: Refine Edge Engine (Inner vs Outer Edge Awareness)
       ========================================================= */
    refineEdge(alphaData, type) {
        let shiftEdge = 0, featherRadius = 1, contrastBoost = 1.0;
        
        switch(type) {
            case 'hard':     shiftEdge = -1; featherRadius = 2; contrastBoost = 1.15; break;
            case 'hair':     
                // For fine details like hair/feathers, DO NOT erode the edge, and DO NOT over-contrast it.
                // It destroys the subtle Alpha tips! 
                shiftEdge = 0; featherRadius = 0; contrastBoost = 1.0; 
                break;
            case 'logo':     shiftEdge = -2; featherRadius = 1; contrastBoost = 1.25; break;
            case 'soft':
            default:         shiftEdge = 0;  featherRadius = 5; contrastBoost = 1.0; break;
        }

        let currentAlpha = new Uint8ClampedArray(alphaData);
        
        // 1. Shift Edge
        if (shiftEdge !== 0) {
            const shifted = new Uint8ClampedArray(alphaData.length);
            for (let y = 1; y < this.h - 1; y++) {
                for (let x = 1; x < this.w - 1; x++) {
                    const idx = y * this.w + x;
                    const a = currentAlpha[idx];
                    
                    if (a > 10 && a < 250) {
                        let vals = [];
                        for (let oy=-1; oy<=1; oy++) {
                            for (let ox=-1; ox<=1; ox++) {
                                vals.push(currentAlpha[(y+oy)*this.w + (x+ox)]);
                            }
                        }
                        shifted[idx] = shiftEdge > 0 ? Math.max(...vals) : Math.min(...vals);
                    } else {
                        shifted[idx] = a; 
                    }
                }
            }
            currentAlpha = shifted;
        }

        // 2. Feather Edge (ONLY feather transitions, not solid cores!)
        if (featherRadius > 0) {
            for (let pass = 0; pass < featherRadius; pass++) {
                const feathered = new Uint8ClampedArray(alphaData.length);
                for (let y = 1; y < this.h - 1; y++) {
                    for (let x = 1; x < this.w - 1; x++) {
                        const idx = y * this.w + x;
                        const a = currentAlpha[idx];
                        
                        // Structure Conservation: Only feather actual edges!
                        if (a > 5 && a < 250) {
                            let sum = 0, count = 0;
                            for (let oy=-1; oy<=1; oy++) {
                                for (let ox=-1; ox<=1; ox++) {
                                    sum += currentAlpha[(y+oy)*this.w + (x+ox)];
                                    count++;
                                }
                            }
                            feathered[idx] = sum / count;
                        } else {
                            feathered[idx] = a;
                        }
                    }
                }
                currentAlpha = feathered;
            }
        }

        // 3. Contrast Boost
        if (contrastBoost !== 1.0) {
            for (let i = 0; i < currentAlpha.length; i++) {
                let a = currentAlpha[i] / 255;
                a = ((a - 0.5) * contrastBoost) + 0.5;
                currentAlpha[i] = Math.max(0, Math.min(1, a)) * 255;
            }
        }

        return currentAlpha;
    }

    /* =========================================================
       STEP 8 & 9: Anti-Halo Cleanup, Unmix & Color Decontamination
       ========================================================= */
    applyAntiHaloAndOutput(alphaData, type) {
        if (type === 'solid') {
            // For perfectly calculated global spatial masks (fastChromaKey),
            // alpha AND color despill are ALREADY fully resolved in fastChromaKey!
            // Applying mathematical unmix here will re-apply erosion and destroy smooth solid objects.
            for (let i = 0; i < this.data.length; i += 4) {
                this.data[i+3] = alphaData[i/4];
            }
            return;
        }

        // LOCAL BACKGROUND ESTIMATION (Spatial search for nearest true background color)
        // Instead of treating the whole image as having one average background, we look around 
        // each edge pixel to find what color is ACTUALLY behind it.
        const w = this.w;
        const h = this.h;
        const searchRadius = 15;
        
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = y * w + x;
                const dataIdx = i * 4;
                let a = alphaData[i];
                
                if (a > 5 && a < 250) {
                    let r = this.data[dataIdx];
                    let g = this.data[dataIdx+1];
                    let b = this.data[dataIdx+2];

                    // 1. Local Background Retrieval
                    let locBgR = this.bgR, locBgG = this.bgG, locBgB = this.bgB;
                    let foundBg = false;
                    let numBgPixels = 0;
                    let sumBgR = 0, sumBgG = 0, sumBgB = 0;

                    // Spiral or simple box search for nearby alpha == 0
                    const step = 2;
                    for(let dy = -searchRadius; dy <= searchRadius; dy += step) {
                        for(let dx = -searchRadius; dx <= searchRadius; dx += step) {
                            const nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                const nIdx = ny * w + nx;
                                if (alphaData[nIdx] <= 10) { // Found nearby deep background
                                    sumBgR += this.data[nIdx*4];
                                    sumBgG += this.data[nIdx*4+1];
                                    sumBgB += this.data[nIdx*4+2];
                                    numBgPixels++;
                                }
                            }
                        }
                    }

                    if (numBgPixels > 0) {
                        locBgR = sumBgR / numBgPixels;
                        locBgG = sumBgG / numBgPixels;
                        locBgB = sumBgB / numBgPixels;
                        foundBg = true;
                    }

                    let aNorm = Math.max(0.1, a / 255.0); 
                    let bgRatio = 1.0 - aNorm;

                    // 2. TRUE Mathematical Unmix with Bounded Noise Control
                    // IMPORTANT V5.4 HALO SUPPRESSION: We skip Unmix division entirely for 'channel' (volumetric) 
                    // and 'channel_fabric' (curtains). Unmix creates negative RGBs (clamped to black), 
                    // which causes the ugly dark/grey outline ring.
                    if (type !== 'channel' && type !== 'channel_hair' && type !== 'channel_fabric') {
                        let fR = (r - locBgR * bgRatio) / aNorm;
                        let fG = (g - locBgG * bgRatio) / aNorm;
                        let fB = (b - locBgB * bgRatio) / aNorm;
    
                        r = Math.max(0, Math.min(255, fR));
                        g = Math.max(0, Math.min(255, fG));
                        b = Math.max(0, Math.min(255, fB));
                    }

                    // 3. EDGE-AWARE DESPILL & HALO SUPPRESSION
                    const gray = (r + g + b) / 3;

                    if (type === 'channel') {
                        // V5.5: Luminance Conservation Desaturation (极致剥夺色相，守恒亮度)
                        // Fluids like clouds reflect light and should be stripped of the blue sky background hue.
                        // Instead of averaging (which darkens), we sync all RGB channels to the highest brightness (maxVal).
                        // This bleaches the blue bleed into pure luminous white/gray without losing ANY brightness!
                        const maxVal = Math.max(r, g, b);
                        if (a < 15) {
                            a = 0;
                            r = 0; g = 0; b = 0;
                        } else if (a < 230) {
                            r = maxVal;
                            g = maxVal;
                            b = maxVal;
                        } else {
                            // Smoothly blend the transition for the solid core (230-255) to retain original solid color
                            const mix = Math.max(0, Math.min(1, (255 - a) / 25.0)); 
                            r = r * (1 - mix) + maxVal * mix;
                            g = g * (1 - mix) + maxVal * mix;
                            b = b * (1 - mix) + maxVal * mix;
                        }
                    } else if (type === 'channel_hair' || type === 'hair') {
                        // For extreme fine objects (feathers, hair, trees)
                        // channelMatting was used, which prevents alpha clamping, BUT doesn't despill RGB inherently.
                        // We must strongly despill the colors here to remove dark backgrounds (like dark blue sky behind white feather),
                        // while NOT destroying the true bright highlights like the white feather body.
                        if (a < 5) {
                            a = 0;
                            r = 0; g = 0; b = 0;
                        } else if (a < 150) {
                            // Use luminance locking to prevent the colors from falling into deep black pits
                            const luma = Math.max(r, g, b);
                            const mix = Math.pow((150 - a) / 150.0, 1.2); 
                            // Push towards pure luminance to bleach out the original dark background hue,
                            r = r * (1 - mix) + luma * mix;
                            g = g * (1 - mix) + luma * mix;
                            b = b * (1 - mix) + luma * mix;
                        }
                    } else if (type === 'channel_fabric') {
                        // V6.1 TRANSLUCENT FABRIC optimization (窗帘、窗纱)
                        // Imitates Photoshop "Extract Mask + Keep RGB", but handles backlit silhouettes!
                        if (a < 5) {
                            a = 0;
                            r = 0; g = 0; b = 0;
                        } else {
                            // "参考云...不要那么激进"
                            // Curtains backlit by bright windows are recorded as grey/dark (silhouette).
                            // If we don't boost them, they look like "grey dirt".
                            // We gently push the color towards its own illumination (luma) to clear out the blue sky bleed,
                            // AND boost the brightness so transparent fabrics look luminous.
                            const luma = Math.max(r, g, b); // Brightest channel (usually correct illumination)
                            const mix = Math.pow((255 - a) / 255.0, 1.0) * 0.6; // Max 60% luma mix in highly transparent areas
                            r = Math.min(255, (r * (1 - mix) + luma * mix) * 1.15); // gentle 15% brightness boost
                            g = Math.min(255, (g * (1 - mix) + luma * mix) * 1.15);
                            b = Math.min(255, (b * (1 - mix) + luma * mix) * 1.15);
                        }
                    } else if (type !== 'green_screen' && type !== 'solid') {
                        // For hard objects, kill the very lowest alpha garbage to clean up the outline
                        if (a < 20) {
                            a = 0;
                            r = 0; g = 0; b = 0;
                        } else if (a < 150) {
                            const mix = Math.pow((150 - a) / 150.0, 1.5); // easing curve
                            r = r * (1 - mix * 0.8) + gray * (mix * 0.8);
                            g = g * (1 - mix * 0.8) + gray * (mix * 0.8);
                            b = b * (1 - mix * 0.8) + gray * (mix * 0.8);
                        }
                    }

                    // Final clamp and assignment
                    this.data[dataIdx]     = Math.max(0, Math.min(255, r));
                    this.data[dataIdx+1]   = Math.max(0, Math.min(255, g));
                    this.data[dataIdx+2]   = Math.max(0, Math.min(255, b));

                } else if (a <= 5) {
                    // Pure transparent - strip color completely to avoid any lingering artifacts
                    this.data[dataIdx] = 0;
                    this.data[dataIdx+1] = 0;
                    this.data[dataIdx+2] = 0;
                    a = 0; // Snap to 0 Alpha
                }
                
                // Output the finalized refined alpha
                this.data[dataIdx+3] = a;
            }
        }
    }

    /* =========================================================
       STEP 10: Filtering Isolated Islands
       ========================================================= */
    filterIsolatedIslands(alphaData, minAreaThreshold) {
        const w = this.w;
        const h = this.h;
        const visited = new Uint8Array(w * h);
        
        // Fast queue for BFS (pre-allocated to max possible size)
        const queue = new Int32Array(w * h); 
        
        if (!minAreaThreshold) {
             // 面积阈值：万分之五的图像面积，或者至少 100 像素
             minAreaThreshold = Math.max(100, Math.floor(w * h * 0.0005));
        }

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = y * w + x;
                if (alphaData[idx] > 0 && visited[idx] === 0) {
                    let head = 0;
                    let tail = 0;
                    
                    queue[tail++] = idx;
                    visited[idx] = 1;
                    
                    while (head < tail) {
                        const curr = queue[head++];
                        
                        const cx = curr % w;
                        const cy = Math.floor(curr / w); // integer division
                        
                        // 8-connected neighborhood
                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dx = -1; dx <= 1; dx++) {
                                if (dx === 0 && dy === 0) continue;
                                const nx = cx + dx;
                                const ny = cy + dy;
                                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                    const nidx = ny * w + nx;
                                    if (alphaData[nidx] > 0 && visited[nidx] === 0) {
                                        visited[nidx] = 1;
                                        queue[tail++] = nidx;
                                    }
                                }
                            }
                        }
                    }
                    
                    // If the connected component area (tail) is less than the threshold, erase it.
                    if (tail < minAreaThreshold) {
                        for (let i = 0; i < tail; i++) {
                            alphaData[queue[i]] = 0;
                        }
                    }
                }
            }
        }
        return alphaData;
    }
}
