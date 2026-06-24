export class MatteTaskSystem {
    constructor(maxWorkers = 2) {
        this.maxWorkers = maxWorkers;
        this.queue = [];
        this.cache = new Map();
        this.activeWorkers = new Map();
        this.idleWorkers = [];
        
        // Initialize pool
        for (let i = 0; i < maxWorkers; i++) {
            this._spawnWorker();
        }
    }

    _spawnWorker() {
        const worker = new Worker(new URL('./matte.worker.js', import.meta.url), { type: 'module' });
        worker.onmessage = (e) => this._handleWorkerMessage(worker, e);
        // Error handling can be added here
        this.idleWorkers.push(worker);
    }

    _handleWorkerMessage(worker, e) {
        const { taskId, status, progressMsg, payload, error } = e.data;
        const task = this.activeWorkers.get(worker);

        if (!task || task.id !== taskId) return;

        if (status === 'progress') {
            if (task.onProgress) task.onProgress(progressMsg);
        } else if (status === 'done') {
            this.cache.set(task.hash, payload); // Cache the final ImageData payload
            task.resolve(payload);
            this._releaseWorker(worker);
        } else if (status === 'error') {
            task.reject(new Error(error));
            this._releaseWorker(worker);
        }
    }

    _releaseWorker(worker) {
        this.activeWorkers.delete(worker);
        this.idleWorkers.push(worker);
        this._processQueue();
    }

    _processQueue() {
        if (this.queue.length === 0 || this.idleWorkers.length === 0) return;

        const worker = this.idleWorkers.pop();
        const task = this.queue.shift();
        
        this.activeWorkers.set(worker, task);

        const transferables = [task.imageData.data.buffer];
        if (task.deepMaskArray) {
            transferables.push(task.deepMaskArray.buffer);
        }

        // Send job to worker
        worker.postMessage({
            taskId: task.id,
            imageDataArray: task.imageData.data,
            width: task.imageData.width,
            height: task.imageData.height,
            layerName: task.layerName,
            deepMaskArray: task.deepMaskArray,
            options: task.options
        }, transferables); 
        // Note: Transferring ownership blocks access on main thread, which is fine since we recreate it anyway usually
    }

    // Mock Cloud API / Local AI Router
    async getDeepMaskRobust(dataUrl, mode = 'local') {
        // Since Gemini is now directly returning a grayscale segmentation mask as requested,
        // we DO NOT run it through any external cutout model. Running it through any background
        // removal tool would interpret the black areas as background, which is redundant and can 
        // cause premultiplied alpha contamination. We just fetch the raw mask blob!
        return await fetch(dataUrl).then(res => res.blob());
    }

    async extractMaskArrayFromBlob(blob, width, height, layerName) {
        const bitmap = await createImageBitmap(blob);
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0, width, height);
        const rgba = ctx.getImageData(0, 0, width, height).data;

        const mask = new Uint8ClampedArray(width * height);

        // Check semantic classification to determine mask handling strategy
        const hardKeywords = /灯|家具|车|杯|瓶|桌|椅|画|框|商品|摆件|沙发|柜|床|钟|盆|鞋|包|器|木|铁|门|窗|电视/i;
        const isHardObject = layerName && hardKeywords.test(layerName);

        for (let i = 0; i < rgba.length; i += 4) {
            let lum = (rgba[i] + rgba[i+1] + rgba[i+2]) / 3;

            if (isHardObject) {
                // threshold incredibly low (e.g. > 10) to artificially inflate the mask size. 
                // We WANT it to bleed onto the background so that Math.min + color subtract can
                // precisely cut the true edge using pixel colors.
                mask[i / 4] = lum > 15 ? 255 : 0;
            } else {
                // Soft compression to remove mist (grey backgrounds) for clouds, smoke, hair
                lum = (lum - 30) * 1.4;
                lum = Math.max(0, Math.min(255, lum));
                lum = Math.pow(lum / 255, 1.2) * 255;
                mask[i / 4] = lum;
            }
        }

        return mask;
    }

    async hashImage(imageData) {
        // High-perf simplified hash (e.g., sample subset of RGB + dims)
        const d = imageData.data;
        const len = d.length;
        let p = 0;
        for (let i = 0; i < len; i += Math.floor(len / 100)) p = (p * 31 + d[i]) | 0;
        return `${imageData.width}x${imageData.height}_${p}`;
    }

    async enqueueProcess(taskId, imageData, dataUrl, layerName, onProgress, options = {}) {
        const hash = await this.hashImage(imageData);
        
        if (this.cache.has(hash)) {
            console.log("[TaskSystem] Used cached result for ", layerName);
            return this.cache.get(hash);
        }

        return new Promise(async (resolve, reject) => {
            try {
                if(onProgress) onProgress("🚀 [AI推理层] 正在进行语义分割分析...");
                
                let deepMaskArray;
                if (dataUrl) {
                    // Hybrid Layering: AI Inference happens on main/imgly-worker thread so it can easily use WASM context bounds
                    const blob = await this.getDeepMaskRobust(dataUrl, 'local');
                    deepMaskArray = await this.extractMaskArrayFromBlob(blob, imageData.width, imageData.height, layerName);
                } else {
                    deepMaskArray = null;
                }
                
                const task = {
                    id: taskId,
                    hash,
                    imageData, // Will be transferred
                    deepMaskArray, // Will be transferred
                    layerName,
                    options,
                    onProgress,
                    resolve,
                    reject
                };

                this.queue.push(task);
                this._processQueue();
            } catch(e) {
                reject(e);
            }
        });
    }

    cancelTask(taskId) {
        // Cancel logic if still in queue
        this.queue = this.queue.filter(t => t.id !== taskId);
        // If actively running, we could terminate worker, but respawn cost is high. 
        // For now, simpler queue filter is provided.
    }
}

export const globalMatteTaskSystem = new MatteTaskSystem();
