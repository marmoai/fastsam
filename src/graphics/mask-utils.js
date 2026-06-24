
/**
 * Smart Masking Utility
 * Compares two images and generates a mask based on their differences.
 * This allows AI-generated content to "overflow" the original bounding box
 * while keeping the unchanged background 100% consistent.
 */

export async function createDifferenceMask(originalCanvas, aiCanvas, threshold = 25) {
    const width = originalCanvas.width;
    const height = originalCanvas.height;
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = width;
    maskCanvas.height = height;
    const mCtx = maskCanvas.getContext('2d');
    
    const oCtx = originalCanvas.getContext('2d');
    const aCtx = aiCanvas.getContext('2d');
    
    const oData = oCtx.getImageData(0, 0, width, height).data;
    const aData = aCtx.getImageData(0, 0, width, height).data;
    const mData = mCtx.createImageData(width, height);
    
    for (let i = 0; i < oData.length; i += 4) {
        const diff = Math.abs(oData[i] - aData[i]) + 
                     Math.abs(oData[i+1] - aData[i+1]) + 
                     Math.abs(oData[i+2] - aData[i+2]);
        
        // If difference is significant, it's new content
        const val = diff > threshold ? 255 : 0;
        mData.data[i] = val;     // R
        mData.data[i+1] = val;   // G
        mData.data[i+2] = val;   // B
        mData.data[i+3] = 255;   // A
    }
    
    mCtx.putImageData(mData, 0, 0);
    
    // Post-processing: Blur the mask to create soft edges for blending
    const finalMask = document.createElement('canvas');
    finalMask.width = width;
    finalMask.height = height;
    const fCtx = finalMask.getContext('2d');
    fCtx.filter = 'blur(4px)'; // Soften the edges
    fCtx.drawImage(maskCanvas, 0, 0);
    
    return finalMask;
}

/**
 * Dilates the alpha channel of an image data object.
 * This expands the opaque areas (foreground) into the transparent areas (background),
 * helping to recover edges that were over-segmented.
 * 
 * @param {ImageData} imageData - The ImageData object to modify.
 * @param {number} radius - The dilation radius in pixels.
 * @returns {ImageData} A new ImageData object with the dilated alpha channel.
 */
export function dilateAlphaChannel(imageData, radius = 2) {
    const width = imageData.width;
    const height = imageData.height;
    const srcData = imageData.data;
    
    // Create a copy for the output
    const outImageData = new ImageData(new Uint8ClampedArray(srcData), width, height);
    const outData = outImageData.data;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            
            // If the current pixel is already fully opaque, skip
            if (srcData[idx + 3] === 255) continue;
            
            let maxAlpha = srcData[idx + 3];
            let bestR = srcData[idx];
            let bestG = srcData[idx + 1];
            let bestB = srcData[idx + 2];
            
            // Check neighborhood
            const minY = Math.max(0, y - radius);
            const maxY = Math.min(height - 1, y + radius);
            const minX = Math.max(0, x - radius);
            const maxX = Math.min(width - 1, x + radius);
            
            for (let ny = minY; ny <= maxY; ny++) {
                for (let nx = minX; nx <= maxX; nx++) {
                    const nIdx = (ny * width + nx) * 4;
                    if (srcData[nIdx + 3] > maxAlpha) {
                        maxAlpha = srcData[nIdx + 3];
                        bestR = srcData[nIdx];
                        bestG = srcData[nIdx + 1];
                        bestB = srcData[nIdx + 2];
                    }
                }
            }
            
            if (maxAlpha > outData[idx + 3]) {
                outData[idx] = bestR;
                outData[idx + 1] = bestG;
                outData[idx + 2] = bestB;
                outData[idx + 3] = maxAlpha;
            }
        }
    }
    
    return outImageData;
}
