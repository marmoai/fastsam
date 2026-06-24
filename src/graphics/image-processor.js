import { getProxiedUrl } from '../core/utils.js';

export const createImageWithHole = (baseImageFile, maskDataUrl) => {
    return new Promise((resolve, reject) => {
        const baseImg = new Image();
        const maskImg = new Image();
        let loadedCount = 0;

        const onImageLoad = () => {
            loadedCount++;
            if (loadedCount < 2) return;

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = baseImg.naturalWidth;
            canvas.height = baseImg.naturalHeight;
            ctx.drawImage(baseImg, 0, 0);
            ctx.globalCompositeOperation = 'destination-out';
            ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
            ctx.globalCompositeOperation = 'source-over';
            canvas.toBlob(blob => {
                if (blob) {
                    const newFile = new File([blob], `edited-${baseImageFile.name || 'image.png'}`, { type: 'image/png' });
                    resolve(newFile);
                } else {
                    reject(new Error('Canvas toBlob failed.'));
                }
            }, 'image/png');
        };
        
        baseImg.onload = onImageLoad;
        maskImg.onload = onImageLoad;
        baseImg.onerror = (err) => reject(err);
        maskImg.onerror = (err) => reject(err);

        if (typeof baseImageFile === 'string') {
            if (!baseImageFile.startsWith('data:') && !baseImageFile.startsWith('blob:')) {
                baseImg.crossOrigin = "anonymous";
            }
            baseImg.src = getProxiedUrl(baseImageFile);
        } else {
            baseImg.src = URL.createObjectURL(baseImageFile);
        }
        maskImg.src = maskDataUrl;
    });
};

export const cropImageByBox = (file, box) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = typeof file === 'string' ? getProxiedUrl(file) : URL.createObjectURL(file);
        if (typeof file === 'string' && !file.startsWith('data:') && !file.startsWith('blob:')) {
            img.crossOrigin = "anonymous";
        }
        img.onload = () => {
            if (typeof file !== 'string') URL.revokeObjectURL(url);
            const [ymin, xmin, ymax, xmax] = box;
            
            const realWidth = img.naturalWidth;
            const realHeight = img.naturalHeight;
            
            const x = Math.floor((xmin / 1000) * realWidth);
            const y = Math.floor((ymin / 1000) * realHeight);
            const w = Math.floor(((xmax - xmin) / 1000) * realWidth);
            const h = Math.floor(((ymax - ymin) / 1000) * realHeight);
            
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext('2d');
            
            ctx.drawImage(img, x, y, w, h, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob(blob => {
                if (blob) resolve(blob);
                else reject(new Error("Cropping failed"));
            }, 'image/png');
        };
        img.onerror = (e) => {
            if (typeof file !== 'string') URL.revokeObjectURL(url);
            reject(e);
        };
        img.src = url;
    });
};
