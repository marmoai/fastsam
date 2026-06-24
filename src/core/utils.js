export const isRemovalRequest = (text) => {
    if (!text || typeof text !== 'string') return false;
    const keywords = ['移除', '删掉', '去掉', 'remove', 'erase', 'delete', '擦除'];
    return keywords.some(k => text.toLowerCase().includes(k));
};

export const showToast = (message, duration = 2000, isSuccess = true) => {
    console.log(`[Toast]: ${message} (Success: ${isSuccess})`);
    const toast = document.createElement('div');
    toast.className = `toast ${isSuccess ? 'success' : 'error'}`;
    toast.textContent = message;
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.padding = '10px 20px';
    toast.style.backgroundColor = isSuccess ? '#4caf50' : '#f44336';
    toast.style.color = 'white';
    toast.style.zIndex = '10000';
    document.body.appendChild(toast);
    setTimeout(() => document.body.removeChild(toast), duration);
};

export const isMaterialRequest = (text) => {
    if (!text || typeof text !== 'string') return false;
    const keywords = ['材质', '纹理', '换成', 'material', 'texture', 'pattern', 'surface'];
    return keywords.some(k => text.toLowerCase().includes(k));
};

export const fileToDataURL = (file) => {
    return new Promise((resolve, reject) => {
        if (!file) {
            return resolve('');
        }
        if (typeof file === 'string') {
            if (file.startsWith('data:')) {
                return resolve(file);
            }
            fetch(getProxiedUrl(file))
                .then(res => res.blob())
                .then(blob => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                })
                .catch(reject);
            return;
        }
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

export const fileToBase64 = (file) => fileToDataURL(file).then(url => url ? url.split(',')[1] : '');

export const OSS_BACKEND_URL = 'https://oss-key-cpggotpipg.cn-hangzhou.fcapp.run';
export const RECOGNIZE_BACKEND_URL = 'https://recognize-fzltzjhmeu.cn-hangzhou.fcapp.run';
export const OSS_PUBLIC_ORIGINS = [
    'https://www.marmoai.cn',
    'https://marmoai.cn'
];

export const getAuthToken = () => localStorage.getItem('authing_token') || '';

export const buildAuthorizedHeaders = (headers = {}) => {
    const token = getAuthToken();
    return token ? { ...headers, Authorization: `Bearer ${token}` } : { ...headers };
};

export const isDirectOssUrl = (url) => {
    if (!url || typeof url !== 'string') return false;
    return OSS_PUBLIC_ORIGINS.some(origin => url.startsWith(origin)) || url.includes('.aliyuncs.com/');
};

export const isInvalidImageSrc = (url) => {
    if (typeof url !== 'string') {
        return !url;
    }

    const normalized = url.trim();
    if (!normalized || normalized === 'undefined' || normalized === 'null') {
        return true;
    }

    return normalized.endsWith('/undefined') || normalized.endsWith('/null');
};

export const getProxiedUrl = (url) => {
    if (isInvalidImageSrc(url)) return '';
    if (url.startsWith('data:')) return url;
    if (url.startsWith('blob:')) return url;
    if (url.includes('localhost')) return url;
    if (isDirectOssUrl(url)) return url;
    // If it's already a proxied URL, don't proxy it again
    if (url.startsWith(OSS_BACKEND_URL)) return url;
    return `${OSS_BACKEND_URL}/proxy?url=${encodeURIComponent(url)}`;
};

export const dataURLToFile = async (dataUrl, filename) => {
    if (dataUrl.startsWith('data:')) {
        return dataURLtoFileSync(dataUrl, filename);
    }
    const res = await fetch(getProxiedUrl(dataUrl));
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || 'image/png' });
};

export const dataURLtoFileSync = (dataurl, filename) => {
    if (!dataurl || typeof dataurl !== 'string') {
        console.error('Invalid data URL (not a string):', dataurl);
        return null;
    }
    const arr = dataurl.split(',');
    if (arr.length < 2) {
        console.error('Invalid data URL format (missing comma):', dataurl.substring(0, 50) + '...');
        return null;
    }
    
    // Check if the data part is actually base64 encoded
    const base64Part = arr[1];
    try {
        atob(base64Part);
    } catch (e) {
        console.error('Invalid base64 string in data URL:', base64Part.substring(0, 50) + '...', e);
        return null;
    }

    const mimeMatch = arr[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'image/png';
    const bstr = atob(base64Part);
    const n = bstr.length;
    const u8arr = new Uint8Array(n);
    let i = n;
    while(i--){ u8arr[i] = bstr.charCodeAt(i); }
    return new File([u8arr], filename, {type:mime});
};

export const blobToBase64 = (blob) => {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
    });
};

export const isImageGenerationRequest = (text) => {
    if (!text) return false;
    const keywords = ['生成', '画', '绘制', '创造', '图片', '图像', '照片', '插画', '海报', 'logo', '图标', '风格', '场景', '人物', '建筑', '风景', '写实', '抽象', '4K', '高清', '渲染'];
    return keywords.some(keyword => text.toLowerCase().includes(keyword));
};

export const addWatermark = async (imageDataUrl) => {
    if (!imageDataUrl || typeof imageDataUrl !== 'string') {
        console.error('addWatermark: Invalid input', imageDataUrl);
        return imageDataUrl; // Return as is if invalid
    }
    
    console.log('addWatermark input:', imageDataUrl.substring(0, 50) + '...');
    
    return new Promise((resolve, reject) => {
        const img = new Image();
        
        // Only set crossOrigin for remote URLs
        if (!imageDataUrl.startsWith('data:') && !imageDataUrl.startsWith('blob:')) {
            img.crossOrigin = "anonymous";
        }
        
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                
                if (canvas.width === 0 || canvas.height === 0) {
                    console.warn('addWatermark: Image has zero dimensions');
                    resolve(imageDataUrl);
                    return;
                }

                ctx.drawImage(img, 0, 0);
                const fontSize = Math.max(12, Math.min(canvas.width * 0.03, 60));
                ctx.font = `normal ${fontSize}px 'Arial', sans-serif`;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'bottom';
                const padding = canvas.width * 0.02;
                ctx.fillText('摹图', canvas.width - padding, canvas.height - padding);
                
                const dataUrl = canvas.toDataURL('image/png');
                console.log('addWatermark output:', dataUrl.substring(0, 50) + '...');
                resolve(dataUrl);
            } catch (e) {
                console.error('addWatermark: Canvas processing failed', e);
                resolve(imageDataUrl); // Fallback to original
            }
        };
        
        img.onerror = (err) => {
            console.error("Failed to load image for watermarking. URL:", imageDataUrl.substring(0, 100) + "...", err);
            // Instead of rejecting, we can resolve with the original image to avoid breaking the workflow
            resolve(imageDataUrl);
        };
        
        img.src = getProxiedUrl(imageDataUrl);
    });
};

export const compressImage = async (fileOrBlob, maxWidth = 4096, quality = 0.92) => {
    if (!fileOrBlob || !(fileOrBlob instanceof Blob || fileOrBlob instanceof File)) {
        throw new Error('Invalid file or blob provided to compressImage');
    }
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(fileOrBlob);
        img.onload = () => {
            URL.revokeObjectURL(url);
            let width = img.width;
            let height = img.height;

            if (width > maxWidth || height > maxWidth) {
                if (width > height) {
                    height = Math.round((height * maxWidth) / width);
                    width = maxWidth;
                } else {
                    width = Math.round((width * maxWidth) / height);
                    height = maxWidth;
                }
            } else if (
                fileOrBlob.size < 2 * 1024 * 1024 &&
                ['image/jpeg', 'image/png', 'image/webp'].includes(fileOrBlob.type)
            ) {
                // If the image doesn't need resizing and is under 2MB, 
                // we can just return the original file to save processing time and keep max quality.
                resolve(fileOrBlob);
                return;
            }

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            // Use original format if possible, fallback to jpeg
            let outputType = 'image/jpeg';
            let outputExt = '.jpg';
            
            if (fileOrBlob.type === 'image/png') {
                outputType = 'image/png';
                outputExt = '.png';
            } else if (fileOrBlob.type === 'image/webp') {
                outputType = 'image/webp';
                outputExt = '.webp';
            }
            
            let originalName = fileOrBlob.name || 'compressed';
            if (originalName.lastIndexOf('.') !== -1) {
                originalName = originalName.substring(0, originalName.lastIndexOf('.'));
            }

            canvas.toBlob((blob) => {
                if (blob) {
                    const newFile = new File([blob], originalName + outputExt, {
                        type: outputType,
                        lastModified: Date.now(),
                    });
                    resolve(newFile);
                } else {
                    reject(new Error('Canvas to Blob failed'));
                }
            }, outputType, quality);
        };
        img.onerror = (err) => {
            URL.revokeObjectURL(url);
            reject(err);
        };
        img.src = url;
    });
};

export const getUserId = () => {
    // 0. 允许通过 URL 参数强制指定 ID (用于找回丢失的数据)
    // 注意：在 iframe 中，我们需要检查顶层窗口的 URL 或者当前 iframe 的 URL
    let forceId = null;
    try {
        const urlParams = new URLSearchParams(window.location.search);
        forceId = urlParams.get('force_user_id');
        
        // 如果在 iframe 里没找到，尝试从父窗口获取（如果同源的话）
        if (!forceId && window.parent !== window) {
            try {
                const parentParams = new URLSearchParams(window.parent.location.search);
                forceId = parentParams.get('force_user_id');
            } catch (e) {
                // 跨域限制，忽略
            }
        }
    } catch (e) {
        console.error("Error parsing URL params:", e);
    }

    if (forceId) {
        // 清理可能带入的斜杠
        forceId = forceId.replace(/\/$/, '');
        localStorage.setItem('temp_user_id', forceId);
        return forceId;
    }

    // 1. 尝试从 Authing 登录信息中获取 (优先获取明确的 last_known_userId，其次是 userinfo 快照与 authing_token 反解析)
    const lastKnownId = localStorage.getItem('last_known_userId');
    if (lastKnownId) {
        return lastKnownId;
    }

    const userInfoStr = localStorage.getItem('userinfo');
    if (userInfoStr) {
        try {
            const userInfo = JSON.parse(userInfoStr);
            if (userInfo && userInfo.id) return userInfo.id;
            if (userInfo && userInfo.sub) return userInfo.sub;
        } catch (e) {
            console.error('解析用户信息失败', e);
        }
    }

    const authingToken = localStorage.getItem('authing_token');
    if (authingToken) {
        try {
            const parts = authingToken.split('.');
            if (parts.length === 3) {
                const payload = JSON.parse(decodeURIComponent(escape(atob(parts[1]))));
                if (payload && (payload.sub || payload.id)) {
                    const actualSub = payload.sub || payload.id;
                    localStorage.setItem('last_known_userId', actualSub);
                    return actualSub;
                }
            }
        } catch (e) {}
    }

    // 2. 如果没有登录信息，使用设备指纹作为临时 ID
    let tempId = localStorage.getItem('temp_user_id');
    if (!tempId) {
        tempId = 'tmp_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
        localStorage.setItem('temp_user_id', tempId);
    }
    return tempId;
};

export const getImageDimensions = (fileOrUrl) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = reject;
        if (typeof fileOrUrl === 'string') {
            img.src = getProxiedUrl(fileOrUrl);
        } else if (fileOrUrl instanceof Blob || fileOrUrl instanceof File) {
            img.src = URL.createObjectURL(fileOrUrl);
        } else {
            reject(new Error('Invalid fileOrUrl provided'));
        }
    });
};

export const getClosestSupportedAspectRatio = (width, height) => {
    const supportedRatios = [
        { str: '1:1', val: 1 },
        { str: '4:3', val: 4/3 },
        { str: '3:4', val: 3/4 },
        { str: '16:9', val: 16/9 },
        { str: '9:16', val: 9/16 }
    ];
    const imgRatio = width / height;
    let bestRatio = supportedRatios[0];
    let minDiff = Math.abs(imgRatio - bestRatio.val);
    for (const r of supportedRatios) {
        const diff = Math.abs(imgRatio - r.val);
        if (diff < minDiff) {
            minDiff = diff;
            bestRatio = r;
        }
    }
    return bestRatio.str;
};
