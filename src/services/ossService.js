import { getUserId, OSS_BACKEND_URL, getProxiedUrl, compressImage, buildAuthorizedHeaders } from '../core/utils.js';

const UPLOAD_CACHE_KEY = 'oss_upload_cache';

function guessExtensionFromMimeType(mimeType = '') {
    if (mimeType === 'image/png') return 'png';
    if (mimeType === 'image/webp') return 'webp';
    if (mimeType === 'image/gif') return 'gif';
    return 'jpg';
}

function ensureNamedUploadFile(fileLike, fallbackBaseName = 'upload') {
    if (fileLike instanceof File) {
        return fileLike;
    }
    if (fileLike instanceof Blob) {
        const mimeType = fileLike.type || 'image/jpeg';
        const extension = guessExtensionFromMimeType(mimeType);
        return new File([fileLike], `${fallbackBaseName}.${extension}`, {
            type: mimeType,
            lastModified: Date.now()
        });
    }
    return fileLike;
}

function getCurrentSessionIdForUpload() {
    return window.state?.currentSessionId || null;
}

async function calculateHash(fileOrBase64) {
    try {
        let buffer;
        if (typeof fileOrBase64 === 'string' && fileOrBase64.startsWith('data:')) {
            const encoder = new TextEncoder();
            buffer = encoder.encode(fileOrBase64);
        } else if (fileOrBase64 instanceof Blob) {
            buffer = await fileOrBase64.arrayBuffer();
        } else {
            return null;
        }
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } catch (e) {
        console.warn('Failed to calculate hash:', e);
        return null;
    }
}

function getCachedUrl(hash) {
    if (!hash) return null;
    try {
        const cache = JSON.parse(localStorage.getItem(UPLOAD_CACHE_KEY) || '{}');
        return cache[hash] || null;
    } catch (e) {
        return null;
    }
}

function setCachedUrl(hash, url) {
    if (!hash || !url) return;
    try {
        const cache = JSON.parse(localStorage.getItem(UPLOAD_CACHE_KEY) || '{}');
        cache[hash] = url;
        
        // Limit cache size to prevent localStorage overflow (e.g. 200 items)
        const keys = Object.keys(cache);
        if (keys.length > 200) {
            delete cache[keys[0]];
        }
        
        localStorage.setItem(UPLOAD_CACHE_KEY, JSON.stringify(cache));
    } catch (e) {
        console.warn('Failed to save to upload cache:', e);
    }
}

export const uploadImageToOSS = async (fileOrBase64, options = {}) => {
    // 如果已经是目标 OSS 的 URL，直接返回，不必重新上传
    if (typeof fileOrBase64 === 'string' && (fileOrBase64.startsWith(OSS_BACKEND_URL) || fileOrBase64.startsWith('https://www.marmoai.cn/'))) {
        return fileOrBase64;
    }

    const userId = getUserId();
    const sessionId = options.sessionId || getCurrentSessionIdForUpload();
    
    // 1. Calculate hash and check cache
    const hash = await calculateHash(fileOrBase64);
    if (hash) {
        const cachedUrl = getCachedUrl(hash);
        if (cachedUrl) {
            console.log('OSS Upload: Cache hit, skipping upload.');
            return cachedUrl;
        }
    }

    let file = fileOrBase64;
    
    console.log('OSS Upload: Incoming type:', typeof file, 'Value:', file?.constructor?.name || file);

    // If it's an object with a blob property (common pattern in this app)
    if (file && typeof file === 'object' && file.blob) {
        file = file.blob;
    }
    
    // If it's a string, it might be base64 or a URL
    if (typeof file === 'string') {
        if (file.startsWith('data:') || file.startsWith('http')) {
            const res = await fetch(file);
            file = await res.blob();
        }
    }

    // Ensure it's a Blob or File
    if (!(file instanceof Blob || file instanceof File)) {
        console.error('OSS Upload: Invalid file type after conversion:', file);
        throw new Error('Invalid file or blob provided to OSS upload');
    }

    file = ensureNamedUploadFile(file, `upload_${Date.now()}`);

    // Compress the image before uploading (max 2048px, 85% quality)
    try {
        file = await compressImage(file, 2048, 0.85);
    } catch (compressErr) {
        console.warn('Image compression failed, falling back to original file:', compressErr);
        // Ensure the original file is still a Blob/File
        if (!(file instanceof Blob || file instanceof File)) {
            throw new Error('Original file is not a valid Blob or File');
        }
    }

    file = ensureNamedUploadFile(file, `upload_${Date.now()}`);
    if (!file.size) {
        throw new Error('待上传文件为空');
    }

    console.log('OSS Upload: Prepared file:', {
        name: file.name,
        type: file.type,
        size: file.size,
        sessionId
    });

    const formData = new FormData();
    formData.append('file', file, file.name || `upload_${Date.now()}.${guessExtensionFromMimeType(file.type)}`);
    formData.append('userId', userId);
    if (sessionId) {
        formData.append('sessionId', sessionId);
    }

    try {
        const response = await fetch(`${OSS_BACKEND_URL}/upload`, {
            method: 'POST',
            headers: buildAuthorizedHeaders(),
            body: formData,
        });

        if (!response.ok) {
            let errorMessage = `OSS 上传失败 (${response.status})`;
            try {
                const errorPayload = await response.json();
                errorMessage = errorPayload?.message || errorPayload?.debug || errorMessage;
            } catch (parseError) {
                try {
                    const errorText = await response.text();
                    if (errorText) errorMessage = errorText;
                } catch (readError) {
                    // Ignore secondary parsing failures and keep the generic message.
                }
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();
        if (result.status === 'success') {
            const url = result.data.url;
            // 2. Save to cache
            if (hash) {
                setCachedUrl(hash, url);
            }
            return url;
        } else {
            throw new Error(result.message || 'OSS 上传失败');
        }
    } catch (error) {
        console.error('OSS 上传出错:', error);
        throw error;
    }
};

export const uploadEmbeddedMedia = async (value, visited = new WeakSet(), options = {}) => {
    if (!value) return value;

    // If it is a string and starts with "data:", upload it immediately
    if (typeof value === 'string') {
        if (value.startsWith('data:')) {
            try {
                return await uploadImageToOSS(value, options);
            } catch (err) {
                console.error('Failed to auto-upload embedded base64 data to OSS:', err);
                return value; // Keep fallback
            }
        }
        return value;
    }

    // If it is a Blob or File, upload it immediately
    if (value instanceof Blob || value instanceof File) {
        try {
            return await uploadImageToOSS(value, options);
        } catch (err) {
            console.error('Failed to auto-upload embedded Blob/File to OSS:', err);
            return value;
        }
    }

    // Guard against circular references (only for objects/arrays)
    if (typeof value === 'object') {
        if (visited.has(value)) {
            return value;
        }
        visited.add(value);
    }

    // If it is an array, map over it recursively
    if (Array.isArray(value)) {
        return await Promise.all(value.map(item => uploadEmbeddedMedia(item, visited, options)));
    }

    // If it is a standard object
    if (typeof value === 'object') {
        const cloned = { ...value };
        const keys = Object.keys(cloned);

        // If there's a `.blob` property in the object (common schema in this workbench app)
        if (cloned.blob && (cloned.blob instanceof Blob || cloned.blob instanceof File || (typeof cloned.blob === 'string' && cloned.blob.startsWith('data:')))) {
            try {
                const url = await uploadImageToOSS(cloned.blob, options);
                cloned.dataUrl = url;
                cloned.sourceImage = url; // Some objects use sourceImage instead of dataUrl
                delete cloned.blob; // Save database/OSS file size space
            } catch (err) {
                console.error('Failed to auto-upload cloned.blob:', err);
            }
        }

        // Process each key concurrently
        await Promise.all(
            keys.map(async (key) => {
                // If key is 'blob' and we just deleted it, skip
                if (key === 'blob' && !cloned.blob) return;
                cloned[key] = await uploadEmbeddedMedia(cloned[key], visited, options);
            })
        );
        return cloned;
    }

    return value;
};

export const saveWorkbenchStateToOSS = async (state) => {
    console.info('[OSS] saveWorkbenchStateToOSS 已停用，工作台状态统一通过 session/runtimeWorkspace 持久化。');
    return { status: 'success', message: 'skipped_legacy_state_sync' };
};

export const getWorkbenchStateFromOSS = async () => {
    return null;
};

export const saveAssetsToOSS = async (assets) => {
    console.info('[OSS] saveAssetsToOSS 已停用，资产统一通过 session/runtimeWorkspace 持久化。');
    return { status: 'success', message: 'skipped_legacy_assets_sync' };
};

export const getAssetsFromOSS = async () => {
    return [];
};

export const saveSessionsToOSS = async (sessions) => {
    const userId = getUserId();
    try {
        const sessionList = Array.isArray(sessions) ? sessions : [];
        const manifestSessions = sessionList.map(session => ({
            id: session.id,
            title: session.title || '未命名项目',
            timestamp: session.timestamp || Date.now(),
            updatedAt: session.updatedAt || session.timestamp || Date.now(),
            isAutoRenamed: !!session.isAutoRenamed,
            deletedAt: session.deletedAt || null
        }));

        for (const session of sessionList) {
            if (!session?.id) continue;
            const payload = JSON.stringify({ userId, session });
            console.log(`Syncing session ${session.id} to OSS, payload size: ${(payload.length / 1024 / 1024).toFixed(2)} MB`);

            const response = await fetch(`${OSS_BACKEND_URL}/save-session`, {
                method: 'POST',
                headers: buildAuthorizedHeaders({
                    'Content-Type': 'application/json',
                }),
                body: payload,
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`保存单个会话到 OSS 失败, status: ${response.status}, message: ${errorText}`);
                throw new Error(`保存单个会话到 OSS 失败: ${response.status} ${errorText}`);
            }
        }

        const manifestPayload = JSON.stringify({ userId, sessions: manifestSessions });
        console.log(`Syncing sessions manifest to OSS, payload size: ${(manifestPayload.length / 1024).toFixed(2)} KB`);

        const response = await fetch(`${OSS_BACKEND_URL}/save-sessions-manifest`, {
            method: 'POST',
            headers: buildAuthorizedHeaders({
                'Content-Type': 'application/json',
            }),
            body: manifestPayload,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`保存会话清单到 OSS 失败, status: ${response.status}, message: ${errorText}`);
            throw new Error(`保存会话清单到 OSS 失败: ${response.status} ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('保存会话到 OSS 出错:', error);
        throw error;
    }
};

export const getSessionsFromOSS = async () => {
    const userId = getUserId();
    try {
        const response = await fetch(`${OSS_BACKEND_URL}/get-sessions?userId=${userId}`, {
            headers: buildAuthorizedHeaders()
        });
        if (!response.ok) throw new Error('从 OSS 获取会话列表失败');
        
        const result = await response.json();
        if (result.status === 'success') {
            return result.sessions;
        } else {
            throw new Error(result.message || '获取会话列表失败');
        }
    } catch (error) {
        console.error('从 OSS 获取会话列表出错:', error);
        return null;
    }
};

export const saveSatisfactionRulesToOSS = async (rules) => {
    const userId = getUserId();
    try {
        const payload = JSON.stringify({ userId, rules });
        console.log(`Syncing satisfaction rules to OSS, payload size: ${(payload.length / 1024).toFixed(2)} KB`);
        
        const response = await fetch(`${OSS_BACKEND_URL}/save-satisfaction-rules`, {
            method: 'POST',
            headers: buildAuthorizedHeaders({
                'Content-Type': 'application/json',
            }),
            body: payload,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`保存满意定稿规则到 OSS 失败, status: ${response.status}, message: ${errorText}`);
            throw new Error(`保存满意定稿规则到 OSS 失败: ${response.status} ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('保存满意定稿规则到 OSS 出错:', error);
        throw error;
    }
};

export const getSatisfactionRulesFromOSS = async () => {
    const userId = getUserId();
    try {
        const response = await fetch(`${OSS_BACKEND_URL}/get-satisfaction-rules?userId=${userId}`, {
            headers: buildAuthorizedHeaders()
        });
        if (response.status === 404) {
            console.warn('[OSS] get-satisfaction-rules endpoint is not yet active on Aliyun FC. Using local rules cache.');
            return [];
        }
        if (!response.ok) throw new Error(`HTTP status ${response.status}`);
        
        const result = await response.json();
        if (result.status === 'success') {
            return result.rules || [];
        } else {
            throw new Error(result.message || '获取失败');
        }
    } catch (error) {
        console.warn(`[ImplicitMemory] 从 OSS 云端获取定稿规则失败或尚未初始化（可能由于刚新建账户未保存过数据，或云端函数更新中有短暂延迟）。已自动无缝切换到本地 IndexedDB：${error.message}`);
        return [];
    }
};
