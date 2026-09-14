import { RECOGNIZE_BACKEND_URL } from '../core/utils.js';

const DEFAULT_MODEL = 'Wan-AI/Wan2.2-I2V-A14B';
const DEFAULT_IMAGE_SIZE = '1280x720';
const DEFAULT_POLL_INTERVAL_MS = 5000;
const DEFAULT_MAX_ATTEMPTS = 48;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function pickWanImageSize(width, height) {
    const safeWidth = Number(width) || 1;
    const safeHeight = Number(height) || 1;
    const ratio = safeWidth / safeHeight;

    if (ratio > 1.15) return '1280x720';
    if (ratio < 0.85) return '720x1280';
    return '960x960';
}

function extractVideoUrl(payload) {
    return payload?.videoUrl
        || payload?.video
        || payload?.url
        || payload?.videos?.[0]?.url
        || payload?.results?.videos?.[0]?.url
        || payload?.result?.videos?.[0]?.url
        || payload?.data?.videos?.[0]?.url
        || payload?.output?.video
        || payload?.output?.url
        || null;
}

function extractStatus(payload) {
    return payload?.status
        || payload?.state
        || payload?.result?.status
        || payload?.data?.status
        || '';
}

function extractFailureMessage(payload) {
    return payload?.error?.message
        || payload?.message
        || payload?.detail
        || payload?.error
        || '视频生成失败';
}

export async function submitWanVideo({
    prompt,
    image,
    imageSize = DEFAULT_IMAGE_SIZE,
    negativePrompt = '',
    seed,
    model = DEFAULT_MODEL
}) {
    const response = await fetch(RECOGNIZE_BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mode: 'siliconflow_video_submit',
            model,
            prompt,
            image,
            image_size: imageSize,
            negative_prompt: negativePrompt,
            seed
        })
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data?.error || data?.details || `视频任务提交失败: ${response.status}`);
    }

    const requestId = data?.requestId || data?.request_id;
    if (!requestId) {
        throw new Error('视频任务提交成功，但未返回 requestId');
    }

    return {
        ...data,
        requestId
    };
}

export async function fetchWanVideoStatus(requestId) {
    const response = await fetch(RECOGNIZE_BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mode: 'siliconflow_video_status',
            requestId
        })
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data?.error || data?.details || `视频任务状态查询失败: ${response.status}`);
    }

    return data;
}

export async function pollWanVideoStatus(requestId, options = {}) {
    const {
        intervalMs = DEFAULT_POLL_INTERVAL_MS,
        maxAttempts = DEFAULT_MAX_ATTEMPTS,
        onProgress
    } = options;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        const payload = await fetchWanVideoStatus(requestId);
        const status = extractStatus(payload);
        const videoUrl = extractVideoUrl(payload);

        if (typeof onProgress === 'function') {
            onProgress({
                attempt,
                requestId,
                status,
                payload,
                videoUrl
            });
        }

        if (status === 'Succeed' || videoUrl) {
            if (!videoUrl) {
                throw new Error('视频任务已成功，但未拿到视频地址');
            }
            return {
                requestId,
                status: status || 'Succeed',
                videoUrl,
                payload
            };
        }

        if (status === 'Failed') {
            throw new Error(extractFailureMessage(payload));
        }

        if (attempt < maxAttempts) {
            await sleep(intervalMs);
        }
    }

    throw new Error('视频生成超时，请稍后重试');
}

export async function generateWanMotionPreview({
    prompt,
    image,
    imageSize = DEFAULT_IMAGE_SIZE,
    negativePrompt = '',
    seed,
    model = DEFAULT_MODEL,
    onProgress
}) {
    const submitResult = await submitWanVideo({
        prompt,
        image,
        imageSize,
        negativePrompt,
        seed,
        model
    });

    const finalResult = await pollWanVideoStatus(submitResult.requestId, { onProgress });
    return {
        ...finalResult,
        submitResult
    };
}
