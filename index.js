const http = require('http');
const https = require('https');

// Use native fetch when available, otherwise fall back to node-fetch
const fetch = typeof globalThis.fetch !== 'undefined' ? globalThis.fetch : require('node-fetch');

const API_ENDPOINT = 'https://api.siliconflow.cn/v1/chat/completions';
const API_TOKEN = process.env.API_TOKEN || ''; // SiliconFlow API Token
const QUICKROUTER_API_TOKEN = process.env.QUICKROUTER_API_TOKEN || ''; // QuickRouter API Token
const SILICONFLOW_IMAGE_API_ENDPOINT = process.env.SILICONFLOW_IMAGE_API_ENDPOINT || 'https://api.siliconflow.cn/v1/images/generations';
const SILICONFLOW_IMAGE_EDIT_API_TOKEN = process.env.SILICONFLOW_IMAGE_EDIT_API_TOKEN || process.env.SILICONFLOW_API_TOKEN || '';
const SILICONFLOW_IMAGE_EDIT_MODEL = process.env.SILICONFLOW_IMAGE_EDIT_MODEL || 'Qwen/Qwen-Image-Edit-2509';
const DASHSCOPE_API_ENDPOINT = process.env.DASHSCOPE_API_ENDPOINT || 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation';
const DASHSCOPE_API_KEY = process.env.DASHSCOPE_API_KEY || '';
// The old alias can remain in an already-deployed FC environment. Treat it as
// the current Pro model instead of silently sending requests to the old route.
const configuredDashscopeImageModel = String(process.env.DASHSCOPE_IMAGE_MODEL || '').trim().toLowerCase();
const DASHSCOPE_IMAGE_MODEL = !configuredDashscopeImageModel || configuredDashscopeImageModel === 'qwen-image-3.0'
  ? 'qwen-image-3.0-pro'
  : configuredDashscopeImageModel;
const SILICONFLOW_VIDEO_API_TOKEN = process.env.SILICONFLOW_VIDEO_EDIT_API_TOKEN || '';
const SILICONFLOW_VIDEO_SUBMIT_ENDPOINT = process.env.SILICONFLOW_VIDEO_SUBMIT_ENDPOINT || 'https://api.siliconflow.cn/v1/video/submit';
const SILICONFLOW_VIDEO_STATUS_ENDPOINT = process.env.SILICONFLOW_VIDEO_STATUS_ENDPOINT || 'https://api.siliconflow.cn/v1/video/status';
const SILICONFLOW_VIDEO_MODEL = process.env.SILICONFLOW_VIDEO_MODEL || 'Wan-AI/Wan2.2-I2V-A14B';
const FASTSAM_CLOUD_ENDPOINT = (process.env.FASTSAM_CLOUD_ENDPOINT || '').replace(/\/+$/, '');
const FASTSAM_CLOUD_TOKEN = process.env.FASTSAM_CLOUD_TOKEN || '';

const PORT = 9000;

// 允许跨域的响应头
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*', 
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400'
};

// 智能获取和检索商品关联的推荐系统数据库
const ProductDatabase = {
  async findRecommendations(product) {
    try {
      console.log('🔍 [ProductDatabase] 正在检索相似商品，当前输入商品:', product.name, product.category);
      const res = await fetch('https://www.marmoai.cn/images/sku-data/product_database.json');
      if (!res.ok) throw new Error(`HTTP 异常 ${res.status}`);
      const data = await res.json();
      const products = data.products || [];
      
      // 按类别或者关键字匹配相似度进行推荐
      const matches = products.filter(p => {
        // 排除当前输入的同款商品自身
        if (p.id === product.id || p.skuId === product.skuId) return false;
        
        // 1. 同级品系分类优先关联
        if (product.category && p.category === product.category) return true;
        
        // 2. 文本语义词汇关键词召回
        const kwMatches = product.name && p.name && (p.name.includes(product.name) || product.name.includes(p.name));
        return !!kwMatches;
      });
      
      return matches.slice(0, 6);
    } catch (e) {
      console.error('⚠️ [ProductDatabase] 获取并分析产品数据失败:', e.message);
      return [];
    }
  }
};

function normalizeImageInputToDataUrl(imageInput, fallbackMimeType = 'image/png') {
  if (!imageInput || typeof imageInput !== 'string') return null;
  const trimmed = imageInput.trim();
  if (!trimmed) return null;
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (/^data:image\//i.test(trimmed)) return trimmed;
  return `data:${fallbackMimeType};base64,${trimmed}`;
}

async function fetchImageUrlAsBase64(imageUrl) {
  const response = await fetch(imageUrl, {
    method: 'GET',
    headers: {
      'Accept': 'image/*,*/*'
    }
  });

  if (!response.ok) {
    throw new Error(`下载生成图片失败: ${response.status}`);
  }

  const mimeType = response.headers.get('content-type') || 'image/png';
  const arrayBuffer = await response.arrayBuffer();
  const base64 = Buffer.from(arrayBuffer).toString('base64');
  return { base64, mimeType };
}

function extractSiliconflowImageUrl(responseData) {
  return responseData?.data?.[0]?.url
    || responseData?.images?.[0]?.url
    || responseData?.output?.[0]?.url
    || responseData?.url
    || null;
}

function summarizeImageApiResponse(responseData, rawLength = 0) {
  const items = Array.isArray(responseData?.data) ? responseData.data : [];
  return {
    topLevelKeys: responseData && typeof responseData === 'object'
      ? Object.keys(responseData)
      : [],
    rawLength,
    imageCount: items.length,
    b64Lengths: items.map(item => typeof item?.b64_json === 'string' ? item.b64_json.length : 0),
    hasImageUrl: items.some(item => typeof item?.url === 'string'),
    hasError: Boolean(responseData?.error),
    errorType: typeof responseData?.error?.type === 'string' ? responseData.error.type : null,
    errorCode: typeof responseData?.error?.code === 'string' ? responseData.error.code : null
  };
}

function normalizeDashscopeImageSize(size) {
  if (typeof size !== 'string' || !size.trim()) return '1024*1024';
  return size.trim().replace(/[xX]/g, '*');
}

function isQwenImage3Request(requestBody) {
  const mode = String(requestBody?.mode || '').toLowerCase();
  const model = String(requestBody?.model || '').toLowerCase();

  // A clean-plate task is not inherently a Qwen 3.0 task. Respect its
  // explicit model so Qwen Image Edit 2509 never gets redirected to DashScope.
  if (isSiliconflowQwenImageEditRequest(requestBody)) return false;

  return model === 'qwen-image-3.0-pro'
    || model === 'qwen-cleanplate'
    || model === DASHSCOPE_IMAGE_MODEL.toLowerCase()
    || mode === 'qwen_image_generation'
    || mode === 'qwen_image_edit'
    || mode === 'qwen-image-3.0-pro';
}

function isSiliconflowQwenImageEditRequest(requestBody) {
  const mode = String(requestBody?.mode || '').toLowerCase();
  const model = String(requestBody?.model || '').toLowerCase();
  const editModel = SILICONFLOW_IMAGE_EDIT_MODEL.toLowerCase();
  return model === editModel
    || model === 'qwen/qwen-image-edit-2509'
    || mode === 'qwen-image-edit-2509'
    || ((mode === 'image_edit' || mode === 'image_generation') && model === editModel);
}

function getDashscopeImageInputs(requestBody) {
  const inputs = [];
  const addImage = (value, fallbackMimeType = 'image/png') => {
    const normalized = normalizeImageInputToDataUrl(value, fallbackMimeType);
    if (normalized) inputs.push(normalized);
  };

  addImage(requestBody?.image, requestBody?.imageMimeType || 'image/png');
  addImage(requestBody?.image2, requestBody?.image2MimeType || 'image/png');
  addImage(requestBody?.image3, requestBody?.image3MimeType || 'image/png');

  if (Array.isArray(requestBody?.images)) {
    requestBody.images.slice(0, 4).forEach(image => addImage(image, 'image/png'));
  }

  return inputs;
}

function extractDashscopeImageCandidate(responseData) {
  const roots = [
    responseData?.output,
    responseData?.data,
    responseData?.result,
    responseData?.results
  ];
  const preferredKeys = new Set(['image', 'image_url', 'b64_json', 'base64']);
  const visited = new Set();

  const visit = (value, key = '', depth = 0) => {
    if (value == null || depth > 8) return null;
    if (typeof value === 'string') {
      if (preferredKeys.has(key) || key === 'url') return value;
      return null;
    }
    if (typeof value !== 'object' || visited.has(value)) return null;
    visited.add(value);

    if (Array.isArray(value)) {
      for (const entry of value) {
        const candidate = visit(entry, '', depth + 1);
        if (candidate) return candidate;
      }
      return null;
    }

    for (const [entryKey, entryValue] of Object.entries(value)) {
      if (preferredKeys.has(entryKey) && typeof entryValue === 'string') return entryValue;
    }
    for (const [entryKey, entryValue] of Object.entries(value)) {
      const candidate = visit(entryValue, entryKey, depth + 1);
      if (candidate) return candidate;
    }
    return null;
  };

  for (const root of roots) {
    const candidate = visit(root);
    if (candidate) return candidate;
  }
  return null;
}

async function normalizeDashscopeImageResult(candidate) {
  if (typeof candidate !== 'string' || !candidate.trim()) return null;
  const value = candidate.trim();

  if (/^https?:\/\//i.test(value)) {
    const downloaded = await fetchImageUrlAsBase64(value);
    return { base64: downloaded.base64, mimeType: downloaded.mimeType, imageUrl: value };
  }

  const dataUrlMatch = value.match(/^data:(image\/[^;]+);base64,(.+)$/i);
  if (dataUrlMatch) {
    return { base64: dataUrlMatch[2], mimeType: dataUrlMatch[1] };
  }

  return { base64: value.replace(/^base64,/, ''), mimeType: 'image/png' };
}

async function callDashscopeQwenImage(requestBody) {
  if (!DASHSCOPE_API_KEY) {
    return {
      status: 500,
      body: { error: '未配置 DASHSCOPE_API_KEY 环境变量' }
    };
  }

  if (!requestBody?.prompt) {
    return { status: 400, body: { error: '缺少 prompt 参数' } };
  }

  const imageInputs = getDashscopeImageInputs(requestBody);
  const content = imageInputs.map(image => ({ image }));
  const maskInput = normalizeImageInputToDataUrl(requestBody?.mask, requestBody?.maskMimeType || 'image/png');
  if (maskInput) {
    content.push({ image: maskInput });
  }
  const textInstructions = [
    requestBody.prompt,
    maskInput
      ? 'If the final reference image is a mask, apply the requested edit only to the masked target area and preserve the rest of the scene.'
      : ''
  ].filter(Boolean).join('\n\n');
  // DashScope Qwen Image accepts exactly one text item per user message.
  content.push({ text: textInstructions });

  const suppliedParameters = requestBody?.parameters && typeof requestBody.parameters === 'object'
    ? requestBody.parameters
    : {};
  const parameters = {
    ...suppliedParameters,
    watermark: requestBody.watermark ?? suppliedParameters.watermark ?? false,
    prompt_extend: requestBody.prompt_extend ?? suppliedParameters.prompt_extend ?? true,
    size: normalizeDashscopeImageSize(requestBody.size || suppliedParameters.size)
  };
  if (requestBody.negative_prompt || suppliedParameters.negative_prompt) {
    parameters.negative_prompt = requestBody.negative_prompt || suppliedParameters.negative_prompt;
  }

  const payload = {
    model: DASHSCOPE_IMAGE_MODEL,
    input: {
      messages: [{
        role: 'user',
        content
      }]
    },
    parameters
  };

  try {
    console.log(`🎬 正在转发 Qwen Image 请求到 DashScope: model=${DASHSCOPE_IMAGE_MODEL}, size=${parameters.size}, images=${imageInputs.length}, textItems=1, hasMask=${!!maskInput}`);
    const apiRes = await fetch(DASHSCOPE_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${DASHSCOPE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const rawText = await apiRes.text();
    const responseData = tryParseJson(rawText) || { raw: rawText };
    console.log(`📦 DashScope Qwen Image 返回状态: ${apiRes.status}`);

    if (!apiRes.ok) {
      return { status: apiRes.status, body: responseData };
    }

    const candidate = extractDashscopeImageCandidate(responseData);
    if (!candidate) {
      return {
        status: 502,
        body: {
          error: 'DashScope Qwen Image 未返回图片结果',
          upstream: responseData
        }
      };
    }

    const normalizedImage = await normalizeDashscopeImageResult(candidate);
    if (!normalizedImage?.base64) {
      return {
        status: 502,
        body: { error: 'DashScope Qwen Image 返回的图片格式无效', upstream: responseData }
      };
    }

    return {
      status: 200,
      body: {
        ...responseData,
        model: DASHSCOPE_IMAGE_MODEL,
        mimeType: normalizedImage.mimeType,
        imageData: normalizedImage.base64,
        imageUrl: normalizedImage.imageUrl || null,
        data: [{ b64_json: normalizedImage.base64 }]
      }
    };
  } catch (error) {
    console.error('⚠️ 请求 DashScope Qwen Image 出错:', error);
    return {
      status: 500,
      body: {
        error: '转发 Qwen Image 到 DashScope 时出错',
        details: error.message
      }
    };
  }
}

function normalizeVideoImageSize(size, fallback = '1280x720') {
  const allowedSizes = new Set(['1280x720', '720x1280', '960x960']);
  return allowedSizes.has(size) ? size : fallback;
}

function extractSiliconflowVideoUrl(responseData) {
  const candidates = [
    responseData?.video,
    responseData?.url,
    responseData?.videoUrl,
    responseData?.output?.video,
    responseData?.output?.url,
    responseData?.videos?.[0]?.url,
    responseData?.videos?.[0]?.video,
    responseData?.results?.videos?.[0]?.url,
    responseData?.results?.video?.url,
    responseData?.result?.videos?.[0]?.url,
    responseData?.result?.video?.url,
    responseData?.data?.videos?.[0]?.url,
    responseData?.data?.video?.url
  ];

  for (const value of candidates) {
    if (typeof value === 'string' && /^https?:\/\//i.test(value)) {
      return value;
    }
  }

  const visited = new Set();
  const scan = (value, depth = 0) => {
    if (!value || depth > 6) return null;
    if (typeof value === 'string') {
      return /^https?:\/\//i.test(value) && /\.(mp4|mov|webm)(\?|$)/i.test(value) ? value : null;
    }
    if (typeof value !== 'object') return null;
    if (visited.has(value)) return null;
    visited.add(value);

    if (Array.isArray(value)) {
      for (const entry of value) {
        const nested = scan(entry, depth + 1);
        if (nested) return nested;
      }
      return null;
    }

    for (const [key, nestedValue] of Object.entries(value)) {
      if ((key === 'url' || key === 'videoUrl') && typeof nestedValue === 'string' && /^https?:\/\//i.test(nestedValue)) {
        return nestedValue;
      }
      const nested = scan(nestedValue, depth + 1);
      if (nested) return nested;
    }

    return null;
  };

  return scan(responseData);
}

function tryParseJson(rawText) {
  if (!rawText || typeof rawText !== 'string') return null;
  try {
    return JSON.parse(rawText);
  } catch (err) {
    return null;
  }
}

function normalizeQuickRouterErrorStatus(upstreamStatus, responseData) {
  const errorPayload = responseData?.error || responseData || {};
  const code = errorPayload?.code || '';
  const type = errorPayload?.type || '';
  const message = errorPayload?.message || '';
  const looksLikeModerationBlock =
    code === 'moderation_blocked' ||
    /rejected by the safety system/i.test(message) ||
    /moderation/i.test(message);

  return {
    normalizedStatus: looksLikeModerationBlock ? 403 : upstreamStatus,
    normalizedError: looksLikeModerationBlock ? 'safety_moderation' : null,
    upstreamStatus
  };
}

const server = http.createServer(async (req, res) => {
  // 处理预检请求
  if (req.method === 'OPTIONS') {
    res.writeHead(204, CORS_HEADERS);
    res.end();
    return;
  }
  
  // 处理非POST请求
  if (req.method !== 'POST') {
    res.writeHead(405, {
      'Content-Type': 'application/json',
      ...CORS_HEADERS
    });
    res.end(JSON.stringify({ error: '仅支持 POST 请求' }));
    return;
  }

  let body = '';

  req.on('data', chunk => {
    body += chunk.toString();
  });

  req.on('end', async () => {
    let requestBody = null;
    try {
      console.log('========== 请求开始 ==========');
      console.log('method:', req.method);
      console.log('url:', req.url);
      console.log('raw body length:', body.length);
      console.log('raw body preview:', body.slice(0, 500));

      requestBody = JSON.parse(body);

      console.log('parsed body:', requestBody);
      console.log('mode:', requestBody.mode);

      // 新增：处理获取数据库URL请求
      if (requestBody.mode === 'get_database_url') {
        res.writeHead(200, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify({ 
          database_url: 'https://www.marmoai.cn/images/sku-data/product_database.json'
        }));
        return;
      }

      if (requestBody.mode === 'segment_proxy') {
        console.log('🎬 正在转发分割请求到 FastSAM/SAM 云端函数...');

        if (!FASTSAM_CLOUD_ENDPOINT) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '未配置 FASTSAM_CLOUD_ENDPOINT 环境变量' }));
          return;
        }

        if (!requestBody.payload || typeof requestBody.payload !== 'object') {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 payload 参数' }));
          return;
        }

        try {
          const headers = {
            'Content-Type': 'application/json'
          };
          if (FASTSAM_CLOUD_TOKEN) {
            headers.Authorization = `Bearer ${FASTSAM_CLOUD_TOKEN}`;
          }

          const apiRes = await fetch(`${FASTSAM_CLOUD_ENDPOINT}/segment`, {
            method: 'POST',
            headers,
            body: JSON.stringify(requestBody.payload)
          });

          const rawText = await apiRes.text();
          const responseData = tryParseJson(rawText) || { raw: rawText };
          console.log(`📦 FastSAM 云端返回状态: ${apiRes.status}`);

          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify(responseData));
        } catch (errApi) {
          console.error('⚠️ 请求 FastSAM 云端函数出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            error: '转发分割请求到 FastSAM 云端函数时出错',
            details: errApi.message
          }));
        }
        return;
      }

      if (requestBody.mode === 'siliconflow_video_submit' || requestBody.mode === 'magic_motion_preview_submit') {
        console.log('🎬 正在提交 SiliconFlow Wan 视频任务...');

        if (!requestBody.prompt || !requestBody.image) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 prompt 或 image 参数' }));
          return;
        }

        if (!SILICONFLOW_VIDEO_API_TOKEN) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '未配置 SILICONFLOW_VIDEO_EDIT_API_TOKEN 环境变量' }));
          return;
        }

        const normalizedImage = normalizeImageInputToDataUrl(requestBody.image, requestBody.imageMimeType || 'image/png');
        if (!normalizedImage) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: 'image 参数无效，需提供 base64/data URL/http(s) URL' }));
          return;
        }

        const apiPayload = {
          model: requestBody.model || SILICONFLOW_VIDEO_MODEL,
          prompt: requestBody.prompt,
          image_size: normalizeVideoImageSize(requestBody.image_size || requestBody.imageSize),
          image: normalizedImage
        };

        if (requestBody.negative_prompt) apiPayload.negative_prompt = requestBody.negative_prompt;
        if (requestBody.seed !== undefined && requestBody.seed !== null && requestBody.seed !== '') {
          apiPayload.seed = Number(requestBody.seed);
        }

        try {
          const apiRes = await fetch(SILICONFLOW_VIDEO_SUBMIT_ENDPOINT, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${SILICONFLOW_VIDEO_API_TOKEN}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiPayload)
          });

          const rawText = await apiRes.text();
          const responseData = tryParseJson(rawText) || { raw: rawText };
          const traceId = apiRes.headers.get('x-siliconcloud-trace-id') || '';

          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            ...responseData,
            traceId,
            requestId: responseData?.requestId || responseData?.request_id || null
          }));
        } catch (errApi) {
          console.error('⚠️ 提交 SiliconFlow Wan 视频任务出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            error: '提交 SiliconFlow Wan 视频任务时出错',
            details: errApi.message
          }));
        }
        return;
      }

      if (requestBody.mode === 'siliconflow_video_status' || requestBody.mode === 'magic_motion_preview_status') {
        console.log('🎬 正在查询 SiliconFlow Wan 视频任务状态...');

        if (!requestBody.requestId) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 requestId 参数' }));
          return;
        }

        if (!SILICONFLOW_VIDEO_API_TOKEN) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '未配置 SILICONFLOW_VIDEO_EDIT_API_TOKEN 环境变量' }));
          return;
        }

        try {
          const apiRes = await fetch(SILICONFLOW_VIDEO_STATUS_ENDPOINT, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${SILICONFLOW_VIDEO_API_TOKEN}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ requestId: requestBody.requestId })
          });

          const rawText = await apiRes.text();
          const responseData = tryParseJson(rawText) || { raw: rawText };
          const traceId = apiRes.headers.get('x-siliconcloud-trace-id') || '';
          const videoUrl = extractSiliconflowVideoUrl(responseData);

          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            ...responseData,
            traceId,
            requestId: requestBody.requestId,
            videoUrl
          }));
        } catch (errApi) {
          console.error('⚠️ 查询 SiliconFlow Wan 视频任务状态出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            error: '查询 SiliconFlow Wan 视频任务状态时出错',
            details: errApi.message
          }));
        }
        return;
      }

      // Qwen Image 3.0 Pro uses DashScope's multimodal conversation API.
      // Keep this before the GPT routes so the existing clean-plate modes and
      // future explicit qwen-image-3.0-pro requests share one implementation.
      if (isQwenImage3Request(requestBody)) {
        const dashscopeResult = await callDashscopeQwenImage(requestBody);
        res.writeHead(dashscopeResult.status, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify(dashscopeResult.body));
        return;
      }

      // 新增：处理 GPT-image-2 图片生成中转接口 (Text-to-Image)
      if (
        (requestBody.mode === 'image_generation' || requestBody.mode === 'gpt-image-2-generation') &&
        !isSiliconflowQwenImageEditRequest(requestBody)
      ) {
        console.log('🎬 正在转发图片生成请求到 GPT-image-2 (QuickRouter)...');
        
        // 确保包含必需的参数
        if (!requestBody.prompt) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 prompt 参数' }));
          return;
        }

        const apiPayload = {
          model: "gpt-image-2",
          prompt: requestBody.prompt,
          n: requestBody.n || 1,
          size: requestBody.size || "1024x1024",
          quality: requestBody.quality || "auto",
          output_format: requestBody.output_format || requestBody.format || "jpeg"
        };

        const apiRes = await fetch('https://api.quickrouter.us/v1/images/generations', {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${QUICKROUTER_API_TOKEN}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(apiPayload)
        });

        const rawText = await apiRes.text();
        const responseData = tryParseJson(rawText) || { raw: rawText };
        const statusInfo = normalizeQuickRouterErrorStatus(apiRes.status, responseData);
        console.log('📦 GPT-image-2 生成返回层摘要：', {
          httpStatus: apiRes.status,
          ...summarizeImageApiResponse(responseData, rawText.length)
        });

        const outboundBody = apiRes.ok ? responseData : {
          ...responseData,
          upstream_status: statusInfo.upstreamStatus,
          normalized_status: statusInfo.normalizedStatus,
          error: responseData?.error ? {
            ...responseData.error,
            normalized_reason: statusInfo.normalizedError
          } : responseData.error
        };

        res.writeHead(apiRes.ok ? apiRes.status : statusInfo.normalizedStatus, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify(outboundBody));
        return;
      }

      // 新增：处理 GPT-image-2 图片编辑中转接口
      // Keep explicit SiliconFlow Qwen Image Edit requests ahead of this
      // generic image_edit branch. The old mode-only condition would route
      // every image_edit request, including Qwen/Qwen-Image-Edit-2509, to GPT.
      if (
        (requestBody.mode === 'image_edit' || requestBody.mode === 'gpt-image-2') &&
        !isSiliconflowQwenImageEditRequest(requestBody)
      ) {
        console.log('🎬 正在转发图片编辑请求到 GPT-image-2 (QuickRouter)...');
        
        // 确保包含必需的参数
        if (!requestBody.image || !requestBody.prompt) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 image (base64) 或 prompt 参数' }));
          return;
        }

        function base64ToBuffer(base64String) {
          if (!base64String) return null;
          const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');
          return Buffer.from(base64Data, 'base64');
        }

        const imageBuf = base64ToBuffer(requestBody.image);
        if (!imageBuf) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '无效的 image 图像 base64 数据' }));
          return;
        }

        // 构造 OpenAI 图像编辑 API 标准 multipart 字段
        const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
        
        const fields = {
          model: "gpt-image-2",
          prompt: requestBody.prompt,
          n: String(requestBody.n || 1),
          size: requestBody.size || "1024x1024",
          output_format: requestBody.output_format || requestBody.format || "jpeg",
          image: {
            buffer: imageBuf,
            filename: 'image.png',
            contentType: 'image/png'
          }
        };

        if (requestBody.mask) {
          const maskBuf = base64ToBuffer(requestBody.mask);
          if (maskBuf) {
            fields.mask = {
              buffer: maskBuf,
              filename: 'mask.png',
              contentType: 'image/png'
            };
          }
        }

        if (requestBody.image2) {
          const image2Buf = base64ToBuffer(requestBody.image2);
          if (image2Buf) {
            fields.image2 = {
              buffer: image2Buf,
              filename: 'image2.png',
              contentType: 'image/png'
            };
          }
        }

        if (requestBody.image3) {
          const image3Buf = base64ToBuffer(requestBody.image3);
          if (image3Buf) {
            fields.image3 = {
              buffer: image3Buf,
              filename: 'image3.png',
              contentType: 'image/png'
            };
          }
        }

        const resolveMultipartFieldName = (key) => {
          if (key === 'image2' || key === 'image3') return 'image[]';
          return key;
        };

        // 手动构造 multipart/form-data 缓冲区块
        const parts = [];
        for (const [key, value] of Object.entries(fields)) {
          if (value === undefined || value === null) continue;
          const fieldName = resolveMultipartFieldName(key);

          parts.push(Buffer.from(`--${boundary}\r\n`));
          if (value && typeof value === 'object' && value.buffer) {
            // 文件部分
            parts.push(Buffer.from(`Content-Disposition: form-data; name="${fieldName}"; filename="${value.filename}"\r\n`));
            parts.push(Buffer.from(`Content-Type: ${value.contentType}\r\n\r\n`));
            parts.push(value.buffer);
          } else {
            // 普通文本字段
            parts.push(Buffer.from(`Content-Disposition: form-data; name="${fieldName}"\r\n\r\n`));
            parts.push(Buffer.from(String(value)));
          }
          parts.push(Buffer.from('\r\n'));
        }
        parts.push(Buffer.from(`--${boundary}--\r\n`));

        const multipartBody = Buffer.concat(parts);

        // 使用 fetch 转发请求，极大地提升容错并规避 https.request 在特定容器下面死锁和超时的问题
        try {
          const apiRes = await fetch('https://api.quickrouter.us/v1/images/edits', {
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Authorization': `Bearer ${QUICKROUTER_API_TOKEN}`,
              'Content-Type': `multipart/form-data; boundary=${boundary}`,
              'Content-Length': String(multipartBody.length)
            },
            body: multipartBody
          });

          const resBody = await apiRes.text();
          const responseData = tryParseJson(resBody) || { raw: resBody };
          const statusInfo = normalizeQuickRouterErrorStatus(apiRes.status, responseData);
          console.log(`📦 GPT-image-2 返回层摘要: ${apiRes.status}`, {
            httpStatus: apiRes.status,
            ...summarizeImageApiResponse(responseData, resBody.length)
          });

          const outboundBody = apiRes.ok ? resBody : JSON.stringify({
            ...responseData,
            upstream_status: statusInfo.upstreamStatus,
            normalized_status: statusInfo.normalizedStatus,
            error: responseData?.error ? {
              ...responseData.error,
              normalized_reason: statusInfo.normalizedError
            } : responseData.error
          });
          
          res.writeHead(apiRes.ok ? apiRes.status : statusInfo.normalizedStatus, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(outboundBody);
        } catch (errApi) {
          console.error('⚠️ 请求 QuickRouter 出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '转发图片编辑请求到 QuickRouter 时出错', details: errApi.message }));
        }
        return;
      }

      // 新增：处理背景净化/遮挡补全测试图片编辑中转接口（优先给 clean plate / completion 使用）
      if (
        isSiliconflowQwenImageEditRequest(requestBody) ||
        requestBody.mode === 'image_edit_cleanplate_test' ||
        requestBody.mode === 'clean_plate_test_edit' ||
        requestBody.mode === 'occlusion_completion_test_edit'
      ) {
        console.log(`🎬 正在转发图片任务到 SiliconFlow Qwen Image Edit: model=${requestBody.model || SILICONFLOW_IMAGE_EDIT_MODEL}, mode=${requestBody.mode}, size=${requestBody.size || 'provider-default'}`);

        const isTextToImage = requestBody.mode === 'image_generation' && !requestBody.image;
        if (!requestBody.prompt || (!requestBody.image && !isTextToImage)) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 prompt 或 image 参数' }));
          return;
        }

        if (!SILICONFLOW_IMAGE_EDIT_API_TOKEN) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '未配置 SILICONFLOW_IMAGE_EDIT_API_TOKEN 环境变量' }));
          return;
        }

        const primaryImage = normalizeImageInputToDataUrl(requestBody.image, requestBody.imageMimeType || 'image/png');
        const secondaryImage = normalizeImageInputToDataUrl(requestBody.image2, requestBody.image2MimeType || 'image/png');
        const tertiaryImage = normalizeImageInputToDataUrl(requestBody.image3, requestBody.image3MimeType || 'image/png');

        if (!primaryImage && !isTextToImage) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: 'image 参数无效，需提供 base64/data URL/http(s) URL' }));
          return;
        }

        const apiPayload = {
          model: requestBody.model || SILICONFLOW_IMAGE_EDIT_MODEL,
          prompt: requestBody.prompt,
          num_inference_steps: Number(requestBody.num_inference_steps || requestBody.steps || 40),
          cfg: Number(requestBody.cfg || 4)
        };

        if (primaryImage) apiPayload.image = primaryImage;

        if (secondaryImage) apiPayload.image2 = secondaryImage;
        if (tertiaryImage) apiPayload.image3 = tertiaryImage;

        if (requestBody.size) apiPayload.size = requestBody.size;
        if (requestBody.seed !== undefined && requestBody.seed !== null) apiPayload.seed = requestBody.seed;
        if (requestBody.negative_prompt) apiPayload.negative_prompt = requestBody.negative_prompt;

        try {
          const apiRes = await fetch(SILICONFLOW_IMAGE_API_ENDPOINT, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${SILICONFLOW_IMAGE_EDIT_API_TOKEN}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiPayload)
          });

          const rawText = await apiRes.text();
          let responseData = null;
          try {
            responseData = JSON.parse(rawText);
          } catch (parseErr) {
            responseData = { raw: rawText };
          }

          console.log(`📦 SiliconFlow Qwen Image Edit 返回状态: ${apiRes.status}`);

          const imageUrl = extractSiliconflowImageUrl(responseData);
          if (!imageUrl) {
            res.writeHead(apiRes.status, {
              'Content-Type': 'application/json',
              ...CORS_HEADERS
            });
            res.end(JSON.stringify(responseData));
            return;
          }

          const { base64, mimeType } = await fetchImageUrlAsBase64(imageUrl);
          const normalizedResponse = {
            ...responseData,
            mimeType,
            imageData: base64,
            data: [
              {
                b64_json: base64
              }
            ]
          };

          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify(normalizedResponse));
        } catch (errApi) {
          console.error('⚠️ 请求 SiliconFlow Image Edit 出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({
            error: '转发测试图片编辑请求到 SiliconFlow 时出错',
            details: errApi.message
          }));
        }
        return;
      }

      // 新增：处理 Gemini Lite 中转接口
      if (requestBody.mode === 'gemini_lite_generate') {
        console.log('🎬 正在转发 Gemini Lite 请求到 QuickRouter...');
        
        if (!requestBody.model || !requestBody.payload) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 model 或 payload 参数' }));
          return;
        }

        const modelUrl = `https://api.quickrouter.us/v1beta/models/${requestBody.model}:generateContent`;

        try {
          const apiRes = await fetch(modelUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-goog-api-key': QUICKROUTER_API_TOKEN
            },
            body: JSON.stringify(requestBody.payload)
          });

          const responseData = await apiRes.json();
          console.log(`📦 Gemini Lite 返回状态: ${apiRes.status}`);

          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify(responseData));
        } catch (errApi) {
          console.error('⚠️ 请求 QuickRouter 出错:', errApi);
          res.writeHead(500, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '转发请求到 QuickRouter 时出错', details: errApi.message }));
        }
        return;
      }

      // 识别请求
      if (requestBody.mode === 'recognize') {
        const apiRes = await fetch(API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${API_TOKEN}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: "Qwen/Qwen3-VL-32B-Instruct",
            messages: [
              { role: "user", content: requestBody.content }
            ],
            temperature: 0.2
          })
        });
      
        const data = await apiRes.json();
      
        const raw = data?.choices?.[0]?.message?.content || "";
      
        let parsed = {};
        try {
          parsed = JSON.parse(raw);
        } catch (e) {
          parsed = {};
        }
      
        const objects = (parsed.objects || parsed.家具 || []);
      
        // 🔥 关键：在这里统一 bbox
        const finalObjects = objects.map(o => {
          let bbox = o.bbox;
      
          if (bbox && (bbox[2] > 1 || bbox[3] > 1)) {
            bbox = [
              bbox[0] / 1000,
              bbox[1] / 1000,
              bbox[2] / 1000,
              bbox[3] / 1000
            ];
          }
      
          return {
            ...o,
            bbox
          };
        });
      
        res.writeHead(200, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
      
        return res.end(JSON.stringify({
          success: true,
          objects: finalObjects
        }));
      }

      // 推荐请求
      if (requestBody.mode === 'recommend') {
        const inputObj = requestBody.product;
        if (!inputObj || !inputObj.name) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: 'product 对象不完整' }));
          return;
        }

        const recommendations = await ProductDatabase.findRecommendations(inputObj);
        res.writeHead(200, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify({ recommendations }));
        return;
      }

      // 对话请求
      if (requestBody.mode === 'chat') {
        if (!requestBody.question || !requestBody.objects) {
          res.writeHead(400, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(JSON.stringify({ error: '缺少 question 或 objects 参数' }));
          return;
        }
    
        const prompt = `用户问题：${requestBody.question}\n\n相关物品：${
          requestBody.objects.map(obj => 
            `${obj.name} (材质:${obj.material}, 形状:${obj.shape})`
          ).join(', ')
        }`;
    
        const apiRes = await fetch(API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${API_TOKEN}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: "Qwen/Qwen3-VL-32B-Instruct",
            messages: [
              { 
                role: "system", 
                content: "你是一个专业室内设计助手，请根据用户问题和相关物品回答问题" 
              },
              { role: "user", content: prompt }
            ],
            temperature: 0.7
          })
        });
    
        const data = await apiRes.json();
        res.writeHead(200, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify(data));
        return;
      }

      // 无效的模式参数
      res.writeHead(400, {
        'Content-Type': 'application/json',
        ...CORS_HEADERS
      });
      res.end(JSON.stringify({ error: '无效的 mode 参数' }));

    } catch (err) {
      console.error('处理请求时出错:', err);
      res.writeHead(500, {
        'Content-Type': 'application/json',
        ...CORS_HEADERS
      });
      
      res.end(JSON.stringify({
        error: '发生服务器内部错误',
        receivedMode: requestBody?.mode,
        message: err.message
      }));
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
