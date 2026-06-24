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

      // 新增：处理 GPT-image-2 图片生成中转接口 (Text-to-Image)
      if (requestBody.mode === 'image_generation' || requestBody.mode === 'gpt-image-2-generation') {
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

        const apiRes = await fetch('https://api.quickrouter.ai/v1/images/generations', {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${QUICKROUTER_API_TOKEN}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(apiPayload)
        });

        const responseData = await apiRes.json();
        console.log('📦 GPT-image-2 生成返回层响应：', responseData);

        res.writeHead(apiRes.status, {
          'Content-Type': 'application/json',
          ...CORS_HEADERS
        });
        res.end(JSON.stringify(responseData));
        return;
      }

      // 新增：处理 GPT-image-2 图片编辑中转接口
      if (requestBody.mode === 'image_edit' || requestBody.mode === 'gpt-image-2') {
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

        // 手动构造 multipart/form-data 缓冲区块
        const parts = [];
        for (const [key, value] of Object.entries(fields)) {
          if (value === undefined || value === null) continue;

          parts.push(Buffer.from(`--${boundary}\r\n`));
          if (value && typeof value === 'object' && value.buffer) {
            // 文件部分
            parts.push(Buffer.from(`Content-Disposition: form-data; name="${key}"; filename="${value.filename}"\r\n`));
            parts.push(Buffer.from(`Content-Type: ${value.contentType}\r\n\r\n`));
            parts.push(value.buffer);
          } else {
            // 普通文本字段
            parts.push(Buffer.from(`Content-Disposition: form-data; name="${key}"\r\n\r\n`));
            parts.push(Buffer.from(String(value)));
          }
          parts.push(Buffer.from('\r\n'));
        }
        parts.push(Buffer.from(`--${boundary}--\r\n`));

        const multipartBody = Buffer.concat(parts);

        // 使用 fetch 转发请求，极大地提升容错并规避 https.request 在特定容器下面死锁和超时的问题
        try {
          const apiRes = await fetch('https://api.quickrouter.ai/v1/images/edits', {
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
          console.log(`📦 GPT-image-2 返回层状态: ${apiRes.status}, 响应:`, resBody);
          
          res.writeHead(apiRes.status, {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
          });
          res.end(resBody);
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
        requestBody.mode === 'image_edit_cleanplate_test' ||
        requestBody.mode === 'clean_plate_test_edit' ||
        requestBody.mode === 'occlusion_completion_test_edit'
      ) {
        console.log('🎬 正在转发测试图片编辑请求到 SiliconFlow Qwen Image Edit...');

        if (!requestBody.prompt || !requestBody.image) {
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

        if (!primaryImage) {
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
          num_inference_steps: Number(requestBody.num_inference_steps || requestBody.steps || 20),
          cfg: Number(requestBody.cfg || 4),
          image: primaryImage
        };

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

          console.log(`📦 SiliconFlow Image Edit 返回状态: ${apiRes.status}`);

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

        const modelUrl = `https://api.quickrouter.ai/v1beta/models/${requestBody.model}:generateContent`;

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
