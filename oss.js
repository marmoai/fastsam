const http = require('http');
const https = require('https');
const OSS = require('ali-oss');
const { formidable } = require('formidable');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');
const CUSTOM_DOMAIN = 'https://www.marmoai.cn'; 
const WINDOW_MS = 60 * 1000;
const RATE_LIMITS = {
  proxy: { limit: 120, scope: 'ip' },
  upload: { limit: 20, scope: 'user' },
  saveState: { limit: 30, scope: 'user' },
  saveAssets: { limit: 30, scope: 'user' },
  saveSessions: { limit: 30, scope: 'user' },
  saveSession: { limit: 120, scope: 'user' },
  saveSessionsManifest: { limit: 60, scope: 'user' }
};
const rateBuckets = new Map();

const client = new OSS({
  region: process.env.OSS_REGION,
  accessKeyId: process.env.OSS_ACCESS_KEY_ID,
  accessKeySecret: process.env.OSS_ACCESS_KEY_SECRET,
  bucket: process.env.OSS_BUCKET,
  secure: true, // 强制HTTPS
  timeout: 30000 // 30秒超时
});
const AUTHING_DOMAIN = process.env.AUTHING_DOMAIN || '';
const AUTHING_JWKS_URL = process.env.AUTHING_JWKS_URL || (AUTHING_DOMAIN ? `https://${AUTHING_DOMAIN}/oidc/.well-known/jwks.json` : '');
const JWKS_CACHE_TTL_MS = 10 * 60 * 1000;
let jwksCache = {
  fetchedAt: 0,
  keys: []
};
function validateSkuData(data) {
  const requiredFields = ['name', 'category', 'material'];
  for (const field of requiredFields) {
    if (!data[field]) throw new Error(`缺少必要字段: ${field}`);
  }
  
  
  
}

// 新增：生成SKU文件路径
function generateSkuPaths(skuId, category) {
  const baseDir = `images/sku-data/category_${category}`; 
  return {
    jsonPath: `${baseDir}/${skuId}.json` 
  };
}

// 新增：更新全局索引
async function updateSkuIndex(skuData) {
  const indexKey = 'images/sku-data/sku_index.json';
  let index = [];
  
  try {
    // 尝试读取现有索引
    const result = await client.get(indexKey);
    index = JSON.parse(result.content.toString());
  } catch (e) {
    console.log('首次创建索引文件');
  }
  
  // 添加新条目
  index.push({
    skuId: skuData.skuId,
    name: skuData.name,
    category: skuData.category,
    mainImage: skuData.mainImage,
    createdAt: new Date().toISOString()
  });
  
  // 写回OSS
  await client.put(indexKey, Buffer.from(JSON.stringify(index, null, 2)), {
    headers: { 'Content-Type': 'application/json' }
  });
}
async function updateProductDatabase(skuData) {
  const dbKey = 'images/sku-data/product_database.json';
  let database = {
    version: "2.0",
    products: []
  };
  
  try {
    // 尝试读取现有数据库
    const result = await client.get(dbKey);
    database = JSON.parse(result.content.toString());
  } catch (e) {
    console.log('首次创建产品数据库');
  }
  
  // 添加或更新产品信息
  const existingIndex = database.products.findIndex(p => p.skuId === skuData.skuId);
  const productData = {
    id: skuData.skuId.toLowerCase().replace(/_/g, '-'), // 转换SKU_xxx为sku-xxx格式
    name: skuData.name,
    search_keywords: skuData.search_keywords || [],
    category: skuData.category,
    material: skuData.material || {
      primary: '',
      secondary: '',
      full: ''
    },
    shape: skuData.shape || '',
    pricing: skuData.pricing || {
      price: ''
    },
    assets: {
      main_image: skuData.assets?.main_image || ''
    }
  };
  
  if (existingIndex >= 0) {
    // 更新现有产品
    database.products[existingIndex] = productData;
  } else {
    // 添加新产品
    database.products.push(productData);
  }
  
  // 写回OSS
  await client.put(dbKey, Buffer.from(JSON.stringify(database, null, 2)), {
    headers: { 'Content-Type': 'application/json' }
  });
}
// 环境变量验证
const requiredEnvVars = ['OSS_REGION', 'OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET', 'OSS_BUCKET', 'AUTHING_DOMAIN'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.error(`❌ 缺少必需环境变量: ${envVar}`);
    process.exit(1);
  }
}

function cleanupRateBucket(now = Date.now()) {
  for (const [key, bucket] of rateBuckets.entries()) {
    if (bucket.resetAt <= now) {
      rateBuckets.delete(key);
    }
  }
}

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded.trim()) {
    return forwarded.split(',')[0].trim();
  }
  return req.socket?.remoteAddress || 'unknown_ip';
}

function decodeJwtPayload(token) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const normalized = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = normalized + '='.repeat((4 - normalized.length % 4) % 4);
    return JSON.parse(Buffer.from(padded, 'base64').toString('utf8'));
  } catch (error) {
    return null;
  }
}

function decodeBase64Url(value) {
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized + '='.repeat((4 - normalized.length % 4) % 4);
  return Buffer.from(padded, 'base64');
}

function parseJwt(token) {
  const parts = token.split('.');
  if (parts.length !== 3) return null;

  try {
    const header = JSON.parse(decodeBase64Url(parts[0]).toString('utf8'));
    const payload = JSON.parse(decodeBase64Url(parts[1]).toString('utf8'));
    return {
      header,
      payload,
      signingInput: `${parts[0]}.${parts[1]}`,
      signature: decodeBase64Url(parts[2])
    };
  } catch (error) {
    return null;
  }
}

function httpGetJson(url) {
  return new Promise((resolve, reject) => {
    const transport = url.startsWith('https:') ? https : http;
    const request = transport.get(url, (response) => {
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Failed to fetch ${url}: HTTP ${response.statusCode}`));
        return;
      }

      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => {
        body += chunk;
      });
      response.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });

    request.on('error', reject);
    request.setTimeout(5000, () => {
      request.destroy(new Error(`Timeout fetching ${url}`));
    });
  });
}

async function getJwksKeys() {
  const now = Date.now();
  if (jwksCache.keys.length > 0 && (now - jwksCache.fetchedAt) < JWKS_CACHE_TTL_MS) {
    return jwksCache.keys;
  }

  const jwks = await httpGetJson(AUTHING_JWKS_URL);
  const keys = Array.isArray(jwks?.keys) ? jwks.keys : [];
  if (keys.length === 0) {
    throw new Error('JWKS 中未找到可用公钥');
  }

  jwksCache = {
    fetchedAt: now,
    keys
  };
  return keys;
}

function toBase64Url(buffer) {
  return buffer.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function createPemFromJwk(jwk) {
  if (!jwk || jwk.kty !== 'RSA' || !jwk.n || !jwk.e) {
    throw new Error('仅支持 RSA JWK');
  }
  const keyObject = crypto.createPublicKey({ key: jwk, format: 'jwk' });
  return keyObject.export({ type: 'spki', format: 'pem' });
}

async function verifyJwtToken(token) {
  const parsed = parseJwt(token);
  if (!parsed) {
    return null;
  }

  const { header, payload, signingInput, signature } = parsed;
  if (header.alg !== 'RS256' || !header.kid) {
    return null;
  }

  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < now) {
    return null;
  }
  if (payload.nbf && payload.nbf > now) {
    return null;
  }

  let keys = await getJwksKeys();
  let jwk = keys.find(item => item.kid === header.kid);

  if (!jwk) {
    jwksCache = { fetchedAt: 0, keys: [] };
    keys = await getJwksKeys();
    jwk = keys.find(item => item.kid === header.kid);
  }

  if (!jwk) {
    return null;
  }

  const pem = createPemFromJwk(jwk);
  const verifier = crypto.createVerify('RSA-SHA256');
  verifier.update(signingInput);
  verifier.end();

  const isValid = verifier.verify(pem, signature);
  if (!isValid) {
    return null;
  }

  const userId = payload.sub || payload.id || null;
  if (!userId) {
    return null;
  }

  return { token, payload, userId };
}

async function getAuthContext(req) {
  const authHeader = req.headers.authorization || '';
  if (!authHeader.startsWith('Bearer ')) return null;
  const token = authHeader.slice(7).trim();
  if (!token) return null;

  return verifyJwtToken(token);
}

async function requireAuth(req, res, headers, expectedUserId = null) {
  const auth = await getAuthContext(req);
  if (!auth) {
    res.writeHead(401, headers);
    res.end(JSON.stringify({ status: 'error', message: '未授权或登录已过期' }));
    return null;
  }

  if (expectedUserId && auth.userId !== expectedUserId) {
    res.writeHead(403, headers);
    res.end(JSON.stringify({ status: 'error', message: '用户身份不匹配' }));
    return null;
  }

  return auth;
}

function getUserRootDir(userId) {
  return `images/projects/${userId}`;
}

function getManifestKey(userId) {
  return `${getUserRootDir(userId)}/manifest.json`;
}

function getLegacySessionsKey(userId) {
  return `${getUserRootDir(userId)}/sessions.json`;
}

function getSessionDir(userId, sessionId) {
  return `${getUserRootDir(userId)}/sessions/${sessionId}`;
}

function getSessionDataKey(userId, sessionId) {
  return `${getSessionDir(userId, sessionId)}/session.json`;
}

async function readJsonFromOss(key, fallbackValue = null) {
  try {
    const result = await client.get(key);
    return JSON.parse(result.content.toString());
  } catch (error) {
    const isMissing =
      error.code === 'NoSuchKey' ||
      error.status === 404 ||
      (typeof error.message === 'string' && error.message.includes('Not Found'));

    if (isMissing) {
      return fallbackValue;
    }
    throw error;
  }
}

async function writeJsonToOss(key, value) {
  await client.put(
    key,
    Buffer.from(JSON.stringify(value, null, 2)),
    {
      headers: { 'Content-Type': 'application/json' }
    }
  );
}

function buildSessionManifestEntry(session) {
  return {
    id: session.id,
    title: session.title || '未命名项目',
    timestamp: session.timestamp || Date.now(),
    updatedAt: session.updatedAt || session.timestamp || Date.now(),
    isAutoRenamed: !!session.isAutoRenamed,
    deletedAt: session.deletedAt || null
  };
}

async function writeLegacySessionsPointer(userId, manifest) {
  await writeJsonToOss(getLegacySessionsKey(userId), {
    version: 2,
    storageMode: 'project_storage_v2',
    updatedAt: manifest.updatedAt || Date.now(),
    sessions: Array.isArray(manifest.sessions) ? manifest.sessions : []
  });
}

async function saveSessionsToProjectStorage(userId, sessions = []) {
  const manifest = {
    version: 2,
    updatedAt: Date.now(),
    sessions: []
  };

  for (const session of sessions) {
    if (!session?.id) continue;
    const entry = buildSessionManifestEntry(session);
    manifest.sessions.push(entry);
    await writeJsonToOss(getSessionDataKey(userId, session.id), session);
  }

  manifest.sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  await writeJsonToOss(getManifestKey(userId), manifest);
  await writeLegacySessionsPointer(userId, manifest);
  return manifest;
}

async function saveSessionToProjectStorage(userId, session) {
  if (!session?.id) {
    throw new Error('缺少 session.id');
  }

  await writeJsonToOss(getSessionDataKey(userId, session.id), session);

  const existingManifest = await readJsonFromOss(getManifestKey(userId), {
    version: 2,
    updatedAt: Date.now(),
    sessions: []
  });
  const manifestSessions = Array.isArray(existingManifest.sessions)
    ? existingManifest.sessions.filter(entry => entry?.id && entry.id !== session.id)
    : [];

  manifestSessions.push(buildSessionManifestEntry(session));
  manifestSessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  const manifest = {
    version: 2,
    updatedAt: Date.now(),
    sessions: manifestSessions
  };
  await writeJsonToOss(getManifestKey(userId), manifest);
  await writeLegacySessionsPointer(userId, manifest);
  return manifest;
}

async function saveSessionsManifestToProjectStorage(userId, entries = []) {
  const manifestSessions = entries
    .filter(entry => entry?.id)
    .map(buildSessionManifestEntry)
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  const manifest = {
    version: 2,
    updatedAt: Date.now(),
    sessions: manifestSessions
  };
  await writeJsonToOss(getManifestKey(userId), manifest);
  await writeLegacySessionsPointer(userId, manifest);
  return manifest;
}

async function loadSessionsFromProjectStorage(userId) {
  const manifest = await readJsonFromOss(getManifestKey(userId), null);
  if (!manifest || !Array.isArray(manifest.sessions)) {
    return null;
  }

  const sessions = [];
  for (const entry of manifest.sessions) {
    if (!entry?.id || entry.deletedAt) continue;
    const session = await readJsonFromOss(getSessionDataKey(userId, entry.id), null);
    if (!session) continue;
    sessions.push({
      ...session,
      title: entry.title || session.title,
      timestamp: entry.timestamp || session.timestamp,
      updatedAt: entry.updatedAt || session.updatedAt,
      isAutoRenamed: typeof entry.isAutoRenamed === 'boolean' ? entry.isAutoRenamed : !!session.isAutoRenamed
    });
  }

  sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  return sessions;
}

function enforceRateLimit(req, res, headers, routeKey, identity) {
  const policy = RATE_LIMITS[routeKey];
  if (!policy) return true;

  cleanupRateBucket();
  const subject = identity || 'anonymous';
  const key = `${routeKey}:${subject}`;
  const now = Date.now();
  let bucket = rateBuckets.get(key);

  if (!bucket || bucket.resetAt <= now) {
    bucket = { count: 0, resetAt: now + WINDOW_MS };
    rateBuckets.set(key, bucket);
  }

  bucket.count += 1;
  if (bucket.count > policy.limit) {
    const retryAfter = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000));
    res.writeHead(429, {
      ...headers,
      'Retry-After': String(retryAfter)
    });
    res.end(JSON.stringify({
      status: 'error',
      message: '请求过于频繁，请稍后再试',
      retryAfter
    }));
    return false;
  }

  return true;
}

const server = http.createServer(async (req, res) => {
  // 设置CORS头
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
    'Access-Control-Max-Age': '86400',
    'Content-Type': 'application/json'
  };

  try {
    // 处理预检请求
    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      return res.end();
    }

    // 新增：图片代理，解决前端 Canvas 跨域问题
    if (req.url.startsWith('/proxy') && req.method === 'GET') {
      try {
        if (!enforceRateLimit(req, res, headers, 'proxy', getClientIp(req))) {
          return;
        }

        const urlParams = new URL(req.url, `http://${req.headers.host || 'localhost'}`).searchParams;
        const targetUrl = urlParams.get('url');
        
        if (!targetUrl) {
          res.writeHead(400, headers);
          return res.end(JSON.stringify({ status: 'error', message: '缺少 url 参数' }));
        }

        console.log('🌐 代理请求图片:', targetUrl);
        
        // 优先尝试通过 OSS 客户端读取（如果是本 Bucket 的文件）
        let ossKey = '';
        if (targetUrl.includes('marmoai.cn/')) {
          ossKey = targetUrl.split('marmoai.cn/')[1].split('?')[0];
        } else if (targetUrl.includes('.aliyuncs.com/')) {
          const parts = targetUrl.split('.aliyuncs.com/');
          if (parts.length > 1) ossKey = parts[1].split('?')[0];
        }

        if (ossKey) {
          try {
            const result = await client.getStream(ossKey);
            res.writeHead(200, {
              ...headers,
              'Content-Type': result.res.headers['content-type'] || 'image/jpeg',
              'Cache-Control': 'public, max-age=86400'
            });
            result.stream.pipe(res);
            return;
          } catch (ossError) {
            console.warn('OSS 代理读取失败，尝试直接请求:', ossError.message);
          }
        }

        // 备选：直接通过 HTTP/HTTPS 请求
        const protocol = targetUrl.startsWith('https') ? https : http;
        const proxyReq = protocol.get(targetUrl, (proxyRes) => {
          const proxyHeaders = {
            ...headers,
            'Content-Type': proxyRes.headers['content-type'] || 'image/jpeg',
            'Cache-Control': 'public, max-age=86400'
          };
          
          res.writeHead(proxyRes.statusCode, proxyHeaders);
          proxyRes.pipe(res);
        });

        proxyReq.on('error', (e) => {
          console.error('代理请求失败:', e);
          res.writeHead(500, headers);
          res.end(JSON.stringify({ status: 'error', message: '代理请求失败' }));
        });

        return;
      } catch (error) {
        console.error('代理处理出错:', error);
        res.writeHead(500, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
        return;
      }
    }

    
    
// 在 server.createServer 的回调函数中添加以下代码
// 在 server.createServer 的回调函数中添加以下代码
if (req.url === '/designs' && req.method === 'GET') {
    try {
        console.log('尝试列出 OSS 设计文件...');
        const result = await client.list({
            prefix: 'images/design/',
            'max-keys': 100
        });
        
        console.log('OSS 设计文件返回:', result.objects?.length, '个文件');
        
        const designUrls = result.objects.map(obj => {
            return `https://www.marmoai.cn/${obj.name}`;
        });

        res.writeHead(200, {
            'Content-Type': 'application/json',
            ...headers
        });
        res.end(JSON.stringify({ 
            status: 'success', 
            designs: designUrls 
        }));
        
    } catch (error) {
        console.error('OSS 设计列表读取失败:', error);
        res.writeHead(500, {
            'Content-Type': 'application/json',
            ...headers
        });
        res.end(JSON.stringify({ 
            status: 'error',
            message: '无法读取设计文件',
            ...(process.env.NODE_ENV === 'development' && { debug: error.message })
        }));
    }
    return;
}

if (req.url === '/products' && req.method === 'GET') {
    try {
        console.log('尝试列出 OSS 产品文件...');
        const result = await client.list({
            prefix: 'images/products/',
            'max-keys': 100
        });
        
        console.log('OSS 产品文件返回:', result.objects?.length, '个文件');
        
        const productUrls = result.objects.map(obj => {
            return `https://www.marmoai.cn/${obj.name}`;
        });

        res.writeHead(200, {
            'Content-Type': 'application/json',
            ...headers
        });
        res.end(JSON.stringify({ 
            status: 'success', 
            products: productUrls 
        }));
        
    } catch (error) {
        console.error('OSS 产品列表读取失败:', error);
        res.writeHead(500, {
            'Content-Type': 'application/json',
            ...headers
        });
        res.end(JSON.stringify({ 
            status: 'error',
            message: '无法读取产品文件',
            ...(process.env.NODE_ENV === 'development' && { debug: error.message })
        }));
    }
    return;
}

    if (req.url === '/gallery' && req.method === 'GET') {
        try {
            console.log('尝试列出 OSS 图库文件...');
            const result = await client.list({
                prefix: 'images/gallery/',
                'max-keys': 100
            });
            
            console.log('OSS 图库文件返回:', result.objects?.length, '个文件');
            
            const galleryUrls = result.objects.map(obj => {
                return `https://www.marmoai.cn/${obj.name}`;
            });

            res.writeHead(200, {
                'Content-Type': 'application/json',
                ...headers
            });
            res.end(JSON.stringify({ 
                status: 'success', 
                gallery: galleryUrls 
            }));
            
        } catch (error) {
            console.error('OSS 图库列表读取失败:', error);
            res.writeHead(500, {
                'Content-Type': 'application/json',
                ...headers
            });
            res.end(JSON.stringify({ 
                status: 'error',
                message: '无法读取图库文件',
                ...(process.env.NODE_ENV === 'development' && { debug: error.message })
            }));
        }
        return;
    }

    // 新增：保存画板状态
    if (req.url === '/save-state' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, state } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!state) throw new Error('缺少 state 数据');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
        if (!enforceRateLimit(req, res, headers, 'saveState', auth.userId)) return;

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '状态保存请求已忽略，主存储现为 session runtimeWorkspace' }));
      } catch (error) {
        console.error('保存状态失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：获取画板状态
    if (req.url.startsWith('/get-state') && req.method === 'GET') {
      try {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const userId = url.searchParams.get('userId');
        if (!userId) throw new Error('缺少 userId');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', state: null, message: '状态已并入 session runtimeWorkspace' }));
      } catch (error) {
        console.error('获取状态失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 保存单个会话详情，避免一次性上传完整 sessions 数组触发 FC 请求体上限
    if (req.url === '/save-session' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, session } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!session) throw new Error('缺少 session 数据');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
        if (!enforceRateLimit(req, res, headers, 'saveSession', auth.userId)) return;

        await saveSessionToProjectStorage(userId, session);

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '会话保存成功', mode: 'project_storage_v2_single' }));
      } catch (error) {
        console.error('保存单个会话失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 保存轻量会话清单，用于同步删除/排序；不包含 messages/runtimeWorkspace 等重数据
    if (req.url === '/save-sessions-manifest' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, sessions } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!Array.isArray(sessions)) throw new Error('缺少 sessions 清单');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
        if (!enforceRateLimit(req, res, headers, 'saveSessionsManifest', auth.userId)) return;

        await saveSessionsManifestToProjectStorage(userId, sessions);

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '会话清单保存成功', mode: 'project_storage_v2_manifest' }));
      } catch (error) {
        console.error('保存会话清单失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：保存会话列表（兼容旧前端；新前端不再走这个大请求）
    if (req.url === '/save-sessions' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, sessions } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!sessions) throw new Error('缺少 sessions 数据');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
        if (!enforceRateLimit(req, res, headers, 'saveSessions', auth.userId)) return;

        await saveSessionsToProjectStorage(userId, sessions);

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '会话列表保存成功', mode: 'project_storage_v2' }));
      } catch (error) {
        console.error('保存会话失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：获取会话列表
    if (req.url.startsWith('/get-sessions') && req.method === 'GET') {
      try {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const userId = url.searchParams.get('userId');
        if (!userId) throw new Error('缺少 userId');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;

        let sessions = await loadSessionsFromProjectStorage(userId);
        if (!sessions) {
          const legacyData = await readJsonFromOss(getLegacySessionsKey(userId), { sessions: [] });
          sessions = Array.isArray(legacyData?.sessions) ? legacyData.sessions : [];
        }

        res.writeHead(200, headers);
        res.end(JSON.stringify({ 
          status: 'success', 
          sessions,
          mode: sessions.length > 0 ? 'project_storage_v2' : 'legacy_or_empty'
        }));
      } catch (error) {
        console.error('获取会话失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }


    // 新增：保存资产列表
    if (req.url === '/save-assets' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, assets } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!assets) throw new Error('缺少 assets 数据');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
        if (!enforceRateLimit(req, res, headers, 'saveAssets', auth.userId)) return;

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '资产列表写入已停用，资产以 session/runtimeWorkspace 为准' }));
      } catch (error) {
        console.error('保存资产失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：获取资产列表
    if (req.url.startsWith('/get-assets') && req.method === 'GET') {
      try {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const userId = url.searchParams.get('userId');
        if (!userId) throw new Error('缺少 userId');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', assets: [], message: '资产已并入 session/runtimeWorkspace' }));
      } catch (error) {
        console.error('获取资产失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：保存满意定稿规则（创意思路和满意参数）
    if (req.url === '/save-satisfaction-rules' && req.method === 'POST') {
      try {
        let body = '';
        for await (const chunk of req) {
          body += chunk;
        }
        const { userId, rules } = JSON.parse(body);
        if (!userId) throw new Error('缺少 userId');
        if (!rules) throw new Error('缺少 rules 数据');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;

        const ossPath = `images/projects/${userId}/satisfaction_rules.json`;
        await client.put(ossPath, Buffer.from(JSON.stringify({ rules })), {
          headers: { 'Content-Type': 'application/json' }
        });

        res.writeHead(200, headers);
        res.end(JSON.stringify({ status: 'success', message: '满意定稿规则保存成功' }));
      } catch (error) {
        console.error('保存满意定稿规则失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    // 新增：获取满意定稿规则
    if (req.url.startsWith('/get-satisfaction-rules') && req.method === 'GET') {
      try {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const userId = url.searchParams.get('userId');
        if (!userId) throw new Error('缺少 userId');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;

        const ossPath = `images/projects/${userId}/satisfaction_rules.json`;
        try {
          const result = await client.get(ossPath);
          const data = JSON.parse(result.content.toString());
          res.writeHead(200, headers);
          res.end(JSON.stringify({ 
            status: 'success', 
            rules: data.rules || [] 
          }));
        } catch (e) {
          const isMissingFile =
            e.code === 'NoSuchKey' ||
            e.status === 404 ||
            (typeof e.message === 'string' && (
              e.message.includes('Not Found') ||
              e.message.includes('Unexpected close tag')
            ));

          if (isMissingFile) {
            res.writeHead(200, headers);
            res.end(JSON.stringify({ status: 'success', rules: [], message: '未找到保存的满意定稿规则' }));
          } else {
            throw e;
          }
        }
      } catch (error) {
        console.error('获取满意定稿规则失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
      return;
    }

    if (req.url === '/upload-sku' && req.method === 'POST') {
      console.log('=== 收到SKU上传请求 ===');
      console.log('Headers:', req.headers);
      
      try {
        // 1. 解析请求数据
        const contentType = req.headers['content-type'] || '';
        let requestData;
    
        if (contentType.includes('multipart/form-data')) {
          const form = formidable({
            multiples: true,
            maxFileSize: 50 * 1024 * 1024,
            filter: ({ name }) => name === 'data'
          });
    
          const { fields } = await new Promise((resolve, reject) => {
            form.parse(req, (err, fields, files) => {
              if (err) reject(err);
              else resolve({ fields, files });
            });
          });
    
          requestData = JSON.parse(fields.data);
        } else if (contentType.includes('application/json')) {
          let body = '';
          for await (const chunk of req) {
            body += chunk;
          }
          requestData = JSON.parse(body);
        } else {
          throw new Error('不支持的Content-Type');
        }
    
        // 2. 兼容多种数据结构
        let skuItems;
        if (Array.isArray(requestData)) {
          skuItems = requestData;
        } else if (requestData.objects && Array.isArray(requestData.objects)) {
          skuItems = requestData.objects;
        } else if (requestData.name && requestData.category) {
          skuItems = [requestData];
        } else {
          throw new Error('无法识别的数据结构');
        }
    
        // 3. 处理每个SKU项
        const results = await Promise.all(skuItems.map(async (item) => {
          try {
            // 验证数据
            validateSkuData(item);
    
            // 生成SKU ID和路径
            const skuId = `SKU_${uuidv4().replace(/-/g, '').substring(0, 12)}`;
            const paths = generateSkuPaths(skuId, item.category);
    
            // 转换图片URL为自定义域名（如果存在）
            if (item.assets?.main_image) {
              item.assets.main_image = item.assets.main_image.replace(
                /https:\/\/[^\/]+/,
                'https://www.marmoai.cn'
              );
            }
    
            // 创建要存储的数据对象（排除bbox和confidence）
            const { bbox, confidence, ...skuToStore } = item;
            skuToStore.skuId = skuId;
            skuToStore.createdAt = new Date().toISOString();
    
            // 上传JSON到OSS
            await client.put(
              paths.jsonPath,
              Buffer.from(JSON.stringify(skuToStore, null, 2)),
              {
                headers: { 
                  'Content-Type': 'application/json'       
                }
              }
            );
    
            // 更新索引
            await updateSkuIndex({
              skuId,
              name: item.name,
              category: item.category,
              createdAt: new Date().toISOString(),
              mainImage: skuToStore.assets?.main_image || '' // 添加自定义域名的图片URL
            });
            await updateProductDatabase(skuToStore);
            return {
              status: 'success',
              data: {
                skuId,
                name: item.name,
                category: item.category,
                jsonUrl: `https://www.marmoai.cn/${paths.jsonPath}` // 使用自定义域名
              }
            };
          } catch (error) {
            return {
              status: 'error',
              message: error.message,
              code: error.code || 'VALIDATION_ERROR',
              sku: item.name || '未知SKU'
            };
          }
        }));
    
        // 4. 返回响应
        const hasErrors = results.some(result => result.status === 'error');
        res.writeHead(hasErrors ? 207 : 200, headers);
        res.end(JSON.stringify({
          status: hasErrors ? 'partial_success' : 'success',
          results: results,
          count: {
            total: results.length,
            success: results.filter(r => r.status === 'success').length,
            failed: results.filter(r => r.status === 'error').length
          }
        }));
    
      } catch (error) {
        console.error('SKU上传失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({
          status: 'error',
          message: error.message,
          code: error.code || 'INVALID_REQUEST'
        }));
      }
      return;
    }
    
    
    if (req.url === '/upload-image' && req.method === 'POST') {
      try {
        const form = formidable({
          multiples: false,
          maxFileSize: 50 * 1024 * 1024, // 50MB
          filter: ({ mimetype }) => mimetype && mimetype.startsWith('image/')
        });
    
        const { files } = await new Promise((resolve, reject) => {
          form.parse(req, (err, fields, files) => {
            if (err) reject(err);
            else resolve({ fields, files });
          });
        });
    
        const file = Array.isArray(files.image) ? files.image[0] : files.image;
        if (!file) throw new Error('未收到图片文件');
    
        // 生成OSS文件名
        const extname = path.extname(file.originalFilename || '') || '.jpg';
        const ossFilename = `images/uploads/${Date.now()}_${Math.random().toString(36).slice(2, 8)}${extname}`;
    
        // 上传到OSS
        const result = await client.put(ossFilename, fs.createReadStream(file.filepath), {
          headers: {
            'Content-Type': file.mimetype,
            'Content-Disposition': 'inline'
          }
        });
    
        res.writeHead(200, headers);
        res.end(JSON.stringify({
          status: 'success',
          url: `${CUSTOM_DOMAIN}/${ossFilename}` 
        }));
    
      } catch (error) {
        console.error('图片上传失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({
          status: 'error',
          message: error.message
        }));
      }
      return;
    }
     

    
    
    if (req.url === '/upload-avatar' && req.method === 'POST') {
      try {
        const form = formidable({
          multiples: false,
          maxFileSize: 2 * 1024 * 1024,
          keepExtensions: true,
          allowEmptyFiles: false,
          filter: ({ mimetype }) => ['image/jpeg', 'image/png'].includes(mimetype)
        });
    
        const { fields, files } = await new Promise((resolve, reject) => {
          form.parse(req, (err, fields, files) => {
            if (err) reject(err);
            else resolve({ fields, files });
          });
        });
    
        const userId = Array.isArray(fields.userId) ? fields.userId[0] : fields.userId;
        if (!userId) throw new Error('缺少 userId');
        const auth = await requireAuth(req, res, headers, userId);
        if (!auth) return;
    
        console.log('📦 收到文件对象:', files);
    
        let file = files.file;
        if (Array.isArray(file)) {
          file = file[0]; // 取第一个文件
        }
    
        if (!file) {
          const debugInfo = { files, fields };
          throw new Error('未收到头像文件（debug: ' + JSON.stringify(debugInfo) + '）');
        }
    
        if (!file.filepath || typeof file.filepath !== 'string') {
          const debugInfo = { file, filepathType: typeof file.filepath, fields };
          throw new Error('文件路径无效，无法读取头像文件（debug: ' + JSON.stringify(debugInfo) + '）');
        }
    
        const stream = fs.createReadStream(file.filepath);
        const ossFilename = `images/avatars/${userId}.jpg`;
    
        await client.putStream(ossFilename, stream, {
          headers: {
            'Content-Disposition': 'inline',
            'x-oss-forbid-overwrite': 'false'
          }
        });
    
        res.writeHead(200, headers);
        res.end(JSON.stringify({
          status: 'success',
          data: {
            avatarUrl: `https://www.marmoai.cn/${ossFilename}?t=${Date.now()}`
          }
        }));
      } catch (error) {
        console.error('❌ 头像上传失败:', error);
        res.writeHead(400, headers);
        res.end(JSON.stringify({
          status: 'error',
          message: error.message
        }));
      }
      return;
    }
    
    
    // 只接受POST请求
    if (req.method !== 'POST') {
      res.writeHead(405, headers);
      return res.end(JSON.stringify({ 
        status: 'error',
        message: '仅支持POST请求' 
      }));
    }

    // 解析表单数据
    const form = formidable({
      multiples: false,      // 单文件上传
      keepExtensions: true,  // 保留文件扩展名
      maxFileSize: 100 * 1024 * 1024, // 100MB限制
      filter: ({ mimetype }) => {
        // 文件类型白名单
        const allowedTypes = [
          'image/jpeg', 
          'image/png',
          'image/gif',
          'image/webp'
        ];
        return allowedTypes.includes(mimetype);
      }
    });

    const { fields, files } = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        else resolve({ fields, files });
      });
    });

    const userId = Array.isArray(fields.userId) ? fields.userId[0] : fields.userId;
    const sessionId = Array.isArray(fields.sessionId) ? fields.sessionId[0] : fields.sessionId;
    if (!userId) {
      throw new Error('缺少 userId');
    }
    const auth = await requireAuth(req, res, headers, userId);
    if (!auth) return;
    if (!enforceRateLimit(req, res, headers, 'upload', auth.userId)) return;

    // 获取文件对象（兼容formidable v1和v2+）
    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file || !file.filepath) {
      throw new Error('未收到有效文件');
    }

    console.log('📤 收到上传请求:', {
      filename: file.originalFilename,
      size: (file.size / 1024 / 1024).toFixed(2) + 'MB',
      type: file.mimetype
    });

    // 初始化OSS客户端
   

    // 生成安全的OSS文件名
    const extname = path.extname(file.originalFilename || '') || '.jpg';
    const projectDir = sessionId ? getSessionDir(userId, sessionId) : getUserRootDir(userId);
    const ossFilename = `${projectDir}/assets/${Date.now()}_${Math.random().toString(36).slice(2)}${extname.toLowerCase()}`;

    // 上传到OSS
    console.log('⬆️ 正在上传到OSS:', ossFilename);
    const ossResult = await client.put(ossFilename, file.filepath, {
      headers: {
        'Content-Disposition': 'inline'
      }
    });

    console.log('✅ 上传成功:', {
      url: `https://www.marmoai.cn/${ossFilename}`,
      size: ossResult.res.size,
      status: ossResult.res.status
    });

    // 返回标准响应
    res.writeHead(200, headers);
    res.end(JSON.stringify({
      status: 'success',
      data: {
        url: `https://www.marmoai.cn/${ossFilename}`, // 强制使用自定义域名
        filename: ossFilename,
        size: file.size,
        mimeType: file.mimetype,
        ossMeta: {
          bucket: ossResult.bucket,
          etag: ossResult.etag,
          requestId: ossResult.requestId
        }
      }
    }));

  } catch (error) {
    console.error('❌ 处理请求出错:', {
      error: error.message,
      stack: error.stack,
      code: error.code,
      time: new Date().toISOString()
    });

    const statusCode = error.code === 'AccessDenied' ? 403 : 
                      error.code === 'FileSizeLimitExceeded' ? 413 : 
                      (error.message.includes('文件类型') || error.message.includes('未收到有效文件')) ? 400 : 500;

    res.writeHead(statusCode, headers);
    res.end(JSON.stringify({
      status: 'error',
      message: getFriendlyErrorMessage(error),
      code: error.code || 'SERVER_ERROR',
      ...(process.env.NODE_ENV !== 'production' && { debug: error.message })
    }));
  }
});

// 友好的错误消息映射
function getFriendlyErrorMessage(error) {
  if (error.code === 'AccessDenied') return 'OSS访问被拒绝，请检查权限配置';
  if (error.code === 'FileSizeLimitExceeded') return '文件大小不能超过100MB';
  if (error.message.includes('文件类型')) return '仅支持JPEG/PNG/GIF/WEBP图片格式';
  return '文件上传服务暂时不可用';
}

// 启动服务器
const PORT = process.env.PORT || 9000;
server.listen(PORT, () => {
  console.log(`
  🚀 服务器已启动:
  - 地址: http://localhost:${PORT}
  - OSS配置: 
    Bucket: ${process.env.OSS_BUCKET}
    Region: ${process.env.OSS_REGION}
  - 上传端点: POST /upload
  `);
});

// 处理未捕获的异常
process.on('uncaughtException', (err) => {
  console.error('⚠️ 未捕获异常:', err);
});

process.on('unhandledRejection', (reason) => {
  console.error('⚠️ 未处理的Promise拒绝:', reason);
});
