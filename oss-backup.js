const http = require('http');
const OSS = require('ali-oss');
const { formidable } = require('formidable');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const CUSTOM_DOMAIN = 'https://www.marmoai.cn'; 

const client = new OSS({
  region: process.env.OSS_REGION,
  accessKeyId: process.env.OSS_ACCESS_KEY_ID,
  accessKeySecret: process.env.OSS_ACCESS_KEY_SECRET,
  bucket: process.env.OSS_BUCKET,
  secure: true, // 强制HTTPS
  timeout: 30000 // 30秒超时
});
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
const requiredEnvVars = ['OSS_REGION', 'OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET', 'OSS_BUCKET'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.error(`❌ 缺少必需环境变量: ${envVar}`);
    process.exit(1);
  }
}

const server = http.createServer(async (req, res) => {
  // 设置CORS头
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Content-Type': 'application/json'
  };

  try {
    // 处理预检请求
    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      return res.end();
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
    
        if (!fields.userId) throw new Error('缺少 userId');
    
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
        const ossFilename = `images/avatars/${fields.userId}.jpg`;
    
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
    const userId = fields.userId || 'unknown'; // 从表单获取userId
    const ossFilename = `images/projects/${userId}/${Date.now()}_${Math.random().toString(36).slice(2)}${extname.toLowerCase()}`;

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
                      error.message.includes('文件类型') ? 415 : 500;

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