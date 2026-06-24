const AUTH_BACKEND_URL = 'https://auth-id-gsggenqbpo.cn-hangzhou.fcapp.run';

// 获取Authing配置（从后端获取）
let AUTHING_CONFIG = {
    domain: '',
    appId: '',
    redirectUri: window.location.origin + '/art'
};

// 初始化Authing配置
async function initAuthingConfig() {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(`${AUTH_BACKEND_URL}/auth-config`, { signal: controller.signal });
        clearTimeout(timeout);
        if (!response.ok) throw new Error('获取配置失败');
        const config = await response.json();
        AUTHING_CONFIG = {
            ...AUTHING_CONFIG,
            domain: config.domain,
            appId: config.appId
        };
    } catch (error) {
        console.error('加载Authing配置失败:', error);
        throw error;
    }
}
  const API_ENDPOINTS = {
    CREDITS_API: "https://get-credits-ydzngaecdu.cn-hangzhou.fcapp.run", // 替换为你的实际地址
    GENERATE_API: "https://generate-image-khthgyzuec.cn-hangzhou.fcapp.run" // 替换为你的实际地址
  };
  
  // 共享状态对象
  const authState = {
    isLoggedIn: false,
    userInfo: null,
    checkInterval: null
  };
  
  // 防止重复执行的锁
  let isUpdatingUI = false;
  
  // 初始化认证系统
  export async function initAuth() {
    try {
        // 初始化配置
        await initAuthingConfig();
        
        // 处理回调
        const callbackHandled = await handleAuthingCallback();
        
        // 检查登录状态
        await checkLoginStatus();
        
        // 设置状态同步和受保护链接
        setupStateSync();
        setupProtectedLinks();

        // 如果是登录状态，尝试加载点数
        if (authState.isLoggedIn && authState.userInfo?.sub) {
            await updateCreditCounter(authState.userInfo.sub);
        }
  
    } catch (error) {
      console.error('认证初始化失败:', error);
      // 初始化失败时，如果已经登录过，也尝试加载点数（可能 token 过期但 localStorage 有旧用户 id）
      // 虽然不太完美，但可以尝试用 localStorage 里的 userId 加载点数
       const fallbackUserId = localStorage.getItem('last_known_userId');
       if(fallbackUserId) {
           await updateCreditCounter(fallbackUserId);
       }
    }
  }
  
  // 处理Authing回调
  async function handleAuthingCallback() {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');

    if (code) {
        try {
            const response = await fetch(`${AUTH_BACKEND_URL}/auth-config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    code, 
                    redirectUri: AUTHING_CONFIG.redirectUri 
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error_description || '认证失败');
            }

            const tokenData = await response.json();
            localStorage.setItem('authing_token', tokenData.access_token);
            if (tokenData.id_token) {
                localStorage.setItem('authing_id_token', tokenData.id_token);
            }

            // 清除URL中的code参数
            window.history.replaceState({}, document.title, window.location.pathname);
            return true;
        } catch (error) {
            console.error('认证回调处理失败:', error);
            localStorage.removeItem('authing_token');
            localStorage.removeItem('authing_id_token');
            return false;
        }
    }
    return false;
}
  
  // 辅助函数：安全地解密 JWT 密文 payload
  function decodeJWT(token) {
    try {
      if (!token) return null;
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      const payloadStr = decodeURIComponent(escape(atob(parts[1])));
      return JSON.parse(payloadStr);
    } catch (e) {
      console.error('解密 Authing JWT Token 失败:', e);
      return null;
    }
  }

  // 获取用户信息
  async function fetchUserInfo() {
      const token = localStorage.getItem('authing_token');
      if (!token) return null;

      // 1. 优先尝试提取本地存储或 JWT 包里的数据，以此作为瞬间就绪的 Fallback
      let fallbackUser = null;
      try {
          const cachedUserInfo = localStorage.getItem('userinfo');
          if (cachedUserInfo) {
              fallbackUser = JSON.parse(cachedUserInfo);
          }
      } catch(e) {}

      if (!fallbackUser) {
          const idToken = localStorage.getItem('authing_id_token') || token;
          const decoded = decodeJWT(idToken);
          if (decoded) {
              fallbackUser = {
                  sub: decoded.sub || decoded.id,
                  nickname: decoded.nickname || decoded.name || decoded.username || '已登录用户',
                  username: decoded.username || decoded.name,
                  picture: decoded.picture || decoded.avatar
              };
          }
      }

      try {
        if (!AUTHING_CONFIG.domain) {
            console.warn('OIDC 账号后端未就绪，使用 JWT 状态离线解密恢复');
            if (fallbackUser && fallbackUser.sub) {
                localStorage.setItem('last_known_userId', fallbackUser.sub);
            }
            return fallbackUser;
        }
        const response = await fetch(`https://${AUTHING_CONFIG.domain}/oidc/me`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/json'
          }
        });
  
        if (!response.ok) {
             // 如果 token 无效或过期，清除 token
             if (response.status === 401) {
                 console.warn('Token 无效或已过期，清除本地 token');
                 localStorage.removeItem('authing_token');
                 localStorage.removeItem('authing_id_token');
             }
             throw new Error('获取用户信息失败');
        }
  
        const user = await response.json();
  
        // 正常存入本地，并缓存 userinfo
        user.nickname = user.nickname || user.username || '用户';
        if (user.sub) {
            localStorage.setItem('last_known_userId', user.sub);
            localStorage.setItem('userinfo', JSON.stringify(user));
        }
        return user;
      } catch (error) {
        console.warn('向 OIDC 服务器同步状态失败，使用不离线本地登录载体替代：', error);
        if (fallbackUser && fallbackUser.sub) {
            localStorage.setItem('last_known_userId', fallbackUser.sub);
            localStorage.setItem('userinfo', JSON.stringify(fallbackUser));
        }
        return fallbackUser;
      }
    }
  
  // 检查登录状态
  async function checkLoginStatus() {
    const token = localStorage.getItem('authing_token');
    authState.isLoggedIn = !!token;
  
    if (authState.isLoggedIn) {
      try {
        authState.userInfo = await fetchUserInfo();
        if (!authState.userInfo) {
           authState.isLoggedIn = false;
        }
      } catch (error) {
         console.warn('读取用户信息捕获非毁灭性错误，尝试使用本地快照进行不掉线填充:', error);
         const lastKnownId = localStorage.getItem('last_known_userId');
         if (lastKnownId) {
             authState.isLoggedIn = true;
             let cachedObj = { sub: lastKnownId, nickname: '已登录用户' };
             try {
                 const localUser = localStorage.getItem('userinfo');
                 if (localUser) cachedObj = JSON.parse(localUser);
             } catch(e) {}
             authState.userInfo = cachedObj;
         } else {
             authState.isLoggedIn = false;
         }
      }
    } else {
      authState.userInfo = null;
    }
  
    updateUI(); // 更新 UI
    return authState.isLoggedIn;
  }
  
  // 验证Token (JWT 客户端验证，非必须，fetchUserInfo 会做更严格的验证)
  function validateToken(token) {
    try {
      if (!token) return false;
      const parts = token.split('.');
      if (parts.length !== 3) return false;
  
      const payload = JSON.parse(atob(parts[1]));
      return payload.exp > Date.now() / 1000;
    } catch {
      return false;
    }
  }
  
  
  // 更新UI
  function updateUI() {
    if (isUpdatingUI) return;
    isUpdatingUI = true;
  
    const container = document.getElementById('authContainer') || document.getElementById('authButton');
    if (!container) {
      isUpdatingUI = false;
      return;
    }
    let creditCounter = document.getElementById('creditCounter');
    // 如果 creditCounter 不存在，动态创建它并插入到 nav-menu 的前面
    if (!creditCounter) {
      const navMenu = document.querySelector('.nav-menu');
      if (navMenu) {
        navMenu.insertAdjacentHTML('beforeend', `
          <div class="credit-counter" id="creditCounter" style="display:none">
            <i class="fas fa-bolt"></i>
            <span id="remainingCredits">--</span>
          </div>
        `);
        creditCounter = document.getElementById('creditCounter');
      }
    }
  
    const userId = authState.userInfo?.sub;
    const displayName = authState.userInfo?.nickname || authState.userInfo?.username || '用户';
  
    // 统一使用 www.marmoai.cn 的地址作为基础（虽然这里的头像 URL 看起来是固定的）
    // 更好的做法是 Authing 返回头像 URL 或通过其 API 获取
    const avatarBaseUrl = 'https://www.marmoai.cn/images/avatars/';
    const avatarUrl = userId
      ? `${avatarBaseUrl}${userId}.jpg?t=${Date.now()}`
      : `${avatarBaseUrl}default-avatar.png`;
  
  
    if (authState.isLoggedIn) {
       // 更新 authContainer
       container.innerHTML = `
         <div class="user-menu">
          <img src="${avatarUrl}"
               class="user-avatar"
               onerror="this.src='${avatarBaseUrl}default-avatar.png'">
          <span>${displayName}</span>
          <div class="user-dropdown">
            <a href="/user-profile">个人中心</a>
            <a href="#" onclick="logout()">退出登录</a>
          </div>
        </div>
      `;
      // 如果 creditCounter 存在且用户已登录，显示并更新点数
      if (creditCounter) {
        creditCounter.style.display = 'flex';
        // **确保这里调用 updateCreditCounter 时传入 userId**
        if(userId) {
           updateCreditCounter(userId).catch(console.error);
        } else {
           // 理论上 isLoggedIn 为 true 时 userId 应该存在，但作为兜底
           console.warn("用户已登录但无法获取 userId，无法更新点数");
           creditCounter.style.display = 'none'; // 无法获取 userId 则隐藏点数
        }
      }
    } else {
      // 未登录状态，更新 authContainer
      container.innerHTML = `
        <a href="#" class="authing-login-btn" onclick="auth.redirectToAuthing()">
          <i class="fas fa-user"></i>
          <span>登录注册</span>
        </a>
      `;
      // 未登录状态，隐藏点数
      if (creditCounter) {
          creditCounter.style.display = 'none';
          // 可选：清空点数显示
          const remainingCreditsSpan = document.getElementById('remainingCredits');
          if(remainingCreditsSpan) remainingCreditsSpan.textContent = '--';
      }
    }
  
    // 将事件监听器移到模块顶层（已在你提供的代码中）
    // window.addEventListener('avatarUpdated', ...);
  
  
    // 更新发布菜单状态 (保持原样)
    const publishLinks = document.querySelectorAll('.nav-link[href="#"]');
    publishLinks.forEach(link => {
      if (authState.isLoggedIn) {
        // 如果是实际需要保护的链接，应该判断其特定 class 或 data 属性
        // 这里示例只是隐藏 href="#" 的占位链接
        if(link.getAttribute('href') === '#') {
             link.style.display = 'inline-block'; // 或者 'flex' 根据你的布局
        }
      } else {
         if(link.getAttribute('href') === '#') {
            link.style.display = 'none';
         }
      }
    });
  
    isUpdatingUI = false;
  }
  
  
  // 异步函数 updateCreditCounter，接受 userId 参数
  async function updateCreditCounter(userId) {
    if (!userId) {
        console.warn("updateCreditCounter 需要 userId");
        return;
    }
    // 使用 API_ENDPOINTS.CREDITS_API 作为基础 URL
    const API_URL = `${API_ENDPOINTS.CREDITS_API}/credits?userId=${encodeURIComponent(userId)}`; // **这里加上 userId 参数**
  
    const counter = document.getElementById('remainingCredits');
    const creditCounterDiv = document.getElementById('creditCounter'); // 获取包含点数和图标的 div
  
    try {
      // 先从本地存储获取上次记录的点数（作为备用值）
      const localStorageKey = `credits_${userId}`; // 每个用户单独存储本地点数
      const fallbackCredits = parseInt(localStorage.getItem(localStorageKey)) || 0;
      if (counter) counter.textContent = fallbackCredits;
      if (creditCounterDiv) creditCounterDiv.classList.toggle('low', fallbackCredits < 5);
  
  
      // 尝试从后端获取最新点数
      const response = await fetch(API_URL, {
        method: 'GET', // 明确是 GET 方法
        headers: {
          // 'Authorization': `Bearer ${localStorage.getItem('authing_token')}`, // GET 请求获取点数一般不需要 token，除非你的后端要求
          'Accept': 'application/json' // 指定接收 JSON
        }
      });
  
      if (!response.ok) {
           // 如果不是 400 (缺少 userId), 打印详细错误
           if (response.status !== 400) {
              const errorText = await response.text();
              console.warn(`信用点数 API HTTP错误 ${response.status}:`, errorText);
           } else {
               console.warn(`信用点数 API 缺少 userId 错误 (预期在 auth.js 中已解决): ${response.status}`);
           }
          throw new Error(`获取点数失败，HTTP状态: ${response.status}`);
      }
  
      const data = await response.json();
  
      // 确保 data.credits 是有效数字
      const credits = typeof data?.credits === 'number' ? data.credits : fallbackCredits;
  
      // 更新UI和本地存储
      if (counter) {
        counter.textContent = credits;
      }
      if (creditCounterDiv) {
          creditCounterDiv.classList.toggle('low', credits < 5);
      }
      localStorage.setItem(localStorageKey, credits.toString()); // 按用户存储本地点数
  
  
    } catch (error) {
      console.warn('信用点数更新失败 (fetch 错误或 JSON 解析错误):', error);
      // 保持使用本地存储的备用值，并显示错误提示
       if (counter) {
          const localStorageKey = `credits_${userId}`;
          counter.textContent = localStorage.getItem(localStorageKey) || '--'; // 使用用户对应的本地存储
          if (counter.textContent === '--') showTemporaryTooltip('点数加载失败');
          else showTemporaryTooltip(`点数加载失败，显示上次记录 (${counter.textContent})`);
        } else {
           showTemporaryTooltip('点数加载失败');
        }
        if (creditCounterDiv) {
           const fallbackCredits = parseInt(localStorage.getItem(`credits_${userId}`)) || 0; // 使用用户对应的本地存储
           creditCounterDiv.classList.toggle('low', fallbackCredits < 5);
        }
    }
  }
  
  
  // 辅助函数：显示临时提示 (已在你提供的代码中)
  function showTemporaryTooltip(message) {
    const tooltip = document.createElement('div');
    tooltip.className = 'credit-tooltip'; // 确保你的 CSS 有 .credit-tooltip 样式
    tooltip.textContent = message;
    // 避免重复添加
    if (!document.querySelector('.credit-tooltip')) {
       document.body.appendChild(tooltip);
       setTimeout(() => {
         tooltip.classList.add('fade-out'); // 确保你的 CSS 有 .fade-out 动画
         setTimeout(() => tooltip.remove(), 1000);
       }, 3000);
    }
  }
  
  
  // 状态同步 (保持原样，增加对 userId 变化的处理)
  function setupStateSync() {
    let lastKnownUserId = authState.userInfo?.sub;
  
    window.addEventListener('storage', async (e) => {
      if (e.key === 'authing_token') {
        await checkLoginStatus();
        // 如果登录状态变化或用户变化，重新加载点数
        const currentUserId = authState.userInfo?.sub;
        if (currentUserId && currentUserId !== lastKnownUserId) {
            lastKnownUserId = currentUserId;
            await updateCreditCounter(currentUserId);
        } else if (!currentUserId && lastKnownUserId) {
            // 用户登出
            lastKnownUserId = null;
            // 隐藏点数显示等
             const creditCounterDiv = document.getElementById('creditCounter');
             if(creditCounterDiv) creditCounterDiv.style.display = 'none';
             const remainingCreditsSpan = document.getElementById('remainingCredits');
             if(remainingCreditsSpan) remainingCreditsSpan.textContent = '--';
        }
      }
    });
  
    // 定时检查登录状态（已在你提供的代码中）
    // authState.checkInterval = setInterval(() => { ... });
    // window.addEventListener('beforeunload', () => { ... });
  }
  
  // 保护链接处理 (保持原样)
  function setupProtectedLinks() {
    document.addEventListener('click', function(e) {
      // 检查是否是受保护的链接
      const protectedLink = e.target.closest('.protected-link'); // 使用特定的 class 来标识受保护链接
      // 检查是否是发布菜单的链接 (假设发布菜单下的链接都需要登录)
      const isPublishDropdownLink = e.target.closest('#publishDropdown .dropdown-menu a');
  
  
      if ((protectedLink || isPublishDropdownLink) && !authState.isLoggedIn) {
        e.preventDefault();
        const confirmLogin = window.confirm('需要登录后才能使用该功能，是否立即登录？');
        if (confirmLogin) redirectToAuthing();
      }
    });
  }
  
  
  // 登录跳转 (保持原样)
  export async function redirectToAuthing() {
    try {
        await initAuthingConfig();
        const params = new URLSearchParams({
            app_id: AUTHING_CONFIG.appId,
            redirect_uri: AUTHING_CONFIG.redirectUri,
            response_type: 'code',
            scope: 'openid profile'
        });
        window.location.href = `https://${AUTHING_CONFIG.domain}/login?${params}`;
    } catch (error) {
        console.error('登录跳转失败:', error);
        alert('认证服务不可用，请稍后重试');
    }
}
  
  // 退出登录 (保持原样)
  window.logout = function() {
    console.log('logout 函数被调用！', new Date().toISOString()); // 添加这行日志
    localStorage.removeItem('authing_token');
    localStorage.removeItem('authing_id_token');
    localStorage.removeItem('last_known_userId'); // 退出登录时清除备用 userId
    // 清除当前用户的本地点数缓存（可选，因为下次登录会用新的 key）
    // const currentUserId = authState.userInfo?.sub;
    // if(currentUserId) localStorage.removeItem(`credits_${currentUserId}`);
  
  
    authState.isLoggedIn = false;
    authState.userInfo = null;
    updateUI();
    // 触发 storage 事件，通知其他页面（如果你的应用是多tab/多窗口的）
    window.dispatchEvent(new Event('storage'));
  };
  
  // 全局暴露
  if (typeof window !== 'undefined') {
    window.auth = {
      initAuth, // 暴露 initAuth 供其他脚本调用
      checkLoginStatus,
      redirectToAuthing,
      logout,
      getState: () => authState // 暴露 getState 方法获取当前认证状态
    };
  }
  
  // 考虑在 DOMContentLoaded 后立即初始化
  document.addEventListener('DOMContentLoaded', () => {
      // 检查 window.auth 是否已经被其他方式初始化过
      // 如果没有，则调用 initAuth
      if (typeof window !== 'undefined') {
        window.auth = {
          initAuth, // 暴露 initAuth 供其他脚本调用
          checkLoginStatus,
          redirectToAuthing,
          logout,
          getState: () => authState, // 暴露 getState 方法获取当前认证状态
          // **新增：暴露 API_ENDPOINTS 和 updateCreditCounter**
          API_ENDPOINTS: API_ENDPOINTS, // 将内部常量暴露出去
          updateCreditCounter: updateCreditCounter // 将内部函数暴露出去
        };
      }
  });