import { marked } from "marked";
import { getProxiedUrl } from "../core/utils.js";
import { state } from "../core/state.js";
import { updateChatSourceRail } from "./chat-source-rail.js";

function persistMessageIfNeeded(message) {
    if (message?.persist === false || message?.visibility === 'progress' || message?.visibility === 'internal') return;
    if (window.__isRenderingChatHistory) return;
    if (!state.currentSessionId) return;

    const currentSession = state.sessions.find(session => session.id === state.currentSessionId);
    if (!currentSession) return;

    if (!Array.isArray(currentSession.messages)) {
        currentSession.messages = [];
    }

    // Skip if this exact message object is already part of the session history.
    if (currentSession.messages.includes(message)) return;

    currentSession.messages.push(message);
    currentSession.updatedAt = Date.now();

    if (typeof window.renderHistoryList === 'function') {
        window.renderHistoryList();
    }

    if (window.dbHelper?.saveSession) {
        window.dbHelper.saveSession(currentSession).catch((error) => {
            console.error('Failed to persist chat message:', error);
        });
    }
}

export function buildMessageContentHTML(message) {
    let html = '';
    
    // Support html type for agent debate
    if (message.type === 'html' && message.content) {
        return message.content;
    }
    
    if (message.content) {
        html += `<div class="message-text">${marked.parse(message.content)}</div>`;
    }

    if (message.imageData) {
        if (Array.isArray(message.imageData)) {
            html += '<div class="image-grid-container">';
            message.imageData.forEach((imgData, index) => {
                if (!imgData.src) return;
                const proxiedSrc = getProxiedUrl(imgData.src);
                if (!proxiedSrc) return;
                html += `
                    <div class="grid-item">
                        <img src="${proxiedSrc}" class="image-message generated-image" alt="Generated variation ${index + 1}" crossorigin="anonymous">
                        <button class="select-image-btn" data-src="${imgData.src}" data-mime="${imgData.mimeType}">
                            选用此图
                        </button>
                    </div>
                `;
            });
            html += '</div>';
        } else if (message.imageData.src) {
            const proxiedSrc = getProxiedUrl(message.imageData.src);
            if (proxiedSrc) {
                html += `<div class="message-image"><img src="${proxiedSrc}" class="image-message generated-image" alt="Generated or uploaded content" crossorigin="anonymous"></div>`;
            }
        }
    } else if (message.type === 'image' && message.imageData?.src) {
         const proxiedSrc = getProxiedUrl(message.imageData.src);
         if (proxiedSrc) {
             html = `<div class="message-image"><img src="${proxiedSrc}" class="image-message generated-image" alt="Generated or uploaded content" crossorigin="anonymous"></div>`;
         }
    }

    return `<div class="message-content">${html || '<div></div>'}</div>`;
}

if (typeof window !== 'undefined') {
    window.buildMessageContentHTML = buildMessageContentHTML;
}

export function addMessage(message) {
    // Internal Workbench diagnostics update state elsewhere and never belong in chat UI.
    if (message?.visibility === 'internal') return null;

    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;

    persistMessageIfNeeded(message);

    const { sender } = message;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    if (message.visibility) messageDiv.dataset.visibility = message.visibility;
    if (message.persist === false) messageDiv.dataset.persist = 'false';
    
    const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
    const htmlContent = buildMessageContentHTML(message);
    
    messageDiv.innerHTML = sender === 'bot' ? botAvatar + htmlContent : htmlContent;
    
    // Add update method for dynamic content updates
    messageDiv.update = (newContent) => {
        message.content = newContent;
        const newHtmlContent = buildMessageContentHTML(message);
        messageDiv.innerHTML = sender === 'bot' ? botAvatar + newHtmlContent : newHtmlContent;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Update minimap
    updateChatSourceRail();
    
    return messageDiv;
}

export function yieldToChatPaint() {
    return new Promise(resolve => {
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(() => requestAnimationFrame(resolve));
        } else {
            setTimeout(resolve, 0);
        }
    });
}

export function renderMessages(messages, onActionClick) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    chatMessages.innerHTML = '';
    window.__isRenderingChatHistory = true;
    try {
        if (!messages || messages.length === 0) {
            renderWelcomeGuide(chatMessages);
        } else {
            messages.forEach(msg => addMessage(msg));
        }
    } finally {
        window.__isRenderingChatHistory = false;
    }
    
    // Update minimap
    updateChatSourceRail();
}

/**
 * 渲染极简对话流风格的欢迎语 (Minimalist Chat)
 */
export function renderWelcomeGuide(container) {
    const welcomeText = `你好！我是你的 AI 创意助手 小M。✨

我可以帮你生成精美的图片、修改设计细节，或者陪你聊聊创意灵感。

试试对我说：‘帮我画一个赛博朋克风格的猫’。`;

    addMessage({ sender: 'bot', type: 'text', content: welcomeText });
}

export function appendSuggestionButtons(messageElement, suggestions = [], onSuggestionClick) {
    const container = document.createElement('div');
    container.className = 'suggestion-container';
    
    const defaultSuggestions = [
        { label: '添加戏剧性光影', prompt: '为这张图添加更富戏剧性的光影效果' },
        { label: '转为电影风格', prompt: '将这张图的风格转换为电影质感' },
        { label: '让它更生动', prompt: '让这张图的色彩更生动鲜艳' }
    ];

    const suggestionsToRender = (suggestions && suggestions.length > 0) ? suggestions : defaultSuggestions;

    suggestionsToRender.forEach(suggestion => {
        const button = document.createElement('button');
        button.className = 'suggestion-btn';
        button.textContent = `[ ${suggestion.label} ]`;
        button.onclick = () => {
            if (suggestion.action) {
                suggestion.action();
            } else if (onSuggestionClick) {
                onSuggestionClick(suggestion.prompt);
            }
        };
        container.appendChild(button);
    });
    
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.appendChild(container);
    } else {
        messageElement.appendChild(container);
    }
}

function escapeAgentJobText(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

const AGENT_JOB_STATUS_LABELS = {
    created: '已创建',
    planned: '已规划',
    running: '执行中',
    verifying: '验证中',
    waiting_confirmation: '等待确认 / 处理验证',
    committing: '保存中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
};

function getAgentStepState(job, index) {
    if (job.status === 'failed' && index === job.currentStep) return 'failed';
    if (job.status === 'cancelled' && index === job.currentStep) return 'cancelled';
    if (job.status === 'completed' || index < job.currentStep) return 'completed';
    if (index === job.currentStep && ['running', 'verifying', 'waiting_confirmation', 'committing'].includes(job.status)) return 'active';
    return 'pending';
}

function buildAgentJobCardHTML(job) {
    const planSteps = Array.isArray(job?.plan?.steps) ? job.plan.steps : [];
    const steps = planSteps.map((step, index) => {
        const state = getAgentStepState(job, index);
        const icon = state === 'completed' ? '✓' : state === 'failed' ? '!' : state === 'active' ? '●' : '○';
        return `<li class="agent-job-step is-${state}"><span class="agent-job-step-icon">${icon}</span><span>${escapeAgentJobText(step.label || step.commandType)}</span></li>`;
    }).join('');
    const error = job.error?.message
        ? `<div class="agent-job-error">${escapeAgentJobText(job.error.message)}</div>`
        : '';
    const failedChecks = job.verification?.checks?.filter(check => !check.passed).slice(0, 3) || [];
    const verification = job.verification
        ? `<div class="agent-job-verification ${job.verification.passed ? 'is-passed' : 'is-failed'}">${job.verification.passed ? '✓ 结构与视觉入口验证通过' : '⚠ 验证未通过，结果已保留待处理'}${failedChecks.length ? `<div class="agent-job-checks">${failedChecks.map(check => escapeAgentJobText(check.message)).join('<br>')}</div>` : ''}</div>`
        : '';
    const confirmButton = job.status === 'waiting_confirmation' && job.verification?.passed !== false
        ? '<button type="button" class="agent-job-action agent-job-confirm">确认保存</button>'
        : '';
    const cancelButton = ['created', 'planned', 'running', 'verifying', 'waiting_confirmation'].includes(job.status)
        ? '<button type="button" class="agent-job-action agent-job-cancel">取消任务</button>'
        : '';
    const retryButton = job.status === 'failed' || (job.status === 'waiting_confirmation' && job.verification?.passed === false)
        ? '<button type="button" class="agent-job-action agent-job-retry">重试验证</button>'
        : '';
    return `
        <div class="agent-job-card" data-job-id="${escapeAgentJobText(job.id)}">
            <div class="agent-job-header">
                <div><span class="agent-job-kicker">AGENT JOB</span><div class="agent-job-goal">${escapeAgentJobText(job.goal)}</div></div>
                <span class="agent-job-status is-${escapeAgentJobText(job.status)}">${escapeAgentJobText(AGENT_JOB_STATUS_LABELS[job.status] || job.status)}</span>
            </div>
            <ol class="agent-job-steps">${steps}</ol>
            ${verification}
            ${error}
            <div class="agent-job-actions">${confirmButton}${retryButton}${cancelButton}</div>
        </div>
    `;
}

/**
 * Render a transient Job card. Job state is deliberately not written into the
 * conversation history; the Runtime remains the source of truth for recovery.
 */
export function addAgentJobCard(job, handlers = {}) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message agent-job-message';
    const botAvatar = '<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">';
    messageDiv.innerHTML = botAvatar + buildAgentJobCardHTML(job);

    const controller = {
        element: messageDiv,
        update(nextJob) {
            messageDiv.innerHTML = botAvatar + buildAgentJobCardHTML(nextJob);
            const confirm = messageDiv.querySelector('.agent-job-confirm');
            const cancel = messageDiv.querySelector('.agent-job-cancel');
            const retry = messageDiv.querySelector('.agent-job-retry');
            const bind = (button, handler) => {
                if (!button || !handler) return;
                button.addEventListener('click', async () => {
                    button.disabled = true;
                    try {
                        await handler(nextJob);
                    } catch (error) {
                        if (handlers.onError) handlers.onError(error, nextJob);
                        button.disabled = false;
                    }
                });
            };
            bind(confirm, handlers.onConfirm);
            bind(cancel, handlers.onCancel);
            bind(retry, handlers.onRetry);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    };
    chatMessages.appendChild(messageDiv);
    controller.update(job);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    updateChatSourceRail();
    return controller;
}

export function focusAgentJobCard(jobId) {
    if (!jobId) return false;
    const escapedId = typeof CSS !== 'undefined' && typeof CSS.escape === 'function'
        ? CSS.escape(String(jobId))
        : String(jobId).replace(/(["\\])/g, '\\$1');
    const card = document.querySelector(`.agent-job-card[data-job-id="${escapedId}"]`);
    if (!card) return false;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.remove('is-focused');
    void card.offsetWidth;
    card.classList.add('is-focused');
    window.setTimeout(() => card.classList.remove('is-focused'), 1800);
    return true;
}

if (typeof window !== 'undefined') {
    window.addEventListener('marmo:agent-task-focus', event => {
        focusAgentJobCard(event.detail?.jobId);
    });
}

export function buildBotFallbackText(userInstruction, { isEditTask, isGenTask }) {
    if (isEditTask) return '图片已处理完成。';
    if (isGenTask) return '图像已生成。';
    return '已完成您的请求。';
}

export function renderAgentDebateInChat(critiques, itemId) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const containerId = `debate-${Date.now()}`;
    let html = `
        <div class="agent-debate-container" id="${containerId}">
            <div class="agent-debate-header">
                <i class="fas fa-users"></i> 专家评审团建议
            </div>
            <div class="agent-debate-list">
    `;

    critiques.forEach(c => {
        const escapedSuggestion = c.suggestion.replace(/'/g, "\\'").replace(/"/g, '&quot;').replace(/\n/g, ' ');
        html += `
            <div class="agent-msg-item" onclick="window.applyAgentSuggestion('${escapedSuggestion}', '${itemId}')">
                <div class="agent-avatar" style="background: ${c.agent.color}">
                    <i class="fas ${c.agent.icon}"></i>
                </div>
                <div class="agent-content">
                    <div class="agent-name" style="color: ${c.agent.color}">
                        ${c.agent.name}
                    </div>
                    <div class="agent-text">"${c.critique}"</div>
                    <div class="agent-action">
                        <i class="fas fa-magic"></i> 点击采纳修改
                    </div>
                </div>
            </div>
        `;
    });

    html += `</div></div>`;
    
    return html;
}
