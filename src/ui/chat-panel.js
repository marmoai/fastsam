import { marked } from "marked";
import { getProxiedUrl } from "../core/utils.js";
import { state } from "../core/state.js";
import { updateChatSourceRail } from "./chat-source-rail.js";

function persistMessageIfNeeded(message) {
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
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;

    persistMessageIfNeeded(message);

    const { sender } = message;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
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
