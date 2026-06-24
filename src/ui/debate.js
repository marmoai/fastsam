import { state } from '../core/state.js';
import { addMessage, renderAgentDebateInChat } from './chat-panel.js';
import { fileToBase64 } from '../core/utils.js';
import { AGENTS } from '../ai-services/agents.js';
import { analyzeWithAgent } from '../ai-services/skills-engine.js';

export async function triggerAgentDebate(itemId) {
    const item = state.workbenchItems.get(itemId);
    if (!item) return;

    // UI Feedback
    addMessage({ sender: 'bot', type: 'text', content: '👥 正在召集专家团队（色彩、构图、甲方）对图片进行评审，请稍候...' });

    try {
        const base64Data = await fileToBase64(item.file || item.dataUrl);
        
        // Create promises for parallel execution
        const promises = Object.values(AGENTS).map(agent => 
            analyzeWithAgent(agent, base64Data, item.file?.type || 'image/png')
        );

        const results = await Promise.all(promises);
        const validResults = results.filter(r => r !== null);

        if (item) {
            item.critiquesData = validResults;
            state.workbenchItems.set(itemId, item); // 确保更新到 Map 中
        }

        // Render in Chat
        const debateHtml = renderAgentDebateInChat(validResults, itemId);
        
        if (window.addWorkbenchActionToChat) {
            await window.addWorkbenchActionToChat('专家评审团辩论', '对图片进行了多角度的评审', null, null, debateHtml);
        } else {
            addMessage({ sender: 'bot', type: 'html', content: debateHtml });
        }

        // Render Sticky Notes on Workbench
        renderStickyNotesOnWorkbench(itemId, validResults);

    } catch (e) {
        console.error("Agent debate failed", e);
        addMessage({ sender: 'bot', type: 'text', content: '❌ 评审团连接失败，请重试。' });
    }
}

export function renderStickyNotesOnWorkbench(itemId, critiques) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.el) return;

    // 移除旧的便签
    item.el.querySelectorAll('.sticky-note').forEach(n => n.remove());

    // 使用索引来垂直堆叠便签
    let noteIndex = 0;

    critiques.forEach(c => {
        const note = document.createElement('div');
        note.className = 'sticky-note';
        
        // --- 尺寸与初始定位 ---
        note.style.width = '150px';
        note.style.padding = '8px';
        note.style.fontSize = '11px';
        note.style.borderTopWidth = '12px';
        note.style.left = '102%'; 
        
        const desiredVisualOffsetY = 40; 
        const verticalOffset = noteIndex * (desiredVisualOffsetY / state.workbenchZoom); 
        note.style.top = `${verticalOffset}px`;

        const inverseScale = 1 / state.workbenchZoom;
        note.style.transformOrigin = `top left`;
        note.style.transform = `scale(${inverseScale}) rotate(1deg)`;

        // 确保事件触发时，总是使用全局最新的 workbenchZoom 值
        note.addEventListener('mouseenter', () => {
            const currentInverseScale = 1 / state.workbenchZoom; // 读取实时缩放值
            note.style.transform = `scale(${1.05 * currentInverseScale}) rotate(0deg)`;
            note.style.zIndex = '150';
        });
        note.addEventListener('mouseleave', () => {
            const currentInverseScale = 1 / state.workbenchZoom; // 读取实时缩放值
            note.style.transform = `scale(${currentInverseScale}) rotate(1deg)`; // 保持旋转角度一致
            note.style.zIndex = '100';
        });

        note.style.borderTopColor = c.agent.color + '40';

        note.innerHTML = `
            <div class="sticky-header" style="color:${c.agent.color}">
                <div class="sticky-icon" style="background:${c.agent.color}">
                    <i class="fas ${c.agent.icon}"></i>
                </div>
                ${c.agent.name}
            </div>
            <div>${c.critique}</div>
            <button class="sticky-action-btn">采纳建议</button>
        `;

        const btn = note.querySelector('button');
        if (btn) {
            btn.onclick = (e) => {
                e.stopPropagation();
                if (typeof window.applyAgentSuggestion === 'function') {
                    window.applyAgentSuggestion(c.suggestion, itemId); 
                }
                note.remove(); 
            };
        }
        
        note.onclick = (e) => e.stopPropagation();

        item.el.appendChild(note);
        noteIndex++;
    });
}

export function applyAgentSuggestion(prompt, itemId) {
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    if (!userInput) return;

    // 1. 填充文字
    userInput.value = prompt;

    // 2. 如果提供了图片ID，自动将图片设为当前编辑对象
    if (itemId) {
        const item = state.workbenchItems.get(itemId);
        if (item && (item.file || item.dataUrl)) {
            try {
                state.mainImageFile = item.file || item.dataUrl;
                state.lastGeneratedImageForEditing = item.file || item.dataUrl;
                
                // 重置其他状态
                state.isContextPreviewHidden = false;
                state.maskDataUrl = null;
                state.referenceImageFiles = []; // 清空参考图，专注于单图修改
                state.pendingBaseImageShare = true;
                state.pendingReferenceImageShares.clear();
                
                // 刷新输入框上方的预览区
                if (typeof window.updateImagePreview === 'function') {
                    window.updateImagePreview();
                }
            } catch (e) {
                console.error("Auto-select image failed:", e);
            }
        }
    }

    // 3. 更新按钮状态并自动发送
    if (typeof window.updateSendBtnState === 'function') {
        window.updateSendBtnState();
    }
    
    // 只有当发送按钮可用时（意味着有图或有字），才触发发送
    if (sendBtn && !sendBtn.disabled) {
        // 给一个小延迟，让UI先渲染预览图，体验更好
        setTimeout(() => {
            sendBtn.click();
        }, 100);
    }
}

// Expose to window for inline onclick
window.applyAgentSuggestion = applyAgentSuggestion;

