import { state } from '../core/state.js';
import { 
    generateSessionTitle, 
    editOrQueryImageWithGemini, 
    editOrQueryImageWithGemini_Multiple,
    getSmartSuggestions,
    generateImage
} from '../ai-services/skills-engine.js';
import {
    generateTextWithGemini
} from '../ai-services/gemini-client.js';
import { 
    fileToDataURL, 
    dataURLtoFileSync, 
    addWatermark,
    isMaterialRequest,
    isRemovalRequest,
    isImageGenerationRequest
} from '../core/utils.js';
import { 
    addImageToWorkbench
} from './workbench-core.js';
import { 
    buildMessageContentHTML, 
    appendSuggestionButtons,
    renderMessages,
    addMessage,
    buildBotFallbackText
} from './chat-panel.js';
import { renderHistoryList } from './sidebar.js';
import { performPreciseEdit } from './layer-manager.js';
import { openMaskEditor } from '../graphics/mask-drawer.js';
import { showCustomConfirm } from './modals.js';
import { dbHelper } from '../core/session.js';
import { implicitMemoryEngine } from '../runtime/ImplicitMemoryEngine';
import { hidePredictiveOverlay } from './predictive-prompt.js';

const { 
    sessions, 
    currentSessionId, 
    isSending, 
    updateSendBtnState,
    mainImageFile,
    lastGeneratedImageForEditing,
    pendingBaseImageShare,
    pendingReferenceImageShares,
    referenceImageFiles,
    maskDataUrl,
    lastGenerationContext,
    preciseEditMode,
    currentIntentLock,
    fileToWorkbenchIdMap,
    workbenchItems
} = state;

const userInput = document.getElementById('userInput');
const chatMessages = document.getElementById('chatMessages');

export const isImageEditRequest = (text, hasMask, hasRefImages) => {
    if (hasMask || hasRefImages) return true;
    if (!text) return false;
    const keywords = [
        '修改', '添加', '变成', '改成', '改为', '换成', '删除', '擦掉', '移除', '让它', 
        '把它', '增加', '画上', '调整', '编辑', '替换', '风格转换', '重绘', '扩图', 
        '角度', '方向', '朝向', '移动', '旋转', '翻转', '颜色', '色调', '亮度', '对比度', 
        '饱和度', '材质', '质感', '效果', '滤镜', '模糊', '锐化', '加上', '放入', 
        '去掉', '拿走', '改变', '变换', '变成', '变为'
    ];
    return keywords.some(keyword => text.toLowerCase().includes(keyword));
};

export async function handleRegenerate() {
    if (!state.lastGenerationContext || state.isSending) return;
    
    const { prompt, baseImage, referenceImages, mask, isGenTask, isEditTask } = state.lastGenerationContext;
    
    // Add user message for visual feedback
    const displayContent = "[ 指令: 重新生成 ]";
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if (currentSession) {
        const userMsg = { sender: 'user', type: 'text', content: displayContent };
        currentSession.updatedAt = Date.now();
        currentSession.messages.push(userMsg);
        addMessage(userMsg);
    }

    state.isSending = true;
    if (state.updateSendBtnState) state.updateSendBtnState();

    const placeholderDiv = document.createElement('div');
    placeholderDiv.className = 'message bot-message';
    const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
    
    const isMasklessMaterialEdit = isEditTask && !mask && isMaterialRequest(prompt);
    let placeholderText = isMasklessMaterialEdit ? "正在为您重新生成多个版本..." : "收到，正在为您重新生成...";
    
    let placeholderContent = `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">${placeholderText}</div></div>`;
    placeholderDiv.innerHTML = botAvatar + placeholderContent;
    state.chatMessages.appendChild(placeholderDiv); 
    state.chatMessages.scrollTop = state.chatMessages.scrollHeight;
    
    try {
        let result;
        if (isEditTask) {
            if (isMasklessMaterialEdit) result = await editOrQueryImageWithGemini_Multiple(prompt, baseImage, referenceImages, mask);
            else result = await editOrQueryImageWithGemini(prompt, baseImage, referenceImages, mask);
        } else if (isGenTask) {
            result = await generateImage(prompt);
        } else throw new Error("Regeneration is only available for image tasks.");
        
        if (result.success) {
            const botMessage = { sender: 'bot', type: 'bot-rich' };

            if (Array.isArray(result.imageData)) {
                botMessage.imageData = await Promise.all(result.imageData.map(async (img) => {
                    const originalSrc = `data:${img.mimeType};base64,${img.imageData}`;
                    return { src: await addWatermark(originalSrc), mimeType: img.mimeType };
                }));
                botMessage.content = result.text || '已为您生成多个版本，请选用一张以继续。';
                
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
                if (currentSession) {
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                }
                if (window.updateImagePreview) await window.updateImagePreview();

                botMessage.imageData.forEach(async (imgData) => {
                    try {
                        const file = dataURLtoFileSync(imgData.src, `regen-${Date.now()}.png`);
                        await addImageToWorkbench(file, 'AI生成', {
                            dataUrl: imgData.src,
                            parentId: baseImage ? fileToWorkbenchIdMap.get(baseImage) : null,
                            generationParams: { prompt: prompt }
                        });
                    } catch(e) {}
                });

                getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                state.lastGeneratedImageForEditing = null;

            } else if (result.imageData) {
                const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                botMessage.imageData = { src: watermarkedImgSrc };
                botMessage.content = result.text || buildBotFallbackText(prompt, {isEditTask, isGenTask});

                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
                if (currentSession) {
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                }
                if (window.updateImagePreview) await window.updateImagePreview();

                const newFile = dataURLtoFileSync(watermarkedImgSrc, `regen-${Date.now()}.png`);
                state.lastGeneratedImageForEditing = newFile;
                state.mainImageFile = newFile;

                let finalX = 50000, finalY = 50000;
                const baseId = fileToWorkbenchIdMap.get(baseImage);
                if (baseId) {
                    const baseItem = workbenchItems.get(baseId);
                    if (baseItem && baseItem.el) {
                        const baseW = parseFloat(baseItem.el.style.width) || 300;
                        finalX = parseFloat(baseItem.el.style.left) + (baseW * 0.35);
                        finalY = parseFloat(baseItem.el.style.top) + (baseW * 0.35);
                    }
                }

                await addImageToWorkbench(newFile, '重新生成', {
                    x: finalX, y: finalY,
                    dataUrl: watermarkedImgSrc,
                    parentId: baseId,
                    type: 'regenerate',
                    generationParams: { prompt: prompt }
                });

                getSmartSuggestions(result.imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
            }
        }
    } catch (error) {
        placeholderDiv.innerHTML = botAvatar + `<div>重新生成失败: ${error.message}</div>`;
    } finally {
        state.isSending = false; 
        if (state.updateSendBtnState) state.updateSendBtnState(); 
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (currentSession) await dbHelper.saveSession(currentSession); 
        renderHistoryList();
    }
}

export async function handleSend() {
    let textFromInput = state.userInput.value.trim();
    
    // Hide predictive autocomplete card on submission
    hidePredictiveOverlay();

    if (textFromInput) {
        implicitMemoryEngine.recordPromptStep(textFromInput, state.currentSessionId || 'default-session');
    }
    
    if (state.currentIntentLock === 'material') {
        if (!textFromInput) textFromInput = "替换材质";
        else if (!isMaterialRequest(textFromInput)) textFromInput = "材质替换为：" + textFromInput;
    } else if (state.currentIntentLock === 'erase') {
        if (!textFromInput) textFromInput = "移除选中区域";
        else if (!isRemovalRequest(textFromInput)) textFromInput = "局部重绘并移除：" + textFromInput;
    }
    state.currentIntentLock = null;

    const regenerationKeywords = ["不满意", "重来", "再试一次", "重新生成", "换一个", "另一个版本"];
    if (regenerationKeywords.some(k => textFromInput.toLowerCase().includes(k)) && state.lastGenerationContext) {
        handleRegenerate(); return;
    }
    
    if (state.preciseEditMode.pendingBox && state.preciseEditMode.pendingItemId && textFromInput) {
        const promptText = textFromInput.replace(/^修改框选区域：\s*/, '');
        if (promptText) {
            if (!state.currentSessionId) {
                state.currentSessionId = `session-${Date.now()}`;
                state.sessions.unshift({ id: state.currentSessionId, title: '精准修图', timestamp: Date.now(), messages: [] });
            }
            const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
            const userMsg = { sender: 'user', type: 'text', content: textFromInput };
            currentSession.messages.push(userMsg);
            renderMessages(currentSession.messages); renderHistoryList();
            state.userInput.value = ''; state.userInput.style.height = '';
            const box = state.preciseEditMode.pendingBox; const itemId = state.preciseEditMode.pendingItemId;
            state.preciseEditMode.pendingBox = null; state.preciseEditMode.pendingItemId = null;
            await performPreciseEdit(itemId, box, promptText); return;
        }
    }
    
    const effectiveBaseImage = state.mainImageFile || state.lastGeneratedImageForEditing;
    if (state.isSending || (!textFromInput && !effectiveBaseImage)) return;
    state.isSending = true; 
    if (state.updateSendBtnState) state.updateSendBtnState();

    let userMessages = [];
    if (textFromInput) userMessages.push({ sender: 'user', type: 'text', content: textFromInput });
    if (state.pendingBaseImageShare && state.mainImageFile) {
        userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(state.mainImageFile) } });
        state.pendingBaseImageShare = false;
    }
    for (const file of [...state.pendingReferenceImageShares]) {
        userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(file) } });
        state.pendingReferenceImageShares.delete(file);
    }
    
    if (!state.currentSessionId) {
        state.currentSessionId = `session-${Date.now()}`;
        const newSession = { id: state.currentSessionId, title: textFromInput.substring(0, 20) || '图片对话', timestamp: Date.now(), messages: [] };
        state.sessions.unshift(newSession);
        (async () => {
            const smartTitle = await generateSessionTitle(textFromInput);
            const s = state.sessions.find(s => s.id === state.currentSessionId);
            if (s) {
                if (!Array.isArray(s.messages)) {
                    s.messages = [];
                }
                s.title = smartTitle;
                s.updatedAt = Date.now();
                await dbHelper.saveSession(s);
                renderHistoryList();
            }
        })();
    }
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    currentSession.updatedAt = Date.now();
    currentSession.messages.push(...userMessages);
    renderMessages(currentSession.messages); renderHistoryList();
    state.userInput.value = ''; state.userInput.style.height = '';
    
    const referenceImagesSent = [...state.referenceImageFiles];
    const maskSent = state.maskDataUrl;
    const isGenTask = isImageGenerationRequest(textFromInput) && !effectiveBaseImage;
    const isEditTask = !!effectiveBaseImage && isImageEditRequest(textFromInput, !!maskSent, referenceImagesSent.length > 0);
    const isQueryTask = !!effectiveBaseImage && !isEditTask;

    state.lastGenerationContext = (isGenTask || isEditTask) ? { prompt: textFromInput, baseImage: effectiveBaseImage, referenceImages: [...referenceImagesSent], mask: maskSent, isGenTask, isEditTask } : null;

    let placeholderDiv = document.createElement('div');
    placeholderDiv.className = 'message bot-message';
    const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
    const isMasklessMaterialEdit = isEditTask && !maskSent && isMaterialRequest(textFromInput);
    
    let placeholderText = isGenTask || isEditTask ? "图片正在创建..." : "思考中...";
    if (isMasklessMaterialEdit) placeholderText = "正在为您生成多个版本...";

    let placeholderContent = (isGenTask || isEditTask)
        ? `<div class="image-placeholder-container"><div class=\"image-placeholder-box\"><div class=\"spinner\"></div></div><div class=\"placeholder-text\">${placeholderText}</div></div>`
        : `<div>${placeholderText}</div>`;
    placeholderDiv.innerHTML = botAvatar + placeholderContent;
    state.chatMessages.appendChild(placeholderDiv); state.chatMessages.scrollTop = state.chatMessages.scrollHeight;
    
    state.mainImageFile = null; state.referenceImageFiles = []; state.maskDataUrl = null;
    if (window.updateImagePreview) await window.updateImagePreview();

    try {
        let result;
        if (isGenTask) {
            result = await generateImage(textFromInput);
        } else if (isEditTask) {
            if (isMasklessMaterialEdit) result = await editOrQueryImageWithGemini_Multiple(textFromInput, effectiveBaseImage, referenceImagesSent, maskSent);
            else result = await editOrQueryImageWithGemini(textFromInput, effectiveBaseImage, referenceImagesSent, maskSent);
        } else if (isQueryTask) {
            result = await editOrQueryImageWithGemini(textFromInput, effectiveBaseImage, referenceImagesSent, maskSent);
        } else result = await generateTextWithGemini(textFromInput, state.currentSessionId, currentSession.messages);

        if (result.success) {
            const botMessage = { sender: 'bot', type: 'bot-rich' };
            
            if (Array.isArray(result.imageData)) {
                botMessage.imageData = await Promise.all(result.imageData.map(async (img) => {
                    const watermarkedSrc = await addWatermark(`data:${img.mimeType};base64,${img.imageData}`);
                    try {
                        const file = dataURLtoFileSync(watermarkedSrc, `gen-${Date.now()}.png`);
                        await addImageToWorkbench(file, 'AI生成', {
                            dataUrl: watermarkedSrc,
                            parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                            generationParams: { prompt: textFromInput }
                        });
                    } catch(e){}
                    return { src: watermarkedSrc, mimeType: img.mimeType };
                }));
                botMessage.content = result.text || '已为您生成多个版本，请选用一张以继续。';
                
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (window.updateImagePreview) await window.updateImagePreview();
                getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                state.lastGeneratedImageForEditing = null;
            } else if (result.imageData) {
                const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                botMessage.imageData = { src: watermarkedImgSrc };
                botMessage.content = result.text || buildBotFallbackText(textFromInput, {isEditTask, isGenTask});
                
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (window.updateImagePreview) await window.updateImagePreview();

                try {
                    state.lastGeneratedImageForEditing = dataURLtoFileSync(watermarkedImgSrc, `gen-${Date.now()}.png`);
                    state.mainImageFile = state.lastGeneratedImageForEditing;
                    await addImageToWorkbench(state.lastGeneratedImageForEditing, 'AI生成', {
                        dataUrl: watermarkedImgSrc,
                        parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                        generationParams: { prompt: textFromInput }
                    });
                } catch(e){}
                
                getSmartSuggestions(result.imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
            } else {
                botMessage.content = result.text;
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (!isQueryTask) state.lastGeneratedImageForEditing = null;
            }
        }
    } catch (error) {
        placeholderDiv.innerHTML = botAvatar + `<div>处理失败: ${error.message}</div>`;
    } finally {
        state.isSending = false; 
        if (state.updateSendBtnState) state.updateSendBtnState(); 
        await dbHelper.saveSession(currentSession); 
        renderHistoryList();
    }
}

export function handleSuggestionClick(prompt) {
    state.userInput.value = prompt;
    handleSend();
}

export function triggerAIActiveGuidance() {
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if (!currentSession || !Array.isArray(currentSession.messages) || currentSession.messages.length > 2) return;

    const messageDiv = addMessage({ sender: 'bot', content: '图片已上传。您可以直接输入指令进行编辑，或尝试以下操作：' });
    appendSuggestionButtons(messageDiv, [
        { label: '绘制蒙版并替换', action: () => { if(state.mainImageFile) openMaskEditor(state.mainImageFile) } },
        { label: '咨询这张图片', prompt: '详细描述这张图的内容' },
    ], handleSuggestionClick);
}
