import { state } from '../core/state.js';
import { 
    generateSessionTitle, 
    editOrQueryImageWithGemini, 
    editOrQueryImageWithGemini_Multiple,
    generateImage,
    getSmartSuggestions
} from '../ai-services/skills-engine.js';
import { 
    fileToDataURL, 
    dataURLtoFileSync, 
    addWatermark,
    getExplicitRequestedImageCount,
    isMaterialRequest,
    isRemovalRequest
} from '../core/utils.js';
import { determineConversationIntent, executeConversationIntent, composeExecutionPrompt, resolveExecutionPrompt, mergeRegeneratePrompt, getIntentPlaceholderText, shouldUseImagePlaceholder } from '../ai-services/intent-router.js';
import { interpretConversationTurn } from '../ai-services/conversation-interpreter.js';
import { resolveStateConflict } from '../ai-services/state-conflict-resolver.js';
import { createDefaultWorkingMemory, ensureSessionWorkingMemory, mergeInterpreterResultIntoSession, finalizeWorkingMemoryWithAssistantResult } from '../ai-services/session-working-memory.js';
import { 
    addImageToWorkbench
} from './workbench-core.js';
import { 
    buildMessageContentHTML, 
    appendSuggestionButtons,
    renderMessages,
    addMessage,
    yieldToChatPaint,
    buildBotFallbackText
} from './chat-panel.js';
import { renderHistoryList } from './sidebar.js';
import { performPreciseEdit } from './layer-manager.js';
import { openMaskEditor } from '../graphics/mask-drawer.js';
import { showCustomConfirm } from './modals.js';
import { dbHelper, setLastActiveSessionId } from '../core/session.js';
import { implicitMemoryEngine } from '../runtime/ImplicitMemoryEngine';
import { hidePredictiveOverlay } from './predictive-prompt.js';
import { getAssistantWorkspaceContext } from '../services/workspace-context.js';
import { applyWorkspaceCommand, buildWorkspaceCommandResponse, detectWorkspaceCommand } from '../services/workspace-intents.js';
import { executeLensSearch, looksLikeLensSearchRequest } from '../services/capability-orchestrator.js';
import { getImageModel } from '../ai-services/gemini-client.js';
import { shouldUseQwenLocalImageRouting as shouldUseQwenLocalImageRoutingShared } from '../ai-services/qwen-local-routing.js';

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
const GENERIC_SESSION_TITLES = new Set(['新建项目', '图片对话', '新对话', 'Untitled', 'New Project', 'New Chat', '开启新对话', '精准修图']);

function shouldUseQwenLocalImageRouting(text, baseImage, mask) {
    return shouldUseQwenLocalImageRoutingShared({
        model: getImageModel(),
        text,
        baseImage,
        mask
    });
}

function buildRenameContextFromMessages(messages = []) {
    const actionKeywords = ['正在', '完成', '生成', '提取', '拆解', '融合', '扩图', '重光照', '重绘', '编辑', '处理'];
    const meaningfulMsgs = messages.filter(msg => {
        if (msg?.type !== 'text') return false;
        const text = String(msg.content || '').trim();
        if (!text || text.includes('[ 已选用图片 ]') || text.includes('图片已上传')) return false;
        if (msg.sender === 'user') return true;
        return msg.sender === 'bot' && actionKeywords.some(keyword => text.includes(keyword));
    });

    return meaningfulMsgs
        .slice(-3)
        .map(msg => `${msg.sender === 'user' ? '用户' : '系统'}: ${msg.content}`)
        .join('\n');
}

async function autoRenameSessionIfNeeded(session) {
    if (!session) return;
    const contextText = buildRenameContextFromMessages(session.messages || []);
    if (!contextText || session.lastRenameContext === contextText) return;

    try {
        const smartTitle = await generateSessionTitle(contextText);
        if (!smartTitle) return;
        session.title = smartTitle;
        session.lastRenameContext = contextText;
        session.updatedAt = Date.now();
        session.isAutoRenamed = true;
        await dbHelper.saveSession(session);
        renderHistoryList();
    } catch (error) {
        console.error('autoRenameSessionIfNeeded failed:', error);
    }
}

export async function handleRegenerate() {
    if (!state.lastGenerationContext || state.isSending) return;
    
    const actualInput = state.userInput.value.trim();
    const { prompt, corePrompt, baseImage, referenceImages, mask, isGenTask, isEditTask } = state.lastGenerationContext;
    
    // Add user message for visual feedback
    const displayContent = actualInput || "[ 指令: 重新生成 ]";
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if (currentSession) {
        const userMsg = { sender: 'user', type: 'text', content: displayContent };
        currentSession.updatedAt = Date.now();
        currentSession.messages.push(userMsg);
        addMessage(userMsg);
    }

    state.userInput.value = '';
    state.userInput.style.height = '';

    state.isSending = true;
    if (state.updateSendBtnState) state.updateSendBtnState();

    const placeholderDiv = document.createElement('div');
    placeholderDiv.className = 'message bot-message';
    const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
    
    const requestedImageCount = getExplicitRequestedImageCount(corePrompt || prompt, 6);
    let placeholderText = requestedImageCount > 1 ? "正在为您生成多个版本..." : "收到，正在为您重新生成...";
    
    let placeholderContent = `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">${placeholderText}</div></div>`;
    placeholderDiv.innerHTML = botAvatar + placeholderContent;
    state.chatMessages.appendChild(placeholderDiv); 
    state.chatMessages.scrollTop = state.chatMessages.scrollHeight;
    
    try {
        const retryCorePrompt = mergeRegeneratePrompt(corePrompt || prompt, actualInput);
        const retryPrompt = composeExecutionPrompt(
            { route: isEditTask ? 'image_edit' : 'image_generation' },
            {
                prompt: retryCorePrompt,
                baseImage,
                referenceImages,
                mask,
                history: currentSession?.messages || [],
                workingMemory: currentSession?.workingMemory || null,
                workspaceContext: getAssistantWorkspaceContext(state)
            }
        );
        let result;
        if (isEditTask) {
            if (requestedImageCount > 1) result = await editOrQueryImageWithGemini_Multiple(retryPrompt, baseImage, referenceImages, mask, null, requestedImageCount);
            else result = await editOrQueryImageWithGemini(retryPrompt, baseImage, referenceImages, mask);
        } else if (isGenTask) {
            result = await generateImage(retryPrompt, '1:1', {
                baseImage,
                referenceImages,
                history: currentSession?.messages || [],
                imageCount: requestedImageCount
            });
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
                            generationParams: { prompt: retryPrompt }
                        });
                    } catch(e) {}
                });

                getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                state.lastGeneratedImageForEditing = null;

            } else if (result.imageData) {
                const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                botMessage.imageData = { src: watermarkedImgSrc };
                botMessage.content = result.text || buildBotFallbackText(retryPrompt, {isEditTask, isGenTask});

                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
                if (currentSession) {
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                }
                if (window.updateImagePreview) await window.updateImagePreview();

                const newFile = dataURLtoFileSync(watermarkedImgSrc, `regen-${Date.now()}.png`);
                // 不再自动把 AI 结果设为下一轮对话底图，避免后续纯文本请求被误路由到图片链路
                state.lastGeneratedImageForEditing = null;
                state.mainImageFile = null;

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
                    generationParams: { prompt: retryPrompt }
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
                state.sessions.unshift({ id: state.currentSessionId, title: '精准修图', timestamp: Date.now(), messages: [], workingMemory: createDefaultWorkingMemory(), isAutoRenamed: false });
                setLastActiveSessionId(state.currentSessionId);
            }
            const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
            const hadSessionMessages = currentSession.messages.length > 0;
            const userMsg = { sender: 'user', type: 'text', content: textFromInput };
            currentSession.messages.push(userMsg);
            if (hadSessionMessages) addMessage(userMsg);
            else renderMessages(currentSession.messages);
            renderHistoryList();
            state.userInput.value = ''; state.userInput.style.height = '';
            const box = state.preciseEditMode.pendingBox; const itemId = state.preciseEditMode.pendingItemId;
            state.preciseEditMode.pendingBox = null; state.preciseEditMode.pendingItemId = null;
            await yieldToChatPaint();
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
        const newSession = { id: state.currentSessionId, title: '图片对话', timestamp: Date.now(), messages: [], workingMemory: createDefaultWorkingMemory(), isAutoRenamed: false };
        state.sessions.unshift(newSession);
        setLastActiveSessionId(state.currentSessionId);
    }
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    ensureSessionWorkingMemory(currentSession);
    const hadSessionMessages = currentSession.messages.length > 0;
    currentSession.updatedAt = Date.now();
    currentSession.messages.push(...userMessages);
    if (hadSessionMessages) {
        userMessages.forEach(message => addMessage(message));
    } else {
        renderMessages(currentSession.messages);
    }
    renderHistoryList();
        state.userInput.value = ''; state.userInput.style.height = '';
    const useQwenLocalImageRouting = shouldUseQwenLocalImageRouting(
        textFromInput,
        effectiveBaseImage,
        state.maskDataUrl
    );
    if (useQwenLocalImageRouting) {
        console.log('[Qwen Local Routing] 图片任务跳过 Gemini 解释器与意图分类器');
    }
    if (!useQwenLocalImageRouting) autoRenameSessionIfNeeded(currentSession);
    
    const referenceImagesSent = [...state.referenceImageFiles];
    const maskSent = state.maskDataUrl;
    let placeholderDiv = document.createElement('div');
    placeholderDiv.className = 'message bot-message';
    const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
        placeholderDiv.innerHTML = botAvatar + `<div>思考中...</div>`;
        state.chatMessages.appendChild(placeholderDiv); state.chatMessages.scrollTop = state.chatMessages.scrollHeight;
        await yieldToChatPaint();
    
    state.mainImageFile = null; state.referenceImageFiles = []; state.maskDataUrl = null;
    if (window.updateImagePreview) await window.updateImagePreview();

    try {
        let workspaceContext = getAssistantWorkspaceContext(state);
        const interpretation = await interpretConversationTurn({
            text: textFromInput,
            baseImage: effectiveBaseImage,
            referenceImages: referenceImagesSent,
            mask: maskSent,
            history: currentSession.messages,
            workingMemory: currentSession.workingMemory,
            selectedModel: state.selectedModel,
            workspaceContext,
            skipModel: useQwenLocalImageRouting
        });
        const resolvedInterpretation = resolveStateConflict({
            text: textFromInput,
            baseImage: effectiveBaseImage,
            referenceImages: referenceImagesSent,
            mask: maskSent,
            history: currentSession.messages,
            workingMemory: currentSession.workingMemory,
            workspaceContext
        }, interpretation);
        const updatedWorkingMemory = mergeInterpreterResultIntoSession(currentSession, resolvedInterpretation);

        const routeDecision = await determineConversationIntent({
            text: textFromInput,
            baseImage: effectiveBaseImage,
            referenceImages: referenceImagesSent,
            mask: maskSent,
            history: currentSession.messages,
            selectedModel: state.selectedModel,
            workingMemory: updatedWorkingMemory,
            workspaceContext,
            localImageRouting: useQwenLocalImageRouting
        });

        const workspaceCommand = detectWorkspaceCommand(textFromInput);
        if (workspaceCommand) {
            workspaceContext = applyWorkspaceCommand(state, workspaceCommand, textFromInput);
        }
        const lensSearchRequest = looksLikeLensSearchRequest(textFromInput);
        let lensSearchResult = null;

        const isGenTask = routeDecision.route === 'image_generation';
        const isEditTask = routeDecision.route === 'image_edit';
        const isQueryTask = routeDecision.route === 'image_query';
        const shouldFinalizeBriefFromResponse = resolvedInterpretation.briefSource === 'assistant_result' && !isGenTask && !isEditTask;

        const resolvedExecutionPrompt = resolveExecutionPrompt(routeDecision, {
            prompt: textFromInput,
            baseImage: effectiveBaseImage,
            referenceImages: referenceImagesSent,
            mask: maskSent,
            targetAspectRatio: '1:1',
            history: currentSession.messages,
            sessionId: state.currentSessionId,
            workingMemory: updatedWorkingMemory,
            workspaceContext
        });
        const executionPrompt = composeExecutionPrompt(routeDecision, {
            prompt: resolvedExecutionPrompt || textFromInput,
            baseImage: effectiveBaseImage,
            referenceImages: referenceImagesSent,
            mask: maskSent,
            targetAspectRatio: '1:1',
            history: currentSession.messages,
            sessionId: state.currentSessionId,
            workingMemory: updatedWorkingMemory,
            workspaceContext
        });

        state.lastGenerationContext = (isGenTask || isEditTask) ? {
            prompt: textFromInput,
            corePrompt: resolvedExecutionPrompt || textFromInput,
            resolvedPrompt: executionPrompt,
            baseImage: effectiveBaseImage,
            referenceImages: [...referenceImagesSent],
            mask: maskSent,
            isGenTask,
            isEditTask
        } : null;

        const placeholderText = lensSearchRequest
            ? '正在搜索同款...'
            : getIntentPlaceholderText(routeDecision, {
            prompt: textFromInput,
            mask: maskSent
        });
        const placeholderContent = shouldUseImagePlaceholder(routeDecision)
            ? `<div class="image-placeholder-container"><div class=\"image-placeholder-box\"><div class=\"spinner\"></div></div><div class=\"placeholder-text\">${placeholderText}</div></div>`
            : `<div>${placeholderText}</div>`;
        placeholderDiv.innerHTML = botAvatar + placeholderContent;

        if (lensSearchRequest) {
            lensSearchResult = await executeLensSearch(state, textFromInput, workspaceContext);
        }

        let result = workspaceCommand
            ? {
                success: true,
                text: buildWorkspaceCommandResponse(workspaceCommand, workspaceContext),
                functionCalls: []
            }
            : lensSearchResult
            ? lensSearchResult
            : await executeConversationIntent(routeDecision, {
                prompt: resolvedExecutionPrompt || textFromInput,
                composedPrompt: executionPrompt,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                targetAspectRatio: '1:1',
                history: currentSession.messages,
                sessionId: state.currentSessionId,
                workingMemory: updatedWorkingMemory,
                workspaceContext
            });

        if (routeDecision.route === 'text_chat' && result.functionCalls && result.functionCalls.length > 0) {
            let originalText = result.text;
            let hasEdit = false;
            for (const call of result.functionCalls) {
                const { name, args } = call;
                if (name === 'edit_image') {
                    const targetItem = state.workbenchItems.get(args.itemId);
                    if (targetItem && targetItem.file) {
                        if (!hasEdit) {
                            placeholderDiv.innerHTML = botAvatar + `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">正在编辑图片...</div></div>`;
                            hasEdit = true;
                        }
                        const requestedCount = getExplicitRequestedImageCount(args.prompt || '', 6);
                        result = requestedCount > 1
                            ? await editOrQueryImageWithGemini_Multiple(args.prompt, targetItem.file, [], null, null, requestedCount)
                            : await editOrQueryImageWithGemini(args.prompt, targetItem.file, [], null);
                        if (!result.text) result.text = originalText !== "正在执行操作..." ? originalText : "已为您修改图片。";
                    } else {
                        if (!result.text) result.text = originalText !== "正在执行操作..." ? originalText : "找不到指定的图片进行编辑。";
                    }
                }
            }
        }

        if (result.success || result.handled) {
            const botMessage = { sender: 'bot', type: 'bot-rich' };
            
            if (Array.isArray(result.imageData)) {
                botMessage.imageData = await Promise.all(result.imageData.map(async (img) => {
                    const watermarkedSrc = await addWatermark(`data:${img.mimeType};base64,${img.imageData}`);
                    try {
                        const file = dataURLtoFileSync(watermarkedSrc, `gen-${Date.now()}.png`);
                        await addImageToWorkbench(file, 'AI生成', {
                            dataUrl: watermarkedSrc,
                            parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                            generationParams: { prompt: executionPrompt }
                        });
                    } catch(e){}
                    return { src: watermarkedSrc, mimeType: img.mimeType };
                }));
                botMessage.content = result.text || '已为您生成多个版本，请选用一张以继续。';
                
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (shouldFinalizeBriefFromResponse) {
                    finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                }
                if (window.updateImagePreview) await window.updateImagePreview();
                getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                state.lastGeneratedImageForEditing = null;
            } else if (result.imageData) {
                const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                botMessage.imageData = { src: watermarkedImgSrc };
                botMessage.content = result.text || buildBotFallbackText(executionPrompt, {isEditTask, isGenTask});
                
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (shouldFinalizeBriefFromResponse) {
                    finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                }
                if (window.updateImagePreview) await window.updateImagePreview();

                try {
                    const generatedFile = dataURLtoFileSync(watermarkedImgSrc, `gen-${Date.now()}.png`);
                    // 不再自动把 AI 结果设为下一轮对话底图，避免后续纯文本请求被误路由到图片链路
                    state.lastGeneratedImageForEditing = null;
                    state.mainImageFile = null;
                    await addImageToWorkbench(generatedFile, 'AI生成', {
                        dataUrl: watermarkedImgSrc,
                        parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                        generationParams: { prompt: executionPrompt }
                    });
                } catch(e){}
                
                getSmartSuggestions(result.imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
            } else {
                botMessage.content = result.text;
                placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                currentSession.updatedAt = Date.now();
                currentSession.messages.push(botMessage);
                if (shouldFinalizeBriefFromResponse) {
                    finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                }
                if (!isQueryTask) state.lastGeneratedImageForEditing = null;
            }
        }
    } catch (error) {
        placeholderDiv.innerHTML = botAvatar + `<div>处理失败: ${error.message}</div>`;
    } finally {
        state.isSending = false; 
        if (state.updateSendBtnState) state.updateSendBtnState(); 
        await dbHelper.saveSession(currentSession);
        if (!currentSession.isAutoRenamed || GENERIC_SESSION_TITLES.has(currentSession.title)) {
            await autoRenameSessionIfNeeded(currentSession);
        }
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

    const messageDiv = addMessage({ sender: 'bot', content: '图片已上传。您可以让我先读图、提炼结构、整理成文本总纲，或者继续编辑与生成：' });
    appendSuggestionButtons(messageDiv, [
        { label: '绘制蒙版并替换', action: () => { if(state.mainImageFile) openMaskEditor(state.mainImageFile) } },
        { label: '提炼结构总纲', prompt: '请把这张图整理成后续都要遵循的结构化文本总纲' },
        { label: '咨询这张图片', prompt: '详细描述这张图的内容' },
    ], handleSuggestionClick);
}
