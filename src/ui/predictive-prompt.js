import { implicitMemoryEngine } from '../runtime/ImplicitMemoryEngine';
import { state } from '../core/state';

let activeMatch = null;

// Dynamically inject styles for the Predictive Overlay
function injectPredictiveStyles() {
    if (document.getElementById('mvr-predictive-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'mvr-predictive-styles';
    style.innerHTML = `
        .predictive-outer-container {
            position: absolute;
            bottom: calc(100% + 8px);
            left: 0;
            right: 0;
            z-index: 999;
            pointer-events: none;
            display: flex;
            justify-content: center;
            opacity: 0;
            transform: translateY(10px) scale(0.98);
            transition: all 0.25s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        
        .predictive-outer-container.active {
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }
        
        .predictive-capsule {
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(139, 92, 246, 0.35);
            box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.2), 
                        0 8px 16px -6px rgba(0, 0, 0, 0.05);
            border-radius: 14px;
            padding: 10px 14px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
            cursor: pointer;
            user-select: none;
        }

        body.dark-mode .predictive-capsule {
            background: rgba(30, 30, 36, 0.95);
            border-color: rgba(168, 85, 247, 0.45);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 
                        0 4px 15px rgba(139, 92, 246, 0.15);
        }

        .predictive-capsule:hover {
            border-color: #a855f7;
            box-shadow: 0 12px 30px -5px rgba(139, 92, 246, 0.35);
        }

        .predictive-sparkle {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(99, 102, 241, 0.15));
            color: #8b5cf6;
            flex-shrink: 0;
            position: relative;
        }

        body.dark-mode .predictive-sparkle {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(99, 102, 241, 0.2));
            color: #c084fc;
        }

        /* Pulsing ring around the sparkle icon */
        .predictive-sparkle::after {
            content: '';
            position: absolute;
            top: -2px; left: -2px; right: -2px; bottom: -2px;
            border-radius: 50%;
            border: 1px solid rgba(139, 92, 246, 0.25);
            animation: pulseRing 2s infinite ease-out;
        }

        @keyframes pulseRing {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(1.3); opacity: 0; }
        }

        .predictive-content-area {
            flex-grow: 1;
            min-width: 0; /* for text-truncation in flexbox */
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .predictive-badge-row {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .predictive-badge {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 6px;
            border-radius: 6px;
            background: rgba(139, 92, 246, 0.1);
            color: #8b5cf6;
            letter-spacing: 0.05em;
        }

        body.dark-mode .predictive-badge {
            background: rgba(168, 85, 247, 0.2);
            color: #c084fc;
        }

        .predictive-count-pill {
            font-size: 10px;
            color: #6b7280;
        }
        body.dark-mode .predictive-count-pill {
            color: #9ca3af;
        }

        .predictive-text-preview {
            font-size: 13.5px;
            line-height: 1.4;
            color: #4b5563;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        body.dark-mode .predictive-text-preview {
            color: #d1d5db;
        }

        .matched-text {
            font-weight: 500;
            color: #1f2937;
        }
        body.dark-mode .matched-text {
            color: #f3f4f6;
        }

        .completion-text {
            color: #8b5cf6;
            font-weight: 500;
            text-shadow: 0 0 8px rgba(139, 92, 246, 0.1);
        }

        body.dark-mode .completion-text {
            color: #a78bfa;
        }

        .predictive-action-tips {
            text-align: right;
            font-size: 11px;
            color: #9cb3a8;
            flex-shrink: 0;
            padding-left: 8px;
            border-left: 1px solid rgba(0, 0, 0, 0.06);
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 2px;
        }

        body.dark-mode .predictive-action-tips {
            border-left-color: rgba(255, 255, 255, 0.08);
            color: #6b7280;
        }

        .tip-key-pill {
            font-family: inherit;
            background: rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 4px;
            padding: 1px 4px;
            font-size: 10px;
            color: #4b5563;
            font-weight: 600;
        }

         body.dark-mode .tip-key-pill {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.12);
            color: #d1d5db;
        }

        .predictive-tag-pill {
            font-size: 10px;
            padding: 1px 5px;
            border-radius: 4px;
            background: rgba(139, 92, 246, 0.08);
            border: 1px solid rgba(139, 92, 246, 0.2);
            color: #8b5cf6;
            margin-right: 2px;
            font-family: var(--font-sans);
        }
        
        body.dark-mode .predictive-tag-pill {
            background: rgba(168, 85, 247, 0.15);
            border-color: rgba(168, 85, 247, 0.3);
            color: #c084fc;
        }

        /* Success toast notice when parameters are auto-applied */
        .parameter-toast {
            position: absolute;
            bottom: calc(100% + 50px);
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: #10b981;
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
        }
        
        .parameter-toast.active {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }

        /* Input field flash spotlight effect on autocomplete completion */
        @keyframes spotlightGlow {
            0% { box-shadow: 0 0 0 0px rgba(139, 92, 246, 0.4); background: rgba(139, 92, 246, 0.05); }
            50% { box-shadow: 0 0 0 6px rgba(139, 92, 246, 0.2); background: rgba(139, 92, 246, 0.02); }
            100% { box-shadow: 0 0 0 0px rgba(139, 92, 246, 0); background: transparent; }
        }

        .spotlight-active {
            animation: spotlightGlow 0.8s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
        }
    `;
    document.head.appendChild(style);
}

export function initPredictivePromptEngine(userInput) {
    if (!userInput) return;
    
    injectPredictiveStyles();

    // Create the container element dynamically
    let container = document.getElementById('predictiveContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'predictiveContainer';
        container.className = 'predictive-outer-container';
        
        // Target wrapper and insert it
        const wrapper = userInput.closest('.input-container-wrapper');
        if (wrapper) {
            wrapper.appendChild(container);
        } else {
            userInput.parentNode.insertBefore(container, userInput);
        }
    }

    // Monitor input events
    userInput.addEventListener('input', () => {
        const value = userInput.value;
        
        if (!value || value.trim().length < 2) {
            hidePredictiveOverlay();
            return;
        }

        const match = implicitMemoryEngine.findMatch(value);
        if (match) {
            activeMatch = match;
            renderMatchInOverlay(match, value);
        } else {
            hidePredictiveOverlay();
        }
    });

    // Handle keys: Tab or Enter to autocomplete
    userInput.addEventListener('keydown', (e) => {
        if (!activeMatch) return;

        // If user presses Tab, intercept and autocomplete instantly
        if (e.key === 'Tab') {
            e.preventDefault();
            applyAutocomplete();
        }
    });

    // Also hide predictions if user clicks outside of the chat panel area
    document.addEventListener('click', (e) => {
        const wrapper = userInput.closest('.input-container-wrapper');
        if (wrapper && !wrapper.contains(e.target)) {
            hidePredictiveOverlay();
        }
    });
}

function renderMatchInOverlay(match, rawInput) {
    const container = document.getElementById('predictiveContainer');
    if (!container) return;

    // Split finalPrompt into "what is typed" versus "what is proposed"
    const lowerInput = rawInput.toLowerCase();
    const finalPrompt = match.finalPrompt;
    
    let matchedPart = rawInput;
    let extensionPart = '';

    // Check if the finalPrompt starts with user's typed input, split cleanly
    if (finalPrompt.toLowerCase().startsWith(lowerInput)) {
        matchedPart = finalPrompt.substring(0, rawInput.length);
        extensionPart = finalPrompt.substring(rawInput.length);
    } else {
        // Fuzzy/Keyword split: find where the user's last keyword or overlap ends
        const lastIndex = Math.max(0, ...implicitMemoryEngine.extractStemKeywords(rawInput).map(kw => {
            const index = finalPrompt.toLowerCase().lastIndexOf(kw.toLowerCase());
            return index !== -1 ? index + kw.length : 0;
        }));

        if (lastIndex > 0 && lastIndex < finalPrompt.length) {
            matchedPart = finalPrompt.substring(0, lastIndex);
            extensionPart = finalPrompt.substring(lastIndex);
        } else {
            // General fallback
            matchedPart = rawInput;
            extensionPart = ' ' + finalPrompt.substring(Math.min(finalPrompt.length, rawInput.length));
        }
    }

    const badgeLabel = match.id.startsWith('default') ? '脑电波配方' : '已心有灵犀 💖';
    const usageTip = `关联 ${match.count} 次满意定稿`;

    // Render preset tags
    const tags = match.presetTags || ['双通道融合', '参数自适应'];
    const tagsHtml = tags.map(t => `<span class="predictive-tag-pill">✨ ${t}</span>`).join('');

    container.innerHTML = `
        <div class="predictive-capsule" id="predictiveCapsuleInner" title="点击或按 Tab 键一键定稿">
            <div class="predictive-sparkle">
                <i class="fas fa-magic" style="font-size: 13px;"></i>
            </div>
            <div class="predictive-content-area">
                <div class="predictive-badge-row">
                    <span class="predictive-badge">${badgeLabel}</span>
                    <span class="predictive-count-pill">${usageTip}</span>
                </div>
                <div class="predictive-text-preview" style="margin-bottom: 4px;">
                    <span class="matched-text">${escapeHTML(matchedPart)}</span><span class="completion-text">${escapeHTML(extensionPart)}</span>
                </div>
                <div class="predictive-tags-row" style="display: flex; gap: 4px; flex-wrap: wrap;">
                    ${tagsHtml}
                </div>
            </div>
            <div class="predictive-action-tips">
                <div><span class="tip-key-pill">Tab</span></div>
                <div style="font-size: 9px; opacity: 0.8; margin-top: 2px;">一键套用</div>
            </div>
        </div>
    `;

    // Click handler for direct complete
    const innerBubble = document.getElementById('predictiveCapsuleInner');
    if (innerBubble) {
        innerBubble.addEventListener('click', (e) => {
            e.stopPropagation();
            applyAutocomplete();
        });
    }

    container.classList.add('active');
}

export function hidePredictiveOverlay() {
    const container = document.getElementById('predictiveContainer');
    if (container) {
        container.classList.remove('active');
    }
    activeMatch = null;
}

function applyAutocomplete() {
    if (!activeMatch) return;

    const userInput = document.getElementById('userInput');
    if (userInput) {
        // Update value with beautiful selection highlight
        userInput.value = activeMatch.finalPrompt;
        
        // Dispatch simple HTML animation spotlight focus
        const wrapper = userInput.closest('.input-container-wrapper');
        if (wrapper) {
            wrapper.classList.remove('spotlight-active');
            // Trigger reflow
            void wrapper.offsetWidth;
            wrapper.classList.add('spotlight-active');
        }

        // Adjust scrollHeight since input expanded
        userInput.style.height = ''; 
        const newHeight = Math.min(userInput.scrollHeight, 200);
        userInput.style.height = `${newHeight}px`;

        userInput.focus();
        if (state.updateSendBtnState) {
            state.updateSendBtnState();
        }

        // FULL-STACK PARAMETER REUSE:
        // 1) Apply Fusion parameters to the active selected item on workbench
        let appliedFusion = false;
        if (activeMatch.fusionProperties && window.mvrRuntime) {
            const currentAssetId = state.currentActiveWorkbenchItemId;
            if (currentAssetId) {
                const ws = window.mvrRuntime.getCurrentWorkspace();
                if (ws && ws.dispatcher) {
                    try {
                        ws.dispatcher.dispatch({
                            type: 'UPDATE_FUSION',
                            payload: {
                                uid: currentAssetId,
                                fusionProperties: activeMatch.fusionProperties
                            }
                        });
                        appliedFusion = true;
                        
                        // Synchronize any open slider control panels
                        if (typeof window.syncFusionPanelWithSelectedAsset === 'function') {
                            window.syncFusionPanelWithSelectedAsset(currentAssetId);
                        }
                    } catch(e) {
                        console.error('[PredictivePrompt] Live fusion preloading failed:', e);
                    }
                }
            }
        }

        // 2) Cache inside global state so next image generation/editing inherits these ratios/options
        state.pendingImplicitMemoryOverride = {
            fusionProperties: activeMatch.fusionProperties || null,
            crop: activeMatch.crop || null,
            aspectRatio: activeMatch.aspectRatio || null
        };

        // 3) Show parameter auto-apply success toast with beautiful styling inside the input area wrap
        const tags = activeMatch.presetTags || ['双通道融合', '高保真参数'];
        const tagsStr = tags.join(' + ');

        let toast = document.getElementById('mvrParameterToast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'mvrParameterToast';
            toast.className = 'parameter-toast';
            if (wrapper) {
                wrapper.appendChild(toast);
            } else {
                document.body.appendChild(toast);
            }
        }

        toast.innerHTML = `
            <i class="fas fa-check-circle" style="color: #6ee7b7; font-size: 12px;"></i>
            <span>✨ 已自动套用：[${tagsStr}] 全栈参数级效果！</span>
        `;

        toast.classList.remove('active');
        void toast.offsetWidth; // trigger reflow
        toast.classList.add('active');

        setTimeout(() => {
            toast.classList.remove('active');
        }, 3000);
    }

    hidePredictiveOverlay();
}

function escapeHTML(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
