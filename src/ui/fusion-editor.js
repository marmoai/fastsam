import { state } from '../core/state.js';
import { addMessage } from './chat-panel.js';
import { SemanticRecommender } from '../runtime/SemanticRecommender';
import { evolutionEngine } from '../runtime/EvolutionEngine';

const { workbenchItems } = state;

export let preciseEditMode = {
    active: false,
    itemId: null,
    selectionBox: null,
    overlay: null,
    helperText: null,
    pendingBox: null,
    pendingItemId: null,
    menuEl: null
};

export function startPreciseEditMode(itemId) {
    const item = workbenchItems.get(itemId);
    if (!item) {
        console.error('精准修图：找不到工作台项目', itemId);
        return;
    }

    console.log('启动精准修图模式', itemId);
    preciseEditMode.active = true;
    preciseEditMode.itemId = itemId;

    // 使用 MarmoLens 的框选功能
    if (window.marmoLens && typeof window.marmoLens.startSelectionMode === 'function') {
        try {
            if (preciseEditMode.menuEl) {
                preciseEditMode.menuEl.remove();
            }
            const menu = document.createElement('div');
            menu.className = 'selection-mode-menu';
            menu.innerHTML = `
                <button class="selection-mode-btn active" data-mode="rect" style="background: #f0f0f0;">
                    <i class="fas fa-crop-alt"></i>
                    <span>框选</span>
                </button>
                <button class="selection-mode-btn" data-mode="lasso">
                    <i class="fas fa-pencil-alt"></i>
                    <span>套索</span>
                </button>
                <button class="selection-mode-btn" data-mode="cancel" style="color: #ff4444;">
                    <i class="fas fa-times"></i>
                    <span>取消</span>
                </button>
            `;
            const rect = item.el.getBoundingClientRect();
            
            let posX = rect.left + rect.width / 2;
            let posY = rect.top - 80;
            
            // Boundary check: If not enough space above, show below
            if (posY < 10) posY = rect.top + rect.height + 20;
            
            menu.style.left = posX + 'px';
            menu.style.top = posY + 'px';
            menu.style.transform = 'translateX(-50%)';
            document.body.appendChild(menu);
            preciseEditMode.menuEl = menu;

            const btns = menu.querySelectorAll('.selection-mode-btn');
            btns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const mode = btn.dataset.mode;
                    if (mode === 'cancel') {
                        menu.remove();
                        preciseEditMode.menuEl = null;
                        preciseEditMode.active = false;
                        window.marmoLens.cleanupOverlay();
                        if (window.marmoLens._originalHandleMouseUp) {
                            window.marmoLens.handleMouseUp = window.marmoLens._originalHandleMouseUp;
                        }
                        return;
                    }
                    btns.forEach(b => {
                        b.classList.remove('active');
                        b.style.background = 'transparent';
                    });
                    btn.classList.add('active');
                    btn.style.background = '#f0f0f0';
                    window.marmoLens.startSelectionMode(itemId, mode);
                });
            });

            // 默认使用框选模式
            window.marmoLens.startSelectionMode(itemId, 'rect');
            
            // 临时存储原始 handleMouseUp，以便在修图模式结束后恢复
            if (!window.marmoLens._originalHandleMouseUp) {
                window.marmoLens._originalHandleMouseUp = window.marmoLens.handleMouseUp.bind(window.marmoLens);
            }
            
            // 修改 MarmoLens 的 handleMouseUp，使其支持修图模式
            window.marmoLens.handleMouseUp = async function(e, item) {
                if (!preciseEditMode.active) {
                    return window.marmoLens._originalHandleMouseUp(e, item);
                }
                
                if (!this.isSelecting) return;
                this.isSelecting = false;
                
                window.removeEventListener('mousemove', this.moveHandler);
                window.removeEventListener('mouseup', this.upHandler);

                let x, y, w, h;
                if (this.selectionType === 'lasso') {
                    if (this.lassoPoints.length < 3) {
                        this.cleanupOverlay();
                        preciseEditMode.active = false;
                        if (preciseEditMode.menuEl) { preciseEditMode.menuEl.remove(); preciseEditMode.menuEl = null; }
                        window.marmoLens.handleMouseUp = window.marmoLens._originalHandleMouseUp;
                        return;
                    }
                    const xs = this.lassoPoints.map(p => p.x);
                    const ys = this.lassoPoints.map(p => p.y);
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    x = minX; y = minY; w = maxX - minX; h = maxY - minY;
                } else {
                    const style = this.selectionBox.style;
                    x = parseFloat(style.left);
                    y = parseFloat(style.top);
                    w = parseFloat(style.width);
                    h = parseFloat(style.height);
                }

                if (w < 10 || h < 10) {
                    this.cleanupOverlay();
                    preciseEditMode.active = false;
                    if (preciseEditMode.menuEl) { preciseEditMode.menuEl.remove(); preciseEditMode.menuEl = null; }
                    // 恢复原始 handleMouseUp
                    window.marmoLens.handleMouseUp = window.marmoLens._originalHandleMouseUp;
                    return;
                }

                const imgW = item.el.offsetWidth;
                const imgH = item.el.offsetHeight;
                
                const xmin = (x / imgW) * 1000;
                const ymin = (y / imgH) * 1000;
                const xmax = ((x + w) / imgW) * 1000;
                const ymax = ((y + h) / imgH) * 1000;

                const box = [ymin, xmin, ymax, xmax];
                
                // 清理框选界面和提示
                this.cleanupOverlay();
                preciseEditMode.active = false;
                if (preciseEditMode.menuEl) { preciseEditMode.menuEl.remove(); preciseEditMode.menuEl = null; }
                
                // 恢复原始 handleMouseUp
                window.marmoLens.handleMouseUp = window.marmoLens._originalHandleMouseUp;
                
                // 框选后直接显示输入框（在聊天输入框）
                const userInput = document.getElementById('userInput');
                if (userInput) {
                    userInput.value = "修改框选区域：";
                    userInput.focus();
                    userInput.setSelectionRange(userInput.value.length, userInput.value.length);
                }
                
                // 存储框选信息，等待用户输入指令后执行
                preciseEditMode.pendingBox = box;
                preciseEditMode.pendingItemId = itemId;
                
                // 添加提示消息
                addMessage({ 
                    sender: 'bot', 
                    type: 'text', 
                    content: '✅ **已框选区域**\n请在输入框中输入修改指令（如：沙发改成椅子、换成红色等），然后发送即可完成精准修图。' 
                });
            };
        } catch (error) {
            console.error('启动 MarmoLens 框选模式失败:', error);
            // 即使 MarmoLens 失败，也保留提示，让用户知道功能已激活
            addMessage({ 
                sender: 'bot', 
                type: 'text', 
                content: '⚠️ 框选功能初始化失败，请刷新页面后重试。' 
            });
        }
    } else {
        console.warn('MarmoLens 不可用，无法启动框选模式');
        // 如果 MarmoLens 不存在，至少显示提示
        addMessage({ 
            sender: 'bot', 
            type: 'text', 
            content: '⚠️ 框选功能不可用，请确保智能搜索功能已加载。' 
        });
    }
}

let currentMode = 'isolated'; // Module level state to remember last selected mode
let isSlidersExpanded = false; // Module level state to remember manual sliders fold state
let isHistoryExpanded = false;
let isRecommendationArchiveExpanded = false;

function renderSlider(id, label, value, min, max, unit) {
    return `
        <div class="fusion-slider-group" data-slider-prop="${id}" style="display: flex; flex-direction: column; gap: 4px; transition: opacity 0.25s ease;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #475569; align-items: center;">
                <span class="slider-label" style="display: flex; align-items: center; gap: 4px;">${label}</span>
                <span class="val-${id}" style="font-weight: 500; font-variant-numeric: tabular-nums; font-size: 10px;">${value}${unit}</span>
            </div>
            <input type="range" 
                data-prop="${id}" 
                data-unit="${unit}"
                min="${min}" max="${max}" value="${value}"
                style="width: 100%; accent-color: #6366f1; cursor: pointer; transition: all 0.2s;">
        </div>
    `;
}

// Style declarations used for elegant popups and animations
const editorStyles = `
@keyframes ledPulseRefined {
    0% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 6px rgba(165, 180, 252, 0.4); }
    50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 12px rgba(165, 180, 252, 0.8), 0 0 6px rgba(165, 180, 252, 0.4); }
    100% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 6px rgba(165, 180, 252, 0.4); }
}

@keyframes bubblePop {
    0% { opacity: 0; transform: scale(0.9) translateY(10px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes ledAlarmAnim {
    0% { background: #8b5cf6; box-shadow: 0 0 4px #8b5cf6; }
    50% { background: #f43f5e; box-shadow: 0 0 16px #f43f5e, 0 0 8px #f43f5e; transform: scale(1.2); }
    100% { background: #8b5cf6; box-shadow: 0 0 4px #8b5cf6; }
}

/* Space Capsule Main Container */
.ai-inspiration-capsule {
    position: fixed;
    right: calc(280px + 40px); /* 默认右侧空间避让聊天面板：外加 40px 间距 */
    top: 69px; /* 调整为与 chat panel 折叠状态下的 toggle-chat 居中同高 */
    width: 144px;
    height: 42px;
    border-radius: 21px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 27, 75, 0.9));
    border: 1.5px solid rgba(129, 140, 248, 0.38);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25), 0 0 15px rgba(99, 102, 241, 0.2);
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    cursor: pointer;
    z-index: 10004;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    transform-origin: center center;
    user-select: none;
    opacity: 0.45;
}

/* 当右侧聊天栏收起时，自动移到右侧并在 toggle-chat 按钮左侧，保持优雅的距离 */
.container.chat-collapsed .ai-inspiration-capsule {
    right: 80px; /* 避开 toggle-chat 按钮范围 (5px 至 45px)，保留 35px 优雅间隔 */
}

.ai-inspiration-capsule:hover {
    transform: scale(1.08); /* 移除 translateY(-50%) 避免干扰定位 */
    border-color: rgba(129, 140, 248, 0.7);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.35), 0 0 25px rgba(139, 92, 246, 0.3);
    opacity: 1;
}

.ai-inspiration-capsule.active-bright {
    opacity: 1 !important;
    border-color: rgba(168, 85, 247, 0.75);
    box-shadow: 0 15px 35px rgba(168, 85, 247, 0.35), 0 0 25px rgba(99, 102, 241, 0.4);
}

.capsule-led-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981, 0 0 4px #10b981;
    transition: all 0.3s;
}

.capsule-led-pulse.pulsing {
    animation: ledAlarmAnim 1s infinite ease-in-out;
}

.capsule-glowing-emblem {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.18);
    color: #a78bfa;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    transition: all 0.3s;
}

.ai-inspiration-capsule:hover .capsule-glowing-emblem {
    color: #c084fc;
    background: rgba(139, 92, 246, 0.3);
}

/* Capsules speech bubble alert */
.capsule-speech-tooltip {
    position: fixed;
    right: calc(280px + 196px); /* 320px (capsule) + 144px (width) + 12px (gap) = 476px */
    top: 72px; /* 垂直居中于 capsule 的中心 (capsule top 69, height 42) */
    transform: scale(0.9) translateX(15px);
    background: linear-gradient(135deg, #121026, #1e1b4b);
    color: #e2e8f0;
    border: 1px solid rgba(139, 92, 246, 0.35);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    padding: 10px 14px;
    border-radius: 12px;
    z-index: 10003;
    font-size: 11px;
    font-weight: 500;
    pointer-events: none;
    opacity: 0;
    transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
    font-family: var(--font-sans);
}

.container.chat-collapsed .capsule-speech-tooltip {
    right: 236px; /* 80px (capsule) + 144px (width) + 12px (gap) = 236px */
}

.capsule-speech-tooltip.visible {
    opacity: 1;
    transform: scale(1) translateX(0); /* 移除 translateY(-50%) */
}

.capsule-speech-tooltip::after {
    content: '';
    position: absolute;
    right: -6px;
    top: 50%;
    transform: translateY(-50%) rotate(45deg);
    width: 10px;
    height: 10px;
    background: #1e1b4b;
    border-right: 1px solid rgba(139, 92, 246, 0.35);
    border-top: 1px solid rgba(139, 92, 246, 0.35);
}

.capsule-speech-tooltip.flipped::after {
    right: auto;
    left: -6px;
    border-right: none;
    border-top: none;
    border-left: 1px solid rgba(139, 92, 246, 0.35);
    border-bottom: 1px solid rgba(139, 92, 246, 0.35);
}

/* Intelligent Drawer sliding from right */
.ai-smart-drawer {
    position: fixed;
    top: 115px; /* 紧贴太空舱下方展开 (太空舱 69px top + 42px height + 4px elegant buffer) */
    right: calc(280px + 40px); /* 默认右侧空间：与太空舱右侧精准齐平 */
    width: 280px;
    height: calc(100vh - 145px); /* 留足底部 30px 空隙 */
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(99, 102, 241, 0.16);
    border-radius: 20px;
    box-shadow: 0 15px 45px rgba(15, 23, 42, 0.12), 0 4px 20px rgba(99, 102, 241, 0.03);
    z-index: 10006;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease, right 0.45s cubic-bezier(0.16, 1, 0.3, 1);
    transform: translateY(-20px) scale(0.96);
    transform-origin: top right;
    opacity: 0;
    pointer-events: none;
}

/* 聊天栏关闭时，与胶囊保持同样位置 */
.container.chat-collapsed .ai-smart-drawer {
    right: 80px;
}

.ai-smart-drawer.expanded {
    transform: translateY(0) scale(1);
    opacity: 1;
    pointer-events: auto;
}

/* Timeline vertical connector path styles */
.history-branch-line {
    position: absolute;
    left: 21px;
    top: 10px;
    bottom: 25px;
    width: 2px;
    background: #e2e8f0;
    z-index: 1;
}

.history-branch-node {
    position: relative;
    padding-left: 36px;
    padding-bottom: 14px;
    z-index: 2;
}

.history-branch-node:last-child {
    padding-bottom: 0px;
}

.branch-node-indicator {
    position: absolute;
    left: 17px;
    top: 15px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #cbd5e1;
    border: 2px solid #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    z-index: 3;
    transition: all 0.25s;
}

.history-branch-node.active .branch-node-indicator {
    background: #6366f1;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.6), 0 0 3px rgba(99, 102, 241, 0.3);
    transform: scale(1.2);
}

.branch-node-hover-zone {
    border: 1px solid rgba(0, 0, 0, 0.04);
    background: #ffffff;
    border-radius: 12px;
    padding: 8px 10px;
    cursor: pointer;
    transition: all 0.22s;
    display: flex;
    align-items: center;
    gap: 10px;
}

.branch-node-hover-zone:hover {
    transform: translateY(-1px);
    background: #f8fafc;
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.history-branch-node.active .branch-node-hover-zone {
    background: linear-gradient(135deg, #ffffff, #f5f7ff);
    border-color: rgba(99, 102, 241, 0.22);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.04);
}

.ai-suggestion-card {
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.ai-suggestion-card:hover {
    transform: translateY(-1.5px);
    border-color: #6366f1;
    background: linear-gradient(135deg, #ffffff, #eff6ff) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
}
.ai-suggestion-card:active {
    transform: scale(0.98);
}
.intent-suggest-tag {
    transition: all 0.22s ease-in-out !important;
}
.intent-suggest-tag:hover {
    background: #ffffff !important;
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    box-shadow: 0 4px 10px rgba(124, 58, 237, 0.16) !important;
    transform: translateY(-1.5px) !important;
}
.intent-suggest-tag:active {
    transform: scale(0.94) !important;
}

.ai-insight-card {
    padding: 16px;
    background: linear-gradient(135deg, #f8f5ff, #eef4ff);
    border: 1px solid rgba(124, 58, 237, 0.14);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
}

.ai-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.ai-status-chip {
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(124, 58, 237, 0.12);
    color: #5b21b6;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.ai-mode-toggle {
    display: flex;
    gap: 6px;
    background: rgba(255, 255, 255, 0.65);
    padding: 4px;
    border-radius: 12px;
}

.ai-mode-toggle button {
    flex: 1;
    padding: 7px 10px;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 11px;
    font-weight: 700;
}

.ai-mode-toggle button.active {
    background: #7c3aed;
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.22);
}

.ai-prompt-input-row {
    display: flex;
    gap: 8px;
    align-items: center;
}

.ai-prompt-input-row input {
    flex: 1;
    min-width: 0;
    padding: 9px 12px;
    border-radius: 12px;
    border: 1px solid rgba(124, 58, 237, 0.16);
    font-size: 12px;
    background: rgba(255, 255, 255, 0.96);
    color: #1e293b;
    outline: none;
}

.ai-prompt-input-row button {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    border: none;
    background: #7c3aed;
    color: #ffffff;
    cursor: pointer;
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.22);
}

.ai-prompt-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.ai-prompt-chip {
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(124, 58, 237, 0.12);
    color: #4338ca;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.18s ease;
}

.ai-prompt-chip:hover {
    transform: translateY(-1px);
    border-color: rgba(124, 58, 237, 0.26);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.1);
}

.ai-primary-actions {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
}

.ai-primary-action {
    border: 1px solid rgba(124, 58, 237, 0.1);
    border-radius: 16px;
    background: #ffffff;
    padding: 12px 13px;
    display: flex;
    align-items: center;
    gap: 11px;
    cursor: pointer;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
}

.ai-primary-action:hover {
    transform: translateY(-1px);
    border-color: rgba(124, 58, 237, 0.24);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.ai-primary-action-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, #ede9fe, #e0e7ff);
    color: #6d28d9;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.ai-primary-action-copy {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
    flex: 1;
}

.ai-primary-action-copy strong {
    font-size: 12px;
    color: #1e293b;
}

.ai-primary-action-copy span {
    font-size: 10.5px;
    color: #64748b;
    line-height: 1.45;
}

.ai-section-block {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.ai-section-title {
    font-size: 11px;
    color: #7c3aed;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 6px;
}

.ai-empty-block {
    padding: 14px;
    border-radius: 14px;
    border: 1px dashed rgba(148, 163, 184, 0.45);
    background: rgba(255, 255, 255, 0.8);
    color: #94a3b8;
    font-size: 11.5px;
    text-align: center;
}

.ai-accordion {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.ai-accordion-btn {
    width: 100%;
    border: 1px solid rgba(124, 58, 237, 0.12);
    background: #ffffff;
    border-radius: 14px;
    padding: 11px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: all 0.18s ease;
}

.ai-accordion-btn:hover {
    border-color: rgba(124, 58, 237, 0.24);
}

.ai-accordion-btn-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11.5px;
    color: #4c1d95;
    font-weight: 700;
}

.ai-accordion-panel {
    display: none;
    flex-direction: column;
    gap: 10px;
    padding: 14px;
    background: #ffffff;
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-radius: 14px;
}

.ai-accordion-panel.expanded {
    display: flex;
}

.drawer-section-flash {
    animation: drawerSectionFlash 1s ease;
}

@keyframes drawerSectionFlash {
    0% { box-shadow: 0 0 0 rgba(124, 58, 237, 0); }
    40% { box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.14); }
    100% { box-shadow: 0 0 0 rgba(124, 58, 237, 0); }
}

/* ========================================================== */
/* DARK MODE COMPATIBILITY FOR AI INSPIRATION AND DECISION DRAWER */
/* ========================================================== */
body.dark-mode .ai-smart-drawer {
    background: rgba(15, 23, 42, 0.95) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.5), 0 4px 20px rgba(99, 102, 241, 0.1) !important;
}

body.dark-mode .ai-smart-drawer > div:first-child {
    background: rgba(30, 41, 59, 0.4) !important;
    border-bottom-color: rgba(51, 65, 85, 0.6) !important;
    color: #f1f5f9 !important;
}

body.dark-mode .ai-smart-drawer h4 {
    color: #f1f5f9 !important;
}

body.dark-mode .ai-smart-drawer > div:first-child span {
    color: #94a3b8 !important;
}

body.dark-mode .close-editor-btn {
    background: #334155 !important;
    color: #cbd5e1 !important;
}

body.dark-mode .close-editor-btn:hover {
    background: #475569 !important;
    color: #ffffff !important;
}

body.dark-mode .drawer-scroll-body {
    background: rgba(15, 23, 42, 0.98) !important;
}

body.dark-mode .drawer-scroll-body > div:first-child {
    background: #1e293b !important;
}

body.dark-mode #modeIsolated, body.dark-mode #modeFusion {
    color: #94a3b8 !important;
}

body.dark-mode #modeIsolated[style*="background: rgb(124, 58, 237)"],
body.dark-mode #modeIsolated[style*="background: rgb(124,58,237)"],
body.dark-mode #modeIsolated[style*="background: #7c3aed"],
body.dark-mode #modeIsolated[style*="background:#7c3aed"] {
    color: white !important;
}

body.dark-mode #modeFusion[style*="background: rgb(124, 58, 237)"],
body.dark-mode #modeFusion[style*="background: rgb(124,58,237)"],
body.dark-mode #modeFusion[style*="background: #7c3aed"],
body.dark-mode #modeFusion[style*="background:#7c3aed"] {
    color: white !important;
}

body.dark-mode .drawer-scroll-body > div[style*="dashed"] {
    background: linear-gradient(135deg, #1e1b4b, #2e1065) !important;
    border-color: #a855f7 !important;
}

body.dark-mode .drawer-scroll-body > div[style*="dashed"] label {
    color: #c084fc !important;
}

body.dark-mode #fusionQuickInput {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: rgba(168, 85, 247, 0.4) !important;
}

body.dark-mode .intent-suggest-tag {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: rgba(168, 85, 247, 0.3) !important;
    color: #e2e8f0 !important;
}

body.dark-mode .intent-suggest-tag:hover {
    background: #7c3aed !important;
    color: white !important;
    border-color: #a855f7 !important;
}

body.dark-mode #manualSlidersToggle {
    background: #1e293b !important;
    border-color: rgba(168, 85, 247, 0.3) !important;
}

body.dark-mode #manualSlidersToggle span,
body.dark-mode #manualSlidersToggle i {
    color: #c084fc !important;
}

body.dark-mode #manualSlidersPanel {
    background: #1e293b !important;
    border-color: rgba(51, 65, 85, 0.8) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
}

body.dark-mode .fusion-slider-group div {
    color: #94a3b8 !important;
}

body.dark-mode .history-branch-line {
    background: #334155 !important;
}

body.dark-mode .branch-node-indicator {
    border-color: #1e293b !important;
}

body.dark-mode .branch-node-hover-zone {
    background: #1e293b !important;
    border-color: rgba(51, 65, 85, 0.6) !important;
}

body.dark-mode .branch-node-hover-zone:hover {
    background: #334155 !important;
    border-color: rgba(168, 85, 247, 0.4) !important;
}

body.dark-mode .history-branch-node.active .branch-node-hover-zone {
    background: linear-gradient(135deg, #1e293b, #2e1065) !important;
    border-color: rgba(168, 85, 247, 0.4) !important;
}

body.dark-mode .branch-node-hover-zone span[style*="color: #1e293b"] {
    color: #f1f5f9 !important;
}

body.dark-mode .branch-node-hover-zone span[style*="color: #94a3b8"] {
    color: #64748b !important;
}

body.dark-mode .branch-node-hover-zone span[style*="color: #7c3aed"] {
    background: rgba(168, 85, 247, 0.15) !important;
    color: #c084fc !important;
}

body.dark-mode .ai-suggestion-card {
    background: #1e293b !important;
    border-color: rgba(51, 65, 85, 0.5) !important;
}

body.dark-mode .ai-suggestion-card:hover {
    background: linear-gradient(135deg, #1e293b, #252347) !important;
    border-color: #7c3aed !important;
}

body.dark-mode .ai-suggestion-card p[style*="color: #1e293b"] {
    color: #f1f5f9 !important;
}

body.dark-mode .ai-suggestion-card p[style*="color: #64748b"] {
    color: #94a3b8 !important;
}

body.dark-mode .ai-suggestion-card i {
    color: #a855f7 !important;
}

body.dark-mode .ai-suggestion-card div[style*="background: #e0e7ff"],
body.dark-mode .ai-suggestion-card div[style*="background:#e0e7ff"] {
    background: rgba(99, 102, 241, 0.2) !important;
}

body.dark-mode .ai-suggestion-card div[style*="background: #f1f5f9"],
body.dark-mode .ai-suggestion-card div[style*="background:#f1f5f9"] {
    background: rgba(51, 65, 85, 0.4) !important;
}

body.dark-mode #recContainer div[style*="background: white"],
body.dark-mode #recContainer div[style*="background:white"] {
    background: #111827 !important;
    border-color: #374151 !important;
    color: #4b5563 !important;
}

body.dark-mode .drawer-scroll-body div[style*="color: #7c3aed"],
body.dark-mode .drawer-scroll-body div[style*="color:#7c3aed"] {
    color: #c084fc !important;
}

body.dark-mode .drawer-scroll-body div[style*="color: #4338ca"],
body.dark-mode .drawer-scroll-body div[style*="color:#4338ca"] {
    color: #818cf8 !important;
}

body.dark-mode .drawer-scroll-body div[style*="color: #64748b"],
body.dark-mode .drawer-scroll-body div[style*="color:#64748b"] {
    color: #94a3b8 !important;
}

body.dark-mode .ai-insight-card {
    background: linear-gradient(135deg, rgba(49, 32, 99, 0.85), rgba(30, 41, 59, 0.95)) !important;
    border-color: rgba(168, 85, 247, 0.24) !important;
}

body.dark-mode .ai-status-chip,
body.dark-mode .ai-prompt-chip,
body.dark-mode .ai-mode-toggle,
body.dark-mode .ai-prompt-input-row input,
body.dark-mode .ai-primary-action,
body.dark-mode .ai-accordion-btn,
body.dark-mode .ai-accordion-panel,
body.dark-mode .ai-empty-block {
    background: #1e293b !important;
    border-color: rgba(71, 85, 105, 0.85) !important;
    color: #e2e8f0 !important;
}

body.dark-mode .ai-mode-toggle button {
    color: #94a3b8 !important;
}

body.dark-mode .ai-mode-toggle button.active {
    color: #ffffff !important;
}

body.dark-mode .ai-primary-action-copy strong {
    color: #f8fafc !important;
}

body.dark-mode .ai-primary-action-copy span,
body.dark-mode .ai-empty-block {
    color: #94a3b8 !important;
}

body.dark-mode .ai-primary-action-icon {
    background: linear-gradient(135deg, rgba(91, 33, 182, 0.35), rgba(49, 46, 129, 0.45)) !important;
    color: #d8b4fe !important;
}

body.dark-mode .ai-section-title,
body.dark-mode .ai-accordion-btn-label {
    color: #c084fc !important;
}
`;

// Inject custom styles and bootstrap Space Capsule UI
if (!document.getElementById('floatingFusionEditorStyles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'floatingFusionEditorStyles';
    styleSheet.type = 'text/css';
    styleSheet.innerText = editorStyles;
    document.head.appendChild(styleSheet);
}

// Hover tracking variables for Scheme 1 (Hover & Dropdown Capsule)
let isHoveringCapsule = false;
let isHoveringDrawer = false;
let hoverDelayTimer = null;
let hideDelayTimer = null;

// Helper to update bubble tooltip position to stay perfectly relative to a moved capsule
function updateTooltipPosition() {
    const tooltip = document.getElementById('capsuleSpeechTooltip');
    const cap = document.getElementById('aiInspirationCapsule');
    if (!tooltip || !cap) return;
    
    if (cap.style.left && cap.style.left !== '') {
        const capRect = cap.getBoundingClientRect();
        tooltip.style.right = 'auto';
        
        // Target: place tooltip on the left side of the capsule with a 12px gap
        // Also center it vertically relative to the capsule
        const tooltipWidth = tooltip.offsetWidth > 0 ? tooltip.offsetWidth : 196;
        const tooltipHeight = tooltip.offsetHeight > 0 ? tooltip.offsetHeight : 36;
        
        let leftPos = capRect.left - tooltipWidth - 12;
        
        // Dynamic orientation detection: if placing tooltip on the left makes it go offscreen, flip it to the right of the capsule
        if (leftPos < 10) {
            leftPos = capRect.right + 12;
            tooltip.classList.add('flipped');
        } else {
            tooltip.classList.remove('flipped');
        }
        
        tooltip.style.left = `${leftPos}px`;
        tooltip.style.top = `${capRect.top + (capRect.height - tooltipHeight) / 2}px`;
    } else {
        // Reset to default CSS positioning
        tooltip.style.right = '';
        tooltip.style.left = '';
        tooltip.style.top = '';
        tooltip.classList.remove('flipped');
    }
}

// Track if capsule has been dragged to ignore simple click action
let hasDragged = false;

// Dynamically create or retrieve the Space Capsule element
function initAIInspirationCapsule() {
    let capsule = document.getElementById('aiInspirationCapsule');
    if (!capsule) {
        capsule = document.createElement('div');
        capsule.id = 'aiInspirationCapsule';
        capsule.className = 'ai-inspiration-capsule';
        capsule.title = '✨ AI 灵感太空舱 - 鼠标悬停可往下展开/点击打开创意时光机';
        
        capsule.innerHTML = `
            <div class="capsule-led-pulse" id="capsuleLed" style="margin-right: 2px;"></div>
            <div class="capsule-glowing-emblem" style="width: 28px; height: 28px; font-size: 11px;">
                <i class="fas fa-rocket" style="font-size: 11px;"></i>
            </div>
            <div style="font-size: 11px; font-weight: 700; color: rgba(226, 232, 240, 0.95); letter-spacing: 1px; text-transform: uppercase; white-space: nowrap; margin-left: 6px; margin-right: 6px;">AI 灵感舱</div>
            <div id="capsuleCountBadge" style="width: 18px; height: 18px; border-radius: 50%; background: #6366f1; color: white; font-size: 10px; display: flex; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 0 6px rgba(99, 102, 241, 0.4); display: none;">0</div>
        `;
        
        const container = document.querySelector('.container') || document.body;
        container.appendChild(capsule);
        
        // Tooltip Speech Bubble
        const tooltip = document.createElement('div');
        tooltip.id = 'capsuleSpeechTooltip';
        tooltip.className = 'capsule-speech-tooltip';
        tooltip.style.cursor = 'pointer';
        tooltip.title = '✨ 点击立即开启 AI 创意与决策舱';
        tooltip.innerHTML = `<i class="fas fa-bolt" style="color: #a855f7;"></i> <span id="tooltipText">AI 已为当前图层准备了奇妙灵感</span>`;
        container.appendChild(tooltip);

        // Hover events for drop-down integration
        capsule.addEventListener('mouseenter', () => {
            isHoveringCapsule = true;
            clearTimeout(hideDelayTimer);
            clearTimeout(hoverDelayTimer);

            hoverDelayTimer = setTimeout(() => {
                const itemId = state.currentActiveWorkbenchItemId;
                if (itemId) {
                    showFloatingFusionEditor(itemId);
                    tooltip.classList.remove('visible');
                }
            }, 300); // 300ms flat buffer delay
        });

        capsule.addEventListener('mouseleave', () => {
            isHoveringCapsule = false;
            clearTimeout(hoverDelayTimer);
            hideDelayTimer = setTimeout(() => {
                if (!isHoveringCapsule && !isHoveringDrawer) {
                    hideFloatingFusionEditor();
                }
            }, 400); // 400ms transition cushion space
        });

        // Clicking the tooltip also opens the drawer immediately
        tooltip.addEventListener('click', (e) => {
            e.stopPropagation();
            const itemId = state.currentActiveWorkbenchItemId;
            if (itemId) {
                showFloatingFusionEditor(itemId);
                tooltip.classList.remove('visible');
            } else {
                showFeedbackToast("💡 请先选中您需要灵感加工的画布图层");
            }
        });

        // Click handler: opens right drawer if not just finished dragging
        capsule.addEventListener('click', (e) => {
            e.stopPropagation();
            if (hasDragged) {
                hasDragged = false; // consume drag
                return;
            }
            const itemId = state.currentActiveWorkbenchItemId;
            if (itemId) {
                showFloatingFusionEditor(itemId);
                // Hide bubble tooltip on open
                tooltip.classList.remove('visible');
            } else {
                showFeedbackToast("💡 请先选中您需要灵感加工的画布图层");
            }
        });

        // Long press/Normal drag anywhere support
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        let capStartLeft = 0;
        let capStartTop = 0;

        capsule.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return; // Only left click drag
            isDragging = true;
            hasDragged = false;
            
            const rect = capsule.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            capStartLeft = rect.left;
            capStartTop = rect.top;

            // Instantly remove transition for snappy, high-performance dragging
            capsule.style.transition = 'none';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const dx = e.clientX - startX;
            const dy = e.clientY - startY;

            // Trigger drag state if moved more than 4px
            if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
                hasDragged = true;
            }

            if (hasDragged) {
                let newLeft = capStartLeft + dx;
                let newTop = capStartTop + dy;

                // Keep capsule bounds inside the viewport safely
                const margin = 10;
                const maxLeft = window.innerWidth - capsule.offsetWidth - margin;
                const maxTop = window.innerHeight - capsule.offsetHeight - margin;

                newLeft = Math.max(margin, Math.min(newLeft, maxLeft));
                newTop = Math.max(margin, Math.min(newTop, maxTop));

                capsule.style.left = `${newLeft}px`;
                capsule.style.top = `${newTop}px`;
                capsule.style.right = 'auto';

                // Real-time tooltip hide to keep drag area clear
                tooltip.classList.remove('visible');
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                // Restore default smooth CSS transitions / springiness
                capsule.style.transition = '';
                
                // Update final positioning coordinates
                if (hasDragged) {
                    updateTooltipPosition();
                }
            }
        });

        // Click on workspace/blank space collapses the dropdown panel immediately
        document.addEventListener('click', (e) => {
            const cap = document.getElementById('aiInspirationCapsule');
            const ed = document.getElementById('floatingFusionEditor');
            if (cap && cap.contains(e.target)) return;
            if (ed && ed.contains(e.target)) return;
            hideFloatingFusionEditor();
        });
    }
}

// Run capsule bootstrap
initAIInspirationCapsule();

/**
 * Triggered automatically on selected layer click. Fired from showWorkbenchToolbox hook.
 */
window.triggerCapsuleAlert = function(itemId) {
    initAIInspirationCapsule();
    const item = workbenchItems.get(itemId);
    if (!item) return;

    // Retrieve suggestions counts from SemanticRecommender matching asset
    const { recommended, others } = SemanticRecommender.recommend(item);
    const totalInspirations = (recommended ? recommended.length : 0) + (others ? others.length : 0) + 1; // 1 represents the active base or intent variants

    const capsule = document.getElementById('aiInspirationCapsule');
    const led = document.getElementById('capsuleLed');
    const badge = document.getElementById('capsuleCountBadge');
    const tooltip = document.getElementById('capsuleSpeechTooltip');
    const tooltipText = document.getElementById('tooltipText');

    if (capsule && led && badge) {
        // Shine capsule alerts state
        capsule.classList.add('active-bright');
        led.classList.add('pulsing');
        badge.style.display = 'flex';
        badge.textContent = totalInspirations || '4';

        // Play visual micro alarm with elegant bubble popup text
        if (tooltip && tooltipText) {
            tooltipText.innerHTML = `AI 已为当前图层准备了 <strong>${totalInspirations || 4} </strong> 个奇妙灵感 ✨`;
            updateTooltipPosition();
            tooltip.classList.add('visible');
            
            // Fades away nicely after 4.5 seconds to remain fully clean & minimal
            clearTimeout(window.capsuleTooltipTimeout);
            window.capsuleTooltipTimeout = setTimeout(() => {
                tooltip.classList.remove('visible');
            }, 4500);
        }

        // De-alarm pulsing flash after 6 seconds to remain eye-safe
        clearTimeout(window.capsuleLedTimeout);
        window.capsuleLedTimeout = setTimeout(() => {
            led.classList.remove('pulsing');
            capsule.classList.remove('active-bright');
        }, 6000);
    }
};

const DEFAULT_FUSION_PROPS = {
    brightness: 100,
    contrast: 100,
    saturation: 100,
    blur: 0,
    hueRotate: 0,
    grayscale: 0,
    sepia: 0
};

function isExtractedLayerItem(item) {
    if (!item) return false;
    return !!(item.parentId && (
        item.type === 'layer-explode' ||
        item.type === 'layer-extract' ||
        item.type === 'isolated-edit' ||
        item.type === 'extraction'
    ));
}

window.isExtractedLayerItem = isExtractedLayerItem;

function hasSemanticLayerData(item) {
    return !!(item?.hasFullSemanticAnalysis || (Array.isArray(item?.layers) && item.layers.length > 0));
}

function hasMeaningfulFusionChanges(props = {}) {
    return Object.entries(DEFAULT_FUSION_PROPS).some(([key, value]) => (props?.[key] ?? value) !== value);
}

function getMeaningfulVariants(itemId) {
    return evolutionEngine
        .getVariants(itemId)
        .filter(variant => Object.keys(variant?.stateProps || {}).length > 0);
}

function flashDrawerSection(editor, sectionId) {
    const section = editor.querySelector(`#${sectionId}`);
    if (!section) return;
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    section.classList.remove('drawer-section-flash');
    void section.offsetWidth;
    section.classList.add('drawer-section-flash');
    setTimeout(() => section.classList.remove('drawer-section-flash'), 900);
}

function syncFusionControls(itemId, restoredProps) {
    const syncPanels = [document.getElementById('floatingFusionEditor'), document.getElementById('fusionEditPanel')];
    syncPanels.forEach(panel => {
        if (!panel) return;
        Object.entries(restoredProps).forEach(([key, value]) => {
            const slider = panel.querySelector(`input[data-prop="${key}"]`);
            if (slider) {
                slider.value = value;
                const valBadge = panel.querySelector(`.val-${key}`);
                if (valBadge) valBadge.textContent = value + slider.dataset.unit;
            }
        });
    });
}

function restoreEvolutionVariant(itemId, variant) {
    const ws = window.mvrRuntime?.getCurrentWorkspace();
    if (!ws?.dispatcher || !variant) return;

    const restoredProps = JSON.parse(JSON.stringify(variant.stateProps || {}));
    if (Object.keys(restoredProps).length === 0) {
        Object.assign(restoredProps, DEFAULT_FUSION_PROPS);
    }

    ws.dispatcher.dispatch({
        type: 'UPDATE_FUSION',
        intent: `[AI Insight Panel] Restored state to "${variant.intent}"`,
        payload: {
            uid: itemId,
            fusionProperties: restoredProps
        }
    });

    syncFusionControls(itemId, restoredProps);
    showFeedbackToast(`⏱️ 已回到灵感节点: "${variant.intent}"`);
}

function applySuggestionPattern(pattern, itemId) {
    const ws = window.mvrRuntime?.getCurrentWorkspace();
    if (!ws?.dispatcher || !pattern) return;

    try {
        pattern.apply(ws.dispatcher, itemId);
        showFeedbackToast(`🌱 已套用建议: ${pattern.name}`);
        setTimeout(() => {
            showFloatingFusionEditor(itemId);
        }, 120);
    } catch (error) {
        console.error('Error applying preset design strategy:', error);
    }
}

function buildInsightSummary(item, layerName, variants, suggestions) {
    const statusChips = [];
    const semanticReady = hasSemanticLayerData(item);
    const extracted = isExtractedLayerItem(item);
    const tuned = hasMeaningfulFusionChanges(item?.fusionProperties);

    if (extracted) statusChips.push('已提取图层');
    if (semanticReady) statusChips.push(`已识别 ${item.layers?.length || 0} 个可编辑部件`);
    if (variants.length > 0) statusChips.push(`有 ${variants.length} 条灵感轨迹`);
    if (tuned) statusChips.push('最近做过风格微调');
    if (suggestions.length > 0) statusChips.push(`AI 找到 ${suggestions.length} 条推荐方向`);
    if (statusChips.length === 0) statusChips.push('适合先看 AI 推荐再动手');

    let summary = `${layerName} 现在更适合先做轻量创意判断，再进入具体编辑。`;
    if (semanticReady && extracted) {
        summary = `${layerName} 是一个已提取且已分析过的主体图层，适合直接做图层编辑、局部重绘和风格延展。`;
    } else if (semanticReady) {
        summary = `${layerName} 已具备语义结构，可以直接查看可编辑部件，按区域做更聪明的修改。`;
    } else if (extracted) {
        summary = `${layerName} 是一个独立图层，适合先进入图层编辑，必要时让 AI 自动补做语义分析。`;
    } else if (variants.length > 0) {
        summary = `${layerName} 最近已经形成了创意轨迹，适合沿着已有方向继续推演，而不是重新从零开始。`;
    } else if (suggestions.length > 0) {
        summary = `${layerName} 目前最适合先看几条 AI 判断出的方向，再决定是否进入更深的编辑。`;
    }

    return { summary, statusChips };
}

function buildPromptSeeds(item, variants, suggestions) {
    const seeds = [];
    if (variants[0]?.intent) {
        seeds.push(`${variants[0].intent}，延续这个方向`);
    }
    if (suggestions[0]?.name) {
        seeds.push(`沿着“${suggestions[0].name}”做一个更高级的新版本`);
    }
    if (hasSemanticLayerData(item)) {
        seeds.push('保留主体结构，只优化材质和氛围');
    } else {
        seeds.push('先保持构图不变，给我 3 个更聪明的版本');
    }
    return seeds.slice(0, 3);
}

/**
 * Creates, updates or awakens the merged AI co-creation right-side sliding Drawer
 */
export function showFloatingFusionEditor(itemId, x, y, width) {
    const item = workbenchItems.get(itemId);
    console.log('showFloatingFusionEditor called in Drawer mode for item:', itemId, item);
    if (!item) {
        return;
    }

    // Hide toolbox of active elements to prevent visual clashing with the inspiration lists
    const toolbox = document.getElementById('workbenchToolbox') || window.workbenchToolbox;
    if (toolbox) {
        toolbox.style.display = 'none';
    }

    let editor = document.getElementById('floatingFusionEditor');
    if (!editor) {
        editor = document.createElement('div');
        editor.id = 'floatingFusionEditor';
        editor.className = 'ai-smart-drawer';
        
        // Prevent event propagation so clicking inside does not deselect
        editor.onmousedown = (e) => {
            e.stopPropagation();
        };
        editor.onclick = (e) => {
            e.stopPropagation();
        };

        // Hover events so moving cursor inside keeps it open
        editor.addEventListener('mouseenter', () => {
            isHoveringDrawer = true;
            clearTimeout(hideDelayTimer);
        });

        editor.addEventListener('mouseleave', () => {
            isHoveringDrawer = false;
            hideDelayTimer = setTimeout(() => {
                if (!isHoveringCapsule && !isHoveringDrawer) {
                    hideFloatingFusionEditor();
                }
            }, 400); // 400ms buffer delay
        });

        const container = document.querySelector('.container') || document.body;
        container.appendChild(editor);
    }

    editor.removeAttribute('data-collapsing');

    // Slide out expanded Drawer downwards with a smooth, browser-registered transition!
    requestAnimationFrame(() => {
        // Force DOM layout calculation so browser registers the initial transform/opacity state of newer DOM nodes
        editor.offsetHeight; 
        editor.classList.add('expanded');
    });
    
    // In Dropdown mode, the capsule stays visible. Just hide the bubble tooltip.
    const tooltip = document.getElementById('capsuleSpeechTooltip');
    if (tooltip) tooltip.classList.remove('visible');

    const layerName = item.layerName || item.name || '子图图层';

    const ws = window.mvrRuntime?.getCurrentWorkspace();
    if (!ws) return;

    let actualAsset = item;
    if (state && state.workbenchItems) {
        actualAsset = state.workbenchItems.get(itemId) || item;
    }

    const activeAsset = ws.currentState.assetRegistry.get(itemId) || actualAsset;
    const props = (activeAsset && activeAsset.fusionProperties) ? activeAsset.fusionProperties : DEFAULT_FUSION_PROPS;

    // Get recommendations matching this asset
    const { recommended, others } = SemanticRecommender.recommend(actualAsset);
    const allSuggestions = [...recommended, ...others];
    const meaningfulVariants = getMeaningfulVariants(itemId);
    const lastVariant = meaningfulVariants[meaningfulVariants.length - 1] || null;
    const topSuggestion = recommended[0] || others[0] || null;
    const { summary, statusChips } = buildInsightSummary(actualAsset, layerName, meaningfulVariants, allSuggestions);
    const promptSeeds = buildPromptSeeds(actualAsset, meaningfulVariants, allSuggestions);
    const semanticReady = hasSemanticLayerData(actualAsset);
    const primaryActions = [
        {
            id: 'layer-edit',
            icon: 'fa-layer-group',
            title: '图层编辑',
            description: semanticReady ? '直接查看、拆分或局部重绘语义图层' : '进入图层编辑，必要时自动补做智能分析',
            handler: () => {
                if (typeof window.showLayerManagerModal === 'function') {
                    window.showLayerManagerModal(itemId, !semanticReady);
                } else {
                    showFeedbackToast('图层编辑器暂时不可用');
                }
            }
        },
        {
            id: 'extend-style',
            icon: 'fa-sparkles',
            title: '一键风格延展',
            description: topSuggestion ? `沿着“${topSuggestion.name}”方向快速生成一个新版本` : '用一句意图让 AI 为当前图层延展新版本',
            handler: () => {
                if (topSuggestion) {
                    applySuggestionPattern(topSuggestion, itemId);
                    return;
                }
                const quickInput = editor.querySelector('#fusionQuickInput');
                if (quickInput) {
                    currentMode = 'isolated';
                    quickInput.value = `${layerName} 延展出一个更有表现力的新版本`;
                    quickInput.focus();
                    flashDrawerSection(editor, 'insightPromptComposer');
                }
            }
        }
    ];

    if (lastVariant) {
        primaryActions.push({
            id: 'continue-idea',
            icon: 'fa-clock-rotate-left',
            title: '延续上次灵感',
            description: `沿着“${lastVariant.intent}”继续推一个相邻版本`,
            handler: () => {
                const quickInput = editor.querySelector('#fusionQuickInput');
                if (quickInput) {
                    quickInput.value = `${lastVariant.intent}，延续这个方向再做一个版本`;
                    quickInput.focus();
                    flashDrawerSection(editor, 'insightPromptComposer');
                }
            }
        });
    } else if (!semanticReady) {
        primaryActions.push({
            id: 'analyze-layers',
            icon: 'fa-diagram-project',
            title: '智能分析图层',
            description: '先识别主体、部件和可拆分结构，再决定怎么改',
            handler: () => {
                if (typeof window.showLayerManagerModal === 'function') {
                    window.showLayerManagerModal(itemId, true);
                }
            }
        });
    } else {
        primaryActions.push({
            id: 'see-recommendations',
            icon: 'fa-lightbulb',
            title: '看看 AI 推荐',
            description: allSuggestions.length > 0 ? `这里有 ${Math.min(allSuggestions.length, 3)} 条更贴近当前图层的方向` : '如果当前没有推荐，可以直接补一句需求',
            handler: () => {
                if (allSuggestions.length > 0) {
                    flashDrawerSection(editor, 'insightRecommendations');
                } else {
                    const quickInput = editor.querySelector('#fusionQuickInput');
                    if (quickInput) {
                        quickInput.focus();
                        flashDrawerSection(editor, 'insightPromptComposer');
                    }
                }
            }
        });
    }

    // Build Drawer visual compartments
    editor.innerHTML = `
        <div style="background: #ffffff; padding: 22px 24px 16px 24px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; border-bottom: 1.5px solid rgba(226, 232, 240, 0.8); flex-shrink: 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background: #f5f3ff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(124, 58, 237, 0.15); border: 1px solid rgba(124, 58, 237, 0.1); flex-shrink: 0;">
                    <i class="fas fa-magic" style="color: #7c3aed; font-size: 13px;"></i>
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 15px; font-weight: 800; color: #1e293b; letter-spacing: 0.02em; font-family: var(--font-sans);">AI 理解与决策面板</h4>
                    <span style="font-size: 10.5px; color: #64748b; font-weight: 500; display: flex; align-items: center; gap: 4px; margin-top: 2px;">
                        先判断你最可能想做什么，再展开具体编辑
                    </span>
                </div>
            </div>
            <button class="close-editor-btn" style="background: #f1f5f9; border: none; cursor: pointer; color: #64748b; width: 34px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; transition: all 0.2s;" title="收起创意思路舱">
                <span style="font-size: 13.5px; font-weight: 900; letter-spacing: -2px; color: #7c3aed; font-family: monospace;">&gt;&gt;</span>
            </button>
        </div>

        <div class="drawer-scroll-body" style="flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 18px; scrollbar-width: thin; background: #fcfcfd;">
            <section class="ai-insight-card">
                <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:10px;">
                    <div style="display:flex; flex-direction:column; gap:6px;">
                        <div style="font-size:11px; color:#7c3aed; font-weight:800; letter-spacing:0.05em; text-transform:uppercase; display:flex; align-items:center; gap:6px;">
                            <i class="fas fa-brain"></i> AI 对当前图层的理解
                        </div>
                        <div style="font-size:13px; line-height:1.65; color:#312e81; font-weight:600;">
                            ${summary}
                        </div>
                    </div>
                </div>
                <div class="ai-status-row">
                    ${statusChips.map(chip => `<span class="ai-status-chip">${chip}</span>`).join('')}
                </div>
                <div id="insightPromptComposer" style="display:flex; flex-direction:column; gap:8px;">
                    <div class="ai-mode-toggle">
                        <button id="modeIsolated" class="${currentMode === 'isolated' ? 'active' : ''}">
                            <i class="fas fa-box" style="font-size:10px;"></i> 生成新版本
                        </button>
                        <button id="modeFusion" class="${currentMode === 'fusion' ? 'active' : ''}">
                            <i class="fas fa-object-group" style="font-size:10px;"></i> 同步到当前图层
                        </button>
                    </div>
                    <div class="ai-prompt-input-row">
                        <input type="text" id="fusionQuickInput" placeholder="补一句你的想法，例如：更像高级软装样片..." />
                        <button id="fusionQuickApply" title="发送意图">
                            <i class="fas fa-paper-plane" style="font-size:11px;"></i>
                        </button>
                    </div>
                    <div class="ai-prompt-chip-row">
                        ${promptSeeds.map(seed => `<button class="ai-prompt-chip" data-seed="${seed}">${seed}</button>`).join('')}
                    </div>
                </div>
            </section>

            <section class="ai-section-block">
                <div class="ai-section-title">
                    <i class="fas fa-bolt"></i> AI 认为你现在最可能要做的事
                </div>
                <div class="ai-primary-actions">
                    ${primaryActions.map(action => `
                        <button class="ai-primary-action" data-primary-action="${action.id}">
                            <span class="ai-primary-action-icon">
                                <i class="fas ${action.icon}" style="font-size:14px;"></i>
                            </span>
                            <span class="ai-primary-action-copy">
                                <strong>${action.title}</strong>
                                <span>${action.description}</span>
                            </span>
                            <i class="fas fa-angle-right" style="color:#a78bfa; font-size:11px;"></i>
                        </button>
                    `).join('')}
                </div>
            </section>

            <section class="ai-section-block" id="insightRecommendations">
                <div class="ai-section-title">
                    <i class="fas fa-lightbulb"></i> 贴近当前图层的推荐方向
                </div>
                <div id="recContainer" style="display:flex; flex-direction:column; gap:8px;"></div>
            </section>

            <section class="ai-accordion">
                <button id="manualSlidersToggle" class="ai-accordion-btn">
                    <span class="ai-accordion-btn-label">
                        <i class="fas fa-sliders-h"></i> 手动色调与光影微调
                    </span>
                    <i id="manualSlidersChevron" class="fas ${isSlidersExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}" style="font-size:11px; color:#7c3aed;"></i>
                </button>
                <div id="manualSlidersPanel" class="ai-accordion-panel ${isSlidersExpanded ? 'expanded' : ''}">
                    ${renderSlider('brightness', '亮度', props.brightness, 0, 200, '%')}
                    ${renderSlider('contrast', '对比度', props.contrast, 0, 200, '%')}
                    ${renderSlider('saturation', '饱和度', props.saturation, 0, 200, '%')}
                    ${renderSlider('hueRotate', '色相旋转', props.hueRotate || 0, 0, 360, '°')}
                    ${renderSlider('grayscale', '灰度', props.grayscale || 0, 0, 100, '%')}
                    ${renderSlider('sepia', '怀旧/褐变', props.sepia || 0, 0, 100, '%')}
                    ${renderSlider('blur', '高斯模糊', props.blur, 0, 20, 'px')}
                </div>

                <button id="historyToggle" class="ai-accordion-btn">
                    <span class="ai-accordion-btn-label">
                        <i class="fas fa-history"></i> 创意时光机
                    </span>
                    <i id="historyChevron" class="fas ${isHistoryExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}" style="font-size:11px; color:#7c3aed;"></i>
                </button>
                <div id="bubbleEvoSection" class="ai-accordion-panel ${isHistoryExpanded ? 'expanded' : ''}">
                    <div id="bubbleEvoContainer" style="position: relative; display: flex; flex-direction: column; gap: 8px;"></div>
                </div>

                <button id="recommendationArchiveToggle" class="ai-accordion-btn">
                    <span class="ai-accordion-btn-label">
                        <i class="fas fa-layer-group"></i> 完整推荐档案
                    </span>
                    <i id="recommendationArchiveChevron" class="fas ${isRecommendationArchiveExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}" style="font-size:11px; color:#7c3aed;"></i>
                </button>
                <div id="recommendationArchivePanel" class="ai-accordion-panel ${isRecommendationArchiveExpanded ? 'expanded' : ''}">
                    <div id="allSuggestionsContainer" style="display:flex; flex-direction:column; gap:8px;"></div>
                </div>
            </section>
        </div>
    `;

    // Hook Close Button to retract drawer smoothly
    editor.querySelector('.close-editor-btn').onclick = (e) => {
        e.stopPropagation();
        hideFloatingFusionEditor();
        showFeedbackToast('🧬 AI 创意与决策舱已收起');
    };

    const input = editor.querySelector('#fusionQuickInput');
    const applyBtn = editor.querySelector('#fusionQuickApply');
    const modeIsolatedBtn = editor.querySelector('#modeIsolated');
    const modeFusionBtn = editor.querySelector('#modeFusion');
    const primaryActionButtons = editor.querySelectorAll('[data-primary-action]');
    const promptSeedButtons = editor.querySelectorAll('[data-seed]');

    const syncModeButtons = () => {
        modeIsolatedBtn.classList.toggle('active', currentMode === 'isolated');
        modeFusionBtn.classList.toggle('active', currentMode === 'fusion');
    };

    modeIsolatedBtn.onclick = (e) => {
        e.stopPropagation();
        currentMode = 'isolated';
        syncModeButtons();
    };

    modeFusionBtn.onclick = (e) => {
        e.stopPropagation();
        currentMode = 'fusion';
        syncModeButtons();
    };

    // Execute NL intent triggers
    const executeNLIntent = async () => {
        const promptText = input.value.trim();
        if (!promptText) {
            alert('请输入修改指令');
            return;
        }

        // Use workspace dispatcher so it remains in timeline historical stream
        if (ws && ws.dispatcher) {
            ws.dispatcher.dispatch({
                type: currentMode === 'isolated' ? 'CO_CREATE_ISOLATED' : 'CO_CREATE_SYNC',
                intent: promptText,
                payload: {
                    uid: itemId,
                    prompt: promptText
                }
            });
            showFeedbackToast(`🪄 已部署创意意图: "${promptText}"`);
            input.value = '';
            
            setTimeout(() => {
                showFloatingFusionEditor(itemId);
            }, 120);
        } else {
            const modeText = currentMode === 'isolated' ? '独立资产生成中' : '同步至原图';
            editor.innerHTML = `<div style="text-align:center; padding: 40px; font-size: 13px; color: #64748b;"><i class="fas fa-spinner fa-spin" style="margin-right: 6px;"></i> 正在${modeText}...</div>`;
            
            if (currentMode === 'isolated') {
                if (typeof window.handleIsolatedAssetEdit === 'function') {
                    await window.handleIsolatedAssetEdit(itemId, promptText);
                }
            } else {
                if (typeof window.handleQuickFusionSync === 'function') {
                    await window.handleQuickFusionSync(itemId, promptText);
                }
            }
            hideFloatingFusionEditor();
        }
    };

    primaryActions.forEach(action => {
        const btn = editor.querySelector(`[data-primary-action="${action.id}"]`);
        if (btn) {
            btn.onclick = (e) => {
                e.stopPropagation();
                action.handler();
            };
        }
    });

    applyBtn.onclick = (e) => {
        e.stopPropagation();
        executeNLIntent();
    };

    input.onkeydown = (e) => {
        if (e.key === 'Enter') {
            e.stopPropagation();
            executeNLIntent();
        }
    };

    promptSeedButtons.forEach(tag => {
        tag.onclick = (e) => {
            e.stopPropagation();
            const tagVal = tag.getAttribute('data-seed');
            if (tagVal) {
                input.value = tagVal;
                input.focus();
            }
        };
    });

    const slidersToggle = editor.querySelector('#manualSlidersToggle');
    const slidersPanel = editor.querySelector('#manualSlidersPanel');
    const slidersChevron = editor.querySelector('#manualSlidersChevron');
    const historyToggle = editor.querySelector('#historyToggle');
    const historyPanel = editor.querySelector('#bubbleEvoSection');
    const historyChevron = editor.querySelector('#historyChevron');
    const recommendationArchiveToggle = editor.querySelector('#recommendationArchiveToggle');
    const recommendationArchivePanel = editor.querySelector('#recommendationArchivePanel');
    const recommendationArchiveChevron = editor.querySelector('#recommendationArchiveChevron');
    
    if (slidersToggle && slidersPanel) {
        slidersToggle.onclick = (e) => {
            e.stopPropagation();
            isSlidersExpanded = !isSlidersExpanded;
            if (isSlidersExpanded) {
                slidersPanel.classList.add('expanded');
            } else {
                slidersPanel.classList.remove('expanded');
            }
            slidersChevron.className = `fas ${isSlidersExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}`;
        };
    }

    if (historyToggle && historyPanel) {
        historyToggle.onclick = (e) => {
            e.stopPropagation();
            isHistoryExpanded = !isHistoryExpanded;
            historyPanel.classList.toggle('expanded', isHistoryExpanded);
            historyChevron.className = `fas ${isHistoryExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}`;
        };
    }

    if (recommendationArchiveToggle && recommendationArchivePanel) {
        recommendationArchiveToggle.onclick = (e) => {
            e.stopPropagation();
            isRecommendationArchiveExpanded = !isRecommendationArchiveExpanded;
            recommendationArchivePanel.classList.toggle('expanded', isRecommendationArchiveExpanded);
            recommendationArchiveChevron.className = `fas ${isRecommendationArchiveExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}`;
        };
    }

    // Hook up slider oninput events inside the drawer
    const sliders = editor.querySelectorAll('#manualSlidersPanel input[type="range"]');
    sliders.forEach(slider => {
        slider.oninput = (e) => {
            e.stopPropagation();
            const type = e.target.dataset.prop;
            const val = parseFloat(e.target.value);
            const unit = e.target.dataset.unit;
            
            // Update value display text inside the drawer
            const valDisplay = editor.querySelector(`.val-${type}`);
            if (valDisplay) valDisplay.textContent = val + unit;

            // Update value display text inside sidebar panel (#fusionEditPanel) as well if it exists
            const sidebarPanel = document.getElementById('fusionEditPanel');
            if (sidebarPanel) {
                const sidebarSlider = sidebarPanel.querySelector(`input[data-prop="${type}"]`);
                if (sidebarSlider) sidebarSlider.value = val;
                const sidebarDisplay = sidebarPanel.querySelector(`.val-${type}`);
                if (sidebarDisplay) sidebarDisplay.textContent = val + unit;
            }

            // Gather all slider values from drawer safely & preservation
            const currentAsset = ws.currentState.assetRegistry.get(itemId);
            const existingProps = currentAsset ? (currentAsset.fusionProperties || {}) : {};
            const newProps = { ...existingProps };

            sliders.forEach(s => {
                newProps[s.dataset.prop] = parseFloat(s.value);
            });

            // 60FPS UI Synchronous Filter update for selected layer
            const itemEl = document.querySelector(`[data-item-id="${itemId}"]`);
            if (itemEl) {
                const canvasLayers = itemEl.querySelectorAll('.canvas-layer');
                if (canvasLayers && canvasLayers.length > 0) {
                    canvasLayers.forEach(layerEl => {
                        if (layerEl.classList.contains('selected')) {
                            const img = layerEl.querySelector('img');
                            const target = img || layerEl;
                            target.style.filter = `brightness(${newProps.brightness}%) contrast(${newProps.contrast}%) saturate(${newProps.saturation}%) blur(${newProps.blur}px) hue-rotate(${newProps.hueRotate || 0}deg) grayscale(${newProps.grayscale || 0}%) sepia(${newProps.sepia || 0}%)`;
                        }
                    });
                } else {
                    const img = itemEl.querySelector('img');
                    if (img) {
                        img.style.filter = `brightness(${newProps.brightness}%) contrast(${newProps.contrast}%) saturate(${newProps.saturation}%) blur(${newProps.blur}px) hue-rotate(${newProps.hueRotate || 0}deg) grayscale(${newProps.grayscale || 0}%) sepia(${newProps.sepia || 0}%)`;
                    }
                }
            }
            
            // Dispatch SILENTLY for persistent live preview state in background
            if (ws.dispatcher) {
                try {
                    ws.dispatcher.dispatch({
                        type: 'UPDATE_FUSION',
                        payload: {
                            uid: itemId,
                            fusionProperties: newProps
                        }
                    });
                } catch(err) { console.error("Slider dispatch error from drawer", err); }
            }
        };
    });

    const recContainer = editor.querySelector('#recContainer');
    const recommendedList = allSuggestions.slice(0, 3);
    if (recommendedList.length === 0) {
        recContainer.innerHTML = `<div class="ai-empty-block">当前还没有足够强的推荐，你可以补一句需求，让 AI 更快理解你的意图。</div>`;
    } else {
        recommendedList.forEach(pattern => {
            recContainer.appendChild(renderSuggestionCard(pattern, true, itemId));
        });
    }

    const allSuggestionsContainer = editor.querySelector('#allSuggestionsContainer');
    if (allSuggestions.length === 0) {
        allSuggestionsContainer.innerHTML = `<div class="ai-empty-block">完整推荐档案里暂时还是空的，等你做几次编辑后，这里会越来越懂你。</div>`;
    } else {
        allSuggestions.forEach((pattern, index) => {
            allSuggestionsContainer.appendChild(renderSuggestionCard(pattern, index < recommended.length, itemId));
        });
    }

    const renderTrackHistory = () => {
        const evoContainer = editor.querySelector('#bubbleEvoContainer');
        if (!evoContainer) return;

        const variants = evolutionEngine.getVariants(itemId);

        if (variants.length > 0) {
            evoContainer.innerHTML = '<div class="history-branch-line"></div>';

            variants.forEach((variant, idx) => {
                const isLast = idx === variants.length - 1;
                const activeProps = item.fusionProperties || {};
                
                // Compare values to see what Git node match current parameters
                const isCurrentNode = (JSON.stringify(variant.stateProps) === JSON.stringify(activeProps)) || (isLast && Object.keys(activeProps).length === 0);
                
                // Human readable time label
                const timeStr = new Date(variant.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                
                // Get thumbnail img src
                let imgSrc = 'https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?q=80&w=200&auto=format&fit=crop';
                const canvasImg = item.el ? item.el.querySelector('img') : null;
                if (canvasImg && canvasImg.src) {
                    imgSrc = canvasImg.src;
                }

                const node = document.createElement('div');
                node.className = `history-branch-node ${isCurrentNode ? 'active' : ''}`;
                
                let filterString = `
                    brightness(${variant.stateProps.brightness !== undefined ? variant.stateProps.brightness : 100}%)
                    contrast(${variant.stateProps.contrast !== undefined ? variant.stateProps.contrast : 100}%)
                    saturate(${variant.stateProps.saturation !== undefined ? variant.stateProps.saturation : 100}%)
                    grayscale(${variant.stateProps.grayscale || 0}%)
                    sepia(${variant.stateProps.sepia || 0}%)
                    hue-rotate(${variant.stateProps.hueRotate || 0}deg)
                    blur(${variant.stateProps.blur || 0}px)
                `;

                node.innerHTML = `
                    <div class="branch-node-indicator"></div>
                    <div class="branch-node-hover-zone">
                        <div style="position: relative; width: 34px; height: 34px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(0,0,0,0.08); background: #f1f5f9; flex-shrink: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
                            <img src="${imgSrc}" style="width: 100%; height: 100%; object-fit: cover; filter: ${filterString}; pointer-events: none;">
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0;">
                            <span style="font-size: 11.5px; font-weight: 600; color: #1e293b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${variant.intent}">
                                ${variant.intent}
                            </span>
                            <span style="font-size: 9px; color: #94a3b8; font-family: var(--font-mono); font-weight: 500;">
                                ${timeStr}
                            </span>
                        </div>
                        <span style="font-size: 9.5px; font-weight: 700; color: #7c3aed; padding: 2px 6px; background: rgba(124,58,237,0.06); border-radius: 6px; flex-shrink: 0;">
                            ${isCurrentNode ? '当前' : '回滚'}
                        </span>
                    </div>
                `;

                node.onclick = (e) => {
                    e.stopPropagation();
                    const restoredProps = JSON.parse(JSON.stringify(variant.stateProps));
                    if (Object.keys(restoredProps).length === 0) {
                        Object.assign(restoredProps, { brightness: 100, contrast: 100, saturation: 100, blur: 0, hueRotate: 0, grayscale: 0, sepia: 0 });
                    }

                    ws.dispatcher.dispatch({
                        type: 'UPDATE_FUSION',
                        intent: `[Creative Memory Track] Restored state to "${variant.intent}"`,
                        payload: {
                            uid: itemId,
                            fusionProperties: restoredProps
                        }
                    });

                    showFeedbackToast(`⏱️ 创意决策已跳跃至: "${variant.intent}"`);
                    
                    setTimeout(() => {
                        renderTrackHistory();
                    }, 40);

                    syncFusionControls(itemId, restoredProps);
                };

                evoContainer.appendChild(node);
            });
            
            // Auto scroll container to bottom so latest is always visible
            setTimeout(() => { evoContainer.scrollTop = evoContainer.scrollHeight; }, 100);
        } else {
            evoContainer.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px 16px; background: #ffffff; border: 1.5px dashed rgba(124, 58, 237, 0.15); border-radius: 16px; color: #94a3b8; text-align: center; gap: 8px;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background: #f5f3ff; display: flex; align-items: center; justify-content: center; color: #7c3aed;">
                        <i class="fas fa-history" style="font-size: 15px;"></i>
                    </div>
                    <span style="font-size: 11.5px; font-weight: 600; color: #64748b;">图层还没有形成创意轨迹</span>
                </div>
            `;
        }
    };

    renderTrackHistory();
    syncModeButtons();
    setTimeout(() => input.focus(), 100);
}

/**
 * Renders a clickable elegant card layout for suggestions, featuring Napkin styling
 */
function renderSuggestionCard(pattern, isBestMatch, itemId) {
    const card = document.createElement('div');
    card.className = 'ai-suggestion-card';
    card.style.cssText = `
        padding: 10px 12px;
        background: ${isBestMatch ? '#ffffff' : '#fafafa'};
        border-radius: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 10px;
        box-sizing: border-box;
    `;

    card.innerHTML = `
        <div style="font-size: 20px; width: 34px; height: 34px; border-radius: 10px; background: ${isBestMatch ? '#e0e7ff' : '#f1f5f9'}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            ${pattern.emoji}
        </div>
        <div style="display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0;">
            <p style="margin: 0; font-size: 11.5px; font-weight: 600; color: #1e293b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                ${pattern.name}
            </p>
            <p style="margin: 0; font-size: 9.5px; color: #64748b; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${pattern.description}">
                ${pattern.description}
            </p>
        </div>
        <div style="font-size: 10px; color: ${isBestMatch ? '#6366f1' : '#94a3b8'}; padding: 4px; border-radius: 50%; opacity: 0.7;">
            <i class="fas fa-angle-right"></i>
        </div>
    `;

    card.onclick = (e) => {
        e.stopPropagation();
        applySuggestionPattern(pattern, itemId);
    };

    return card;
}

/**
 * Tiny Elegant design feedback toast inside the system
 */
function showFeedbackToast(msg) {
    const existing = document.getElementById('successFeedbackToast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'successFeedbackToast';
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(8px);
        color: white;
        padding: 10px 18px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 500;
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        animation: bubblePop 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    `;
    toast.innerHTML = `<i class="fas fa-magic" style="color: #6366f1;"></i> <span>${msg}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translate(-50%, 15px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 2800);
}

export function hideFloatingFusionEditor() {
    const el = document.getElementById('floatingFusionEditor');
    if (el && el.classList.contains('expanded')) {
        el.classList.remove('expanded');
        el.setAttribute('data-collapsing', 'true');
        setTimeout(() => {
            if (!el.classList.contains('expanded')) {
                el.remove();
            }
        }, 300); // 300ms transition matching CSS
    }

    // Now restore the workbench toolbox if an item is still actively selected!
    if (state.currentActiveWorkbenchItemId && typeof window.showWorkbenchToolbox === 'function') {
        window.showWorkbenchToolbox(state.currentActiveWorkbenchItemId);
    }
}

export function showPreciseEditDialog(itemId, box) {
    const item = workbenchItems.get(itemId);
    if (!item) return;

    const dialog = document.createElement('div');
    dialog.id = 'preciseEditDialog';
    dialog.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 400px;
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        z-index: 10002;
        display: flex;
        flex-direction: column;
        gap: 12px;
    `;

    dialog.innerHTML = `
        <div style="font-size: 16px; font-weight: 600; color: #2A5C82; display: flex; align-items: center; gap: 8px;">
            <i class="fas fa-magic"></i> 精准修图
        </div>
        <div style="font-size: 13px; color: #666;">
            已框选区域，请输入修改指令（如：沙发改成椅子、换成红色等）
        </div>
        <textarea id="preciseEditInput" placeholder="输入修改指令..." 
            style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; resize: vertical; outline: none; font-family: inherit;"></textarea>
        <div style="display: flex; gap: 10px; justify-content: flex-end;">
            <button id="preciseEditCancel" style="padding: 8px 16px; background: #f0f0f0; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;">取消</button>
            <button id="preciseEditApply" style="padding: 8px 16px; background: #2A5C82; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;">
                <i class="fas fa-check"></i> 应用修改
            </button>
        </div>
    `;

    document.body.appendChild(dialog);

    const input = dialog.querySelector('#preciseEditInput');
    const applyBtn = dialog.querySelector('#preciseEditApply');
    const cancelBtn = dialog.querySelector('#preciseEditCancel');

    cancelBtn.onclick = () => {
        dialog.remove();
        if (window.marmoLens) window.marmoLens.stopSelectionMode();
    };

    applyBtn.onclick = async () => {
        const promptText = input.value.trim();
        if (!promptText) {
            alert('请输入修改指令');
            return;
        }
        dialog.innerHTML = `<div style="text-align:center; padding: 20px; font-size: 14px;"><i class="fas fa-spinner fa-spin"></i> 正在生成精准修改...</div>`;
        if (typeof window.performPreciseEdit === 'function') {
            await window.performPreciseEdit(itemId, box, promptText);
        }
        dialog.remove();
    };

    setTimeout(() => input.focus(), 100);
}

window.showFloatingFusionEditor = showFloatingFusionEditor;
window.hideFloatingFusionEditor = hideFloatingFusionEditor;
window.showPreciseEditDialog = showPreciseEditDialog;
window.startPreciseEditMode = startPreciseEditMode;
