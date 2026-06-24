/**
 * Fusion Edit Panel - A side panel for adjusting Asset Fusion properties.
 * Following the de-logicized pattern: UI gathers intent, Dispatcher handles state.
 * Refactored to only contain direct manual controls (sliders), moving smart operations & history to the floating AI Suggestions.
 * Enhanced with Live-Select Sync & Element Type Tolerance.
 */

import { state } from '../core/state.js';
import { showToast } from '../core/utils.js';

export function createFusionPanel(assetId) {
    // Remove existing panel if any
    const existing = document.getElementById('fusionEditPanel');
    if (existing) existing.remove();

    const workspace = window.mvrRuntime?.getCurrentWorkspace();
    if (!workspace) return;

    const asset = workspace.currentState.assetRegistry.get(assetId);
    if (!asset) return;

    const props = asset.fusionProperties || { 
        brightness: 100, 
        contrast: 100, 
        saturation: 100, 
        blur: 0,
        hueRotate: 0,
        grayscale: 0,
        sepia: 0
    };

    const panel = document.createElement('div');
    panel.id = 'fusionEditPanel';
    panel.className = 'fusion-edit-panel';
    panel.dataset.currentAssetId = assetId;
    
    panel.style.cssText = `
        position: fixed;
        right: 20px;
        top: 80px;
        width: 280px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        z-index: 10005;
        font-family: var(--font-sans);
        display: flex;
        flex-direction: column;
        gap: 20px;
        animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        max-height: 80vh;
        overflow-y: auto;
    `;

    panel.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 12px;">
            <h3 class="panel-heading" style="margin: 0; font-size: 13px; font-weight: 700; color: #1e293b; letter-spacing: -0.01em; display: flex; align-items: center; gap: 6px;">
                <i class="fas fa-sliders-h" style="color: #6366f1;"></i> 
                <span>图层颜色调整 (Filters)</span>
            </h3>
            <button class="close-panel-btn" style="background: #f1f5f9; border: none; cursor: pointer; color: #64748b; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s;">
                <i class="fas fa-times" style="font-size: 11px;"></i>
            </button>
        </div>

        <div class="fusion-controls" style="display: flex; flex-direction: column; gap: 16px;">
            <div style="display: flex; flex-direction: column; gap: 14px;">
                ${renderSlider('brightness', '亮度', props.brightness, 0, 200, '%')}
                ${renderSlider('contrast', '对比度', props.contrast, 0, 200, '%')}
                ${renderSlider('saturation', '饱和度', props.saturation, 0, 200, '%')}
                ${renderSlider('hueRotate', '色相旋转', props.hueRotate || 0, 0, 360, '°')}
                ${renderSlider('grayscale', '灰度', props.grayscale || 0, 0, 100, '%')}
                ${renderSlider('sepia', '怀旧/褐变', props.sepia || 0, 0, 100, '%')}
                ${renderSlider('blur', '高斯模糊', props.blur, 0, 20, 'px')}
            </div>
        </div>
    `;

    document.body.appendChild(panel);

    // Event listeners
    panel.querySelector('.close-panel-btn').onclick = () => {
        panel.remove();
    };

    const sliders = panel.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        slider.oninput = (e) => {
            const currentAssetId = panel.dataset.currentAssetId || assetId;
            const type = e.target.dataset.prop;
            const val = parseFloat(e.target.value);
            const unit = e.target.dataset.unit;
            
            // Update value display
            const valDisplay = panel.querySelector(`.val-${type}`);
            if (valDisplay) valDisplay.textContent = val + unit;

            // Gather all slider values safely preserving other custom attributes like overlays
            const ws = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
            if (!ws) return;

            const currentAsset = ws.currentState.assetRegistry.get(currentAssetId);
            const existingProps = currentAsset ? (currentAsset.fusionProperties || {}) : {};
            const newProps = { ...existingProps };

            sliders.forEach(s => {
                if (!s.disabled) {
                    newProps[s.dataset.prop] = parseFloat(s.value);
                }
            });
            
            // Dispatch SILENTLY for live preview
            if (ws.dispatcher) {
                try {
                    ws.dispatcher.dispatch({
                        type: 'UPDATE_FUSION',
                        // no intent here so we don't flood the memory log
                        payload: {
                            uid: currentAssetId,
                            fusionProperties: newProps
                        }
                    });
                } catch(err) { console.error("Slider dispatch error", err); }
            }
        };
    });

    // Execute first sync immediately to apply element type tolerance/ranges
    if (typeof window.syncFusionPanelWithSelectedAsset === 'function') {
        window.syncFusionPanelWithSelectedAsset(assetId);
    }
}

function renderSlider(id, label, value, min, max, unit) {
    return `
        <div class="fusion-slider-group" data-slider-prop="${id}" style="display: flex; flex-direction: column; gap: 4px; transition: opacity 0.25s ease;">
            <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: #475569; align-items: center;">
                <span class="slider-label" style="display: flex; align-items: center; gap: 4px;">${label}</span>
                <span class="val-${id}" style="font-weight: 500; font-variant-numeric: tabular-nums; font-size: 11px;">${value}${unit}</span>
            </div>
            <input type="range" 
                data-prop="${id}" 
                data-unit="${unit}"
                min="${min}" max="${max}" value="${value}"
                style="width: 100%; accent-color: #6366f1; cursor: pointer; transition: all 0.2s;">
        </div>
    `;
}

/**
 * Sync the fusion panel state and properties with a dynamically selected asset (Live Sync)
 * and perform element-type compatibility/range isolation.
 */
window.syncFusionPanelWithSelectedAsset = function(newAssetId) {
    const panel = document.getElementById('fusionEditPanel');
    if (!panel) return;

    const workspace = window.mvrRuntime?.getCurrentWorkspace();
    if (!workspace) return;

    const asset = workspace.currentState.assetRegistry.get(newAssetId);
    if (!asset) return;

    // Update current active asset tracking on panel DOM
    panel.dataset.currentAssetId = newAssetId;

    // Get fusion properties or safe defaults
    const props = asset.fusionProperties || { 
        brightness: 100, 
        contrast: 100, 
        saturation: 100, 
        blur: 0,
        hueRotate: 0,
        grayscale: 0,
        sepia: 0
    };

    // Update heading dynamically to guide user
    const headingSpan = panel.querySelector('.panel-heading span');
    if (headingSpan) {
        const layerName = asset.layerName || asset.name || `图层 ${newAssetId.slice(-4)}`;
        headingSpan.textContent = `调整: ${layerName}`;
    }

    const sliders = panel.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const type = slider.dataset.prop;
        const unit = slider.dataset.unit;
        const val = props[type] !== undefined ? props[type] : (type === 'brightness' || type === 'contrast' || type === 'saturation' ? 100 : 0);

        // Update raw value
        slider.value = val;

        // Update value display text
        const valDisplay = panel.querySelector(`.val-${type}`);
        if (valDisplay) {
            valDisplay.textContent = val + unit;
        }

        // Always keep sliders active
        slider.disabled = false;
        const sliderGroup = slider.closest('.fusion-slider-group');
        if (sliderGroup) {
            sliderGroup.style.opacity = '1';
            sliderGroup.style.pointerEvents = 'auto';
            const badge = sliderGroup.querySelector('.disabled-badge');
            if (badge) badge.remove();
        }
    });
};

// Automatic workspace selections listening & initialization
(function installLiveSyncHooks() {
    if (typeof window !== 'undefined') {
        // Safe interceptor for showWorkbenchToolbox
        if (!window.originalShowWorkbenchToolbox) {
            const checkAndHookShow = setInterval(() => {
                if (window.showWorkbenchToolbox && !window.originalShowWorkbenchToolbox) {
                    window.originalShowWorkbenchToolbox = window.showWorkbenchToolbox;
                    window.showWorkbenchToolbox = function(id) {
                        try {
                            window.originalShowWorkbenchToolbox(id);
                        } catch (err) {
                            console.error("Error in original showWorkbenchToolbox", err);
                        }
                        if (typeof window.syncFusionPanelWithSelectedAsset === 'function') {
                            window.syncFusionPanelWithSelectedAsset(id);
                        }
                    };
                    clearInterval(checkAndHookShow);
                }
            }, 100);
            setTimeout(() => clearInterval(checkAndHookShow), 6000);
        }

        // Safe interceptor for selectWorkbenchItem
        if (!window.originalSelectWorkbenchItem) {
            const checkAndHookSelect = setInterval(() => {
                if (window.selectWorkbenchItem && !window.originalSelectWorkbenchItem) {
                    window.originalSelectWorkbenchItem = window.selectWorkbenchItem;
                    window.selectWorkbenchItem = function(id) {
                        try {
                            window.originalSelectWorkbenchItem(id);
                        } catch (err) {
                            console.error("Error in original selectWorkbenchItem", err);
                        }
                        if (typeof window.syncFusionPanelWithSelectedAsset === 'function') {
                            window.syncFusionPanelWithSelectedAsset(id);
                        }
                    };
                    clearInterval(checkAndHookSelect);
                }
            }, 100);
            setTimeout(() => clearInterval(checkAndHookSelect), 6000);
        }
    }
})();
