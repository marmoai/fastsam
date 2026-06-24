import { state } from '../../core/state.js';
import { getProxiedUrl, isInvalidImageSrc } from '../../core/utils.js';


export async function hydrateWorkbench() {
    if (!window.mvrRuntime) return;
    const workspace = window.mvrRuntime.getCurrentWorkspace();
    if (!workspace) return;

    const { assetRegistry } = workspace.currentState;
    const assets = assetRegistry.getAll();

    for (const asset of assets) {
        const isOrphanExtractedRuntimeAsset =
            String(asset.uid || '').startsWith('asset_') &&
            asset.type === 'product' &&
            !asset.transform &&
            !asset.parentId &&
            !asset.layerName &&
            !asset.originalBbox;
        if (isOrphanExtractedRuntimeAsset) {
            continue;
        }

        // If DOM item doesn't exist, reconstruct it
        let el = document.querySelector(`[data-item-id="${asset.uid}"]`);
        if (!el) {
            if (window.addImageToWorkbench) {
                await window.addImageToWorkbench(null, asset.layerName || '恢复的图片', {
                    id: asset.uid,
                    dataUrl: asset.cleanPlateDataUrl || asset.sourceImage,
                    x: asset.transform?.x,
                    y: asset.transform?.y,
                    initialWidth: asset.transform?.width,
                    initialHeight: asset.transform?.height,
                    zIndex: asset.transform?.zIndex,
                    type: asset.type,
                    genealogy: asset.genealogy,
                    parentId: asset.parentId,
                    layerName: asset.layerName,
                    originalBbox: asset.originalBbox,
                    layers: asset.layers,
                    scene: asset.scene,
                    semanticViews: asset.semanticViews,
                    hasFullSemanticAnalysis: asset.hasFullSemanticAnalysis,
                    originalDataUrl: asset.originalDataUrl || asset.sourceImage,
                    cleanPlateDataUrl: asset.cleanPlateDataUrl,
                    cleanPlateStatus: asset.cleanPlateStatus || 'idle'
                });
            }
        }
    }
}

export function reconcileAllAssets() {
    if (!window.mvrRuntime) return;
    const workspace = window.mvrRuntime.getCurrentWorkspace();
    if (!workspace) return;

    const { assetRegistry } = workspace.currentState;
    const assets = assetRegistry.getAll();

    assets.forEach(asset => {
        const id = asset.uid;
        if (asset) {
            // Reconcile Transforms
            if (asset.transform && window.syncDOMToScene) {
                window.syncDOMToScene(id);
            }
            
            // Sync fusion properties to state.workbenchItems so it gets picked up by history/session serialization
            const item = state.workbenchItems.get(id);
            if (item) {
                 item.fusionProperties = JSON.parse(JSON.stringify(asset.fusionProperties || {}));
                 // Trigger history debounced save since state updated
                 if (window.workbenchHistory && window.workbenchHistory.pushState && !state.isCropping) {
                     // We don't want to save on every slider tick, so let's check
                     // Wait, pushState is already debounced, so this is safe!
                 }
            }

            // Reconcile Fusion Properties (Filters) & Source Image Updates
            const el = document.querySelector(`[data-item-id="${id}"]`);
            if (el) {
                const img = el.querySelector('img');
                
                // Reconcile image source changes
                const displaySource = asset.cleanPlateDataUrl || asset.sourceImage;
                if (img && !isInvalidImageSrc(displaySource) && img.src !== displaySource) {
                    img.src = displaySource;
                }
                // Show remixing state
                if (asset.status === 'remixing') {
                    el.style.boxShadow = '0 0 0 4px #6366f1, 0 0 20px rgba(99, 102, 241, 0.4)';
                    el.classList.add('pulse-anim');
                } else {
                    el.style.boxShadow = 'none';
                    el.style.outline = 'none';
                    el.classList.remove('pulse-anim');
                }

                if (asset.fusionProperties) {
                    const f = asset.fusionProperties;
                    const img = el.querySelector('img');
                    const target = img || el; // Support both img tag and container
                    target.style.filter = `brightness(${f.brightness}%) contrast(${f.contrast}%) saturate(${f.saturation}%) blur(${f.blur}px) hue-rotate(${f.hueRotate || 0}deg) grayscale(${f.grayscale || 0}%) sepia(${f.sepia || 0}%)`;

                    // Handle Material Overlay (AI Material Infusion)
                    let overlay = el.querySelector('.material-overlay');
                    let grain = el.querySelector('.material-grain');
                    
                    if (f.overlay && f.overlay.opacity > 0) {
                        // 1. Color/Metallic Overlay
                        if (!overlay) {
                            overlay = document.createElement('div');
                            overlay.className = 'material-overlay';
                            overlay.style.cssText = `
                                position: absolute;
                                inset: 0;
                                pointer-events: none;
                                border-radius: inherit;
                                transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                                z-index: 1;
                            `;
                            el.appendChild(overlay);
                        }
                        overlay.style.backgroundColor = f.overlay.color;
                        overlay.style.opacity = f.overlay.opacity.toString();
                        overlay.style.mixBlendMode = f.overlay.blendMode;

                        // 2. Grain/Texture Simulation (SVG Noise)
                        if (!grain) {
                            grain = document.createElement('div');
                            grain.className = 'material-grain';
                            grain.style.cssText = `
                                position: absolute;
                                inset: 0;
                                pointer-events: none;
                                border-radius: inherit;
                                opacity: 0.15;
                                mix-blend-mode: overlay;
                                background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
                                z-index: 2;
                            `;
                            el.appendChild(grain);
                        }
                        // Adjust grain density based on contrast
                        grain.style.opacity = (f.contrast > 120 ? 0.25 : 0.1).toString();
                        
                    } else {
                        if (overlay) overlay.remove();
                        if (grain) grain.remove();
                    }
                }
            }
        }
    });
}

// Start the observer loop
window.hydrateWorkbench = hydrateWorkbench; 
window.reconcileAllAssets = reconcileAllAssets;

if (!window.renderLoopStarted) {
    window.renderLoopStarted = true;
    (async () => {
        // Wait for workspace to be initialized
        while(!window.mvrRuntime || !window.mvrRuntime.getCurrentWorkspace()) {
            await new Promise(r => setTimeout(r, 100));
        }
        const workspace = window.mvrRuntime.getCurrentWorkspace();
        workspace.currentState.subscribe(reconcileAllAssets);
        reconcileAllAssets(); // Initial render
    })();
}

export async function renderSceneToCanvas(itemId, scaleFactor = 1) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.el) return null;

    const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    if (!layers) return null;

    const canvas = document.createElement('canvas');
    canvas.width = item.el.offsetWidth * scaleFactor;
    canvas.height = item.el.offsetHeight * scaleFactor;
    const ctx = canvas.getContext('2d');
    
    // Enable high quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // 1. First, render image layers (background and cutout overlays)
    for (let i = layers.length - 1; i >= 0; i--) {
        const layer = layers[i];
        if (!layer.bbox || (layer.visible === false)) continue;

        const [ymin, xmin, ymax, xmax] = layer.bbox;
        const left = (xmin / 1000) * canvas.width;
        const top = (ymin / 1000) * canvas.height;
        const width = ((xmax - xmin) / 1000) * canvas.width;
        const height = ((ymax - ymin) / 1000) * canvas.height;

        const scale = layer.scale !== undefined ? layer.scale : 1;
        const rotate = layer.rotate !== undefined ? layer.rotate : 0;
        const rotation = rotate * Math.PI / 180;

        const drawW = width * scale;
        const drawH = height * scale;
        const drawX = left + (width - drawW) / 2;
        const drawY = top + (height - drawH) / 2;

        ctx.save();
        ctx.translate(drawX + drawW / 2, drawY + drawH / 2);
        if (rotation !== 0) ctx.rotate(rotation);
        
        const activeVersion = layer.versions?.find(v => v.id === layer.activeVersionId);
        const displayCutoutUrl = activeVersion?.cutoutUrl || layer.cutoutUrl || null;

        if (displayCutoutUrl) {
            const cutoutSrc = getProxiedUrl(displayCutoutUrl);
            if (!cutoutSrc) {
                ctx.restore();
                continue;
            }
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.src = cutoutSrc;
            await new Promise(r => img.onload = r);
            ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
        } else if (layer.category === 'background') {
            const bgSrc = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
            const resolvedBgSrc = getProxiedUrl(bgSrc);
            if (!resolvedBgSrc) {
                ctx.restore();
                continue;
            }
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.src = resolvedBgSrc;
            await new Promise(r => img.onload = r);
            ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
        }
        ctx.restore();
    }

    // 2. Render text-note elements attached to this image
    state.workbenchItems.forEach((wbItem, id) => {
        if (wbItem.type === 'text-note' && wbItem.parentId === itemId) {
            const el = wbItem.el;
            const contentEl = el.querySelector('.note-content');
            if (!contentEl) return;
            
            const txt = contentEl.innerText.trim();
            if (!txt) return;

            // Get position relative to the main item
            const parentRect = item.el.getBoundingClientRect();
            const noteRect = el.getBoundingClientRect();
            
            // Note: coordinates here are based on DOM layout before scale
            const zoom = state.workbenchZoom || 1;
            const rx = (noteRect.left - parentRect.left) / zoom;
            const ry = (noteRect.top - parentRect.top) / zoom;
            
            const compStyle = window.getComputedStyle(contentEl);
            const fontSize = parseFloat(compStyle.fontSize) * scaleFactor;
            
            ctx.save();
            ctx.font = `${fontSize}px ${compStyle.fontFamily}`;
            ctx.fillStyle = compStyle.color;
            ctx.textAlign = compStyle.textAlign === 'center' ? 'center' : (compStyle.textAlign === 'right' ? 'right' : 'left');
            ctx.textBaseline = 'top';
            
            let drawX = rx * scaleFactor;
            if (ctx.textAlign === 'center') drawX += (noteRect.width / zoom) * scaleFactor / 2;
            if (ctx.textAlign === 'right') drawX += (noteRect.width / zoom) * scaleFactor;

            // Simple multi-line text rendering
            const lines = txt.split('\n');
            const lineHeight = parseFloat(compStyle.lineHeight) || (fontSize * 1.2);
            let drawY = ry * scaleFactor;
            
            for (let line of lines) {
                ctx.fillText(line, drawX, drawY);
                drawY += lineHeight * scaleFactor;
            }
            
            ctx.restore();
        }
    });

    return canvas;
}
