import { state } from '../../core/state.js';
import { showLayerEditPrompt } from '../modals.js';
import { editLayerAsset } from './layer-assets.js';
import { runtime } from '../../runtime/CoreRuntime';

export function renderCanvasLayers(itemId) {
    const item = state.workbenchItems.get(itemId);
    if (!item) return;

    let layersToRender = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    
    // --- INTEGRATING MVR ASSET RUNTIME ---
    const workspace = runtime.getCurrentWorkspace();
    if (workspace && workspace.currentState.assetRegistry.getAll().length > 0) {
        // If there are registered Assets, map them to layer format for UI mappings
        const assets = workspace.currentState.assetRegistry.getAll();
        console.log(`[MVR] Rendering ${assets.length} assets from Workspace AssetRegistry`);
        // Right now, MVR handles everything as extracted floating assets. For old visual compatibility, 
        // we merge them or replace them depending on the integration strategy.
        // As a Minimal Viable Runtime, we just ensure they are logged and can be drawn.
    }
    // -------------------------------------

    if (!layersToRender || !layersToRender.length) return;

    const wrapper = item.el;
    const cropContainer = wrapper.querySelector('.crop-container');
    if (!cropContainer) return;

    let layersContainer = cropContainer.querySelector('.canvas-layers-container');
    if (!layersContainer) {
        layersContainer = document.createElement('div');
        layersContainer.className = 'canvas-layers-container';
        layersContainer.style.position = 'absolute';
        layersContainer.style.top = '0';
        layersContainer.style.left = '0';
        layersContainer.style.width = '100%';
        layersContainer.style.height = '100%';
        layersContainer.style.pointerEvents = 'none'; // Let events pass through by default
        cropContainer.appendChild(layersContainer);
    }

    layersContainer.innerHTML = '';

    layersToRender.forEach((layer, index) => {
        console.log(`[Layers] Rendering layer ${index}: ${layer.name || 'unnamed'}, visible=${layer.visible !== false}, bbox=${layer.bbox}`);
        if (!layer.bbox) return;
        
        // Check visibility state
        let isVisible = true;
        let isSelected = false;
        if (item.layerStates && item.layerStates.has(index)) {
            const state = item.layerStates.get(index);
            isVisible = state.visible !== false;
            isSelected = state.selected === true;
        } else if (layer.visible === false) {
            isVisible = false;
        }
        
        if (!isVisible) return;

        const [ymin, xmin, ymax, xmax] = layer.bbox;
        
        const layerEl = document.createElement('div');
        layerEl.className = 'canvas-layer';
        if (layer.category === 'background') {
            layerEl.classList.add('is-background');
        }
        if (isSelected) {
            layerEl.classList.add('selected');
            // 移除 BBOX 边框，让图层本身成为唯一视觉主体
            layerEl.style.filter = 'drop-shadow(0 0 8px rgba(79, 70, 229, 0.8))';
        }
        layerEl.dataset.layerIndex = index;
        layerEl.dataset.layerId = layer.id || `layer-${index}`;
        
        // Convert normalized bbox (0-1000) to percentages
        const top = (ymin / 1000) * 100;
        const left = (xmin / 1000) * 100;
        const width = ((xmax - xmin) / 1000) * 100;
        const height = ((ymax - ymin) / 1000) * 100;

        layerEl.style.position = 'absolute';
        layerEl.style.top = `${top}%`;
        layerEl.style.left = `${left}%`;
        layerEl.style.width = `${width}%`;
        layerEl.style.height = `${height}%`;
        layerEl.style.zIndex = `${layersToRender.length - index}`; // Higher index = lower z-index (rendered first in list, so top layer is index 0)
        const activeVersion = layer.versions?.find(v => v.id === layer.activeVersionId);
        const displayCutoutUrl = activeVersion?.cutoutUrl || layer.cutoutUrl || null;

        layerEl.style.pointerEvents = displayCutoutUrl ? 'auto' : 'none';
        layerEl.style.cursor = displayCutoutUrl ? 'move' : 'default';
        layerEl.style.boxSizing = 'border-box';
        layerEl.style.transition = 'transform 0.1s';
        
        // V3: 支持 CSS transform 缩放和旋转
        const scale = layer.scale !== undefined ? layer.scale : 1;
        const rotate = layer.rotate !== undefined ? layer.rotate : 0;
        layerEl.style.transform = `rotate(${rotate}deg) scale(${scale})`;
        layerEl.style.transformOrigin = 'center center';
        
        // console.log(`Layer ${index} render: status=${layer.assetStatus}, cutout=${displayCutoutUrl ? 'YES' : 'NO'}`);

        // V2.4: Improved Rendering Logic - Use <img> for cutouts to avoid stretching and sync issues.
        // Non-cutout semantic layers are metadata only; rendering cropped original-image
        // previews creates a duplicate image overlay and blocks dragging the parent.
        if (displayCutoutUrl) {
            const imgEl = document.createElement('img');
            imgEl.className = 'cutout-img';
            imgEl.src = displayCutoutUrl;
            imgEl.style.width = '100%';
            imgEl.style.height = '100%';
            imgEl.style.objectFit = 'fill'; // Fill the bbox exactly
            imgEl.style.pointerEvents = 'none';
            imgEl.style.display = 'block';
            
            // Debug: check if image loads
            imgEl.onload = () => console.log(`Layer ${index} img loaded`);
            imgEl.onerror = () => console.error(`Layer ${index} img load failed`, displayCutoutUrl.substring(0, 50));
            
            layerEl.appendChild(imgEl);
        } else {
            layerEl.style.backgroundImage = 'none';
            layerEl.style.backgroundColor = isSelected ? 'rgba(79, 70, 229, 0.08)' : 'transparent';
            layerEl.style.border = isSelected ? '1.5px solid rgba(79, 70, 229, 0.85)' : 'none';
        }

        // Add a loading indicator overlay (independent of background source)
        const isProcessing = layer.assetStatus === 'processing' || 
                           layer.assetStatus === 'editing' || 
                           layer.interactionLock?.reason === 'rendering';

        if (isProcessing) {
            const loader = document.createElement('div');
            loader.className = 'layer-loader-overlay';
            loader.style.position = 'absolute';
            loader.style.top = '0';
            loader.style.left = '0';
            loader.style.width = '100%';
            loader.style.height = '100%';
            loader.style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
            loader.style.display = 'flex';
            loader.style.alignItems = 'center';
            loader.style.justifyContent = 'center';
            loader.style.zIndex = '10';
            
            let statusText = '';
            if (layer.assetStatus === 'editing' || layer.interactionLock?.reason === 'rendering') {
                statusText = '重绘中...';
            } else if (layer.assetStatus === 'processing') {
                statusText = '处理中...';
            }
            
            loader.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                    <i class="fas fa-spinner fa-spin" style="color: #4f46e5; font-size: 24px;"></i>
                    ${statusText ? `<span style="font-size: 10px; color: #4f46e5; font-weight: 600; background: rgba(255,255,255,0.8); padding: 2px 6px; border-radius: 4px;">${statusText}</span>` : ''}
                </div>
            `;
            layerEl.appendChild(loader);
        }

        // Hover effect (removed box-shadow to keep WYSIWYG clean)
        layerEl.addEventListener('mouseenter', () => {
            if (!layerEl.classList.contains('selected')) {
                layerEl.style.filter = 'drop-shadow(0 0 4px rgba(129, 140, 248, 0.8))';
            }
        });
        layerEl.addEventListener('mouseleave', () => {
            if (!layerEl.classList.contains('selected')) {
                layerEl.style.filter = 'none';
            }
        });

        // Click to select
        layerEl.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Prevent workbench item drag
            
            if (layer.interactionLock?.draggingDisabled || layer.assetStatus === 'editing') return; // Lock interaction during editing
            
            // Deselect all other layers in this item
            layersContainer.querySelectorAll('.canvas-layer').forEach(el => {
                el.classList.remove('selected');
                el.style.filter = 'none';
            });
            
            layerEl.classList.add('selected');
            layerEl.style.filter = 'drop-shadow(0 0 8px rgba(79, 70, 229, 0.8))';

            // Setup drag
            let startX = e.clientX;
            let startY = e.clientY;
            const startLeft = parseFloat(layerEl.style.left);
            const startTop = parseFloat(layerEl.style.top);
            
        const onMouseMove = (moveEvent) => {
            const screenDx = moveEvent.clientX - startX;
            const screenDy = moveEvent.clientY - startY;
            
            // Convert dx/dy to percentages based on screen size of container
            const rect = cropContainer.getBoundingClientRect();
            const dxPct = (screenDx / rect.width) * 100;
            const dyPct = (screenDy / rect.height) * 100;
            
            const newLeft = startLeft + dxPct;
            const newTop = startTop + dyPct;
            
            layerEl.style.left = `${newLeft}%`;
            layerEl.style.top = `${newTop}%`;

            // V2.4: If using fallback (original image), we must update background position in real-time
            // Check displayCutoutUrl from the closure
            if (!displayCutoutUrl) {
                const bgPosX = width >= 100 ? 0 : (newLeft / (100 - width)) * 100;
                const bgPosY = height >= 100 ? 0 : (newTop / (100 - height)) * 100;
                layerEl.style.backgroundPosition = `${bgPosX}% ${bgPosY}%`;
            }
        };
            
            const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                
                // Update layer bbox in state (V3: keep as percentage record, no longer for strict reverse mapping)
                const newLeft = parseFloat(layerEl.style.left);
                const newTop = parseFloat(layerEl.style.top);
                const newXmin = (newLeft / 100) * 1000;
                const newYmin = (newTop / 100) * 1000;
                const newXmax = newXmin + (width / 100) * 1000;
                const newYmax = newYmin + (height / 100) * 1000;
                
                layer.bbox = [newYmin, newXmin, newYmax, newXmax];
                
                // Update selection state globally
                if (window.updateLayerState && window.renderLayerList) {
                    // Deselect all others
                    layersToRender.forEach((_, idx) => {
                        window.updateLayerState(itemId, idx, { selected: idx === index });
                    });
                    window.renderLayerList(layersToRender, itemId);
                }

                // Track currently active workbench item ID and trigger capsule alert popup
                state.currentActiveWorkbenchItemId = itemId;
                if (typeof window.triggerCapsuleAlert === 'function') {
                    window.triggerCapsuleAlert(itemId);
                }
            };
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        // Double click to edit (V2.3)
        layerEl.addEventListener('dblclick', async (e) => {
            e.stopPropagation();
            
            // Only allow editing for non-background layers
            if (layer.category === 'background') {
                console.log("Cannot edit background layer");
                return;
            }
            if (layer.interactionLock?.editingDisabled || layer.assetStatus === 'editing' || layer.assetStatus === 'processing') {
                console.log("Layer is currently processing or editing");
                return; 
            }
            
            console.log("Opening layer edit prompt for:", layer.name);
            const prompt = await showLayerEditPrompt(layer.name);
            if (prompt) {
                editLayerAsset(itemId, index, prompt);
            }
        });

        layersContainer.appendChild(layerEl);
    });
}

window.renderCanvasLayers = renderCanvasLayers;
