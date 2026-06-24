import { state } from "../core/state.js";
import { editOrQueryImageWithGemini } from "../ai-services/skills-engine.js";
import { dataURLToFile } from "../core/utils.js";

// Callbacks
let addImageToWorkbenchCallback, addMessageCallback;

export function initGenealogyLines(callbacks) {
    addImageToWorkbenchCallback = callbacks.addImageToWorkbench;
    addMessageCallback = callbacks.addMessage;
}

export function drawGenealogyConnections() {
    const genealogyLayer = document.getElementById('genealogyLayer');
    if (!genealogyLayer) return;

    // Clear existing paths (except defs)
    const defs = genealogyLayer.querySelector('defs');
    genealogyLayer.innerHTML = '';
    if(defs) genealogyLayer.appendChild(defs);

    state.workbenchItems.forEach((item, id) => {
        if (!item?.el) return;
        // Handle hidden items as ghosts for genealogy visualization
        if (item.hidden) {
            item.el.style.display = 'block';
            item.el.style.opacity = '0.3';
            item.el.style.pointerEvents = 'auto';
            item.el.classList.add('genealogy-ghost');
        } else {
            item.el.classList.remove('genealogy-ghost');
            // Reset opacity if it was a ghost (and not currently selected/active)
            if (item.el.style.opacity === '0.3') {
                item.el.style.opacity = '1';
            }
        }

        if (item.genealogy && item.genealogy.parents && item.genealogy.parents.length > 0) {
            item.genealogy.parents.forEach(parentId => {
                const parentItem = state.workbenchItems.get(parentId);
                if (parentItem) {
                    drawConnectionPath(parentItem, item, item.genealogy, genealogyLayer);
                }
            });
        }
    });
}

function drawConnectionPath(parent, child, genealogyData, genealogyLayer) {
    const pRect = {
        x: parseFloat(parent.el.style.left),
        y: parseFloat(parent.el.style.top),
        w: parseFloat(parent.el.style.width) || parent.el.offsetWidth,
        h: parseFloat(parent.el.style.height) || parent.el.offsetHeight
    };
    const cRect = {
        x: parseFloat(child.el.style.left),
        y: parseFloat(child.el.style.top),
        w: parseFloat(child.el.style.width) || child.el.offsetWidth,
        h: parseFloat(child.el.style.height) || child.el.offsetHeight
    };

    // Calculate center points
    const startX = pRect.x + pRect.w / 2;
    const startY = pRect.y + pRect.h / 2;
    const endX = cRect.x + cRect.w / 2;
    const endY = cRect.y + cRect.h / 2;

    // Create Bezier Curve
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.classList.add("genealogy-path");
    
    // Dynamic control points based on relative position
    const deltaX = endX - startX;
    const deltaY = endY - startY;
    
    // If child is mostly to the right, curve horizontally. If mostly below, curve vertically.
    let cp1x, cp1y, cp2x, cp2y;
    
    if (Math.abs(deltaX) > Math.abs(deltaY)) {
        // Horizontal layout
        const offset = Math.abs(deltaX) * 0.5;
        cp1x = startX + offset;
        cp1y = startY;
        cp2x = endX - offset;
        cp2y = endY;
    } else {
        // Vertical layout
        const offset = Math.abs(deltaY) * 0.5;
        cp1x = startX;
        cp1y = startY + offset;
        cp2x = endX;
        cp2y = endY - offset;
    }

    const d = `M ${startX} ${startY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endX} ${endY}`;
    path.setAttribute("d", d);
    path.setAttribute("marker-end", "url(#arrowhead)");
    
    // Add data for interaction
    path.dataset.childId = child.el.dataset.itemId;
    path.dataset.parentId = parent.el.dataset.itemId;
    
    // Double click to rewind/edit
    path.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        openGenealogyModifier(child.el.dataset.itemId, parent.el.dataset.itemId, genealogyData);
    });
    
    // Hover effect for marker color
    path.addEventListener('mouseenter', () => path.setAttribute("marker-end", "url(#arrowhead-active)"));
    path.addEventListener('mouseleave', () => path.setAttribute("marker-end", "url(#arrowhead)"));

    genealogyLayer.appendChild(path);
}

async function openGenealogyModifier(childId, parentId, genealogyData) {
    const childItem = state.workbenchItems.get(childId);
    if (!childItem) return;

    // Custom simple modal for prompt editing
    const prompt = genealogyData.prompt || "No prompt data available.";
    
    // Create modal on the fly
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.6); z-index: 10002; display: flex;
        align-items: center; justify-content: center;
    `;
    
    modal.innerHTML = `
        <div style="background: white; padding: 24px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
            <h3 style="margin-top:0; color:var(--primary-color);">🧬 基因回溯修改</h3>
            <p style="font-size:13px; color:#666; margin-bottom:10px;">修改当时生成这张图的参数（Prompt），重新演化。</p>
            <textarea id="genealogyPrompt" style="width:100%; height:100px; padding:10px; border:1px solid #ddd; border-radius:8px; margin-bottom:15px; font-family:inherit;">${prompt}</textarea>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="display:flex; justify-content:flex-end; gap:10px;">
                    <button id="closeGenealogy" style="padding:8px 16px; background:#f0f4f8; border:none; border-radius:6px; cursor:pointer;">取消</button>
                    <button id="applyGenealogy" style="padding:8px 16px; background:var(--primary-color); color:white; border:none; border-radius:6px; cursor:pointer;">重新生成 (Regenerate)</button>
                </div>
                <div id="restoreParentContainer"></div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);

    // Add Restore Parent button if parent is hidden
    const parentItem = state.workbenchItems.get(parentId);
    if (parentItem && parentItem.hidden) {
        const restoreBtn = document.createElement('button');
        restoreBtn.style.cssText = `
            width: 100%; padding: 10px; background: #e3e9f2; border: 1px solid #d1d9e6;
            border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500;
            color: var(--primary-color); display: flex; align-items: center; justify-content: center; gap: 8px;
        `;
        restoreBtn.innerHTML = '<i class="fas fa-eye"></i> 恢复原图 (Restore Parent)';
        restoreBtn.onclick = () => {
            parentItem.hidden = false;
            parentItem.el.style.display = 'block';
            parentItem.el.style.opacity = '1';
            parentItem.el.classList.remove('genealogy-ghost');
            modal.remove();
            drawGenealogyConnections();
            if (window.historyManager) window.historyManager.pushState();
        };
        document.getElementById('restoreParentContainer').appendChild(restoreBtn);
    }
    
    document.getElementById('closeGenealogy').onclick = () => modal.remove();
    
    document.getElementById('applyGenealogy').onclick = async () => {
        const newPrompt = document.getElementById('genealogyPrompt').value;
        modal.remove();
        
        const executeGenealogyRegen = async (customPrompt) => {
            const promptToUse = customPrompt || newPrompt;
            // Trigger Regeneration
            if (addImageToWorkbenchCallback) addImageToWorkbenchCallback(childItem.file, 'Updating...', { x: parseFloat(childItem.el.style.left), y: parseFloat(childItem.el.style.top) });
            // Remove old item effectively (or replace it later). For now, let's keep it until new one arrives or just indicate loading on it.
            // Better: Show loading on the child item.
            
            const childEl = childItem.el;
            childEl.style.opacity = '0.5';
            const spinner = document.createElement('div');
            spinner.className = 'spinner';
            spinner.style.position = 'absolute'; spinner.style.top = '50%'; spinner.style.left = '50%';
            childEl.appendChild(spinner);

            try {
                // Determine parents files
                const parentItem = state.workbenchItems.get(parentId);
                const refImages = parentItem ? [parentItem.file] : [];                
                // Assuming it was an edit/injection task. We re-run the edit.
                // Note: Logic implies we treat parent as base or ref depending on original action.
                // For simplicity, we assume parent is base for direct mutations, or ref for composite.
                // Let's assume parentId is the BASE image for now if parents.length === 1
                
                const result = await editOrQueryImageWithGemini(promptToUse, parentItem.file);
                
                if (result.success && result.imageData) {
                    const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                    const newFile = await dataURLToFile(imgSrc, `evolved-${Date.now()}.png`);
                    
                    // Update the existing item data
                    childItem.file = newFile;
                    childItem.dataUrl = imgSrc;
                    childEl.querySelector('img').src = imgSrc;
                    
                    // Update genealogy data
                    childItem.genealogy.prompt = promptToUse;
                    
                    // Cleanup
                    childEl.style.opacity = '1';
                    spinner.remove();
                    
                    // Flash effect
                    childEl.style.boxShadow = "0 0 20px #4CAF50";
                    setTimeout(() => childEl.style.boxShadow = "", 1000);
                    
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('基因节点重新生成', promptToUse, imgSrc, executeGenealogyRegen);
                    } else if (addMessageCallback) {
                        addMessageCallback({ sender: 'bot', type: 'text', content: '🧬 基因节点已更新：图像重新生成完成。' });
                    }
                }
            } catch (err) {
                console.error(err);
                alert("重新生成失败: " + err.message);
                childEl.style.opacity = '1';
                spinner.remove();
            }
        };

        await executeGenealogyRegen();
    };
}
