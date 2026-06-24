import { ATMOSPHERE_OPTS } from '../core/config.js';
import { initWorkbenchDrop } from './drag-drop.js';
import { applyRelighting } from '../graphics/relight-engine.js';
import { state } from '../core/state.js';

export async function applyAtmosphereToImage(atmosphereNodeId, targetImageId) {
    const lightNode = state.workbenchItems.get(atmosphereNodeId);
    const targetItem = state.workbenchItems.get(targetImageId);
    if (!lightNode || !targetItem) return;

    const nodeRect = lightNode.el.getBoundingClientRect();
    const targetRect = targetItem.el.getBoundingClientRect();
    
    const localX = (nodeRect.left + nodeRect.width/2 - targetRect.left);
    const localY = (nodeRect.top + nodeRect.height/2 - targetRect.top);

    if (typeof applyRelighting === 'function') {
        await applyRelighting(targetItem, lightNode, localX, localY);
    }
}

export function initAtmospherePalette(workbenchGrid) {
    const toggleBtn = document.getElementById('toggleAtmospherePanel');
    const palette = document.getElementById('atmospherePalette');
    
    if (!toggleBtn || !palette) return;

    toggleBtn.addEventListener('click', () => {
        const isVisible = palette.style.display === 'grid';
        palette.style.display = isVisible ? 'none' : 'grid';
    });

    // Generate Cards
    Object.entries(ATMOSPHERE_OPTS).forEach(([key, opt]) => {
        const card = document.createElement('div');
        card.className = 'atmosphere-card';
        card.innerHTML = `<i class="fas ${opt.icon}"></i><span>${opt.label}</span>`;
        card.draggable = true;
        card.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', JSON.stringify({ type: 'atmosphere', key: key }));
        });
        palette.appendChild(card);
    });

    // Handle Drop on Workbench
    initWorkbenchDrop(workbenchGrid);
}
