import { state } from '../../core/state.js';
import { ATMOSPHERE_OPTS } from '../../core/config.js';

const { workbenchItems } = state;

export function addAtmosphereNode(key, x, y) {
    const opt = ATMOSPHERE_OPTS[key];
    const id = `atm-${Date.now()}`;
    const el = document.createElement('div');
    el.className = 'workbench-item atmosphere-node';
    el.dataset.itemId = id;
    el.dataset.color = JSON.stringify(opt.color);
    el.dataset.label = opt.label;
    el.dataset.atmType = key;
    el.dataset.atmPrompt = opt.prompt;
    
    el.style.left = `${x - 30}px`; // Center offset (width 60)
    el.style.top = `${y - 30}px`;

    const inverseScale = 1 / state.workbenchZoom;
    el.style.transform = `scale(${inverseScale})`;

    el.innerHTML = `
        <i class="fas ${opt.icon}"></i>
        <div class="light-beam-indicator"></div>
        <div class="apply-light-btn" style="display:none">应用</div>
         <button class="atm-close-btn" title="移除卡片">&times;</button>
    `;
    
    window.workbenchGrid.appendChild(el);

    workbenchItems.set(id, { 
        file: null, 
        el: el,
        type: 'atmosphere',
        id: id
    });

    el.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('atm-close-btn') || e.target.closest('.atm-close-btn')) return; // Ignore close btn
        if (e.button !== 0) return;
        if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(e);
    });

    el.querySelector('.atm-close-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.lighting-active').forEach(item => {
            item.classList.remove('lighting-active');
            item.querySelector('.relighting-canvas')?.remove();
        });
        if (document.getElementById('lightConnector')) document.getElementById('lightConnector').style.display = 'none';
        
        if (window.deleteWorkbenchItem) window.deleteWorkbenchItem(id, true);
    });
    
    if (window.historyManager) window.historyManager.pushState();
}
