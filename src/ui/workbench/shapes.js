import { state } from '../../core/state.js';

const { workbenchItems } = state;

export function startDrawingShape(type) {
    window.currentDrawingShape = type;
    window.workbenchGrid.style.cursor = 'crosshair';
}

export function handleShapeDrawingMove(e) {
    if (!window.drawingElement || !window.drawingStartPos) return;
    
    const rect = window.workbenchGrid.getBoundingClientRect();
    const currentX = (e.clientX - rect.left) / state.workbenchZoom;
    const currentY = (e.clientY - rect.top) / state.workbenchZoom;
    
    let width = Math.abs(currentX - window.drawingStartPos.x);
    let height = Math.abs(currentY - window.drawingStartPos.y);
    let left = Math.min(window.drawingStartPos.x, currentX);
    let top = Math.min(window.drawingStartPos.y, currentY);
    
    if (e.shiftKey) {
        if (window.currentDrawingShape === 'rect' || window.currentDrawingShape === 'ellipse') {
            const size = Math.max(width, height);
            width = size;
            height = size;
            left = currentX < window.drawingStartPos.x ? window.drawingStartPos.x - size : window.drawingStartPos.x;
            top = currentY < window.drawingStartPos.y ? window.drawingStartPos.y - size : window.drawingStartPos.y;
        } else if (window.currentDrawingShape === 'line' || window.currentDrawingShape === 'arrow') {
            const dx = currentX - window.drawingStartPos.x;
            const dy = currentY - window.drawingStartPos.y;
            const angle = Math.atan2(dy, dx);
            const snappedAngle = Math.round(angle / (Math.PI / 4)) * (Math.PI / 4);
            const length = Math.sqrt(dx*dx + dy*dy);
            
            const snappedX = window.drawingStartPos.x + length * Math.cos(snappedAngle);
            const snappedY = window.drawingStartPos.y + length * Math.sin(snappedAngle);
            
            width = Math.abs(snappedX - window.drawingStartPos.x);
            height = Math.abs(snappedY - window.drawingStartPos.y);
            left = Math.min(window.drawingStartPos.x, snappedX);
            top = Math.min(window.drawingStartPos.y, snappedY);
            
            window.drawingElement.dataset.endX = snappedX;
            window.drawingElement.dataset.endY = snappedY;
        }
    } else {
        if (window.currentDrawingShape === 'line' || window.currentDrawingShape === 'arrow') {
            window.drawingElement.dataset.endX = currentX;
            window.drawingElement.dataset.endY = currentY;
        }
    }
    
    width = Math.max(width, 2);
    height = Math.max(height, 2);
    
    window.drawingElement.style.width = `${width}px`;
    window.drawingElement.style.height = `${height}px`;
    window.drawingElement.style.left = `${left}px`;
    window.drawingElement.style.top = `${top}px`;
    
    if (window.currentDrawingShape === 'line' || window.currentDrawingShape === 'arrow') {
        const svgLine = window.drawingElement.querySelector('line');
        const endX = parseFloat(window.drawingElement.dataset.endX || currentX);
        const endY = parseFloat(window.drawingElement.dataset.endY || currentY);
        
        const x1 = window.drawingStartPos.x - left;
        const y1 = window.drawingStartPos.y - top;
        const x2 = endX - left;
        const y2 = endY - top;
        
        svgLine.setAttribute('x1', x1);
        svgLine.setAttribute('y1', y1);
        svgLine.setAttribute('x2', x2);
        svgLine.setAttribute('y2', y2);
    }
}

export function handleShapeDrawingEnd(e) {
    document.removeEventListener('mousemove', handleShapeDrawingMove);
    document.removeEventListener('mouseup', handleShapeDrawingEnd);
    
    if (window.drawingElement) {
        const width = parseFloat(window.drawingElement.style.width);
        const height = parseFloat(window.drawingElement.style.height);
        
        if (width < 5 && height < 5) {
            window.drawingElement.remove();
        } else {
            window.drawingElement.style.pointerEvents = 'auto';
            
            if (window.currentDrawingShape === 'line' || window.currentDrawingShape === 'arrow') {
                const svg = window.drawingElement.querySelector('svg');
                svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
                svg.setAttribute('preserveAspectRatio', 'none');
            }
            
            const id = window.drawingElement.dataset.itemId;
            workbenchItems.set(id, {
                file: null,
                el: window.drawingElement,
                type: 'shape',
                shapeType: window.currentDrawingShape,
                id: id
            });
            
            ['nw', 'ne', 'sw', 'se'].forEach(dir => {
                const handle = document.createElement('div');
                handle.className = `resize-handle resize-${dir}`;
                handle.addEventListener('mousedown', (ev) => {
                    if (ev.button !== 0 || state.isSpacePressed) return; 
                    if (window.handleResizeStart) window.handleResizeStart(ev, dir, ev.currentTarget.parentElement);
                });
                window.drawingElement.appendChild(handle);
            });
            
            const inverseScale = 1 / state.workbenchZoom;
            window.drawingElement.querySelectorAll('.resize-handle').forEach(handle => {
                handle.style.transform = `scale(${inverseScale})`;
            });
            
            window.drawingElement.addEventListener('mousedown', (ev) => {
                if (ev.target.classList.contains('resize-handle')) return;
                if (ev.button !== 0) return;
                if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(ev);
                if (window.showShapeAdjuster) window.showShapeAdjuster(ev.currentTarget, ev.clientX, ev.clientY);
            });
            
            if (window.showShapeAdjuster) window.showShapeAdjuster(window.drawingElement, e.clientX, e.clientY);
            
            if (window.historyManager) window.historyManager.pushState();
        }
    }
    
    window.drawingElement = null;
    window.drawingStartPos = null;
    window.currentDrawingShape = null;
    window.workbenchGrid.style.cursor = 'default';
}

export function showShapeAdjuster(shapeEl, clientX, clientY) {
    const shapeAdjuster = document.getElementById('shapeAdjuster');
    const shapeAdjusterColor = document.getElementById('shapeAdjusterColor');
    const shapeAdjusterWidth = document.getElementById('shapeAdjusterWidth');
    const shapeAdjusterWidthVal = document.getElementById('shapeAdjusterWidthVal');

    window.currentAdjustingShape = shapeEl;
    const shapeType = shapeEl.dataset.shapeType;
    
    // Read current values
    let color = '#2A5C82';
    let width = 2;
    
    if (shapeType === 'rect' || shapeType === 'ellipse' || shapeType === 'polygon') {
        color = shapeEl.style.backgroundColor && shapeEl.style.backgroundColor !== 'transparent'
            ? shapeEl.style.backgroundColor
            : (shapeEl.style.borderColor || '#2A5C82');
        width = parseInt(shapeEl.style.borderWidth) || 2;
    } else if (shapeType === 'line' || shapeType === 'arrow') {
        const line = shapeEl.querySelector('line');
        if (line) {
            color = line.getAttribute('stroke') || '#2A5C82';
            width = parseInt(line.getAttribute('stroke-width')) || 2;
        }
    }
    
    // Convert rgb to hex for color input
    if (color.startsWith('rgb')) {
        const rgb = color.match(/\d+/g);
        if (rgb && rgb.length === 3) {
            color = '#' + rgb.map(x => parseInt(x).toString(16).padStart(2, '0')).join('');
        }
    }
    
    shapeAdjusterColor.value = color;
    shapeAdjusterWidth.value = width;
    shapeAdjusterWidthVal.textContent = `${width}px`;
    
    shapeAdjuster.style.display = 'flex';
    
    // Position adjuster
    requestAnimationFrame(() => {
        const rect = shapeAdjuster.getBoundingClientRect();
        let posX = clientX + 10;
        let posY = clientY + 10;
        
        if (posX + rect.width > window.innerWidth) {
            posX = window.innerWidth - rect.width - 10;
        }
        if (posY + rect.height > window.innerHeight) {
            posY = window.innerHeight - rect.height - 10;
        }
        
        shapeAdjuster.style.left = `${posX}px`;
        shapeAdjuster.style.top = `${posY}px`;
    });
}

export function restoreShapeToWorkbench(itemState) {
    const id = itemState.id;
    const wrapper = document.createElement('div');
    wrapper.className = `workbench-item workbench-shape shape-${itemState.shapeType}`;
    wrapper.dataset.itemId = id;
    wrapper.dataset.shapeType = itemState.shapeType;
    
    const left = itemState.rect ? itemState.rect.left : itemState.left;
    if (left !== undefined && left !== null) wrapper.style.left = typeof left === 'number' ? `${left}px` : left;
    const top = itemState.rect ? itemState.rect.top : itemState.top;
    if (top !== undefined && top !== null) wrapper.style.top = typeof top === 'number' ? `${top}px` : top;
    const width = itemState.rect ? itemState.rect.width : itemState.width;
    if (width !== undefined && width !== null) wrapper.style.width = typeof width === 'number' ? `${width}px` : width;
    const height = itemState.rect ? itemState.rect.height : itemState.height;
    if (height !== undefined && height !== null) wrapper.style.height = typeof height === 'number' ? `${height}px` : height;
    const zIndex = itemState.rect ? itemState.rect.zIndex : itemState.zIndex;
    if (zIndex !== undefined && zIndex !== null) wrapper.style.zIndex = zIndex;
    wrapper.style.backgroundColor = itemState.fillColor || 'transparent';
    if (itemState.clipPath) {
        wrapper.style.clipPath = itemState.clipPath;
        wrapper.style.webkitClipPath = itemState.clipPath;
    }
    
    if (itemState.shapeType === 'rect' || itemState.shapeType === 'ellipse' || itemState.shapeType === 'polygon') {
        const borderWidth = itemState.borderWidth !== undefined && itemState.borderWidth !== null ? itemState.borderWidth : 2;
        const borderColor = itemState.borderColor || 'var(--primary-color)';
        const borderWidthCss = typeof borderWidth === 'number' ? `${borderWidth}px` : borderWidth;
        if (itemState.shapeType !== 'polygon' || borderWidth !== 0) {
            wrapper.style.border = `${borderWidthCss} solid ${borderColor}`;
        }
        if (itemState.borderRadius) {
            wrapper.style.borderRadius = typeof itemState.borderRadius === 'number' ? `${itemState.borderRadius}px` : itemState.borderRadius;
        }
        if (itemState.shapeType === 'ellipse') {
            wrapper.style.borderRadius = '50%';
        }
    } else if (itemState.shapeType === 'line' || itemState.shapeType === 'arrow') {
        wrapper.innerHTML = itemState.content || '';
    }
    
    window.workbenchGrid.appendChild(wrapper);
    
    workbenchItems.set(id, {
        file: null,
        el: wrapper,
        type: 'shape',
        shapeType: itemState.shapeType,
        id: id,
        parentId: itemState.parentId || null,
        layerName: itemState.layerName || null,
        originalBbox: itemState.originalBbox || null,
        clipPath: itemState.clipPath || null
    });
    
    ['nw', 'ne', 'sw', 'se'].forEach(dir => {
        const handle = document.createElement('div');
        handle.className = `resize-handle resize-${dir}`;
        handle.addEventListener('mousedown', (ev) => {
            if (ev.button !== 0 || state.isSpacePressed) return; 
            if (window.handleResizeStart) window.handleResizeStart(ev, dir, wrapper);
        });
        wrapper.appendChild(handle);
    });
    
    const inverseScale = 1 / state.workbenchZoom;
    wrapper.querySelectorAll('.resize-handle').forEach(handle => {
        handle.style.transform = `scale(${inverseScale})`;
    });
    
    wrapper.addEventListener('mousedown', (ev) => {
        if (ev.target.classList.contains('resize-handle')) return;
        if (ev.button !== 0) return;
        if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(ev);
        if (window.showShapeAdjuster) window.showShapeAdjuster(wrapper, ev.clientX, ev.clientY);
    });
}
