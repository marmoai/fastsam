import { state } from '../../core/state.js';

const { workbenchItems, selectedWorkbenchItems } = state;

export function findNextPosition(width, height) {
    const items = Array.from(workbenchItems.values()).filter(item => item.el && item.type !== 'text-note' && item.type !== 'shape' && item.type !== 'group-label');
    if (items.length === 0) return { x: 50000, y: 50000 };

    const viewport = document.getElementById('workbench');
    const container = document.getElementById('workbenchZoomContainer');
    
    if (!viewport || !container) return { x: 50000, y: 50000 };
    
    const viewportRect = viewport.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    
    const viewLeft = (viewportRect.left - containerRect.left) / state.workbenchZoom;
    const viewTop = (viewportRect.top - containerRect.top) / state.workbenchZoom;
    const viewWidth = viewportRect.width / state.workbenchZoom;
    const viewHeight = viewportRect.height / state.workbenchZoom;
    const viewRight = viewLeft + viewWidth;
    const viewBottom = viewTop + viewHeight;
    
    const visibleItems = items.filter(item => {
        const leftStr = item.el.style.left;
        const topStr = item.el.style.top;
        if (!leftStr || !topStr) return false;
        
        const x = parseFloat(leftStr);
        const y = parseFloat(topStr);
        const w = parseFloat(item.el.style.width) || 600;
        const h = parseFloat(item.el.style.height) || 600;
        
        return !(x + w < viewLeft || x > viewRight || y + h < viewTop || y > viewBottom);
    });

    const gap = 100;
    const defaultW = width || 600;
    const defaultH = height || 600;

    if (visibleItems.length === 0) {
        return { 
            x: viewLeft + (viewWidth - defaultW) / 2, 
            y: viewTop + (viewHeight - defaultH) / 2 
        };
    }

    visibleItems.sort((a, b) => {
        const topA = parseFloat(a.el.style.top) || 0;
        const topB = parseFloat(b.el.style.top) || 0;
        const leftA = parseFloat(a.el.style.left) || 0;
        const leftB = parseFloat(b.el.style.left) || 0;
        if (Math.abs(topA - topB) < 200) return leftA - leftB;
        return topA - topB;
    });

    const rows = [];
    visibleItems.forEach(item => {
        const top = parseFloat(item.el.style.top) || 0;
        if (rows.length === 0 || Math.abs(top - rows[rows.length - 1][0].top) > 200) {
            rows.push([{item, top}]);
        } else {
            rows[rows.length - 1].push({item, top});
        }
    });

    const lastRow = rows[rows.length - 1];
    lastRow.sort((a, b) => parseFloat(a.item.el.style.left) - parseFloat(b.item.el.style.left));

    const cols = 3;

    if (lastRow.length < cols) {
        const lastItem = lastRow[lastRow.length - 1].item;
        const lastX = parseFloat(lastItem.el.style.left) || 0;
        const lastWidth = parseFloat(lastItem.el.style.width) || defaultW;
        const lastY = parseFloat(lastItem.el.style.top) || 0;
        
        return { x: lastX + lastWidth + gap, y: lastY };
    } else {
        const firstItem = lastRow[0].item;
        const firstX = parseFloat(firstItem.el.style.left) || 0;
        
        let maxY = -Infinity;
        visibleItems.forEach(item => {
            const y = parseFloat(item.el.style.top) || 0;
            const h = parseFloat(item.el.style.height) || defaultH;
            if (y + h > maxY) maxY = y + h;
        });
        
        return { x: firstX, y: maxY + gap };
    }
}

export function autoOrganizeToGrid() {
    let items = [];
    if (selectedWorkbenchItems.size > 0) {
        items = Array.from(selectedWorkbenchItems).map(id => workbenchItems.get(id)).filter(item => item && item.type !== 'text-note');
    } else {
        items = Array.from(workbenchItems.values()).filter(item => item.type !== 'text-note');
    }
    
    if (items.length === 0) return;

    let minX = Infinity, minY = Infinity;
    items.forEach(item => {
        const x = parseFloat(item.el.style.left);
        const y = parseFloat(item.el.style.top);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
    });

    items.sort((a, b) => {
        const topA = parseFloat(a.el.style.top) || 0;
        const topB = parseFloat(b.el.style.top) || 0;
        const leftA = parseFloat(a.el.style.left) || 0;
        const leftB = parseFloat(b.el.style.left) || 0;
        
        if (Math.abs(topA - topB) < 200) { 
            return leftA - leftB;
        }
        return topA - topB;
    });

    const gap = 80;
    const maxRowWidth = 2200;
    
    let currentX = minX;
    let currentY = minY;
    let maxRowHeight = 0;

    items.forEach((item) => {
        const width = parseFloat(item.el.style.width) || 600;
        const height = parseFloat(item.el.style.height) || 600;

        if (currentX + width > minX + maxRowWidth && currentX > minX) {
            currentX = minX;
            currentY += maxRowHeight + gap;
            maxRowHeight = 0;
        }

        item.el.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        item.el.style.left = `${currentX}px`;
        item.el.style.top = `${currentY}px`;

        currentX += width + gap;
        maxRowHeight = Math.max(maxRowHeight, height);
        
        setTimeout(() => {
            if (item.el) item.el.style.transition = '';
        }, 500);
    });
    
    if (window.drawGenealogyConnections) {
        setTimeout(window.drawGenealogyConnections, 550);
    }
    
    if (window.historyManager) window.historyManager.pushState();
}

export function calculateSmartPosition(originalEl, direction = 1) {
    const baseX = parseFloat(originalEl.style.left) || 0;
    const baseY = parseFloat(originalEl.style.top) || 0;
    const itemWidth = parseFloat(originalEl.style.width) || 300;
    const itemHeight = parseFloat(originalEl.style.height) || 300;
    
    // 考虑工作台缩放，计算动态间距
    const spacingFactor = 1 / state.workbenchZoom;
    const baseSpacing = 40 * spacingFactor;
    
    switch(direction) {
        case 1: // 右侧
            return {
                x: baseX + itemWidth + baseSpacing,
                y: baseY
            };
        case 2: // 下方
            return {
                x: baseX,
                y: baseY + itemHeight + baseSpacing
            };
        case 3: // 左侧
            return {
                x: baseX - itemWidth - baseSpacing,
                y: baseY
            };
        case 4: // 上方
            return {
                x: baseX,
                y: baseY - itemHeight - baseSpacing
            };
        default:
            return { x: baseX + itemWidth + baseSpacing, y: baseY };
    }
}
