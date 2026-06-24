import { state } from '../core/state.js';
import { screenToWorkbenchCoord } from './interaction.js';
import { addAtmosphereNode } from './workbench-core.js';

let draggedPreview = null;
let touchDragInfo = null;

export function handlePreviewDragStart(event, info) {
    draggedPreview = info;
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', 'drag');
}

export function handlePreviewDragEnter(event) {
    event.preventDefault();
    const targetWrapper = event.target.closest('.preview-item-wrapper');
    if (targetWrapper && draggedPreview && targetWrapper.dataset.fileId !== document.querySelector(`[data-preview-role='${draggedPreview.role}'][data-ref-index='${draggedPreview.index}']`)?.dataset.fileId) {
        targetWrapper.classList.add('drop-target-highlight');
    }
}

export function handlePreviewDragLeave(event) {
    event.target.closest('.preview-item-wrapper')?.classList.remove('drop-target-highlight');
}

export function handlePreviewDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
}

export function handlePreviewDragEnd() {
    document.querySelectorAll('.drop-target-highlight').forEach(el => el.classList.remove('drop-target-highlight'));
    draggedPreview = null;
}

export function handlePreviewDrop(event, targetInfo) {
    event.preventDefault();
    event.target.closest('.preview-item-wrapper')?.classList.remove('drop-target-highlight');
    if (!draggedPreview || (draggedPreview.role === 'ref' && targetInfo.role === 'ref' && draggedPreview.index === targetInfo.index)) {
        draggedPreview = null;
        return;
    }
    if (draggedPreview.role === 'ref' && targetInfo.role === 'base') promoteReferenceToBase(draggedPreview.index);
    else if (draggedPreview.role === 'base' && targetInfo.role === 'ref') promoteReferenceToBase(targetInfo.index);
    else if (draggedPreview.role === 'ref' && targetInfo.role === 'ref') moveReferenceFile(draggedPreview.index, targetInfo.index);
    draggedPreview = null;
    
    if (window.updateImagePreview) window.updateImagePreview();
    if (window.updateSendBtnState) window.updateSendBtnState();
}

export function handleGridDrop(event) {
    // Drop handled via onMouseUp logic in handleWorkbenchDragStart
}

export function handleTouchDragStart(event, info) {
    if (event.target.tagName === 'BUTTON' || event.target.closest('button')) return;
    if (event.touches.length !== 1) return;
    const wrapper = event.target.closest('.preview-item-wrapper');
    if (!wrapper) return;

    event.preventDefault(); 
    
    touchDragInfo = {
        role: info.role, index: info.refIndex, element: wrapper,
        clone: wrapper.cloneNode(true),
        offsetX: event.touches[0].clientX - wrapper.getBoundingClientRect().left,
        offsetY: event.touches[0].clientY - wrapper.getBoundingClientRect().top,
        lastTarget: null
    };
    const cloneStyle = touchDragInfo.clone.style;
    cloneStyle.position = 'absolute';
    cloneStyle.zIndex = '10000';
    cloneStyle.pointerEvents = 'none';
    cloneStyle.opacity = '0.7';
    cloneStyle.width = wrapper.offsetWidth + 'px';
    cloneStyle.height = wrapper.offsetHeight + 'px';
    document.body.appendChild(touchDragInfo.clone);
    cloneStyle.left = (event.touches[0].clientX - touchDragInfo.offsetX) + 'px';
    cloneStyle.top = (event.touches[0].clientY - touchDragInfo.offsetY) + 'px';
    wrapper.style.opacity = '0.4';

    window.addEventListener('touchmove', handleTouchDragMove, { passive: false });
    window.addEventListener('touchend', handleTouchDragEnd, { once: true });
}

export function handleTouchDragMove(event) {
    if (!touchDragInfo) return;
    event.preventDefault(); 
    const touch = event.touches[0];
    const cloneStyle = touchDragInfo.clone.style;
    cloneStyle.left = (touch.clientX - touchDragInfo.offsetX) + 'px';
    cloneStyle.top = (touch.clientY - touchDragInfo.offsetY) + 'px';
    cloneStyle.display = 'none';
    const elementUnder = document.elementFromPoint(touch.clientX, touch.clientY);
    cloneStyle.display = 'block';

    const currentTarget = elementUnder ? elementUnder.closest('.preview-item-wrapper') : null;
    if (touchDragInfo.lastTarget && touchDragInfo.lastTarget !== currentTarget) {
        touchDragInfo.lastTarget.classList.remove('drop-target-highlight');
    }
    if (currentTarget && currentTarget !== touchDragInfo.element) {
        currentTarget.classList.add('drop-target-highlight');
        touchDragInfo.lastTarget = currentTarget;
    } else if (currentTarget === touchDragInfo.element) {
        if (touchDragInfo.lastTarget) touchDragInfo.lastTarget.classList.remove('drop-target-highlight');
        touchDragInfo.lastTarget = null;
    }
}

export function handleTouchDragEnd() {
    if (!touchDragInfo) return;
    window.removeEventListener('touchmove', handleTouchDragMove);
    if (touchDragInfo.lastTarget) {
        touchDragInfo.lastTarget.classList.remove('drop-target-highlight');
        const targetInfo = { role: touchDragInfo.lastTarget.dataset.previewRole, index: parseInt(touchDragInfo.lastTarget.dataset.refIndex, 10) };
        const draggedInfo = { role: touchDragInfo.role, index: touchDragInfo.index };
        if (draggedInfo.role === 'ref' && targetInfo.role === 'base') promoteReferenceToBase(draggedInfo.index);
        else if (draggedInfo.role === 'base' && targetInfo.role === 'ref') promoteReferenceToBase(targetInfo.index);
        else if (draggedInfo.role === 'ref' && targetInfo.role === 'ref') moveReferenceFile(draggedInfo.index, targetInfo.index);
        
        if (window.updateImagePreview) window.updateImagePreview();
        if (window.updateSendBtnState) window.updateSendBtnState();
    }
    touchDragInfo.element.style.opacity = '1';
    document.body.removeChild(touchDragInfo.clone);
    touchDragInfo = null;
}

export function moveReferenceFile(fromIndex, toIndex) {
    const { referenceImageFiles } = state;
    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || fromIndex >= referenceImageFiles.length || toIndex > referenceImageFiles.length) return;
    const [moved] = referenceImageFiles.splice(fromIndex, 1);
    if (moved) referenceImageFiles.splice(toIndex, 0, moved);
}

export function promoteReferenceToBase(refIndex) {
    const { referenceImageFiles, pendingReferenceImageShares } = state;
    if (refIndex < 0 || refIndex >= referenceImageFiles.length) return;
    const [selected] = referenceImageFiles.splice(refIndex, 1);
    if (!selected) return;
    pendingReferenceImageShares.delete(selected);
    
    if (state.mainImageFile) referenceImageFiles.splice(refIndex, 0, state.mainImageFile);
    else if (state.lastGeneratedImageForEditing) {
        referenceImageFiles.splice(refIndex, 0, state.lastGeneratedImageForEditing);
        state.lastGeneratedImageForEditing = null;
    }
    state.mainImageFile = selected;
    state.maskDataUrl = null;
    state.pendingBaseImageShare = true;
}

export function initWorkbenchDrop(workbenchGrid) {
    workbenchGrid.addEventListener('dragover', (e) => e.preventDefault());
    workbenchGrid.addEventListener('drop', async (e) => {
        e.preventDefault();
        const rawData = e.dataTransfer.getData('text/plain');
        if (!rawData || !rawData.startsWith('{')) return;

        const data = JSON.parse(rawData);
        if (data.type === 'atmosphere') {
            const coord = screenToWorkbenchCoord(e.clientX, e.clientY);
            addAtmosphereNode(data.key, coord.x, coord.y);
        } else if (data.type === 'independent_asset') {
            const coord = screenToWorkbenchCoord(e.clientX, e.clientY);
            // Use dynamic import or pass as callback if addImageToWorkbench isn't readily available
            // Wait, we need addImageToWorkbench
            if (window.addImageToWorkbench) {
                window.addImageToWorkbench(data.sourceImage, {
                    type: 'image',
                    name: data.name,
                    source: 'mvr_asset',
                    x: coord.x,
                    y: coord.y
                });
            }
        }
    });
}
