import { deleteSelectedItems, addImageToWorkbench, deleteWorkbenchItem, clearWorkbench, handleUploadImage, selectWorkbenchItem } from './workbench/items.js';
import { findNextPosition, autoOrganizeToGrid, calculateSmartPosition } from './workbench/layout.js';
import { createGroupLabel, restoreGroupLabelToWorkbench, addTextNoteToWorkbench, showTextAdjuster, setTextAlign } from './workbench/notes.js';
import { addAtmosphereNode } from './workbench/atmosphere.js';
import { startDrawingShape, handleShapeDrawingMove, handleShapeDrawingEnd, showShapeAdjuster, restoreShapeToWorkbench } from './workbench/shapes.js';
import { initWorkbenchEvents } from './workbench/events.js';
import { initContextMenu } from './context-menu.js';

// Expose to window for backward compatibility and main.js
window.deleteSelectedItems = deleteSelectedItems;
window.findNextPosition = findNextPosition;
window.calculateSmartPosition = calculateSmartPosition;
window.addImageToWorkbench = addImageToWorkbench;
window.selectWorkbenchItem = selectWorkbenchItem;
window.autoOrganizeToGrid = autoOrganizeToGrid;
window.createGroupLabel = createGroupLabel;
window.restoreGroupLabelToWorkbench = restoreGroupLabelToWorkbench;
window.addTextNoteToWorkbench = addTextNoteToWorkbench;
window.handleUploadImage = handleUploadImage;
window.deleteWorkbenchItem = deleteWorkbenchItem;
window.clearWorkbench = clearWorkbench;
window.addAtmosphereNode = addAtmosphereNode;
window.startDrawingShape = startDrawingShape;
window.handleShapeDrawingMove = handleShapeDrawingMove;
window.handleShapeDrawingEnd = handleShapeDrawingEnd;
window.showTextAdjuster = showTextAdjuster;
window.setTextAlign = setTextAlign;
window.showShapeAdjuster = showShapeAdjuster;
window.restoreShapeToWorkbench = restoreShapeToWorkbench;

// Initialize events
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initWorkbenchEvents();
        initContextMenu();
    });
} else {
    initWorkbenchEvents();
    initContextMenu();
}

export {
    deleteSelectedItems,
    findNextPosition,
    calculateSmartPosition,
    addImageToWorkbench,
    selectWorkbenchItem,
    autoOrganizeToGrid,
    createGroupLabel,
    restoreGroupLabelToWorkbench,
    addTextNoteToWorkbench,
    handleUploadImage,
    deleteWorkbenchItem,
    clearWorkbench,
    addAtmosphereNode,
    startDrawingShape,
    handleShapeDrawingMove,
    handleShapeDrawingEnd,
    showTextAdjuster,
    setTextAlign,
    showShapeAdjuster,
    restoreShapeToWorkbench
};
