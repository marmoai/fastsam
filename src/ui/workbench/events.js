import { state } from '../../core/state.js';
import { createGroupLabel } from './notes.js';
import { autoOrganizeToGrid } from './layout.js';
import { startDrawingShape, handleShapeDrawingMove, handleShapeDrawingEnd } from './shapes.js';
import { addTextNoteToWorkbench, setTextAlign } from './notes.js';
import { handleUploadImage, addImageToWorkbench } from './items.js';

const { workbenchItems } = state;

export function initWorkbenchEvents() {
    const workbenchPanel = document.getElementById('workbench');

    window.workbenchGrid.addEventListener('mousedown', (e) => {
        if (state.isCropping) {
            if (e.target === window.workbenchGrid || e.target.id === 'workbenchZoomContainer') {
                window.exitCroppingMode(true);
            }
            return;
        }
        if (window.currentDrawingShape) {
            if (e.button !== 0) return; // Only left click
            e.stopPropagation();
            
            if (window.currentDrawingShape === 'text') {
                addTextNoteToWorkbench(e.clientX, e.clientY);
                window.currentDrawingShape = null;
                window.workbenchGrid.style.cursor = 'grab';
                return;
            }
            
            const rect = window.workbenchGrid.getBoundingClientRect();
            const startX = (e.clientX - rect.left) / state.workbenchZoom;
            const startY = (e.clientY - rect.top) / state.workbenchZoom;
            
            window.drawingStartPos = { x: startX, y: startY, clientX: e.clientX, clientY: e.clientY };
            
            const id = `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            window.drawingElement = document.createElement('div');
            window.drawingElement.className = `workbench-item workbench-shape shape-${window.currentDrawingShape}`;
            window.drawingElement.dataset.itemId = id;
            window.drawingElement.dataset.shapeType = window.currentDrawingShape;
            
            window.drawingElement.style.left = `${startX}px`;
            window.drawingElement.style.top = `${startY}px`;
            window.drawingElement.style.width = '0px';
            window.drawingElement.style.height = '0px';
            window.drawingElement.style.backgroundColor = 'transparent';
            
            if (window.currentDrawingShape === 'rect' || window.currentDrawingShape === 'ellipse') {
                window.drawingElement.style.border = '2px solid var(--primary-color)';
                if (window.currentDrawingShape === 'ellipse') {
                    window.drawingElement.style.borderRadius = '50%';
                }
            } else if (window.currentDrawingShape === 'line' || window.currentDrawingShape === 'arrow') {
                window.drawingElement.innerHTML = `
                    <svg width="100%" height="100%" style="overflow: visible; position: absolute; top: 0; left: 0;">
                        ${window.currentDrawingShape === 'arrow' ? `
                        <defs>
                            <marker id="arrowhead-${id}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="var(--primary-color)" />
                            </marker>
                        </defs>
                        ` : ''}
                        <line x1="0" y1="0" x2="0" y2="0" stroke="var(--primary-color)" stroke-width="2" vector-effect="non-scaling-stroke"
                              ${window.currentDrawingShape === 'arrow' ? `marker-end="url(#arrowhead-${id})"` : ''} />
                    </svg>
                `;
                window.drawingElement.style.pointerEvents = 'none';
            }
            
            window.workbenchGrid.appendChild(window.drawingElement);
            
            document.addEventListener('mousemove', handleShapeDrawingMove);
            document.addEventListener('mouseup', handleShapeDrawingEnd);
        }
    });

    // Text Adjuster Listeners
    const textAdjusterColor = document.getElementById('textAdjusterColor');
    const textAdjusterFont = document.getElementById('textAdjusterFont');
    const textAdjusterWeight = document.getElementById('textAdjusterWeight');
    const textAdjusterSize = document.getElementById('textAdjusterSize');
    const textAdjusterSizeVal = document.getElementById('textAdjusterSizeVal');
    const textAlignLeft = document.getElementById('textAlignLeft');
    const textAlignCenter = document.getElementById('textAlignCenter');
    const textAlignRight = document.getElementById('textAlignRight');

    if (textAdjusterColor) {
        textAdjusterColor.addEventListener('input', (e) => {
            if (!window.currentAdjustingText) return;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) contentEl.style.color = e.target.value;
        });
        textAdjusterColor.addEventListener('change', () => {
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    if (textAdjusterFont) {
        textAdjusterFont.addEventListener('change', (e) => {
            if (!window.currentAdjustingText) return;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) contentEl.style.fontFamily = e.target.value;
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    if (textAdjusterWeight) {
        textAdjusterWeight.addEventListener('change', (e) => {
            if (!window.currentAdjustingText) return;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) contentEl.style.fontWeight = e.target.value;
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    if (textAdjusterSize) {
        textAdjusterSize.addEventListener('input', (e) => {
            if (!window.currentAdjustingText) return;
            const size = e.target.value;
            if (textAdjusterSizeVal) textAdjusterSizeVal.textContent = `${size}px`;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) contentEl.style.fontSize = `${size}px`;
        });
        textAdjusterSize.addEventListener('change', () => {
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    const textAdjusterSizeMinus = document.getElementById('textAdjusterSizeMinus');
    const textAdjusterSizePlus = document.getElementById('textAdjusterSizePlus');
    
    if (textAdjusterSizeMinus) {
        textAdjusterSizeMinus.addEventListener('click', () => {
            if (!window.currentAdjustingText) return;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) {
                let currentSize = parseInt(contentEl.style.fontSize) || 24;
                if (!isNaN(currentSize) && currentSize > 1) {
                    currentSize -= 1;
                    contentEl.style.fontSize = `${currentSize}px`;
                    if (textAdjusterSizeVal) textAdjusterSizeVal.textContent = `${currentSize}px`;
                    if (textAdjusterSize) textAdjusterSize.value = currentSize;
                    if (window.historyManager) window.historyManager.pushState();
                }
            }
        });
    }

    if (textAdjusterSizePlus) {
        textAdjusterSizePlus.addEventListener('click', () => {
            if (!window.currentAdjustingText) return;
            const contentEl = window.currentAdjustingText.querySelector('.note-content');
            if (contentEl) {
                let currentSize = parseInt(contentEl.style.fontSize) || 24;
                if (!isNaN(currentSize) && currentSize < 900) {
                    currentSize += 1;
                    contentEl.style.fontSize = `${currentSize}px`;
                    if (textAdjusterSizeVal) textAdjusterSizeVal.textContent = `${currentSize}px`;
                    if (textAdjusterSize) textAdjusterSize.value = currentSize;
                    if (window.historyManager) window.historyManager.pushState();
                }
            }
        });
    }

    if (textAlignLeft) textAlignLeft.addEventListener('click', () => { setTextAlign('left'); if (window.historyManager) window.historyManager.pushState(); });
    if (textAlignCenter) textAlignCenter.addEventListener('click', () => { setTextAlign('center'); if (window.historyManager) window.historyManager.pushState(); });
    if (textAlignRight) textAlignRight.addEventListener('click', () => { setTextAlign('right'); if (window.historyManager) window.historyManager.pushState(); });

    document.addEventListener('mousedown', (e) => {
        const textAdjuster = document.getElementById('textAdjuster');
        if (textAdjuster && textAdjuster.style.display === 'flex') {
            if (!textAdjuster.contains(e.target) && !e.target.closest('.workbench-text-note')) {
                textAdjuster.style.display = 'none';
                window.currentAdjustingText = null;
            }
        }
    });

    // Shape Adjuster Listeners
    const shapeAdjusterColor = document.getElementById('shapeAdjusterColor');
    const shapeAdjusterWidth = document.getElementById('shapeAdjusterWidth');
    const shapeAdjusterWidthVal = document.getElementById('shapeAdjusterWidthVal');

    if (shapeAdjusterColor) {
        shapeAdjusterColor.addEventListener('input', (e) => {
            if (!window.currentAdjustingShape) return;
            const color = e.target.value;
            const shapeType = window.currentAdjustingShape.dataset.shapeType;
            
            if (shapeType === 'rect' || shapeType === 'ellipse' || shapeType === 'polygon') {
                if (window.currentAdjustingShape.style.backgroundColor && window.currentAdjustingShape.style.backgroundColor !== 'transparent') {
                    window.currentAdjustingShape.style.backgroundColor = color;
                } else {
                    window.currentAdjustingShape.style.borderColor = color;
                }
            } else if (shapeType === 'line' || shapeType === 'arrow') {
                const line = window.currentAdjustingShape.querySelector('line');
                if (line) line.setAttribute('stroke', color);
                const polygon = window.currentAdjustingShape.querySelector('polygon');
                if (polygon) polygon.setAttribute('fill', color);
            }
        });
        shapeAdjusterColor.addEventListener('change', () => {
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    if (shapeAdjusterWidth) {
        shapeAdjusterWidth.addEventListener('input', (e) => {
            if (!window.currentAdjustingShape) return;
            const width = e.target.value;
            if (shapeAdjusterWidthVal) shapeAdjusterWidthVal.textContent = `${width}px`;
            const shapeType = window.currentAdjustingShape.dataset.shapeType;
            
            if (shapeType === 'rect' || shapeType === 'ellipse') {
                window.currentAdjustingShape.style.borderWidth = `${width}px`;
            } else if (shapeType === 'line' || shapeType === 'arrow') {
                const line = window.currentAdjustingShape.querySelector('line');
                if (line) line.setAttribute('stroke-width', width);
            }
        });
        shapeAdjusterWidth.addEventListener('change', () => {
            if (window.historyManager) window.historyManager.pushState();
        });
    }

    document.addEventListener('mousedown', (e) => {
        const shapeAdjuster = document.getElementById('shapeAdjuster');
        if (shapeAdjuster && shapeAdjuster.style.display === 'flex') {
            if (!shapeAdjuster.contains(e.target) && !e.target.closest('.workbench-shape')) {
                shapeAdjuster.style.display = 'none';
                window.currentAdjustingShape = null;
            }
        }
        
        const floatingFusionEditor = document.getElementById('floatingFusionEditor');
        const workbenchToolbox = document.getElementById('workbenchToolbox');
        const clickedInsideToast = e.target.closest('#successFeedbackToast');
        const isClickingCapsule = e.target.closest('#aiInspirationCapsule') || e.target.closest('#capsuleSpeechTooltip');

        if (floatingFusionEditor) {
            if (!floatingFusionEditor.contains(e.target) && !e.target.closest('.workbench-item') && (!workbenchToolbox || !workbenchToolbox.contains(e.target)) && !clickedInsideToast && !isClickingCapsule) {
                floatingFusionEditor.classList.add('idle');
            }
        }

        if (workbenchToolbox && workbenchToolbox.style.display === 'flex') {
            if (!workbenchToolbox.contains(e.target) && !e.target.closest('.workbench-item') && (!floatingFusionEditor || !floatingFusionEditor.contains(e.target)) && !clickedInsideToast && !isClickingCapsule) {
                if (typeof window.hideWorkbenchToolbox === 'function') {
                    window.hideWorkbenchToolbox();
                }
            }
        }
    });

    // Global paste listener for images
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.isCropping) {
            window.exitCroppingMode(true);
        }
    });

    document.addEventListener('paste', (e) => {
        // Don't intercept if user is typing in an input or contenteditable
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
            return;
        }
        
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let index in items) {
            const item = items[index];
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                const file = new File([blob], "pasted-image.png", { type: blob.type });
                
                // Paste at the center of the current view
                const rect = window.workbenchGrid.getBoundingClientRect();
                const centerX = (window.innerWidth / 2 - rect.left) / state.workbenchZoom;
                const centerY = (window.innerHeight / 2 - rect.top) / state.workbenchZoom;
                
                addImageToWorkbench(file, 'Pasted Image').then(id => {
                    setTimeout(() => {
                        const wbItem = workbenchItems.get(id);
                        if (wbItem && wbItem.el) {
                            wbItem.el.style.left = `${centerX - 100}px`; // Offset by half of default width
                            wbItem.el.style.top = `${centerY - 100}px`;
                            if (window.historyManager) window.historyManager.pushState();
                        }
                    }, 50);
                });
                
                e.preventDefault();
                break;
            }
        }
    });
}
