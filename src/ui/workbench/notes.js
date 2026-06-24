import { state } from '../../core/state.js';
import { loadFontIfNeeded } from '../text-style-utils.js';

const { workbenchItems, selectedWorkbenchItems } = state;

export function createGroupLabel() {
    if (selectedWorkbenchItems.size === 0) {
        if (window.addMessage) window.addMessage({ sender: 'bot', content: '请先选中要编组的项目。' });
        return;
    }

    const selectedItems = Array.from(selectedWorkbenchItems).map(id => workbenchItems.get(id)).filter(item => item);
    if (selectedItems.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    selectedItems.forEach(item => {
        const el = item.el;
        const left = parseFloat(el.style.left);
        const top = parseFloat(el.style.top);
        const width = parseFloat(el.style.width) || el.offsetWidth;
        const height = parseFloat(el.style.height) || el.offsetHeight;

        minX = Math.min(minX, left);
        minY = Math.min(minY, top);
        maxX = Math.max(maxX, left + width);
        maxY = Math.max(maxY, top + height);
    });

    let maxNum = 0;
    workbenchItems.forEach(item => {
        if (item.type === 'group-label') {
            const content = item.el.querySelector('.label-content').innerText;
            const num = parseInt(content);
            if (!isNaN(num)) {
                maxNum = Math.max(maxNum, num);
            }
        }
    });
    const nextNum = maxNum + 1;

    const id = `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const wrapper = document.createElement('div');
    wrapper.className = 'workbench-item workbench-group-label';
    wrapper.dataset.itemId = id;
    
    const labelSize = 40;
    const centerX = (minX + maxX) / 2;
    
    const visualPadding = 20;
    const logicalPadding = visualPadding / state.workbenchZoom;
    const logicalLabelSize = labelSize / state.workbenchZoom;
    
    wrapper.style.left = `${centerX - (labelSize / 2) / state.workbenchZoom}px`;
    wrapper.style.top = `${minY - logicalLabelSize - logicalPadding}px`;
    wrapper.style.width = `${labelSize}px`;
    wrapper.style.height = `${labelSize}px`;

    wrapper.innerHTML = `<div class="label-content" contenteditable="false" spellcheck="false">${nextNum}</div>`;
    
    window.workbenchGrid.appendChild(wrapper);
    
    workbenchItems.set(id, {
        file: null,
        el: wrapper,
        type: 'group-label',
        id: id
    });

    const contentEl = wrapper.querySelector('.label-content');
    
    wrapper.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        wrapper.classList.add('editing');
        contentEl.contentEditable = "true";
        contentEl.focus();
        const range = document.createRange();
        range.selectNodeContents(contentEl);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    });

    contentEl.addEventListener('blur', () => {
        wrapper.classList.remove('editing');
        contentEl.contentEditable = "false";
        if (contentEl.innerText.trim() === '') {
            contentEl.innerText = '?';
        }
        if (window.historyManager) window.historyManager.pushState();
    });

    wrapper.addEventListener('mousedown', (e) => {
        if (wrapper.classList.contains('editing')) {
            e.stopPropagation();
            return;
        }
        if (e.button !== 0) return;
        if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(e);
    });

    const inverseScale = 1 / state.workbenchZoom;
    wrapper.style.transform = `scale(${inverseScale})`;
    
    if (window.historyManager) window.historyManager.pushState();
}

export function restoreGroupLabelToWorkbench(itemState) {
    const id = itemState.id;
    const wrapper = document.createElement('div');
    wrapper.className = 'workbench-item workbench-group-label';
    wrapper.dataset.itemId = id;
    
    const left = itemState.rect ? itemState.rect.left : itemState.left;
    if (left) wrapper.style.left = typeof left === 'number' ? `${left}px` : left;
    const top = itemState.rect ? itemState.rect.top : itemState.top;
    if (top) wrapper.style.top = typeof top === 'number' ? `${top}px` : top;
    const width = itemState.rect ? itemState.rect.width : itemState.width;
    if (width) wrapper.style.width = typeof width === 'number' ? `${width}px` : width;
    const height = itemState.rect ? itemState.rect.height : itemState.height;
    if (height) wrapper.style.height = typeof height === 'number' ? `${height}px` : height;
    const zIndex = itemState.rect ? itemState.rect.zIndex : itemState.zIndex;
    if (zIndex) wrapper.style.zIndex = zIndex;

    wrapper.innerHTML = `<div class="label-content" contenteditable="false" spellcheck="false">${itemState.content}</div>`;
    
    window.workbenchGrid.appendChild(wrapper);
    
    workbenchItems.set(id, {
        file: null,
        el: wrapper,
        type: 'group-label',
        id: id
    });

    const contentEl = wrapper.querySelector('.label-content');
    
    wrapper.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        wrapper.classList.add('editing');
        contentEl.contentEditable = "true";
        contentEl.focus();
        const range = document.createRange();
        range.selectNodeContents(contentEl);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    });

    contentEl.addEventListener('blur', () => {
        wrapper.classList.remove('editing');
        contentEl.contentEditable = "false";
        if (contentEl.innerText.trim() === '') {
            contentEl.innerText = '?';
        }
        if (window.historyManager) window.historyManager.pushState();
    });

    wrapper.addEventListener('mousedown', (e) => {
        if (wrapper.classList.contains('editing')) {
            e.stopPropagation();
            return;
        }
        if (e.button !== 0) return;
        if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(e);
    });

    const inverseScale = 1 / state.workbenchZoom;
    wrapper.style.transform = `scale(${inverseScale})`;
}

export function addTextNoteToWorkbench(x, y, restoreState = null) {
    const id = restoreState ? restoreState.id : `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const wrapper = document.createElement('div');
    wrapper.className = 'workbench-item workbench-text-note';
    wrapper.style.backgroundColor = 'transparent';
    wrapper.dataset.itemId = id;
    
    if (restoreState) {
        const left = restoreState.rect ? restoreState.rect.left : restoreState.left;
        if (left !== undefined && left !== null) wrapper.style.left = typeof left === 'number' ? `${left}px` : left;
        const top = restoreState.rect ? restoreState.rect.top : restoreState.top;
        if (top !== undefined && top !== null) wrapper.style.top = typeof top === 'number' ? `${top}px` : top;
        const width = restoreState.rect ? restoreState.rect.width : restoreState.width;
        if (width !== undefined && width !== null) wrapper.style.width = typeof width === 'number' ? `${width}px` : width;
        const height = restoreState.rect ? restoreState.rect.height : restoreState.height;
        if (height !== undefined && height !== null) wrapper.style.height = typeof height === 'number' ? `${height}px` : height;
        const zIndex = restoreState.rect ? restoreState.rect.zIndex : restoreState.zIndex;
        if (zIndex !== undefined && zIndex !== null) wrapper.style.zIndex = zIndex;
    } else {
        const rect = window.workbenchGrid.getBoundingClientRect();
        const dropX = (x - rect.left) / state.workbenchZoom;
        const dropY = (y - rect.top) / state.workbenchZoom;
        wrapper.style.left = `${dropX}px`;
        wrapper.style.top = `${dropY}px`;
    }
    
    let content = (restoreState && restoreState.content !== undefined) ? restoreState.content : '双击编辑文本';
    
    // Clean up any previously nested HTML bug deeply
    if (typeof content === 'string') {
        let tempDiv = document.createElement('div');
        tempDiv.innerHTML = content;
        let innerNode = tempDiv.querySelector('.note-content') || tempDiv.querySelector('.label-content');
        while (innerNode) {
            content = innerNode.innerHTML;
            tempDiv.innerHTML = content;
            innerNode = tempDiv.querySelector('.note-content') || tempDiv.querySelector('.label-content');
        }
    }
    
    wrapper.innerHTML = `
        <div class="note-content" contenteditable="false" spellcheck="false">${content}</div>
    `;
    
    const contentEl = wrapper.querySelector('.note-content');
    contentEl.style.minHeight = '0';
    
    // Store original CSS effects
    let customCssStr = '';
    if (restoreState && restoreState.css) {
        try {
            customCssStr = JSON.stringify(restoreState.css);
            wrapper.dataset.customCss = customCssStr;
        } catch(e){}
    }

    const applyCustomCss = (el, cssObj) => {
        if (!cssObj) return;
        console.log(`[applyCustomCss] Applying styles:`, cssObj);
        
        let hasContainerStyle = false;
        let requiresTextClip = false;
        let textFillColor = null;

        for (const key in cssObj) {
            try {
                // Convert camelCase to kebab-case (e.g. WebkitBackgroundClip to -webkit-background-clip)
                let kebabKey = key.replace(/([a-z0-9]|(?=[A-Z]))([A-Z])/g, '$1-$2').toLowerCase();
                if (kebabKey.startsWith('webkit-') || kebabKey.startsWith('moz-') || kebabKey.startsWith('ms-') || kebabKey.startsWith('o-')) {
                    kebabKey = '-' + kebabKey;
                }
                
                // If restoreState provided these specific standard text properties, don't overwrite them with the old custom CSS payload
                if (restoreState) {
                    if (kebabKey === 'font-size' && restoreState.fontSize) continue;
                    if (kebabKey === 'color' && restoreState.fontColor && restoreState.fontColor !== 'transparent') continue;
                    if (kebabKey === 'font-family' && restoreState.fontFamily) continue;
                    if (kebabKey === 'font-weight' && restoreState.fontWeight) continue;
                    if (kebabKey === 'font-style' && restoreState.fontStyle) continue;
                    if (kebabKey === 'line-height' && restoreState.lineHeight) continue;
                    if (kebabKey === 'letter-spacing' && restoreState.letterSpacing) continue;
                    if (kebabKey === 'text-shadow' && restoreState.textShadow) continue;
                    if ((kebabKey === '-webkit-text-stroke' || kebabKey === 'text-stroke') && (restoreState.WebkitTextStroke || restoreState.textStroke)) continue;
                    if (kebabKey === 'text-align' && restoreState.textAlign) continue;
                }
                
                if (kebabKey === 'padding') {
                     // The AI sometimes hallucinates padding. If we find it, just ignore it, 
                     // because width/height 100% flex centering will handle the visual padding much better.
                     continue;
                }

                if (kebabKey === 'transform') {
                    el.dataset.textVisualTransform = cssObj[key];
                    continue;
                }
                
                if (['border', 'background', 'padding'].some(p => kebabKey.startsWith(p)) && cssObj[key] !== 'transparent' && cssObj[key] !== 'none') {
                    // Do not treat background gradient as container if it's for text clipping
                    const isTextGradient = kebabKey.startsWith('background') && (cssObj['-webkit-background-clip'] === 'text' || cssObj['background-clip'] === 'text' || cssObj[key].includes('gradient'));
                    if (!isTextGradient) {
                        hasContainerStyle = true;
                    }
                }
                
                if (kebabKey === 'background-clip' || kebabKey === '-webkit-background-clip') {
                    if (cssObj[key] === 'text') requiresTextClip = true;
                }
                if (kebabKey === 'text-fill-color' || kebabKey === '-webkit-text-fill-color') {
                    textFillColor = cssObj[key];
                }

                el.style.setProperty(kebabKey, cssObj[key]);
                // Fallback for JS object assignment if setProperty fails or is ignored
                if (key in el.style) {
                    el.style[key] = cssObj[key];
                }
            } catch (e) {
                console.warn(`[applyCustomCss] Failed to apply ${key}: ${cssObj[key]}`, e);
            }
        }
        
        // Re-apply critical text clipping properties at the VERY END to ensure 'background' shorthands don't ruin them.
        if (requiresTextClip) {
            el.style.webkitBackgroundClip = 'text';
            el.style.backgroundClip = 'text';
            el.style.color = 'transparent'; // Fallback if no text-fill-color provided
        }
        if (textFillColor) {
            el.style.webkitTextFillColor = textFillColor;
            if (textFillColor === 'transparent') el.style.color = 'transparent';
        }
        
        if (hasContainerStyle) {
            el.style.display = 'flex';
            el.style.alignItems = 'center';
            el.style.justifyContent = 'center';
            el.style.width = '100%';
            el.style.height = '100%';
            el.style.boxSizing = 'border-box';
            el.style.textAlign = 'center';
        }
    };
    const removeCustomCss = (el, cssObj) => {
        if (!cssObj) return;
        let hasContainerStyle = false;
        for (const key in cssObj) {
            // keep standard font properties and color if present so it doesn't become invisible
            if (['color', 'font-family', 'font-weight', 'font-style', 'font-size', 'line-height', 'letter-spacing', 'text-align', 'text-shadow', '-webkit-text-stroke', 'text-stroke'].includes(key)) {
               continue; 
            }
            try {
                let kebabKey = key.replace(/([a-z0-9]|(?=[A-Z]))([A-Z])/g, '\$1-\$2').toLowerCase();
                if (kebabKey.startsWith('webkit-') || kebabKey.startsWith('moz-') || kebabKey.startsWith('ms-') || kebabKey.startsWith('o-')) {
                    kebabKey = '-' + kebabKey;
                }
                
                if (['border', 'background', 'padding'].some(p => kebabKey.startsWith(p))) {
                    const isTextGradient = kebabKey.startsWith('background') && (cssObj['-webkit-background-clip'] === 'text' || cssObj['background-clip'] === 'text');
                    if (!isTextGradient) {
                        hasContainerStyle = true;
                    }
                }
                
                if (['color', 'font-family', 'font-weight', 'font-style', 'font-size', 'line-height', 'letter-spacing', 'text-align', 'text-shadow', '-webkit-text-stroke', 'text-stroke'].includes(kebabKey)) continue;
                if (kebabKey === 'transform') {
                    el.dataset.textVisualTransform = '';
                    continue;
                }

                el.style.removeProperty(kebabKey);
                el.style.removeProperty(key);
                
                if (kebabKey === 'background-clip' || kebabKey === '-webkit-background-clip') {
                    el.style.webkitBackgroundClip = '';
                    el.style.backgroundClip = '';
                }
                if (kebabKey === 'text-fill-color' || kebabKey === '-webkit-text-fill-color') {
                    el.style.webkitTextFillColor = '';
                }
            } catch (e) {}
        }
        
        if (hasContainerStyle) {
            el.style.removeProperty('display');
            el.style.removeProperty('align-items');
            el.style.removeProperty('justify-content');
            el.style.removeProperty('width');
            el.style.removeProperty('height');
            el.style.removeProperty('box-sizing');
            el.style.removeProperty('text-align');
        }

        // Force fallback color if text fill is resetting transparent
        el.style.webkitTextFillColor = '';
        if (restoreState && restoreState.fontColor && el.style.color === 'transparent') {
            el.style.color = restoreState.fontColor;
        }
    };

    if (restoreState) {
        const textForSizing = restoreState.textContent || restoreState.content || content || '';
        const lineCountForSizing = Math.max(1, String(textForSizing).split('\n').length);
        const heightVal = restoreState.rect ? restoreState.rect.height : restoreState.height;
        const h = parseFloat(heightVal) || 50;
        const fallbackFontSize = Math.max(1, h / (lineCountForSizing * 1.1));
        const restoreFontSizeNum = parseFloat(restoreState.fontSize);
        const restoreFontSizeLooksInvalid = restoreState.fontSize &&
            fallbackFontSize >= 6 &&
            (!Number.isFinite(restoreFontSizeNum) || restoreFontSizeNum < fallbackFontSize * 0.35);
        if (restoreState.fontSize && !restoreFontSizeLooksInvalid) {
            contentEl.style.fontSize = restoreState.fontSize;
        } else {
            contentEl.style.fontSize = `${fallbackFontSize}px`;
        }
        if (restoreState.fontColor) contentEl.style.color = restoreState.fontColor;
        if (restoreState.fontFamily) {
            loadFontIfNeeded(restoreState.fontFamily);
            contentEl.style.fontFamily = restoreState.fontFamily;
        }
        if (restoreState.fontWeight) contentEl.style.fontWeight = restoreState.fontWeight;
        if (restoreState.fontStyle) contentEl.style.fontStyle = restoreState.fontStyle;
        if (restoreState.lineHeight && !restoreFontSizeLooksInvalid) contentEl.style.lineHeight = restoreState.lineHeight;
        else contentEl.style.lineHeight = `${Math.max(1, fallbackFontSize * 1.08)}px`;
        if (restoreState.letterSpacing) contentEl.style.letterSpacing = restoreState.letterSpacing;
        if (restoreState.textShadow) contentEl.style.textShadow = restoreState.textShadow;
        if (restoreState.WebkitTextStroke || restoreState.textStroke) {
            contentEl.style.webkitTextStroke = restoreState.WebkitTextStroke || restoreState.textStroke;
        }
        if (restoreState.textAlign) contentEl.style.textAlign = restoreState.textAlign;
        
        if (restoreState.css) {
            applyCustomCss(contentEl, restoreState.css);
        }
    }
    
    // Text enforcing and scaling
    const initialTextContent = restoreState ? (restoreState.textContent || restoreState.content || content || '') : content;
    if (!restoreState?.css?.whiteSpace) {
        if (initialTextContent && !initialTextContent.includes('\n')) {
            contentEl.style.whiteSpace = 'nowrap';
        } else {
            contentEl.style.whiteSpace = 'pre-wrap';
        }
    }
    const fitTextToBox = () => {
        const visualTransform = contentEl.dataset.textVisualTransform || '';
        const buildTransform = (scaleValue = 1, centerInWrapper = false) => {
            const transforms = [];
            if (centerInWrapper) transforms.push('translate(-50%, -50%)');
            if (visualTransform) transforms.push(visualTransform);
            if (scaleValue < 1) transforms.push(`scale(${scaleValue})`);
            return transforms.length ? transforms.join(' ') : 'none';
        };
        if (wrapper.classList.contains('editing')) {
            contentEl.style.transform = buildTransform();
            contentEl.style.transformOrigin = 'center center';
            contentEl.style.position = '';
            contentEl.style.top = '';
            contentEl.style.left = '';
            return;
        }
        contentEl.style.transform = buildTransform();
        contentEl.style.transformOrigin = 'center center';
        contentEl.style.position = '';
        contentEl.style.top = '';
        contentEl.style.left = '';
        
        // Wait a frame if width is 0
        if (wrapper.clientWidth === 0) {
            requestAnimationFrame(fitTextToBox);
            return;
        }

        const pushRenderDiagnostic = (extra = {}) => {
            if (typeof window === 'undefined') return;
            const computed = window.getComputedStyle(contentEl);
            window.__marmoTextRenderDiagnostics = window.__marmoTextRenderDiagnostics || [];
            window.__marmoTextRenderDiagnostics.push({
                id,
                text: initialTextContent,
                wrapper: {
                    width: wrapper.clientWidth,
                    height: wrapper.clientHeight
                },
                content: {
                    scrollWidth: contentEl.scrollWidth,
                    scrollHeight: contentEl.scrollHeight,
                    offsetWidth: contentEl.offsetWidth,
                    offsetHeight: contentEl.offsetHeight
                },
                computed: {
                    fontFamily: computed.fontFamily,
                    fontSize: computed.fontSize,
                    lineHeight: computed.lineHeight,
                    whiteSpace: computed.whiteSpace,
                    transform: computed.transform
                },
                inlineTransform: contentEl.style.transform || '',
                extra
            });
            if (window.__marmoTextRenderDiagnostics.length > 300) {
                window.__marmoTextRenderDiagnostics.splice(0, window.__marmoTextRenderDiagnostics.length - 300);
            }
        };

        const wrapperWidth = wrapper.clientWidth;
        const wrapperHeight = wrapper.clientHeight;
        const preFitComputed = window.getComputedStyle(contentEl);
        const currentFontSize = parseFloat(preFitComputed.fontSize);
        if (initialTextContent?.includes('\n') &&
            wrapperHeight > 0 &&
            (!Number.isFinite(currentFontSize) || currentFontSize < Math.max(6, wrapperHeight / 40))) {
            const lineCount = Math.max(1, String(initialTextContent).split('\n').length);
            const recoveredFontSize = Math.max(6, wrapperHeight / (lineCount * 1.12));
            contentEl.style.fontSize = `${recoveredFontSize}px`;
            contentEl.style.lineHeight = `${Math.max(1, recoveredFontSize * 1.08)}px`;
        }
        let contentWidth = contentEl.scrollWidth;
        let contentHeight = contentEl.scrollHeight;

        // Keep extracted display text visually centered inside the stable OCR bbox.
        // The wrapper is already a flex container; moving the wrapper itself corrupts
        // selection/resize coordinates, so only the inner text transform is adjusted.
        const align = window.getComputedStyle(contentEl).textAlign;
        let origin = 'left top';
        if (align === 'center') origin = 'center top';
        if (align === 'right') origin = 'right top';
        const whiteSpace = window.getComputedStyle(contentEl).whiteSpace;
        const compactText = String(initialTextContent || '').replace(/\s+/g, '');
        const isDisplayText = compactText.length > 0 &&
            compactText.length <= 32 &&
            /^[A-Z0-9&$.,:;!?%+-]+$/.test(compactText);
        const isSingleLineDisplayText = isDisplayText && !String(initialTextContent || '').includes('\n');
        const displayLines = String(initialTextContent || '').split('\n').map(line => line.trim()).filter(Boolean);
        const isDoubleLineDisplayText = isDisplayText &&
            displayLines.length === 2 &&
            displayLines.every(line => /^[A-Z0-9$.,:;!?%+-]{2,}$/.test(line));
        const isLargeSingleLineDisplayText = isSingleLineDisplayText &&
            Number.isFinite(currentFontSize) &&
            currentFontSize >= 40;
        let normalizedDisplayLineHeight = false;
        if (isLargeSingleLineDisplayText) {
            const computedLineHeight = parseFloat(window.getComputedStyle(contentEl).lineHeight);
            const targetLineHeight = currentFontSize * 0.98;
            if (!Number.isFinite(computedLineHeight) ||
                computedLineHeight > currentFontSize * 1.03 ||
                computedLineHeight < currentFontSize * 0.82) {
                contentEl.style.lineHeight = `${targetLineHeight}px`;
                normalizedDisplayLineHeight = true;
                contentWidth = contentEl.scrollWidth;
                contentHeight = contentEl.scrollHeight;
            }
        }
        const widthScale = wrapperWidth > 0 && contentWidth > 0 && contentWidth > wrapperWidth
            ? wrapperWidth / contentWidth
            : 1;
        const heightScale = wrapperHeight > 0 && contentHeight > 0 && contentHeight > wrapperHeight
            ? wrapperHeight / contentHeight
            : 1;
        const scale = Math.min(widthScale, heightScale);
        if (isDisplayText) {
            origin = 'center center';
            contentEl.style.transformOrigin = origin;
        }
        const displayHeightOverflowLimit = isSingleLineDisplayText
            ? 1.34
            : (isDoubleLineDisplayText ? 1.24 : 1.18);
        const displayHeightOverflowIsAcceptable = isDisplayText &&
            wrapperHeight > 0 &&
            contentHeight <= wrapperHeight * displayHeightOverflowLimit;
        const effectiveScale = displayHeightOverflowIsAcceptable ? widthScale : scale;
        const minReadableScale = isDisplayText ? 0.96 : (whiteSpace === 'pre' ? 0.72 : 0.92);
        if (effectiveScale < minReadableScale && whiteSpace === 'pre') {
            contentEl.style.whiteSpace = 'pre-wrap';
            contentWidth = contentEl.scrollWidth;
            contentHeight = contentEl.scrollHeight;
            const reflowWidthScale = wrapperWidth > 0 && contentWidth > 0 && contentWidth > wrapperWidth
                ? wrapperWidth / contentWidth
                : 1;
            const reflowHeightScale = wrapperHeight > 0 && contentHeight > 0 && contentHeight > wrapperHeight
                ? wrapperHeight / contentHeight
                : 1;
            const reflowScale = Math.max(Math.min(reflowWidthScale, reflowHeightScale), 0.92);
            if (reflowScale < 1) {
                contentEl.style.transform = buildTransform(reflowScale, false);
                contentEl.style.transformOrigin = origin;
            }
            pushRenderDiagnostic({ mode: 'reflow', scale, effectiveScale, reflowScale, minReadableScale });
            return;
        }
        const appliedScale = effectiveScale < 1 ? Math.max(effectiveScale, minReadableScale) : 1;
        if (isLargeSingleLineDisplayText) {
            contentEl.style.position = 'absolute';
            contentEl.style.left = '50%';
            contentEl.style.top = '50%';
            contentEl.style.transform = buildTransform(appliedScale, true);
            contentEl.style.transformOrigin = origin;
        } else if (effectiveScale < 1) {
            contentEl.style.transform = buildTransform(appliedScale, false);
            contentEl.style.transformOrigin = origin;
        }

        pushRenderDiagnostic({
            mode: 'fit',
            scale,
            effectiveScale,
            appliedScale,
            minReadableScale,
            displayHeightOverflowIsAcceptable,
            transformOrigin: origin,
            normalizedDisplayLineHeight
        });
    };
    
    requestAnimationFrame(fitTextToBox);

    // Re-evaluate on resize wrapper
    const resizeObserver = new ResizeObserver(() => {
        fitTextToBox();
    });
    resizeObserver.observe(wrapper);

    window.workbenchGrid.appendChild(wrapper);
    
    workbenchItems.set(id, {
        file: null,
        el: wrapper,
        type: 'text-note',
        id: id,
        parentId: restoreState ? restoreState.parentId : null,
        css: restoreState && restoreState.css ? restoreState.css : null
    });
    
    wrapper.addEventListener('dblclick', (e) => {
        wrapper.classList.add('editing');
        const customCssObj = workbenchItems.get(id)?.css;
        if (customCssObj) removeCustomCss(contentEl, customCssObj);
        contentEl.contentEditable = "true";
        contentEl.style.whiteSpace = 'pre-wrap';
        contentEl.focus();
        const range = document.createRange();
        range.selectNodeContents(contentEl);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    });
    
    contentEl.addEventListener('blur', () => {
        wrapper.classList.remove('editing');
        contentEl.contentEditable = "false";
        if (contentEl.innerText.trim() === '') {
            contentEl.innerText = '双击编辑文本';
        }
        if (!contentEl.innerText.includes('\n')) {
            contentEl.style.whiteSpace = 'nowrap';
        } else {
            contentEl.style.whiteSpace = 'pre';
        }
        const customCssObj = workbenchItems.get(id)?.css;
        if (customCssObj) applyCustomCss(contentEl, customCssObj);
        fitTextToBox();
        if (window.historyManager) window.historyManager.pushState();
    });

    wrapper.addEventListener('mousedown', (e) => {
        if (wrapper.classList.contains('editing')) {
            e.stopPropagation();
            return;
        }
        if (e.button !== 0) return;
        if (window.handleWorkbenchDragStart) window.handleWorkbenchDragStart(e);
    });
    
    wrapper.addEventListener('click', (e) => {
        if (!wrapper.classList.contains('editing')) {
            if (window.showTextAdjuster) window.showTextAdjuster(wrapper, e.clientX, e.clientY);
        }
    });

    ['nw', 'ne', 'sw', 'se'].forEach(dir => {
        const handle = document.createElement('div');
        handle.className = `resize-handle resize-${dir}`;
        handle.addEventListener('mousedown', (e) => {
            if (e.button !== 0 || state.isSpacePressed) return; 
            if (window.handleResizeStart) window.handleResizeStart(e, dir, wrapper);
        });
        wrapper.appendChild(handle);
    });
    
    const inverseScale = 1 / state.workbenchZoom;
    wrapper.querySelectorAll('.resize-handle').forEach(handle => {
        handle.style.transform = `scale(${inverseScale})`;
    });
    
    if (!restoreState) {
        if (window.showTextAdjuster) window.showTextAdjuster(wrapper, x, y);
    }
}

export function showTextAdjuster(textEl, clientX, clientY) {
    const textAdjuster = document.getElementById('textAdjuster');
    const textAdjusterColor = document.getElementById('textAdjusterColor');
    const textAdjusterFont = document.getElementById('textAdjusterFont');
    const textAdjusterWeight = document.getElementById('textAdjusterWeight');
    const textAdjusterSize = document.getElementById('textAdjusterSize');
    const textAdjusterSizeVal = document.getElementById('textAdjusterSizeVal');
    const textAlignLeft = document.getElementById('textAlignLeft');
    const textAlignCenter = document.getElementById('textAlignCenter');
    const textAlignRight = document.getElementById('textAlignRight');

    window.currentAdjustingText = textEl;
    const contentEl = textEl.querySelector('.note-content');
    
    // Read current values
    const color = contentEl.style.color || '#000000';
    const font = contentEl.style.fontFamily || 'sans-serif';
    const weight = contentEl.style.fontWeight || 'normal';
    const size = parseInt(contentEl.style.fontSize) || 24;
    const align = contentEl.style.textAlign || 'left';
    
    // Convert rgb to hex if needed
    let hexColor = color;
    if (color.startsWith('rgb')) {
        const rgb = color.match(/\d+/g);
        if (rgb && rgb.length === 3) {
            hexColor = '#' + rgb.map(x => parseInt(x).toString(16).padStart(2, '0')).join('');
        }
    }
    
    if (textAdjusterColor) textAdjusterColor.value = hexColor;
    if (textAdjusterFont) textAdjusterFont.value = font.replace(/['"]/g, '');
    if (textAdjusterWeight) textAdjusterWeight.value = weight;
    if (textAdjusterSize) textAdjusterSize.value = size;
    if (textAdjusterSizeVal) textAdjusterSizeVal.textContent = `${size}px`;
    
    [textAlignLeft, textAlignCenter, textAlignRight].forEach(btn => btn.classList.remove('active'));
    if (align === 'center') textAlignCenter.classList.add('active');
    else if (align === 'right') textAlignRight.classList.add('active');
    else textAlignLeft.classList.add('active');
    
    textAdjuster.style.display = 'flex';
    
    // Position adjuster
    requestAnimationFrame(() => {
        const rect = textAdjuster.getBoundingClientRect();
        let posX = clientX + 10;
        let posY = clientY + 10;
        
        if (posX + rect.width > window.innerWidth) {
            posX = window.innerWidth - rect.width - 10;
        }
        if (posY + rect.height > window.innerHeight) {
            posY = window.innerHeight - rect.height - 10;
        }
        
        textAdjuster.style.left = `${posX}px`;
        textAdjuster.style.top = `${posY}px`;
    });
}

export function setTextAlign(align) {
    if (!window.currentAdjustingText) return;
    const contentEl = window.currentAdjustingText.querySelector('.note-content');
    if (contentEl) contentEl.style.textAlign = align;
    
    const textAlignLeft = document.getElementById('textAlignLeft');
    const textAlignCenter = document.getElementById('textAlignCenter');
    const textAlignRight = document.getElementById('textAlignRight');
    
    [textAlignLeft, textAlignCenter, textAlignRight].forEach(btn => btn.classList.remove('active'));
    if (align === 'center') textAlignCenter.classList.add('active');
    else if (align === 'right') textAlignRight.classList.add('active');
    else textAlignLeft.classList.add('active');
}
