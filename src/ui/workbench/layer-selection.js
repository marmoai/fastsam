export function applyLayerSelectionVisual(layerEl, selected, hasCutout = false) {
    if (!layerEl) return;
    layerEl.classList.toggle('selected', !!selected);
    layerEl.style.borderRadius = '4px';
    layerEl.style.boxShadow = 'none';
    layerEl.style.filter = selected
        ? 'drop-shadow(0 0 8px rgba(79, 70, 229, 0.75))'
        : 'none';
    layerEl.style.backgroundColor = !hasCutout && selected
        ? 'rgba(79, 70, 229, 0.08)' : 'transparent';
    layerEl.style.border = !hasCutout && selected
        ? '1.5px solid rgba(79, 70, 229, 0.85)' : 'none';
}

export function updateCanvasLayerSelection(item, layerIndex, selected) {
    const layers = item.scene?.layers || item.layers || [];
    const layer = layers[layerIndex];
    const container = item.el?.querySelector('.canvas-layers-container');
    if (!layer || !container) return;

    // Hidden layers and standalone children leave gaps in the rendered list.
    // DOM position is never a semantic layer index.
    const matches = el => layer.id != null
        ? el.dataset.layerId === String(layer.id)
        : el.dataset.layerIndex === String(layerIndex);
    const layerEl = [...container.querySelectorAll('.canvas-layer')].find(matches);
    const outline = [...container.querySelectorAll('.canvas-layer-selection-outline')].find(matches);
    if (layerEl) {
        outline?.remove();
        applyLayerSelectionVisual(layerEl, selected, !!layerEl.querySelector('.cutout-img'));
        return;
    }
    if (!selected) {
        outline?.remove();
        return;
    }

    const bbox = layer.bbox;
    if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isFinite)) return;
    const [ymin, xmin, ymax, xmax] = bbox;
    if (ymax <= ymin || xmax <= xmin) return;
    const frame = outline || document.createElement('div');
    frame.className = 'canvas-layer-selection-outline';
    frame.dataset.layerId = String(layer.id ?? `layer-${layerIndex}`);
    frame.dataset.layerIndex = String(layerIndex);
    frame.setAttribute('aria-hidden', 'true');
    Object.assign(frame.style, {
        position: 'absolute', top: `${ymin / 10}%`, left: `${xmin / 10}%`,
        width: `${(xmax - xmin) / 10}%`, height: `${(ymax - ymin) / 10}%`,
        boxSizing: 'border-box', pointerEvents: 'none', zIndex: String(layers.length + 1)
    });
    applyLayerSelectionVisual(frame, true);
    if (!outline) container.appendChild(frame);
}
