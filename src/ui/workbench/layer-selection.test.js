// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { updateCanvasLayerSelection } from './layer-selection.js';
import { renderCanvasLayers } from './layers.js';
import { state } from '../../core/state.js';

vi.mock('../../core/state.js', () => ({ state: { workbenchItems: new Map() } }));
vi.mock('../modals.js', () => ({ showLayerEditPrompt: vi.fn() }));
vi.mock('./layer-assets.js', () => ({ editLayerAsset: vi.fn() }));
vi.mock('../../runtime/CoreRuntime', () => ({ runtime: { getCurrentWorkspace: () => null } }));
vi.mock('../../services/workspace-context.js', () => ({
    syncWorkspaceContext: vi.fn(), recordWorkspaceAction: vi.fn()
}));

function fixture() {
    const el = document.createElement('div');
    el.innerHTML = `<div class="canvas-layers-container">
        <div class="canvas-layer" data-layer-id="header" data-layer-index="1"></div>
        <div class="canvas-layer" data-layer-id="badge" data-layer-index="2"></div>
    </div>`;
    return {
        el,
        scene: { layers: [
            { id: 'subject', bbox: [250, 0, 1000, 1000], visible: false },
            { id: 'header', bbox: [50, 300, 160, 700] },
            { id: 'badge', bbox: [800, 800, 900, 900] }
        ] }
    };
}

describe('canvas layer selection with omitted layers', () => {
    it('matches identity instead of a compacted DOM index', () => {
        const item = fixture();
        updateCanvasLayerSelection(item, 1, true);
        expect(item.el.querySelector('[data-layer-id="header"]').classList.contains('selected')).toBe(true);
        expect(item.el.querySelector('[data-layer-id="badge"]').classList.contains('selected')).toBe(false);
    });

    it('outlines hidden/extracted subjects without showing another cutout', () => {
        const item = fixture();
        const before = JSON.stringify(item.scene);
        updateCanvasLayerSelection(item, 0, true);
        const outline = item.el.querySelector('.canvas-layer-selection-outline');
        expect(outline.dataset.layerId).toBe('subject');
        expect(outline.style.top).toBe('25%');
        expect(outline.style.height).toBe('75%');
        expect(outline.style.pointerEvents).toBe('none');
        expect(item.el.querySelectorAll('.canvas-layer.selected')).toHaveLength(0);
        expect(item.el.querySelectorAll('img')).toHaveLength(0);
        expect(JSON.stringify(item.scene)).toBe(before);
        updateCanvasLayerSelection(item, 0, true);
        expect(item.el.querySelectorAll('.canvas-layer-selection-outline')).toHaveLength(1);
        updateCanvasLayerSelection(item, 0, false);
        expect(item.el.querySelector('.canvas-layer-selection-outline')).toBeNull();
    });

    it('keeps simultaneous selections independent', () => {
        const item = fixture();
        updateCanvasLayerSelection(item, 0, true);
        updateCanvasLayerSelection(item, 1, true);
        updateCanvasLayerSelection(item, 0, false);
        expect(item.el.querySelector('[data-layer-id="header"]').classList.contains('selected')).toBe(true);
    });

    it('uses recorded indices for layers without IDs', () => {
        const item = fixture();
        delete item.scene.layers[2].id;
        updateCanvasLayerSelection(item, 2, true);
        expect(item.el.querySelector('[data-layer-index="2"]').classList.contains('selected')).toBe(true);
    });

    it('removes all selection styling from an existing layer', () => {
        const item = fixture();
        updateCanvasLayerSelection(item, 1, true);
        updateCanvasLayerSelection(item, 1, false);
        const el = item.el.querySelector('[data-layer-id="header"]');
        expect(el.classList.contains('selected')).toBe(false);
        expect(el.style.filter).toBe('none');
        expect(el.style.backgroundColor).toBe('transparent');
    });

    it('preserves the outline across redraws without duplicating split children', () => {
        const item = fixture();
        item.el.innerHTML = '<div class="crop-container"></div>';
        item.scene.layers[0].visible = true;
        item.scene.layers[0].cutoutUrl = 'test-cutout.png';
        item.layerStates = new Map([[0, { selected: true }]]);
        state.workbenchItems.clear();
        state.workbenchItems.set('parent', item);
        state.workbenchItems.set('child', { parentId: 'parent', sourceLayerId: 'subject' });
        renderCanvasLayers('parent');
        renderCanvasLayers('parent');
        expect(item.el.querySelectorAll('.canvas-layer-selection-outline')).toHaveLength(1);
        expect(item.el.querySelector('.canvas-layer[data-layer-id="subject"]')).toBeNull();
        expect(item.el.querySelectorAll('.cutout-img')).toHaveLength(0);
        expect(item.el.querySelector('[data-layer-id="header"]').classList.contains('selected')).toBe(false);
        item.layerStates.get(0).selected = false;
        renderCanvasLayers('parent');
        expect(item.el.querySelector('.canvas-layer-selection-outline')).toBeNull();
    });
});
