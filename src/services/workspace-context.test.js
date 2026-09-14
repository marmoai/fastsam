import { describe, expect, it } from 'vitest';
import {
    getAssistantWorkspaceContext,
    recordWorkspaceAction,
    syncWorkspaceContext
} from './workspace-context.js';

function createState() {
    const item = {
        name: '客厅参考图',
        hasFullSemanticAnalysis: true,
        cleanPlateStatus: 'completed',
        cleanPlateDataUrl: 'data:image/png;base64,plate',
        semanticViews: {
            editableSceneLayers: [
                { id: 'sofa', name: '沙发', semanticType: 'furniture', cutoutUrl: '/sofa.png' },
                { id: 'wall', name: '背景墙', category: 'background' }
            ]
        }
    };

    return {
        currentActiveWorkbenchItemId: 'image-1',
        selectedWorkbenchItems: new Set(['image-1']),
        workbenchItems: new Map([['image-1', item]])
    };
}

describe('workspace context bridge', () => {
    it('summarizes the active and selected Workbench item without chat messages', () => {
        const context = getAssistantWorkspaceContext(createState());

        expect(context.activeItem.name).toBe('客厅参考图');
        expect(context.activeItem.layerCount).toBe(2);
        expect(context.activeItem.extractedLayerCount).toBe(1);
        expect(context.activeItem.editableLayerNames).toEqual(['沙发']);
        expect(context.selectedItems[0].id).toBe('image-1');
    });

    it('records recent actions in the internal state bridge', () => {
        const state = createState();
        recordWorkspaceAction(state, { actionName: 'magic_layers_completed', itemId: 'image-1' });

        expect(state.workspaceContext.lastAction.actionName).toBe('magic_layers_completed');
        expect(state.workspaceContext.recentActions).toHaveLength(1);
        expect(syncWorkspaceContext(state).activeItemId).toBe('image-1');
    });

    it('keeps the selected layer as a stable id and clears it when the image changes', () => {
        const state = createState();
        recordWorkspaceAction(state, {
            actionName: 'semantic_layer_selection_changed',
            itemId: 'image-1',
            layerId: 'sofa',
            layerName: '沙发'
        });

        expect(getAssistantWorkspaceContext(state).activeLayerId).toBe('sofa');

        state.workbenchItems.set('image-2', { name: '另一张图', layers: [] });
        state.currentActiveWorkbenchItemId = 'image-2';
        expect(getAssistantWorkspaceContext(state).activeLayerId).toBe(null);
    });

    it('infers the active image from a single selected Workbench item', () => {
        const state = createState();
        state.currentActiveWorkbenchItemId = null;

        const context = getAssistantWorkspaceContext(state);

        expect(context.activeItemId).toBe('image-1');
        expect(context.activeItem.name).toBe('客厅参考图');
    });
});
