import { showFloatingFusionEditor, hideFloatingFusionEditor } from './fusion-editor.js';

export function showAISuggestionCompanion(itemId) {
    showFloatingFusionEditor(itemId);
}

export function hideAISuggestionCompanion() {
    hideFloatingFusionEditor();
}
