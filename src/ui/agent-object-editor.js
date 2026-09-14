import { isAgentRuntimeFeatureEnabled } from '../core/config.js';
import { state } from '../core/state.js';
import { startObjectEditJob } from './agent-task-controller.js';

function activeItem() {
    const itemId = state.currentActiveWorkbenchItemId || state.workspaceContext?.activeItemId;
    return {
        itemId,
        item: itemId ? state.workbenchItems.get(itemId) : null
    };
}

function canUpdateCurrentLayer(item) {
    const parent = item?.parentId ? state.workbenchItems.get(item.parentId) : null;
    const layers = parent?.scene?.layers || parent?.layers || [];
    const sourceLayerId = item?.sourceLayerId || item?.layerId;
    const hasParentLayer = Boolean(
        (sourceLayerId && layers.some(layer => layer?.id === sourceLayerId)) ||
        (item?.layerName && layers.some(layer => layer?.name === item.layerName)) ||
        (Array.isArray(item?.originalBbox) && layers.some(layer =>
            Array.isArray(layer?.bbox) &&
            layer.bbox.length === 4 &&
            layer.bbox.every((value, index) => Math.abs(Number(value) - Number(item.originalBbox[index])) <= 2)
        ))
    );
    return Boolean(
        item?.parentId &&
        Array.isArray(item.originalBbox) &&
        item.originalBbox.length === 4 &&
        parent &&
        hasParentLayer
    );
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function removeEditor() {
    document.body.classList.remove('agent-object-editor-open');
    document.getElementById('agentObjectEditor')?.remove();
}

/**
 * The Visual Object editor is deliberately a plan composer, not an execution
 * surface. It collects only IDs, mode and prompt; AgentRuntime owns execution,
 * verification and the single Commit boundary.
 */
export function showAgentObjectEditor({ runtime, onUserMessage, onFinished, itemId } = {}) {
    if (!isAgentRuntimeFeatureEnabled('objectEditing')) return;
    const resolved = itemId ? { itemId, item: state.workbenchItems.get(itemId) } : activeItem();
    if (!resolved.itemId || !resolved.item) return;

    removeEditor();
    const item = resolved.item;
    const label = item.layerName || item.label || '当前视觉对象';
    const canUpdate = canUpdateCurrentLayer(item);
    const editor = document.createElement('section');
    editor.id = 'agentObjectEditor';
    editor.className = 'agent-object-editor';
    editor.setAttribute('role', 'dialog');
    editor.setAttribute('aria-modal', 'false');
    editor.innerHTML = `
        <div class="agent-object-editor-head">
            <div>
                <span class="agent-visual-kicker">OBJECT EDITOR</span>
                <strong>编辑对象</strong>
                <span class="agent-object-editor-subtitle">${escapeHtml(label)}</span>
            </div>
            <button type="button" class="agent-object-editor-close" aria-label="关闭对象编辑器">×</button>
        </div>
        <div class="agent-object-editor-modes" role="tablist" aria-label="编辑方式">
            <button type="button" class="is-active" data-object-mode="variant" role="tab">生成新版本</button>
            <button type="button" data-object-mode="current_layer" role="tab" ${canUpdate ? '' : 'disabled title="只有来自 Magic Layers 的可回写图层支持此操作"'}>更新当前图层</button>
        </div>
        <textarea data-object-prompt rows="3" placeholder="告诉 Agent 你想怎么改……"></textarea>
        <div class="agent-object-editor-suggestions">
            <span>建议</span>
            <button type="button" data-object-suggestion="换成米白材质">换成米白材质</button>
            <button type="button" data-object-suggestion="保留构图，提升质感">保留构图，提升质感</button>
            <button type="button" data-object-suggestion="生成电商展示版本">生成电商展示版本</button>
        </div>
        ${!canUpdate ? '<div class="agent-object-editor-note">当前对象没有可回写的父图层，将以新版本方式保存。</div>' : ''}
        <div class="agent-object-editor-footer">
            <span data-object-feedback></span>
            <div>
                <button type="button" class="agent-object-editor-cancel">取消</button>
                <button type="button" class="agent-object-editor-submit"><i class="fas fa-sparkles"></i> 生成计划</button>
            </div>
        </div>
    `;
    document.body.appendChild(editor);
    document.body.classList.add('agent-object-editor-open');

    let mode = 'variant';
    const prompt = editor.querySelector('[data-object-prompt]');
    const feedback = editor.querySelector('[data-object-feedback]');
    const submit = editor.querySelector('.agent-object-editor-submit');
    const modeButtons = [...editor.querySelectorAll('[data-object-mode]')];
    const setMode = nextMode => {
        if (nextMode === 'current_layer' && !canUpdate) return;
        mode = nextMode;
        modeButtons.forEach(button => button.classList.toggle('is-active', button.dataset.objectMode === mode));
    };
    modeButtons.forEach(button => button.addEventListener('click', () => setMode(button.dataset.objectMode)));
    editor.querySelectorAll('[data-object-suggestion]').forEach(button => {
        button.addEventListener('click', () => {
            prompt.value = button.dataset.objectSuggestion || '';
            prompt.focus();
        });
    });
    editor.querySelector('.agent-object-editor-close')?.addEventListener('click', removeEditor);
    editor.querySelector('.agent-object-editor-cancel')?.addEventListener('click', removeEditor);
    editor.addEventListener('mousedown', event => event.stopPropagation());
    editor.addEventListener('click', event => event.stopPropagation());
    prompt.focus();

    submit.addEventListener('click', async () => {
        const value = prompt.value.trim();
        if (!value || submit.disabled) {
            feedback.textContent = '请先告诉 Agent 你想怎么改。';
            prompt.focus();
            return;
        }
        submit.disabled = true;
        feedback.textContent = '正在生成 Agent 计划…';
        try {
            removeEditor();
            await startObjectEditJob({
                runtime,
                itemId: resolved.itemId,
                prompt: value,
                mode,
                onUserMessage,
                onFinished
            });
        } catch (error) {
            feedback.textContent = error?.message || String(error);
            submit.disabled = false;
        }
    });
}

export function hideAgentObjectEditor() {
    removeEditor();
}
