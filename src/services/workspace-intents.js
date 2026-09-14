import {
    getAssistantWorkspaceContext,
    recordWorkspaceAction,
    syncWorkspaceContext
} from './workspace-context.js';

const SCENE_MODES = new Map([
    ['interior_design', '室内设计'],
    ['graphic_design', '平面广告']
]);

export function detectWorkspaceCommand(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return null;

    if (/切换到?|进入|改走|使用|按/.test(normalized) && /室内设计模式|室内模式|室内设计/.test(normalized)) {
        return { type: 'set_scene_mode', sceneMode: 'interior_design' };
    }

    if (/切换到?|进入|改走|使用|按/.test(normalized) && /平面广告模式|平面设计模式|平面模式|广告模式|平面广告/.test(normalized)) {
        return { type: 'set_scene_mode', sceneMode: 'graphic_design' };
    }

    return null;
}

export function applyWorkspaceCommand(appState, command, sourceText = '') {
    if (!appState || command?.type !== 'set_scene_mode' || !SCENE_MODES.has(command.sceneMode)) {
        return getAssistantWorkspaceContext(appState);
    }

    syncWorkspaceContext(appState, { sceneMode: command.sceneMode });
    recordWorkspaceAction(appState, {
        actionName: 'mode_changed',
        prompt: sourceText,
        status: 'completed',
        hasResult: false
    });

    return getAssistantWorkspaceContext(appState);
}

export function buildWorkspaceCommandResponse(command, workspaceContext = {}) {
    if (command?.type !== 'set_scene_mode') return '';

    const label = SCENE_MODES.get(command.sceneMode) || '当前';
    const activeItem = workspaceContext?.activeItem;
    const suffix = activeItem
        ? `当前工作台仍以“${activeItem.name}”为上下文。`
        : '当前工作台还没有活动图片。';

    if (command.sceneMode === 'interior_design') {
        return `已切换到${label}模式。接下来我会优先围绕空间、材质、家具、灯具和图层关系来理解你的操作。${suffix}`;
    }

    return `已切换到${label}模式。接下来我会优先围绕版式、文字、价格信息、主视觉和广告画面来理解你的操作。${suffix}`;
}

export function getSceneModeLabel(sceneMode = '') {
    return SCENE_MODES.get(sceneMode) || '未设置';
}
