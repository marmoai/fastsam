import { describe, expect, it } from 'vitest';
import {
    applyWorkspaceCommand,
    buildWorkspaceCommandResponse,
    detectWorkspaceCommand
} from './workspace-intents.js';
import { applyConversationHardGuards } from '../ai-services/hard-guards.js';

function createState() {
    return {
        currentActiveWorkbenchItemId: 'image-1',
        selectedWorkbenchItems: new Set(),
        workbenchItems: new Map([['image-1', { name: '客厅参考图', layers: [] }]])
    };
}

describe('workspace commands', () => {
    it('detects the two explicit scene mode switches', () => {
        expect(detectWorkspaceCommand('切换到室内设计模式')).toEqual({
            type: 'set_scene_mode',
            sceneMode: 'interior_design'
        });
        expect(detectWorkspaceCommand('进入平面广告模式')).toEqual({
            type: 'set_scene_mode',
            sceneMode: 'graphic_design'
        });
    });

    it('applies a mode change to internal state and records it', () => {
        const state = createState();
        const command = detectWorkspaceCommand('切换到室内设计模式');
        const context = applyWorkspaceCommand(state, command, '切换到室内设计模式');

        expect(context.sceneMode).toBe('interior_design');
        expect(context.lastAction.actionName).toBe('mode_changed');
        expect(buildWorkspaceCommandResponse(command, context)).toContain('室内设计模式');
    });

    it('keeps status and mode commands in text chat even with a base image', () => {
        expect(applyConversationHardGuards({
            text: '告诉我这张图现在能做什么',
            baseImage: {}
        }).route).toBe('text_chat');
        expect(applyConversationHardGuards({
            text: '切换到平面广告模式',
            baseImage: {}
        }).route).toBe('text_chat');
    });
});
