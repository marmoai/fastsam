import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock dependencies
const mockState = {
    sessions: [],
    currentSessionId: null
};

const mockRenderMessages = vi.fn();
const mockRenderHistoryList = vi.fn();
const mockDbHelper = {
    saveSession: vi.fn().mockResolvedValue(true)
};

const mockStartNewSession = vi.fn(async (title) => {
    const newSession = { id: 'session-123', title, messages: [] };
    mockState.sessions.push(newSession);
    mockState.currentSessionId = 'session-123';
    return newSession;
});

// The function we want to test (copied from main.js logic)
const addWorkbenchActionToChat = async (actionName, prompt, resultImageSrc) => {
    if (!mockState.currentSessionId) {
        await mockStartNewSession(actionName);
    }
    const currentSession = mockState.sessions.find(s => s.id === mockState.currentSessionId);
    if (!currentSession) return;

    const userMsg = { sender: 'user', type: 'text', content: `[工作台操作] ${actionName}: ${prompt}` };
    currentSession.messages.push(userMsg);
    
    const botMsg = { sender: 'bot', type: 'text', content: `操作已完成。` };
    if (resultImageSrc) {
        botMsg.type = 'image';
        botMsg.imageData = { src: resultImageSrc };
    }
    currentSession.messages.push(botMsg);
    
    mockRenderMessages(currentSession.messages);
    mockRenderHistoryList();
    if (mockDbHelper) await mockDbHelper.saveSession(currentSession);
};

describe('Full-Scene Interconnection Sync Logic', () => {
    beforeEach(() => {
        mockState.sessions = [];
        mockState.currentSessionId = null;
        vi.clearAllMocks();
    });

    it('should create a new session if none exists and add messages', async () => {
        await addWorkbenchActionToChat('智能融合', '将猫放入森林', 'data:image/png;base64,test');

        expect(mockStartNewSession).toHaveBeenCalledWith('智能融合');
        expect(mockState.sessions.length).toBe(1);
        const session = mockState.sessions[0];
        expect(session.messages.length).toBe(2);
        expect(session.messages[0].content).toContain('将猫放入森林');
        expect(session.messages[1].type).toBe('image');
        expect(mockRenderMessages).toHaveBeenCalled();
        expect(mockDbHelper.saveSession).toHaveBeenCalled();
    });

    it('should append to existing session if it exists', async () => {
        // Setup existing session
        const existingSession = { id: 'existing-1', title: 'Existing', messages: [] };
        mockState.sessions.push(existingSession);
        mockState.currentSessionId = 'existing-1';

        await addWorkbenchActionToChat('重光照', '调亮背景', null);

        expect(mockStartNewSession).not.toHaveBeenCalled();
        expect(existingSession.messages.length).toBe(2);
        expect(existingSession.messages[0].content).toContain('调亮背景');
        expect(existingSession.messages[1].type).toBe('text');
        expect(mockRenderMessages).toHaveBeenCalledWith(existingSession.messages);
    });
});
