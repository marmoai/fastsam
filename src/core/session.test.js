import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock dependencies that session.js imports
vi.mock('./state.js', () => ({
    state: {
        sessions: [],
        currentSessionId: null,
        pendingReferenceImageShares: new Set()
    }
}));

vi.mock('../ui/chat-panel.js', () => ({
    addMessage: vi.fn(),
    renderMessages: vi.fn()
}));

vi.mock('./utils.js', () => ({
    dataURLToFile: vi.fn(),
    fileToDataURL: vi.fn()
}));

vi.mock('../ui/workbench-core.js', () => ({
    addTextNoteToWorkbench: vi.fn(),
    restoreGroupLabelToWorkbench: vi.fn(),
    addAtmosphereNode: vi.fn(),
    addImageToWorkbench: vi.fn()
}));

vi.mock('../ui/sidebar.js', () => ({
    sidebarState: { closeMobileMenu: vi.fn() }
}));

vi.mock('../ui/modals.js', () => ({
    closeMagicWandModal: vi.fn()
}));

// Mock indexedDB
const mockDB = {
    transaction: vi.fn().mockReturnValue({
        objectStore: vi.fn().mockReturnValue({
            put: vi.fn(),
            getAll: vi.fn(),
            delete: vi.fn()
        }),
        oncomplete: null,
        onerror: null
    })
};

global.indexedDB = {
    open: vi.fn().mockReturnValue({
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null
    })
};

import { dbHelper } from './session';

describe('session.js dbHelper tests', () => {
    
    it('should initialize the database', async () => {
        const mockOpenRequest = {
            onupgradeneeded: null,
            onsuccess: null,
            onerror: null
        };
        indexedDB.open.mockReturnValue(mockOpenRequest);

        const initPromise = dbHelper.init();
        
        // Simulate success
        mockOpenRequest.onsuccess({ target: { result: 'mock-db' } });
        
        await initPromise;
        expect(dbHelper.db).toBe('mock-db');
    });

    it('should save a session', async () => {
        dbHelper.db = {
            transaction: vi.fn().mockReturnValue({
                objectStore: vi.fn().mockReturnValue({
                    put: vi.fn()
                }),
                oncomplete: null,
                onerror: null
            })
        };
        
        const session = { id: '1', messages: [] };
        const tx = dbHelper.db.transaction();
        dbHelper.db.transaction.mockReturnValue(tx);
        
        const savePromise = dbHelper.saveSession(session);
        
        // Simulate completion
        tx.oncomplete();
        
        await savePromise;
        expect(dbHelper.db.transaction).toHaveBeenCalledWith('sessions', 'readwrite');
    });
});
