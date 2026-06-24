import localforage from 'localforage';

export interface DecisionLogEntry {
    id: string;
    timestamp: number;
    actionType: string;
    payload: any;
    intent?: string;
    reason?: string;
    context?: string;
    stateDelta?: {
        before: any;
        after: any;
    };
}

const MAX_MEMORY_LOGS = 30;
const MAX_FULL_DELTA_LOGS = 5;

// Deep recursive sanitize helper to prevent base64, blobs, and large images leaking into memory logs
export const deepSanitize = (val: any): any => {
    if (!val) return val;
    if (Array.isArray(val)) {
        return val.map(item => deepSanitize(item));
    }
    if (typeof val === 'object') {
        const cloned: any = {};
        for (const key of Object.keys(val)) {
            const lowerKey = key.toLowerCase();
            const value = val[key];
            
            if (
                lowerKey === 'blob' || 
                lowerKey === 'maskblob' || 
                lowerKey === 'originalblob' || 
                lowerKey === 'cleanplateblob'
            ) {
                cloned[key] = '[blob_hidden]';
            } else if (
                lowerKey === 'dataurl' ||
                lowerKey === 'originaldataurl' ||
                lowerKey === 'cleanplatedataurl' ||
                lowerKey === 'sourceimage' ||
                lowerKey === 'image' ||
                lowerKey === 'mask'
            ) {
                if (typeof value === 'string') {
                    if (value.startsWith('data:') || value.length > 500) {
                        cloned[key] = `[${key}_hidden]`;
                    } else {
                        cloned[key] = value;
                    }
                } else {
                    cloned[key] = '[asset_hidden]';
                }
            } else if (lowerKey === 'content' && typeof value === 'string' && value.length > 1000) {
                cloned[key] = '[content_hidden]';
            } else {
                cloned[key] = deepSanitize(value);
            }
        }
        return cloned;
    }
    if (typeof val === 'string') {
        if (val.startsWith('data:') || val.length > 1000) {
            return '[string_hidden_large]';
        }
    }
    return val;
};

class CreativeMemory {
    private logs: DecisionLogEntry[] = [];
    private listeners: Function[] = [];
    private storageKey = 'mvr_decision_logs';

    constructor() {
        this.loadFromStorage();
    }

    private async loadFromStorage() {
        try {
            const stored = await localforage.getItem<DecisionLogEntry[]>(this.storageKey);
            if (stored && Array.isArray(stored)) {
                this.logs = this.compactLogs(stored);

                console.log(`[MemoryLayer] Restored and compacted ${this.logs.length} logs from storage`);
                // Re-save right away to free oversized legacy stateDelta snapshots from IndexedDB.
                await this.saveToStorage();
            }
        } catch (e) {
            console.error('Failed to load memory layer logs:', e);
        }
    }

    private compactLogs(logs: DecisionLogEntry[]) {
        const recentLogs = logs.slice(-MAX_MEMORY_LOGS);
        const fullDeltaStart = Math.max(0, recentLogs.length - MAX_FULL_DELTA_LOGS);

        return recentLogs.map((entry, index) => {
            const compacted: DecisionLogEntry = {
                id: entry.id,
                timestamp: entry.timestamp,
                actionType: entry.actionType,
                payload: deepSanitize(entry.payload),
                intent: entry.intent,
                reason: entry.reason,
                context: entry.context
            };

            if (index >= fullDeltaStart && entry.stateDelta) {
                compacted.stateDelta = {
                    before: deepSanitize(entry.stateDelta.before),
                    after: deepSanitize(entry.stateDelta.after)
                };
            }

            return compacted;
        });
    }

    private async saveToStorage() {
        try {
            this.logs = this.compactLogs(this.logs);
            await localforage.setItem(this.storageKey, this.logs);
        } catch (e) {
            console.error('Failed to save memory layer logs:', e);
        }
    }

    logDecision(action: any, stateDelta?: { before: any; after: any }) {
        // Skip purely visual cascaded events unless explicitly needed
        if (action.type === 'CASCADE_TRANSFORM') return;

        let safeDelta = undefined;
        if (stateDelta) {
            safeDelta = {
                before: deepSanitize(stateDelta.before),
                after: deepSanitize(stateDelta.after)
            };
        }

        const entry: DecisionLogEntry = {
            id: Date.now().toString(36) + Math.random().toString(36).substring(2, 5),
            timestamp: Date.now(),
            actionType: action.type,
            payload: deepSanitize(action.payload),
            intent: action.intent,
            reason: action.reason,
            context: action.context,
            stateDelta: safeDelta
        };

        this.logs.push(entry);
        this.logs = this.compactLogs(this.logs);
        console.log(`[MemoryLayer] Decision Captured: [${entry.intent || entry.actionType}]`, entry.stateDelta ? 'with Delta' : '');
        this.saveToStorage();
        this.notifyListeners();
    }

    getLogs() {
        return this.logs;
    }

    subscribe(fn: Function) {
        this.listeners.push(fn);
    }
    
    private notifyListeners() {
        this.listeners.forEach(fn => fn());
    }
}

export const memoryLayer = new CreativeMemory();
