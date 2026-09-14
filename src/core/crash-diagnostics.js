const STORAGE_KEY = 'marmoai:crash-diagnostics:v1';
const MAX_BREADCRUMBS = 100;
const MAX_STRING_LENGTH = 600;

let diagnostics = {
    version: 1,
    installedAt: Date.now(),
    breadcrumbs: [],
    task: null,
    lastError: null,
    previousRecovery: null
};

function truncate(value) {
    const text = String(value ?? '');
    return text.length > MAX_STRING_LENGTH ? `${text.slice(0, MAX_STRING_LENGTH)}...` : text;
}

function sanitize(value, depth = 0) {
    if (value == null || depth > 3) return value ?? null;
    if (typeof value === 'string') return truncate(value);
    if (typeof value === 'number' || typeof value === 'boolean') return value;
    if (Array.isArray(value)) return value.slice(0, 30).map(item => sanitize(item, depth + 1));
    if (typeof value !== 'object') return truncate(value);
    return Object.fromEntries(
        Object.entries(value).slice(0, 40).map(([key, item]) => [key, sanitize(item, depth + 1)])
    );
}

function readStored() {
    if (typeof sessionStorage === 'undefined') return null;
    try {
        return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || 'null');
    } catch {
        return null;
    }
}

function environmentSnapshot() {
    const memory = typeof performance !== 'undefined' && performance.memory
        ? {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
        }
        : null;
    return {
        url: typeof location !== 'undefined' ? location.href : null,
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : null,
        deviceMemory: typeof navigator !== 'undefined' ? navigator.deviceMemory || null : null,
        memory,
        visibility: typeof document !== 'undefined' ? document.visibilityState : null
    };
}

function persist() {
    try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
            ...diagnostics,
            heartbeatAt: Date.now()
        }));
    } catch {
        // Diagnostics must never interfere with the application when storage is full.
    }
}

function addBreadcrumb(label, details = {}) {
    diagnostics.breadcrumbs.push({
        at: new Date().toISOString(),
        label,
        details: sanitize(details),
        environment: environmentSnapshot()
    });
    diagnostics.breadcrumbs = diagnostics.breadcrumbs.slice(-MAX_BREADCRUMBS);
    persist();
}

function captureError(kind, error, details = {}) {
    const message = error?.message || String(error || 'unknown_error');
    diagnostics.lastError = {
        at: new Date().toISOString(),
        kind,
        message: truncate(message),
        stack: truncate(error?.stack || ''),
        details: sanitize(details),
        task: sanitize(diagnostics.task),
        environment: environmentSnapshot()
    };
    addBreadcrumb(`error:${kind}`, { message, ...details });
}

export function recordDiagnosticBreadcrumb(label, details = {}) {
    addBreadcrumb(label, details);
}

export function beginDiagnosticTask(name, details = {}) {
    diagnostics.task = {
        name,
        startedAt: new Date().toISOString(),
        stage: 'started',
        details: sanitize(details)
    };
    addBreadcrumb(`task:start:${name}`, details);
}

export function updateDiagnosticTask(stage, details = {}) {
    if (!diagnostics.task) return;
    diagnostics.task = {
        ...diagnostics.task,
        stage,
        updatedAt: new Date().toISOString(),
        details: {
            ...(diagnostics.task.details || {}),
            ...sanitize(details)
        }
    };
    addBreadcrumb(`task:stage:${stage}`, details);
}

export function finishDiagnosticTask(status = 'completed', details = {}) {
    if (!diagnostics.task) return;
    diagnostics.task = {
        ...diagnostics.task,
        stage: status,
        finishedAt: new Date().toISOString(),
        details: {
            ...(diagnostics.task.details || {}),
            ...sanitize(details)
        }
    };
    addBreadcrumb(`task:end:${status}`, details);
}

export function installCrashDiagnostics() {
    if (typeof window === 'undefined' || window.__marmoCrashDiagnosticsInstalled) return;
    window.__marmoCrashDiagnosticsInstalled = true;

    const previous = readStored();
    const previousTaskIsActive = previous?.task && !['completed', 'failed', 'aborted'].includes(previous.task.stage);
    if (previousTaskIsActive && previous.heartbeatAt && Date.now() - previous.heartbeatAt > 10000) {
        diagnostics.previousRecovery = {
            detectedAt: new Date().toISOString(),
            reason: 'previous_task_heartbeat_stopped',
            previousTask: sanitize(previous.task),
            previousLastError: sanitize(previous.lastError),
            previousLastBreadcrumb: sanitize(previous.breadcrumbs?.at(-1) || null)
        };
        addBreadcrumb('possible_page_termination_recovered', diagnostics.previousRecovery);
    }

    window.addEventListener('error', event => {
        captureError('window_error', event.error || new Error(event.message || 'window_error'), {
            filename: event.filename,
            line: event.lineno,
            column: event.colno,
            source: 'window.error'
        });
    }, true);
    window.addEventListener('unhandledrejection', event => {
        const reason = event.reason instanceof Error ? event.reason : new Error(String(event.reason || 'unhandled_rejection'));
        captureError('unhandled_rejection', reason, { source: 'window.unhandledrejection' });
    });
    window.addEventListener('pagehide', event => {
        addBreadcrumb('pagehide', { persisted: Boolean(event?.persisted) });
    });
    window.addEventListener('visibilitychange', () => {
        addBreadcrumb('visibilitychange', { visibility: document.visibilityState });
    });

    window.__marmoRecordDiagnostic = recordDiagnosticBreadcrumb;
    window.__marmoBeginDiagnosticTask = beginDiagnosticTask;
    window.__marmoUpdateDiagnosticTask = updateDiagnosticTask;
    window.__marmoFinishDiagnosticTask = finishDiagnosticTask;
    window.__marmoGetCrashReport = () => JSON.stringify({
        ...diagnostics,
        exportedAt: new Date().toISOString(),
        environment: environmentSnapshot()
    }, null, 2);
    window.__marmoClearCrashReport = () => {
        diagnostics = {
            version: 1,
            installedAt: Date.now(),
            breadcrumbs: [],
            task: null,
            lastError: null,
            previousRecovery: null
        };
        persist();
    };

    addBreadcrumb('diagnostics:installed');
    window.setInterval(() => persist(), 2000);
}
