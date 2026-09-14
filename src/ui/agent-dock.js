const STATUS_LABELS = {
    created: 'Agent · 已创建',
    planned: 'Agent · 已规划',
    running: 'Agent · 执行中',
    verifying: 'Agent · 正在验证',
    waiting_confirmation: 'Agent · 有任务等待确认',
    committing: 'Agent · 正在保存',
    completed: 'Agent · 上次任务已完成',
    failed: 'Agent · 任务失败',
    cancelled: 'Agent · 任务已取消'
};

const ACTIVE_STATUSES = new Set(['created', 'planned', 'running', 'verifying', 'committing']);

function priority(job) {
    if (job?.status === 'waiting_confirmation') return 50;
    if (job?.status === 'failed') return 40;
    if (ACTIVE_STATUSES.has(job?.status)) return 30;
    if (job?.status === 'completed') return 10;
    if (job?.status === 'cancelled') return 5;
    return 0;
}

export function initAgentDock({ runtime } = {}) {
    const button = document.getElementById('toggleChatBtn');
    const label = button?.querySelector('[data-agent-dock-label]');
    const count = button?.querySelector('[data-agent-dock-count]');
    const dot = button?.querySelector('[data-agent-dock-dot]');
    const agent = runtime?.getAgentRuntime?.();
    if (!button || !label || !count || !dot || !agent) return () => {};

    const update = () => {
        const jobs = agent.listJobs?.() || [];
        const waiting = jobs.filter(job => job.status === 'waiting_confirmation').length;
        const active = jobs.filter(job => ACTIVE_STATUSES.has(job.status)).length;
        const current = jobs.slice().sort((a, b) => {
            const score = priority(b) - priority(a);
            return score || (b.updatedAt || 0) - (a.updatedAt || 0);
        })[0];

        const status = waiting > 0
            ? 'waiting_confirmation'
            : active > 0
                ? current?.status || 'running'
                : current?.status || 'idle';
        const statusLabel = STATUS_LABELS[status] || 'Agent · 空闲';

        label.textContent = waiting > 0
            ? `Agent · 有 ${waiting} 个任务等待确认`
            : statusLabel;
        button.dataset.agentStatus = status;
        button.title = waiting > 0 ? '打开 Agent 任务面板处理待确认任务' : '打开 Agent 任务面板';
        button.setAttribute('aria-label', button.title);
        button.classList.toggle('is-busy', active > 0);
        button.classList.toggle('is-attention', waiting > 0 || status === 'failed');
        button.classList.toggle('is-success', status === 'completed');
        button.classList.toggle('is-error', status === 'failed');

        const visibleCount = waiting || active;
        count.hidden = visibleCount === 0;
        count.textContent = visibleCount > 9 ? '9+' : String(visibleCount);
        dot.dataset.status = status;
    };

    update();
    const unsubscribe = agent.subscribe(update);
    return () => unsubscribe?.();
}
