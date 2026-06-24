import { memoryLayer } from '../runtime/CreativeMemoryLayer';

export function initDecisionLog() {
    // Auto update when MemoryLayer captures new decision
    memoryLayer.subscribe(() => {
        showLatestDecisionToast();
    });
}

function showLatestDecisionToast() {
    const logs = memoryLayer.getLogs();
    if (logs.length === 0) return;
    
    // Get the most recent log
    const latestLog = logs[logs.length - 1];
    if (!latestLog.intent) return; // Only show toast if semantic intent was inferred
    
    const container = document.getElementById('smartToastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.style.cssText = `
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(43, 108, 176, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        color: #2b6cb0;
        display: flex;
        align-items: center;
        gap: 8px;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        pointer-events: none;
    `;
    
    toast.innerHTML = `<i class="fas fa-brain" style="font-size: 12px; opacity: 0.8;"></i> <span>系统响应: <strong>${latestLog.intent}</strong></span>`;
    
    container.appendChild(toast);
    
    // Animate in
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    });
    
    // Animate out after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            if (toast.parentNode === container) {
                container.removeChild(toast);
            }
        }, 300);
    }, 3000);
}
