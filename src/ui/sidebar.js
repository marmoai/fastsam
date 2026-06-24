import { state } from '../core/state.js';

export const sidebarState = {
    closeMobileMenu: () => {}
};

export function renderHistoryList() {
    const historyList = document.getElementById('historyList');
    const emptyState = document.getElementById('emptyState');
    if (!historyList || !emptyState) return;

    historyList.innerHTML = '';
    const { sessions, currentSessionId } = state;
    
    if (sessions.length === 0) {
        emptyState.style.display = 'flex';
    } else {
        emptyState.style.display = 'none';
        sessions.forEach(session => {
            const li = document.createElement('li');
            li.dataset.sessionId = session.id;
            if (session.id === currentSessionId) li.classList.add('active');
            li.innerHTML = `<span class="history-title">${session.title}</span><button class="history-more-btn">...</button>`;
            historyList.appendChild(li);
        });
    }
}

export function initSidebar() {
    const container = document.querySelector('.container'), 
          sidebar = document.querySelector('.assets-sidebar');
    const toggleSidebarBtn = document.querySelector('.toggle-sidebar');
    const icon = toggleSidebarBtn ? toggleSidebarBtn.querySelector('i') : null;

    // 初始化默认为收起状态
    if (window.innerWidth > 1024) {
        sidebar.classList.add('collapsed'); 
        container.classList.add('sidebar-collapsed');
        container.classList.remove('sidebar-open');
        if(icon) icon.style.transform = 'rotate(90deg)';
    }

    if (toggleSidebarBtn) {
        const avatarToggle = document.getElementById('avatarToggle');
        const handleToggle = () => {
            if (window.innerWidth > 1024) {
                const isCollapsed = sidebar.classList.toggle('collapsed');
                container.classList.toggle('sidebar-open', !isCollapsed);
                container.classList.toggle('sidebar-collapsed', isCollapsed);
                if(icon) icon.style.transform = isCollapsed ? 'rotate(90deg)' : 'rotate(0deg)';
            }
        };
        toggleSidebarBtn.addEventListener('click', handleToggle);
        if (avatarToggle) avatarToggle.addEventListener('click', handleToggle);
    }

    const chatPanel = document.querySelector('.chat-panel');
    const toggleChatBtn = document.getElementById('toggleChatBtn');
    const chatIcon = toggleChatBtn ? toggleChatBtn.querySelector('i') : null;

    if (toggleChatBtn) {
        window.toggleChat = (forceOpen) => {
            if (window.innerWidth > 1024) {
                const isCurrentlyCollapsed = chatPanel.classList.contains('collapsed');
                if (forceOpen === true && !isCurrentlyCollapsed) return;
                if (forceOpen === false && isCurrentlyCollapsed) return;

                const isCollapsed = chatPanel.classList.toggle('collapsed');
                container.classList.toggle('chat-collapsed', isCollapsed);
                if(chatIcon) chatIcon.style.transform = isCollapsed ? 'rotate(90deg)' : 'rotate(0deg)';
            }
        };
        toggleChatBtn.addEventListener('click', () => window.toggleChat());
    }

    const hamburgerMenu = document.getElementById('hamburgerMenu'), 
          mobileOverlay = document.getElementById('mobileOverlay');
    
    const openMobileMenu = () => { 
        if (window.innerWidth <= 1024) { 
            sidebar.classList.add('mobile-open'); 
            mobileOverlay.classList.add('active'); 
            hamburgerMenu.style.display = 'none'; 
        } 
    };
    
    sidebarState.closeMobileMenu = () => { 
        if (window.innerWidth <= 1024) { 
            sidebar.classList.remove('mobile-open'); 
            mobileOverlay.classList.remove('active'); 
            hamburgerMenu.style.display = ''; 
        } 
    };
    
    if (hamburgerMenu) hamburgerMenu.addEventListener('click', openMobileMenu);
    if (mobileOverlay) mobileOverlay.addEventListener('click', sidebarState.closeMobileMenu);
}
