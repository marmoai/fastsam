
import { state } from '../core/state.js';

export function initChatSourceRail() {
    console.log('Initializing Chat Source Rail...');
    const rail = document.createElement('div');
    rail.id = 'chat-source-rail';
    document.body.appendChild(rail);
    
    window.refreshMinimap = updateChatSourceRail;

    // Initial update
    updateChatSourceRail();

    // Listen for window resize to reposition if needed
    window.addEventListener('resize', updateChatSourceRail);
}

export function updateChatSourceRail() {
    const rail = document.getElementById('chat-source-rail');
    if (!rail) {
        // Silently fail if not initialized yet
        return;
    }

    rail.innerHTML = '';
    
    // Find current session messages
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    const messages = Array.isArray(currentSession?.messages) ? currentSession.messages : [];
    
    // If no messages, hide rail
    if (messages.length === 0) {
        rail.style.display = 'none';
        return;
    } else {
        rail.style.display = 'flex';
    }

    rail.onmouseleave = () => {
        rail.classList.remove('hide-popup');
    };

    // Create a container for the list popup
    const listPopup = document.createElement('div');
    listPopup.className = 'rail-list-popup';
    
    // Limit to the latest 50 messages
    const MAX_ITEMS = 50;
    const startIndex = Math.max(0, messages.length - MAX_ITEMS);
    const displayMessages = messages.slice(startIndex);

    displayMessages.forEach((msg, i) => {
        const absoluteIndex = startIndex + i;
        const bar = document.createElement('div');
        bar.className = `rail-bar ${msg.sender}`;
        if (msg.imageData || (msg.type === 'image')) {
            bar.classList.add('image');
        }
        
        bar.onclick = () => {
            rail.classList.add('hide-popup');
            scrollToMessage(absoluteIndex);
        };

        rail.appendChild(bar);

        // Add item to the list popup
        const listItem = document.createElement('div');
        listItem.className = `rail-list-item ${msg.sender}`;
        
        let snippet = '';
        if (msg.content) {
            snippet = msg.content.replace(/<[^>]*>?/gm, ''); // Strip HTML
            snippet = snippet.substring(0, 50) + (snippet.length > 50 ? '...' : '');
        } else if (msg.imageData || msg.type === 'image') {
            snippet = '[图片消息]';
        } else {
            snippet = '...';
        }
        
        const avatar = msg.sender === 'bot' ? '🤖' : '👤';
        listItem.innerHTML = `<span class="rail-list-avatar">${avatar}</span><span class="rail-list-text">${snippet}</span>`;
        
        listItem.onclick = (e) => {
            e.stopPropagation(); // Prevent rail click
            rail.classList.add('hide-popup');
            scrollToMessage(absoluteIndex);
        };
        
        listPopup.appendChild(listItem);
    });

    rail.appendChild(listPopup);
}

function scrollToMessage(index) {
    // 1. Ensure chat panel is open
    const chatPanel = document.querySelector('.chat-panel');
    const isCollapsed = chatPanel && chatPanel.classList.contains('collapsed');
    
    if (isCollapsed && window.toggleChat) {
        window.toggleChat(true);
    }

    // 2. Find the message element
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const messageElements = chatMessages.querySelectorAll('.message');
    if (messageElements[index]) {
        messageElements[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Highlight effect
        messageElements[index].classList.add('highlight-flash');
        setTimeout(() => {
            messageElements[index].classList.remove('highlight-flash');
        }, 2000);
    }
}
