export function updateHeader(session) {
    // Update Chat Header
    const titleEl = document.getElementById('headerTitle');
    const subtitleEl = document.getElementById('headerSubtitle');
    const logoEl = document.getElementById('headerLogo');
    
    // Update Workbench Header
    const wbTitleEl = document.getElementById('workbenchTitle');
    const wbSubtitleEl = document.getElementById('workbenchSubtitle');
    const wbLogoEl = document.getElementById('workbenchHeaderLogo');
    
    if (session) {
        const titleText = session.title;
        const date = new Date(session.timestamp);
        const dateText = date.toLocaleDateString();

        if (titleEl) titleEl.textContent = titleText;
        if (subtitleEl) subtitleEl.textContent = dateText;
        if (logoEl) logoEl.style.display = 'none';

        if (wbTitleEl) wbTitleEl.textContent = titleText;
        if (wbSubtitleEl) wbSubtitleEl.textContent = dateText;
        if (wbLogoEl) wbLogoEl.style.display = 'none';
    } else {
        if (titleEl) titleEl.textContent = 'MarmoAid 图像助手';
        if (subtitleEl) subtitleEl.textContent = '';
        if (logoEl) logoEl.style.display = 'block';

        if (wbTitleEl) wbTitleEl.textContent = '創作工作台';
        if (wbSubtitleEl) wbSubtitleEl.textContent = '';
        if (wbLogoEl) wbLogoEl.style.display = 'block';
    }
}
