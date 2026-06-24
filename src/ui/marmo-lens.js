import { state } from '../core/state.js';
import { addMessage } from './chat-panel.js';
import { cropImageByBox } from '../graphics/image-processor.js';
import { blobToBase64, dataURLtoFileSync } from '../core/utils.js';
import { generateVisualSearch } from '../ai-services/gemini-client.js';
import { addImageToWorkbench } from './workbench-core.js';
import { preciseEditMode } from './fusion-editor.js';

export class MarmoLens {
    constructor(aiInstance) {
        this.ai = aiInstance;
        this.sidebar = document.getElementById('productSidebar');
        this.resultsContainer = document.getElementById('productResults');
        this.loadingIndicator = document.getElementById('productLoading');
        
        // Bind close event
        const closeBtn = document.getElementById('closeProductSidebar');
        if(closeBtn) {
            closeBtn.onclick = () => {
                this.sidebar.classList.remove('active');
            };
        }
    }

    cleanupOverlay() {
        if (this.currentOverlay) {
            this.currentOverlay.remove();
            this.currentOverlay = null;
        }
        this.isSelecting = false;
        this.selectionBox = null;
        if (this.moveHandler) {
            window.removeEventListener('mousemove', this.moveHandler);
            this.moveHandler = null;
        }
        if (this.upHandler) {
            window.removeEventListener('mouseup', this.upHandler);
            this.upHandler = null;
        }
    }

    startSelectionMode(workbenchItemId, type = 'rect') {
        const item = state.workbenchItems.get(workbenchItemId);
        if (!item) return;

        this.cleanupOverlay();
        this.selectionType = type; // 'rect' or 'lasso'

        // 1. 创建容器
        const container = document.createElement('div');
        container.className = 'lens-selection-container';
        if (type === 'lasso') container.style.cursor = 'crosshair';
        
        // 2. 提示文字
        const helpText = document.createElement('div');
        helpText.className = 'lens-helper-text';
        helpText.innerHTML = type === 'lasso' ? '<i class="fas fa-pencil-alt"></i> 随意套索框选区域' : '<i class="fas fa-crop-alt"></i> 按住左键框选区域';
        const inverseScale = 1 / state.workbenchZoom;
        helpText.style.transform = `translate(-50%, -50%) scale(${inverseScale})`;
        container.appendChild(helpText);

        // 3. 绑定鼠标按下事件
        container.onmousedown = (e) => this.handleMouseDown(e, item);
        
        // 添加到图片容器中
        item.el.appendChild(container);
        this.currentOverlay = container;
    }

    handleMouseDown(e, item) {
        if (state.isSpacePressed) return;
        e.stopPropagation(); // 阻止工作台拖拽
        e.preventDefault();

        if (this.selectionBox) {
            this.selectionBox.remove();
            this.selectionBox = null;
        }
        if (this.lassoSvg) {
            this.lassoSvg.remove();
            this.lassoSvg = null;
        }

        this.isSelecting = true;
        this.startPos = { x: e.offsetX, y: e.offsetY };

        if (this.selectionType === 'lasso') {
            this.lassoPoints = [this.startPos];
            const svgNS = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(svgNS, "svg");
            svg.style.cssText = "position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:10001;";
            
            const path = document.createElementNS(svgNS, "path");
            path.setAttribute("class", "lasso-path");
            svg.appendChild(path);
            
            this.currentOverlay.appendChild(svg);
            this.lassoSvg = svg;
            this.lassoPath = path;
        } else {
            const box = document.createElement('div');
            box.className = 'lens-selection-box';
            if (typeof preciseEditMode !== 'undefined' && preciseEditMode.active) {
                box.classList.add('precise-edit-box');
            }
            
            box.style.left = `${this.startPos.x}px`;
            box.style.top = `${this.startPos.y}px`;
            box.style.width = '0px';
            box.style.height = '0px';
            const inverseScale = 1 / state.workbenchZoom;
            box.style.setProperty('--lens-label-scale', inverseScale);
            
            this.currentOverlay.appendChild(box);
            this.selectionBox = box;
        }
        
        const helpText = this.currentOverlay.querySelector('.lens-helper-text');
        if(helpText) helpText.style.display = 'none';

        this.moveHandler = (ev) => this.handleMouseMove(ev);
        this.upHandler = (ev) => this.handleMouseUp(ev, item);

        window.addEventListener('mousemove', this.moveHandler);
        window.addEventListener('mouseup', this.upHandler);
    }

    handleMouseMove(e) {
        if (!this.isSelecting || !this.currentOverlay) return;
        
        const rect = this.currentOverlay.getBoundingClientRect();
        const relX = (e.clientX - rect.left) * (this.currentOverlay.offsetWidth / rect.width);
        const relY = (e.clientY - rect.top) * (this.currentOverlay.offsetHeight / rect.height);
        
        if (this.selectionType === 'lasso') {
            this.lassoPoints.push({ x: relX, y: relY });
            const d = this.lassoPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
            this.lassoPath.setAttribute("d", d);
        } else if (this.selectionBox) {
            const width = Math.abs(relX - this.startPos.x);
            const height = Math.abs(relY - this.startPos.y);
            const left = Math.min(relX, this.startPos.x);
            const top = Math.min(relY, this.startPos.y);
            
            this.selectionBox.style.width = `${width}px`;
            this.selectionBox.style.height = `${height}px`;
            this.selectionBox.style.left = `${left}px`;
            this.selectionBox.style.top = `${top}px`;
        }
    }

    handleMouseUp(e, item) {
        if (!this.isSelecting) return;
        this.isSelecting = false;
        
        window.removeEventListener('mousemove', this.moveHandler);
        window.removeEventListener('mouseup', this.upHandler);

        let x, y, w, h;
        if (this.selectionType === 'lasso') {
            if (this.lassoPoints.length < 3) {
                this.cleanupOverlay();
                return;
            }
            const xs = this.lassoPoints.map(p => p.x);
            const ys = this.lassoPoints.map(p => p.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            
            x = minX;
            y = minY;
            w = maxX - minX;
            h = maxY - minY;
        } else if (this.selectionBox) {
            const style = this.selectionBox.style;
            x = parseFloat(style.left);
            y = parseFloat(style.top);
            w = parseFloat(style.width);
            h = parseFloat(style.height);
        }

        if (w < 10 || h < 10) {
            this.cleanupOverlay();
            return;
        }

        const imgW = item.el.offsetWidth;
        const imgH = item.el.offsetHeight;
        
        const xmin = (x / imgW) * 1000;
        const ymin = (y / imgH) * 1000;
        const xmax = ((x + w) / imgW) * 1000;
        const ymax = ((y + h) / imgH) * 1000;

        const box = [ymin, xmin, ymax, xmax];

        // 如果是精准修图模式，由其覆盖的 handleMouseUp 处理
        if (typeof preciseEditMode !== 'undefined' && preciseEditMode.active) {
            return;
        }

        this.performVisualSearch(item, { box: box, label: "Manual Selection" });
    }

    renderOverlay(item, objects) {
        const oldSvg = item.el.querySelector('.lens-overlay-svg');
        if (oldSvg) oldSvg.remove();

        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("class", "lens-overlay-svg");
        svg.setAttribute("viewBox", "0 0 1000 1000");
        svg.setAttribute("preserveAspectRatio", "none");

        objects.forEach(obj => {
            if (!obj.box || !Array.isArray(obj.box) || obj.box.length < 4) return;

            const [ymin, xmin, ymax, xmax] = obj.box;
            const width = xmax - xmin;
            const height = ymax - ymin;

            const rect = document.createElementNS(svgNS, "rect");
            rect.setAttribute("x", xmin);
            rect.setAttribute("y", ymin);
            rect.setAttribute("width", width);
            rect.setAttribute("height", height);
            rect.setAttribute("class", "lens-object-path");
            
            const title = document.createElementNS(svgNS, "title");
            title.textContent = `点击搜索同款: ${obj.label}`;
            rect.appendChild(title);

            rect.addEventListener('click', (e) => {
                e.stopPropagation();
                this.performVisualSearch(item, obj);
            });

            svg.appendChild(rect);
        });

        item.el.appendChild(svg);
    }

    async performVisualSearch(item, objectData) {
        this.sidebar.classList.add('active');
        this.resultsContainer.innerHTML = '';
        this.loadingIndicator.style.display = 'flex';

        try {
            // 1. Crop
            const croppedBlob = await cropImageByBox(item.file || item.dataUrl, objectData.box);
            const croppedBase64 = await blobToBase64(croppedBlob);

            // 2. Gemini Call
            const { resultData, webLinks } = await generateVisualSearch(croppedBase64);
            
            // 3. Process Response
            this.renderResults(resultData, croppedBase64, webLinks);

        } catch (e) {
            console.error("Search failed:", e);
            this.resultsContainer.innerHTML = `<p style="text-align:center; padding:20px; color:#e53e3e;">识别失败: ${e.message}</p>`;
        } finally {
            this.loadingIndicator.style.display = 'none';
        }
    }

    renderResults(data, croppedImgBase64, webLinks) {
        this.resultsContainer.innerHTML = '';

        // 1. Anchor Image
        const headerDiv = document.createElement('div');
        headerDiv.style.cssText = "border-bottom:1px solid #eee; padding-bottom:15px; margin-bottom:15px;";
        headerDiv.innerHTML = `
            <div style="font-size:12px; color:#888; margin-bottom:8px; text-align:center;">已识别对象</div>
            <div style="display:flex; gap:12px; align-items:center;">
                <img src="data:image/png;base64,${croppedImgBase64}" style="width:80px; height:80px; object-fit:contain; border:1px solid #ddd; border-radius:8px; background:#f8f9fa;">
                <div>
                    <div style="font-weight:700; color:#2d3748; font-size:15px;">${data.productName}</div>
                    <div style="font-size:12px; color:#666; margin-top:4px;">${data.description}</div>
                </div>
            </div>
        `;
        this.resultsContainer.appendChild(headerDiv);

        // 2. Aggregated Search Links
        const q = encodeURIComponent(data.searchQuery);
        const platforms = [
            { name: 'Google Shopping', icon: 'fa-google', color: '#4285F4', url: `https://www.google.com/search?tbm=shop&q=${q}` },
            { name: '淘宝 / Taobao', icon: 'fa-shopping-bag', color: '#FF5000', url: `https://s.taobao.com/search?q=${q}` },
            { name: 'Amazon', icon: 'fa-amazon', color: '#FF9900', url: `https://www.amazon.com/s?k=${q}` },
            { name: 'Pinterest (灵感)', icon: 'fa-pinterest', color: '#E60023', url: `https://www.pinterest.com/search/pins/?q=${q}` }
        ];

        const actionsDiv = document.createElement('div');
        actionsDiv.innerHTML = `<div style="font-size:13px; font-weight:600; margin-bottom:10px; color:#444;">全网同款搜索 (Real-time)</div>`;
        
        platforms.forEach(plat => {
            const btn = document.createElement('a');
            btn.href = plat.url;
            btn.target = "_blank";
            btn.style.cssText = `
                display: flex; align-items: center; justify-content: space-between;
                padding: 10px 14px; margin-bottom: 8px;
                background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                text-decoration: none; color: #333; transition: all 0.2s;
            `;
            btn.innerHTML = `
                <div style="display:flex; align-items:center; gap:10px;">
                    <i class="fab ${plat.icon}" style="color:${plat.color}; width:20px; text-align:center;"></i>
                    <span style="font-size:14px;">在 ${plat.name} 搜索</span>
                </div>
                <i class="fas fa-external-link-alt" style="font-size:12px; color:#ccc;"></i>
            `;
            btn.onmouseover = () => { btn.style.borderColor = plat.color; btn.style.transform = 'translateY(-1px)'; };
            btn.onmouseout = () => { btn.style.borderColor = '#e2e8f0'; btn.style.transform = 'translateY(0)'; };
            actionsDiv.appendChild(btn);
        });
        this.resultsContainer.appendChild(actionsDiv);

        // 3. Web Grounding Links
        if (webLinks && webLinks.length > 0) {
            const linksDiv = document.createElement('div');
            linksDiv.style.marginTop = "20px";
            linksDiv.innerHTML = `<div style="font-size:13px; font-weight:600; margin-bottom:10px; color:#444;">相关网页结果</div>`;
            
            webLinks.slice(0, 5).forEach(link => {
                const linkEl = document.createElement('a');
                linkEl.href = link.uri;
                linkEl.target = "_blank";
                linkEl.className = "product-card"; 
                linkEl.style.cssText = "display:block; text-decoration:none; padding:10px; margin-bottom:8px; border:1px solid #eee; background: white; border-radius: 8px;";
                linkEl.innerHTML = `
                    <div style="font-size:13px; font-weight:500; color:#2A5C82; margin-bottom:4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${link.title}</div>
                    <div style="font-size:11px; color:#999; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${link.uri}</div>
                `;
                linksDiv.appendChild(linkEl);
            });
            this.resultsContainer.appendChild(linksDiv);
        }
        
        // 4. Add to Workbench
        const tryOnDiv = document.createElement('div');
        tryOnDiv.style.marginTop = "20px";
        tryOnDiv.style.paddingTop = "15px";
        tryOnDiv.style.borderTop = "1px solid #eee";
        
        const tryOnBtn = document.createElement('button');
        tryOnBtn.className = 'btn-primary';
        tryOnBtn.style.width = '100%';
        tryOnBtn.style.padding = '10px';
        tryOnBtn.innerHTML = '<i class="fas fa-cube"></i> 将此截图作为素材添加到工作台';
        tryOnBtn.onclick = async () => {
            const dataUrl = `data:image/png;base64,${croppedImgBase64}`;
            const file = dataURLtoFileSync(dataUrl, `crop-${Date.now()}.png`);
            
            addImageToWorkbench(file, data.productName, { 
                dataUrl: dataUrl,
                parentId: null
            });

            addMessage({ sender: 'bot', type: 'text', content: `✂️ 已将截图素材 **${data.productName}** 添加到工作台。` });
            
            if(window.innerWidth < 1024) this.sidebar.classList.remove('active');
        };
        tryOnDiv.appendChild(tryOnBtn);
        this.resultsContainer.appendChild(tryOnDiv);
    }
}
