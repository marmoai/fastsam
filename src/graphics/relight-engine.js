import { state } from "../core/state.js";
import { generateRelitImage } from "../ai-services/skills-engine.js";
import { getProxiedUrl } from "../core/utils.js";

// Callbacks
let addImageToWorkbenchCallback, addMessageCallback;

export function initRelightEngine(callbacks) {
    addImageToWorkbenchCallback = callbacks.addImageToWorkbench;
    addMessageCallback = callbacks.addMessage;
}

// --- 2.5D Relighting WebGL Renderer ---
export class RelightingRenderer {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'relighting-canvas';
        this.gl = this.canvas.getContext('webgl', { preserveDrawingBuffer: true });
        if(!this.gl) {
            console.error("WebGL not supported");
            return;
        }
        
        this.program = null;
        this.texture = null;
        this.buffer = null;
        this.init();
    }
    
    init() {
        if (!this.gl) return;
        
        const vsSource = document.getElementById('relight-vs').text;
        const fsSource = document.getElementById('relight-fs').text;
        
        const vs = this.compileShader(this.gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(this.gl.FRAGMENT_SHADER, fsSource);
        
        if (!vs || !fs) return;
        
        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vs);
        this.gl.attachShader(this.program, fs);
        this.gl.linkProgram(this.program);
        
        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            console.error('Shader program init failed: ' + this.gl.getProgramInfoLog(this.program));
            return;
        }

        // Quad Buffer
        const vertices = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
        const texCoords = new Float32Array([0,1, 1,1, 0,0, 0,0, 1,1, 1,0]);
        
        this.buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
        const combined = new Float32Array(vertices.length + texCoords.length);
        combined.set(vertices);
        combined.set(texCoords, vertices.length);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, combined, this.gl.STATIC_DRAW);
    }
    
    compileShader(type, source) {
        if (!this.gl) return null;
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile failed: ' + this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        return shader;
    }
    
    attachToItem(itemEl, imgUrl) {
        if (!this.gl) return;
        
        const width = itemEl.offsetWidth;
        const height = itemEl.offsetHeight;
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        // Remove existing if any
        const old = itemEl.querySelector('.relighting-canvas');
        if(old) old.remove();
        
        itemEl.appendChild(this.canvas);
        
        // Load Texture
        const image = new Image();
        image.crossOrigin = "anonymous";
        image.src = getProxiedUrl(imgUrl);
        image.onload = () => {
            if (!this.gl) return;
            this.texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, image);
        };
    }
    
    render(lightRelX, lightRelY, color, intensity) {
        if (!this.program || !this.texture || !this.gl) return;
        
        const gl = this.gl;
        gl.useProgram(this.program);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        
        const aPos = gl.getAttribLocation(this.program, "a_position");
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
        
        const aTex = gl.getAttribLocation(this.program, "a_texCoord");
        gl.enableVertexAttribArray(aTex);
        gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 12 * 4);
        
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
        
        gl.uniform2f(gl.getUniformLocation(this.program, "u_resolution"), this.canvas.width, this.canvas.height);
        gl.uniform3f(gl.getUniformLocation(this.program, "u_lightPos"), lightRelX, lightRelY, 150.0);
        gl.uniform3f(gl.getUniformLocation(this.program, "u_lightColor"), color[0], color[1], color[2]);
        gl.uniform1f(gl.getUniformLocation(this.program, "u_intensity"), intensity);
        
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}

// Single shared renderer instance
export const relighter = new RelightingRenderer();

// --- 2.5D Relighting Preview Functions ---
export let activeLightNode = null;

export function updateRelightingPreview(lightNode) {
    const lightConnector = document.getElementById('lightConnector');
    if (!lightNode) return;
    
    activeLightNode = lightNode;
    const nodeRect = lightNode.el.getBoundingClientRect();
    const nodeCx = nodeRect.left + nodeRect.width/2;
    const nodeCy = nodeRect.top + nodeRect.height/2;

    let target = null;
    let minDist = 400; // Interaction radius

    state.workbenchItems.forEach(item => {
        if (item.el.classList.contains('atmosphere-node')) return; // Skip other lights
        if (!item.dataUrl && !item.src) return; // Skip items without image data
        
        const imgRect = item.el.getBoundingClientRect();
        const imgCx = imgRect.left + imgRect.width/2;
        const imgCy = imgRect.top + imgRect.height/2;
        const dist = Math.hypot(nodeCx - imgCx, nodeCy - imgCy);
        
        if (dist < minDist) {
            minDist = dist;
            target = item;
        }
    });

    if (target) {
        // Attach WebGL renderer if not already there
        const imgUrl = target.dataUrl || target.src;
        if (!target.el.querySelector('.relighting-canvas') && imgUrl) {
            relighter.attachToItem(target.el, imgUrl);
        }
        
        target.el.classList.add('lighting-active');
        
        // Calculate position relative to image Top-Left
        const targetRect = target.el.getBoundingClientRect();
        const localX = (nodeCx - targetRect.left);
        const localY = (nodeCy - targetRect.top);
        
        // Get Light Color from node data
        const colorStr = lightNode.el.dataset.color || "[1.0, 1.0, 1.0]";
        const rgb = JSON.parse(colorStr);
        
        relighter.render(localX, localY, rgb, 1.5);
        
        // Draw Connector
        const gridRect = document.getElementById('workbenchZoomContainer').getBoundingClientRect();
        const lineStartX = (nodeCx - gridRect.left) / state.workbenchZoom;
        const lineStartY = (nodeCy - gridRect.top) / state.workbenchZoom;
        const lineEndX = (targetRect.left + targetRect.width/2 - gridRect.left) / state.workbenchZoom;
        const lineEndY = (targetRect.top + targetRect.height/2 - gridRect.top) / state.workbenchZoom;
        
        const len = Math.hypot(lineEndX - lineStartX, lineEndY - lineStartY);
        const angle = Math.atan2(lineEndY - lineStartY, lineEndX - lineStartX);
        
        if (lightConnector) {
            lightConnector.style.display = 'block';
            lightConnector.style.width = len + 'px';
            lightConnector.style.left = lineStartX + 'px';
            lightConnector.style.top = lineStartY + 'px';
            lightConnector.style.transform = `rotate(${angle}rad)`;
        }
        
        // Show Apply Button
        const btn = lightNode.el.querySelector('.apply-light-btn');
        if(btn) {
            btn.style.display = 'block';
            btn.onclick = (e) => {
                e.stopPropagation();
                applyRelighting(target, lightNode, localX, localY);
            };
        }
    } else {
        // Cleanup
        document.querySelectorAll('.lighting-active').forEach(el => {
            el.classList.remove('lighting-active');
            const c = el.querySelector('.relighting-canvas');
            if(c) c.remove();
        });
        if (lightConnector) lightConnector.style.display = 'none';
        const btn = lightNode.el.querySelector('.apply-light-btn');
        if(btn) btn.style.display = 'none';
    }
}

export async function applyRelighting(targetItem, lightNode, relX, relY) {
    // Calculate logic direction for prompt
    const w = targetItem.el.offsetWidth;
    const h = targetItem.el.offsetHeight;
    const dx = relX - w/2;
    const dy = relY - h/2;
    
    let hDir = dx < 0 ? "Left" : "Right";
    let vDir = dy < 0 ? "Top" : "Bottom";
    const colorName = lightNode.el.dataset.label || 'light';
    
    const defaultPrompt = `Relight this image. Add a ${colorName} light source coming from the ${vDir}-${hDir}. Ensure realistic shadows and highlights consistent with the 3D geometry of the subject.`;
    
    const executeRelight = async (customPrompt) => {
        const lightConnector = document.getElementById('lightConnector');
        const btn = lightNode && lightNode.el ? lightNode.el.querySelector('.apply-light-btn') : null;
        if (btn) btn.innerText = "渲染中...";

        const promptToUse = customPrompt || defaultPrompt;
        try {
            const dataUrl = targetItem.dataUrl || targetItem.src;
            const newImgData = await generateRelitImage(promptToUse, dataUrl);
            
            if(newImgData && newImgData.imageData) {
                const imgSrc = `data:${newImgData.mimeType};base64,${newImgData.imageData}`;
                const blob = await (await fetch(imgSrc)).blob();
                const file = new File([blob], "relit.png", {type: newImgData.mimeType});
                
                const rect = targetItem.el.getBoundingClientRect();
                const containerRect = document.getElementById('workbenchZoomContainer').getBoundingClientRect();
                const x = (rect.left - containerRect.left) / state.workbenchZoom + 220;
                const y = (rect.top - containerRect.top) / state.workbenchZoom;
                
                if (addImageToWorkbenchCallback) addImageToWorkbenchCallback(file, '重光照结果', {x, y});
                
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`重光照 [${colorName} from ${vDir}-${hDir}]`, promptToUse, imgSrc, executeRelight);
                } else if (addMessageCallback) {
                    addMessageCallback({ sender: 'bot', type: 'text', content: '✅ 重光照完成。' });
                }
            }
        } catch(e) {
            console.error(e);
            if (addMessageCallback) addMessageCallback({ sender: 'bot', type: 'text', content: '重光照失败: ' + e.message });
        } finally {
            if (btn) btn.innerText = "应用";
            // Cleanup visuals
            if (targetItem && targetItem.el) {
                targetItem.el.classList.remove('lighting-active');
                targetItem.el.querySelector('.relighting-canvas')?.remove();
            }
            if (lightConnector) lightConnector.style.display = 'none';
            if (btn) btn.style.display = 'none';
            activeLightNode = null;
        }
    };

    await executeRelight();
}
