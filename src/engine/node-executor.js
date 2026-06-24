import * as skills from '../ai-services/skills-engine.js';
import { NODE_REGISTRY } from './node-registry.js';
import { fileToBase64 } from '../core/utils.js';

/**
 * Node Executor
 * Responsible for executing a single node by mapping it to the appropriate skill.
 */

const getMaskBbox = async (maskInput) => {
    return new Promise(async (resolve) => {
        let maskDataUrl = maskInput;
        if (maskInput instanceof File || maskInput instanceof Blob) {
            maskDataUrl = await new Promise((res) => {
                const reader = new FileReader();
                reader.onload = (e) => res(e.target.result);
                reader.readAsDataURL(maskInput);
            });
        }
        
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
            let found = false;
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const i = (y * canvas.width + x) * 4;
                    if ((data[i] > 50 || data[i+1] > 50 || data[i+2] > 50) && data[i+3] > 50) { 
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        found = true;
                    }
                }
            }
            if (!found) resolve(null);
            else {
                resolve([
                    (minY / canvas.height) * 1000,
                    (minX / canvas.width) * 1000,
                    (maxY / canvas.height) * 1000,
                    (maxX / canvas.width) * 1000
                ]);
            }
        };
        img.onerror = () => resolve(null);
        img.src = maskDataUrl.startsWith('data:') ? maskDataUrl : `data:image/png;base64,${maskDataUrl}`;
    });
};

export async function executeNode(node, inputs) {
    const nodeDef = NODE_REGISTRY[node.type];
    if (!nodeDef) {
        throw new Error(`Node type "${node.type}" not found in registry.`);
    }

    const skillFn = skills[nodeDef.skill];
    if (!skillFn) {
        throw new Error(`Skill function "${nodeDef.skill}" not found in skills-engine.`);
    }

    console.log(`[NodeExecutor] Executing node: ${node.id} (${node.type})`);

    // Map inputs based on node type and skill requirements
    // This part is crucial for bridging the Node Graph's structure with the existing Skill functions
    let result;
    
    // Apply default values from node definition
    nodeDef.inputs.forEach(inputDef => {
        if (inputs[inputDef.id] === undefined && inputDef.default !== undefined) {
            inputs[inputDef.id] = inputDef.default;
        }
        if (inputDef.type === 'array' && inputs[inputDef.id] !== undefined) {
            if (!Array.isArray(inputs[inputDef.id])) {
                inputs[inputDef.id] = [inputs[inputDef.id]];
            }
        }
    });

    // Ensure prompt is a string if it exists
    if (inputs.prompt && typeof inputs.prompt !== 'string') {
        if (Array.isArray(inputs.prompt)) {
            inputs.prompt = inputs.prompt.join(' ');
        } else if (typeof inputs.prompt === 'object') {
            inputs.prompt = JSON.stringify(inputs.prompt);
        } else {
            inputs.prompt = String(inputs.prompt);
        }
    }

    switch (node.type) {
        case "generate-image":
            // generateImage(prompt, aspectRatio)
            result = await skillFn(inputs.prompt, inputs.aspectRatio);
            if (!result || !result.imageData) throw new Error("图像生成失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "material-replace":
        case "inpainting":
            let bbox = null;
            if (inputs.mask) {
                bbox = await getMaskBbox(inputs.mask);
            }
            
            if (!bbox) {
                console.log(`[NodeExecutor] No mask provided or mask empty for ${node.type}, attempting automatic bounding box detection...`);
                try {
                    const targetInfo = await skills.analyzeTargetBoundingBox(inputs.image, inputs.prompt);
                    if (targetInfo && targetInfo.bbox) {
                        console.log(`[NodeExecutor] Found target "${targetInfo.name}" at bbox:`, targetInfo.bbox);
                        bbox = targetInfo.bbox;
                    }
                } catch (err) {
                    console.error("[NodeExecutor] Error during automatic bounding box detection:", err);
                }
            }

            if (bbox) {
                try {
                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    img.src = inputs.image;
                    await new Promise(r => img.onload = r);
                    
                    const imgW = img.naturalWidth;
                    const imgH = img.naturalHeight;
                    
                    const originalCanvas = document.createElement('canvas');
                    originalCanvas.width = imgW;
                    originalCanvas.height = imgH;
                    const oCtx = originalCanvas.getContext('2d');
                    oCtx.drawImage(img, 0, 0);
                    
                    const [ymin, xmin, ymax, xmax] = bbox;
                    let bx = (xmin / 1000) * imgW;
                    let by = (ymin / 1000) * imgH;
                    let bw = ((xmax - xmin) / 1000) * imgW;
                    let bh = ((ymax - ymin) / 1000) * imgH;
                    
                    // Expand the bounding box slightly (e.g., 5%) to give context and avoid edge artifacts
                    const paddingX = bw * 0.05;
                    const paddingY = bh * 0.05;
                    bx = Math.max(0, bx - paddingX);
                    by = Math.max(0, by - paddingY);
                    bw = Math.min(imgW - bx, bw + paddingX * 2);
                    bh = Math.min(imgH - by, bh + paddingY * 2);
                    
                    const maskCanvas = document.createElement('canvas');
                    maskCanvas.width = imgW;
                    maskCanvas.height = imgH;
                    const mCtx = maskCanvas.getContext('2d');
                    mCtx.fillStyle = '#000000';
                    mCtx.fillRect(0, 0, imgW, imgH);
                    mCtx.fillStyle = '#FFFFFF';
                    const inset = Math.max(1, Math.round(imgW / 1000));
                    mCtx.fillRect(bx + inset, by + inset, bw - inset * 2, bh - inset * 2);
                    
                    // Calculate closest supported aspect ratio and pad the image
                    const supportedRatios = [
                        { str: '1:1', val: 1 },
                        { str: '4:3', val: 4/3 },
                        { str: '3:4', val: 3/4 },
                        { str: '16:9', val: 16/9 },
                        { str: '9:16', val: 9/16 }
                    ];
                    const imgRatio = imgW / imgH;
                    let bestRatio = supportedRatios[0];
                    let minDiff = Math.abs(imgRatio - bestRatio.val);
                    for (const r of supportedRatios) {
                        const diff = Math.abs(imgRatio - r.val);
                        if (diff < minDiff) {
                            minDiff = diff;
                            bestRatio = r;
                        }
                    }

                    let padW = imgW;
                    let padH = imgH;
                    if (imgRatio > bestRatio.val) {
                        padH = imgW / bestRatio.val;
                    } else {
                        padW = imgH * bestRatio.val;
                    }

                    const offsetX = (padW - imgW) / 2;
                    const offsetY = (padH - imgH) / 2;

                    const paddedOriginalCanvas = document.createElement('canvas');
                    paddedOriginalCanvas.width = padW;
                    paddedOriginalCanvas.height = padH;
                    const poCtx = paddedOriginalCanvas.getContext('2d');
                    poCtx.fillStyle = '#FFFFFF';
                    poCtx.fillRect(0, 0, padW, padH);
                    poCtx.drawImage(originalCanvas, offsetX, offsetY, imgW, imgH);
                    const base64Original = paddedOriginalCanvas.toDataURL('image/png').split(',')[1];

                    const paddedMaskCanvas = document.createElement('canvas');
                    paddedMaskCanvas.width = padW;
                    paddedMaskCanvas.height = padH;
                    const pmCtx = paddedMaskCanvas.getContext('2d');
                    pmCtx.fillStyle = '#000000';
                    pmCtx.fillRect(0, 0, padW, padH);
                    pmCtx.drawImage(maskCanvas, offsetX, offsetY, imgW, imgH);
                    const base64Mask = paddedMaskCanvas.toDataURL('image/png').split(',')[1];
                    
                    // Prepare prompt for precise edit (MATCHING layer-manager.js EXACTLY)
                    let materialProtocol = "";
                    let semanticAction = "";
                    let finalCommandAction = "";

                    if (node.type === 'material-replace') {
                        materialProtocol = `
            ## SPECIAL SKILL: STRUCTURAL-PRESERVING MATERIAL REPLACEMENT
            - **3D MESH LOCK (CRITICAL)**: Treat the object in Image 0 as a strictly fixed 3D mesh. You are a rendering engine. Do not alter its geometry, volume, or silhouette.
            - **SURFACE PROJECTION**: Do not "redraw" the object. Instead, "project" and "wrap" the requested material from Image 2+ onto the existing pixels as a UV texture map. The underlying structural skeleton must not shift by even a single pixel.
            - **NEGATIVE CONSTRAINTS**: FORBIDDEN ACTIONS: Do not add new structural elements (e.g., new wrinkles, folds, seams, buttons, or bumps) that do not exist in Image 0. Do not smooth out or delete existing structural indentations. The micro-structure must remain 100% identical.
            - **TEXTURE PURGE & REFERENCE FIDELITY**: Completely discard the original texture/patterns. Strictly use the textures, materials, and surface properties from Image 2+.
            - **LIGHTING ADAPTATION**: Apply the new material while inheriting the exact lighting, shadows, and reflections from the environment in Image 0.
                        `;
                        semanticAction = `**MATERIAL PROJECTION**: Project and wrap the new material onto the object to match the instruction: "${inputs.prompt}", without altering the underlying structure.`;
                        finalCommandAction = `Perform a high-precision material projection strictly confined to the boxed area.`;
                    } else {
                        semanticAction = `**SEMANTIC REGENERATION**: Redraw the object/area within the box to match the instruction: "${inputs.prompt}".`;
                        finalCommandAction = `Perform a high-precision redraw strictly confined to the boxed area.`;
                    }
                    
                    const systemProtocol = `
            ## SKILL: PRECISION INPAINTING & MATERIAL TRANSFORMATION
            - Image 0: The clean original source image.
            - Image 1: A spatial mask (Black & White). The WHITE area defines the STRICT and ONLY quarantine zone you are allowed to edit. The BLACK area MUST remain 100% identical to Image 0.
            - Image 2+: Reference textures or styles to be applied.
            
            ${materialProtocol}

            CRITICAL DIRECTIVES & PENALTIES:
            1. **THE QUARANTINE ZONE**: You are ONLY permitted to alter the pixels corresponding to the WHITE area in Image 1. 
            2. **ABSOLUTE FREEZE OUTSIDE**: Every single pixel corresponding to the BLACK area in Image 1 MUST remain 100% identical to Image 0. If you alter the background, lighting, or any object in the black zone, the operation is considered a catastrophic failure.
            3. **SEAMLESS INTEGRATION**: ${semanticAction} The new content inside the white zone must blend flawlessly with the frozen pixels outside the zone. Match the original lighting, shadows, and grain perfectly.
            4. **NO HALLUCINATIONS**: Do not invent new objects or structures that were not requested.
                    `;
                    
                    const finalEditPrompt = `
            [SYSTEM INSTRUCTION]
            ${systemProtocol}
            
            FINAL COMMAND: ${finalCommandAction} 
            WARNING: You will be penalized if ANY pixel corresponding to the BLACK area in Image 1 is changed. 
            MANDATORY: Return ONLY the clean, final image.
                    `;
                    
                    const contentsParts = [
                        { inlineData: { data: base64Original, mimeType: 'image/png' } },
                        { inlineData: { data: base64Mask, mimeType: 'image/png' } }
                    ];
                    
                    if (inputs.referenceImages && inputs.referenceImages.length > 0) {
                        for (const refFile of inputs.referenceImages) {
                            if (!refFile) continue;
                            const refBase64 = await fileToBase64(refFile);
                            contentsParts.push({ inlineData: { data: refBase64, mimeType: refFile.type || 'image/png' } });
                        }
                    }
                    
                    contentsParts.push({ text: finalEditPrompt });
                    
                    const resultImageData = await skills.generatePreciseEditImage(contentsParts, bestRatio.str);
                    if (resultImageData) {
                        // Composite final image
                        const aiResultImg = new Image();
                        aiResultImg.crossOrigin = "anonymous";
                        aiResultImg.src = `data:image/png;base64,${resultImageData}`;
                        await new Promise(r => aiResultImg.onload = r);
                        
                        const unpaddedCanvas = document.createElement('canvas');
                        unpaddedCanvas.width = imgW;
                        unpaddedCanvas.height = imgH;
                        const uCtx = unpaddedCanvas.getContext('2d');

                        const scaleX = aiResultImg.width / padW;
                        const scaleY = aiResultImg.height / padH;
                        
                        const sx = offsetX * scaleX;
                        const sy = offsetY * scaleY;
                        const sWidth = imgW * scaleX;
                        const sHeight = imgH * scaleY;

                        uCtx.drawImage(aiResultImg, sx, sy, sWidth, sHeight, 0, 0, imgW, imgH);

                        const finalCanvas = document.createElement('canvas');
                        finalCanvas.width = imgW;
                        finalCanvas.height = imgH;
                        const fCtx = finalCanvas.getContext('2d');
                        
                        fCtx.drawImage(img, 0, 0);
                        
                        // Create mask for compositing
                        const compositeMaskCanvas = document.createElement('canvas');
                        compositeMaskCanvas.width = imgW;
                        compositeMaskCanvas.height = imgH;
                        const cmCtx = compositeMaskCanvas.getContext('2d');
                        cmCtx.clearRect(0, 0, imgW, imgH);
                        
                        if (inputs.mask) {
                            // If user provided a mask, use it for perfect compositing
                            const userMaskImg = new Image();
                            userMaskImg.crossOrigin = "anonymous";
                            let maskDataUrl = inputs.mask;
                            if (inputs.mask instanceof File || inputs.mask instanceof Blob) {
                                maskDataUrl = await new Promise((res) => {
                                    const reader = new FileReader();
                                    reader.onload = (e) => res(e.target.result);
                                    reader.readAsDataURL(inputs.mask);
                                });
                            }
                            userMaskImg.src = maskDataUrl.startsWith('data:') ? maskDataUrl : `data:image/png;base64,${maskDataUrl}`;
                            await new Promise(r => userMaskImg.onload = r);
                            cmCtx.drawImage(userMaskImg, 0, 0, imgW, imgH);
                        } else {
                            // Otherwise use the bounding box
                            cmCtx.fillStyle = 'white';
                            const lineWidth = Math.max(4, Math.round(imgW / 200));
                            const inset = Math.ceil(lineWidth / 2) + 1;
                            cmCtx.fillRect(bx + inset, by + inset, bw - inset * 2, bh - inset * 2);
                        }
                        
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = imgW;
                        tempCanvas.height = imgH;
                        const tCtx = tempCanvas.getContext('2d');
                        tCtx.drawImage(unpaddedCanvas, 0, 0);
                        tCtx.globalCompositeOperation = 'destination-in';
                        tCtx.drawImage(compositeMaskCanvas, 0, 0);
                        
                        fCtx.drawImage(tempCanvas, 0, 0);
                        
                        const finalBase64 = finalCanvas.toDataURL('image/png');
                        return { image: finalBase64 };
                    } else {
                        console.warn("[NodeExecutor] Precise edit failed, falling back to generic edit.");
                    }
                } catch (err) {
                    console.error("[NodeExecutor] Error during precise edit:", err);
                }
            }
            
            // Fallthrough to generic edit if mask is present but precise edit failed, or no bbox found
        case "edit-image":
            const forceMaterial = node.type === 'material-replace';
            result = await skills.editOrQueryImageWithGemini(inputs.prompt, inputs.image, inputs.referenceImages || [], inputs.mask, null, forceMaterial);
            if (!result || !result.imageData) throw new Error("图像编辑失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "relight":
            // generateRelitImage(prompt, base64Image)
            result = await skillFn(inputs.prompt, inputs.image);
            if (!result || !result.imageData) throw new Error("光影重绘失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "analyze":
            // analyzeWithAgent(agent, base64Data, mimeType)
            const agent = { id: inputs.agentType, role: inputs.agentType }; // Simplified agent object
            result = await skillFn(agent, inputs.image, 'image/png');
            if (!result) throw new Error("图像分析失败，未返回数据。");
            return { analysis: result };

        case "visual-search":
            // generateVisualSearch(croppedBase64)
            result = await skillFn(inputs.image);
            if (!result || !result.resultData) throw new Error("视觉搜索失败，未返回数据。");
            return { data: result.resultData, links: result.webLinks };

        case "layer-analysis":
            // analyzeImageLayers(file)
            result = await skillFn(inputs.image);
            if (!result) throw new Error("图层分析失败，未返回数据。");
            return { layers: result };

        case "crop":
            // cropImageByLayer(image, layers, layerName)
            result = await skillFn(inputs.image, inputs.layers, inputs.layerName);
            if (!result || !result.imageData) throw new Error("裁剪失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "outpaint":
            result = await skillFn(inputs.image, inputs.prompt, inputs.aspectRatio);
            if (!result || !result.imageData) throw new Error("外扩失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "camera-lens":
            result = await skillFn(inputs.image, inputs.prompt);
            if (!result || !result.imageData) throw new Error("相机镜头调整失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "video-generation":
            result = await skillFn(inputs.image, inputs.prompt, inputs.aspectRatio);
            if (!result) throw new Error("视频生成失败，未返回数据。");
            // result is a Blob, convert to Object URL or base64?
            // The graph expects a URL or base64. Let's return an Object URL for now.
            return { video: URL.createObjectURL(result) };

        case "upscale":
            result = await skillFn(inputs.image, inputs.prompt);
            if (!result || !result.imageData) throw new Error("高清增强失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        case "multiview":
            result = await skillFn(inputs.image, inputs.prompt);
            if (!result || !result.imageData) throw new Error("多维视图生成失败，未返回数据。");
            return { image: `data:${result.mimeType || 'image/png'};base64,${result.imageData}` };

        default:
            throw new Error(`Execution logic for node type "${node.type}" not implemented.`);
    }
}
