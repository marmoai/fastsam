import { ai, getTextModel, backgroundModel, getImageModel, MODEL_SUITES, proxyGenerateContent } from "./gemini-client.js";
import { ThinkingLevel } from "@google/genai";
import { state } from "../core/state.js";
import * as prompts from "./prompts.js";
import { isRemovalRequest, isMaterialRequest, fileToBase64, dataURLToFile, compressImage, getProxiedUrl, OSS_BACKEND_URL, RECOGNIZE_BACKEND_URL } from "../core/utils.js";
import { createImageWithHole } from "../graphics/image-processor.js";

const getThinkingLevel = () => {
    return state.thinkingLevel === 'MEDIUM' ? ThinkingLevel.HIGH : ThinkingLevel.LOW;
};

const getSemanticModel = () => MODEL_SUITES.flash.text || getTextModel();

export async function extractTextFromImage(base64Image, bbox = [0,0,1000,1000], options = {}) {
    try {
        const semanticModel = getSemanticModel();
        console.log(`[SemanticText] model=${semanticModel}, thinking=ui`);
        let cleanBase64 = '';
        let mimeType = 'image/png';
        if (typeof base64Image === 'string') {
            if (base64Image.startsWith('data:')) {
                mimeType = base64Image.substring(5, base64Image.indexOf(';'));
                cleanBase64 = base64Image.split(',')[1];
            } else if (base64Image.startsWith('blob:') || base64Image.startsWith('http')) {
                const res = await fetch(getProxiedUrl(base64Image));
                const blob = await res.blob();
                mimeType = blob.type || 'image/png';
                const dataUrl = await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.readAsDataURL(blob);
                });
                cleanBase64 = dataUrl.split(',')[1];
            } else {
                cleanBase64 = base64Image.includes(',') ? base64Image.split(',')[1] : base64Image;
            }
        }
        
        const response = await proxyGenerateContent({
            model: semanticModel,
            contents: {
                parts: [
                    { inlineData: { data: cleanBase64, mimeType: mimeType } },
                    { text: "Extract the EXACT text content found within the bounding box [ymin="+bbox[0]+", xmin="+bbox[1]+", ymax="+bbox[2]+", xmax="+bbox[3]+"] (normalized 0-1000). \n" +
"CRITICAL REQUIREMENTS:\n" +
"0. EXTRACTION SCOPE: "+(options.mode === 'document' ? "Document mode: extract all visible text blocks in the requested bbox." : "Strict selection mode: return ONLY text glyphs whose tight bbox is inside or substantially overlaps the requested bbox. Ignore all other text in the image even if visually related.")+"\n" +
"1. EXACT TEXT & LINE BREAKS: If the text is physically on a SINGLE horizontal line in the image, NEVER insert '\\n'. Only use '\\n' if the text is ACTUALLY stacked or spread across multiple distinct lines. NEVER split a word across lines or characters; output 'ROASTED' not 'ROASTE\\nD', and 'FRIED' not 'FRIE\\nD'. Provide a TIGHT bounding box [ymin, xmin, ymax, xmax] for each grouped text block.\n" +
"2. METALLIC & GRADIENT DETECTION (SUPER IMPORTANT): Observe carefully. If the text has ANY shine, gold/silver/metal reflection, or gradient: YOU ABSOLUTELY MUST use a multi-stop 'background-image: linear-gradient(...)'. Include multiple color stops to PERFECTLY simulate the shine/reflection (e.g., light-dark-light-dark for metal). DO NOT output plain 'color' if it's metallic or gradient!\n" +
"3. Typography details: Identify approximate 'font-size', 'line-height', font-weight (e.g., 900 for heavy bold), font-style, 'letter-spacing' (e.g., '0.05em'), 'text-align', and 'transform' (e.g., 'rotate(-5deg)'). Font-size must describe the visible text height inside the provided bbox, not a default CSS size.\n" +
"4. Output a 'css' object containing CSS properties that perfectly mimic the visual effects:\n" +
"   - For solid text: output exact 'color' (HEX/RGBA).\n" +
"   - For gradients/metallic: you MUST use 'background-image: linear-gradient(...)'. IF USING A GRADIENT you MUST ALSO include: '-webkit-background-clip': 'text' AND '-webkit-text-fill-color': 'transparent'.\n" +
"   - Include 'font-family'. For Chinese, MUST use: 'Noto Sans SC', 'Noto Serif SC', 'Ma Shan Zheng', 'Zhi Mang Xing', 'ZCOOL KuaiLe', 'ZCOOL QingKe HuangYou', or 'Long Cang'. For English, use common standard fonts.\n" +
"   - NEVER omit 'font-weight': use 300/400/500/600/700/800/900 based on the visible stroke thickness. Heavy poster/display text is usually 800 or 900.\n" +
"   - Use '-webkit-text-stroke' for clear text outlines, border text, or sticker-like lettering.\n" +
"   - Use 'text-shadow' for drop shadows, glows, or 3D extrusions (e.g., '2px 2px 0px #111'). COMBINE text-shadow with gradients for 3D text.\n" +
"   - Include 'filter' for blur/glow/brightness effects when visible, 'opacity' when translucent, 'text-transform' when uppercase/small-caps styling is visible, and 'transform' for rotation/skew.\n" +
"   - If the source uses a condensed, expanded, handwritten, serif, sans, display, or script style, choose the closest available CSS font-family and add 'font-stretch' when useful.\n" +
"5. TEXT-ONLY LAYERS: If the text sits on a distinct shape, pill, button, card, badge, or panel, still return ONLY the text glyph layer with a tight glyph bbox. Do NOT merge the background container into the text layer and do NOT add panel background-color/border-radius/padding to css; those containers are separate shape layers.\n" +
"6. FLAT DESIGN OCR: Split visible typography by visual editing unit. Different font size, weight, color, baseline, line position, or container relationship means separate lines. Include small captions/body copy and label text on product cards even when the text is on top of a colored panel.\n" +
"\n" +
"Return ONLY a valid JSON object in this exact format without markdown blocks:\n" +
"{\n" +
"  \"lines\": [\n" +
"    {\n" +
"      \"textContent\": \"line 1 text\\\\nline 2 text\",\n" +
"      \"bbox\": [ymin, xmin, ymax, xmax],\n" +
"      \"css\": { \n" +
"         \"font-family\": \"'Impact', sans-serif\",\n" +
"         \"font-size\": \"48px\",\n" +
"         \"font-weight\": \"900\",\n" +
"         \"line-height\": \"1.05\",\n" +
"         \"letter-spacing\": \"0.02em\",\n" +
"         \"text-align\": \"center\",\n" +
"         \"color\": \"#ffffff\",\n" +
"         \"background-image\": \"linear-gradient(to bottom, #ffe5aa, #ffd257 30%, #d89617 50%, #ffe9a6 51%, #fffdf4)\", \n" +
"         \"-webkit-background-clip\": \"text\", \n" +
"         \"-webkit-text-fill-color\": \"transparent\",\n" +
"         \"text-shadow\": \"2px 2px 4px rgba(0,0,0,0.8)\",\n" +
"         \"-webkit-text-stroke\": \"1px #5a3c00\",\n" +
"         \"text-transform\": \"uppercase\",\n" +
"         \"font-stretch\": \"condensed\",\n" +
"         \"filter\": \"drop-shadow(0 2px 2px rgba(0,0,0,0.45))\",\n" +
"         \"transform\": \"rotate(-3deg)\"\n" +
"      }\n" +
"    }\n" +
"  ]\n" +
"}" }
                ]
            },
            config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });
        
        const jsonStr = response.text.trim();
        const startIndex = jsonStr.indexOf('{');
        const endIndex = jsonStr.lastIndexOf('}');
        if (startIndex === -1 || endIndex === -1) throw new Error("No JSON found");
        const validJson = jsonStr.substring(startIndex, endIndex + 1).replace(/```json|```/g, '');
        return JSON.parse(validJson);
    } catch (e) {
        console.error("Text extraction failed:", e);
        return null;
    }
}

async function ensurePngBase64(imageInput) {
    if (!imageInput) return null;
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const pngDataUrl = canvas.toDataURL('image/png');
            resolve(pngDataUrl);
        };
        img.onerror = () => {
            if (typeof imageInput === 'string') {
                resolve(imageInput.includes(',') ? imageInput : 'data:image/png;base64,' + imageInput);
            } else {
                resolve(null);
            }
        };
        if (typeof imageInput === 'string') {
            if (imageInput.startsWith('data:') || imageInput.startsWith('blob:') || imageInput.startsWith('http')) {
                img.src = imageInput;
            } else {
                img.src = 'data:image/png;base64,' + imageInput;
            }
        } else if (imageInput instanceof File || imageInput instanceof Blob) {
            img.src = URL.createObjectURL(imageInput);
        } else {
            resolve(null);
        }
    });
}

async function prepareNativeEdit(imageInput, maskInput, direction, targetRatioName, isOutpaint) {
    const RATIO_MAP = {
        '1:1': 1.0,
        '16:9': 16 / 9,
        '9:16': 9 / 16,
        '4:3': 4 / 3,
        '3:4': 3 / 4
    };
    const targetRatioVal = RATIO_MAP[targetRatioName] || 1.0;

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
            const naturalW = img.naturalWidth;
            const naturalH = img.naturalHeight;
            const originalRatio = naturalW / naturalH;

            let cageW = naturalW;
            let cageH = naturalH;

            if ((isOutpaint || targetRatioName !== '1:1') && targetRatioName) {
                if (targetRatioVal >= originalRatio) {
                    cageW = Math.round(naturalH * targetRatioVal);
                    cageH = naturalH;
                } else {
                    cageW = naturalW;
                    cageH = Math.round(naturalW / targetRatioVal);
                }
            }

            cageW = Math.round(cageW / 16) * 16 || 16;
            cageH = Math.round(cageH / 16) * 16 || 16;

            const canvasImg = document.createElement('canvas');
            canvasImg.width = cageW;
            canvasImg.height = cageH;
            const ctxImg = canvasImg.getContext('2d');

            const canvasMask = document.createElement('canvas');
            canvasMask.width = cageW;
            canvasMask.height = cageH;
            const ctxMask = canvasMask.getContext('2d');

            let x = 0, y = 0;
            if (cageW !== naturalW || cageH !== naturalH) {
                if (isOutpaint) {
                    if (direction === 'left') {
                        x = cageW - naturalW;
                    } else if (direction === 'right') {
                        x = 0;
                    } else if (direction === 'bottom') {
                        y = 0;
                    } else if (direction === 'top') {
                        y = cageH - naturalH;
                    } else {
                        x = Math.round((cageW - naturalW) / 2);
                        y = Math.round((cageH - naturalH) / 2);
                    }
                } else {
                    x = Math.round((cageW - naturalW) / 2);
                    y = Math.round((cageH - naturalH) / 2);
                }
            }

            ctxImg.fillStyle = '#FFFFFF';
            ctxImg.fillRect(0, 0, cageW, cageH);
            ctxImg.drawImage(img, x, y, naturalW, naturalH);

            let finalMaskBase64 = null;
            if (isOutpaint || maskInput || cageW !== naturalW || cageH !== naturalH) {
                ctxMask.fillStyle = 'rgba(0,0,0,0)';
                ctxMask.clearRect(0,0, cageW, cageH);

                ctxMask.fillStyle = '#FFFFFF';
                ctxMask.fillRect(x, y, naturalW, naturalH);

                if (maskInput) {
                    const maskImg = new Image();
                    await new Promise((resMask) => {
                        maskImg.onload = () => {
                            const tempCanvas = document.createElement('canvas');
                            tempCanvas.width = naturalW;
                            tempCanvas.height = naturalH;
                            const tempCtx = tempCanvas.getContext('2d');
                            tempCtx.drawImage(maskImg, 0, 0, naturalW, naturalH);

                            const imgData = tempCtx.getImageData(0, 0, naturalW, naturalH);
                            const data = imgData.data;

                            for (let i = 0; i < data.length; i += 4) {
                                const brightness = Math.round((data[i] + data[i+1] + data[i+2]) / 3);
                                if (brightness > 127 && data[i+3] > 127) {
                                    data[i+3] = 0;
                                } else {
                                    data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255;
                                }
                            }
                            tempCtx.putImageData(imgData, 0, 0);

                            ctxMask.clearRect(x, y, naturalW, naturalH);
                            ctxMask.drawImage(tempCanvas, x, y);
                            resMask();
                        };
                        maskImg.onerror = () => resMask();
                        if (typeof maskInput === 'string') {
                            maskImg.src = maskInput.startsWith('data:') ? maskInput : 'data:image/png;base64,' + maskInput;
                        } else { resMask(); }
                    });
                }
                finalMaskBase64 = canvasMask.toDataURL('image/png').split(',')[1];
            }

            resolve({
                image: canvasImg.toDataURL('image/png').split(',')[1],
                mask: finalMaskBase64,
                width: cageW,
                height: cageH
            });
        };
        img.onerror = reject;

        if (typeof imageInput === 'string') {
            img.src = imageInput.startsWith('data:') || imageInput.startsWith('blob:') || imageInput.startsWith('http') ? imageInput : 'data:image/png;base64,' + imageInput;
        } else if (imageInput instanceof File || imageInput instanceof Blob) {
            img.src = URL.createObjectURL(imageInput);
        }
    });
}

async function prepareNativeMaskAndImage(imageInput, maskInput) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
            const naturalW = img.naturalWidth;
            const naturalH = img.naturalHeight;

            const canvasImg = document.createElement('canvas');
            canvasImg.width = naturalW;
            canvasImg.height = naturalH;
            const ctxImg = canvasImg.getContext('2d');
            
            ctxImg.fillStyle = '#FFFFFF';
            ctxImg.fillRect(0, 0, naturalW, naturalH);
            ctxImg.drawImage(img, 0, 0, naturalW, naturalH);

            let finalMaskBase64 = null;
            if (maskInput) {
                const canvasMask = document.createElement('canvas');
                canvasMask.width = naturalW;
                canvasMask.height = naturalH;
                const ctxMask = canvasMask.getContext('2d');
                ctxMask.fillStyle = '#FFFFFF';
                ctxMask.fillRect(0, 0, naturalW, naturalH);

                const maskImg = new Image();
                await new Promise((resMask) => {
                    maskImg.onload = () => {
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = naturalW;
                        tempCanvas.height = naturalH;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(maskImg, 0, 0, naturalW, naturalH);
                        
                        const imgData = tempCtx.getImageData(0, 0, naturalW, naturalH);
                        const data = imgData.data;
                        
                        // Convert grayscale mask (white edit zone) to transparent (A=0)
                        for (let i = 0; i < data.length; i += 4) {
                            const brightness = Math.round((data[i] + data[i+1] + data[i+2]) / 3);
                            if (brightness > 127 && data[i+3] > 127) {
                                data[i+3] = 0; // EDIT (Transparent)
                            } else {
                                data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255; // PRESERVE (Opaque)
                            }
                        }
                        tempCtx.putImageData(imgData, 0, 0);
                        
                        ctxMask.clearRect(0, 0, naturalW, naturalH);
                        ctxMask.drawImage(tempCanvas, 0, 0);
                        resMask();
                    };
                    maskImg.onerror = () => {
                        resMask();
                    };
                    if (typeof maskInput === 'string') {
                        maskImg.src = maskInput.startsWith('data:') ? maskInput : 'data:image/png;base64,' + maskInput;
                    } else if (maskInput instanceof File || maskInput instanceof Blob) {
                        maskImg.src = URL.createObjectURL(maskInput);
                    } else {
                        resMask();
                    }
                });
                finalMaskBase64 = canvasMask.toDataURL('image/png').split(',')[1];
            }

            resolve({
                image: canvasImg.toDataURL('image/png').split(',')[1],
                mask: finalMaskBase64
            });
        };
        img.onerror = reject;

        if (typeof imageInput === 'string') {
            img.src = imageInput.startsWith('data:') || imageInput.startsWith('blob:') || imageInput.startsWith('http') ? imageInput : 'data:image/png;base64,' + imageInput;
        } else if (imageInput instanceof File || imageInput instanceof Blob) {
            img.src = URL.createObjectURL(imageInput);
        }
    });
}

export async function editOrQueryImageWithGemini(prompt, baseImage, referenceImages = [], mask = null, forcedAspectRatio = null, forceMaterialTask = false) {
    prompt = prompt || "处理图片";
    if (!baseImage) throw new Error("A base image is required for editing or querying.");

    // Direct QuickRouter GPT-image-2 API orchestration
    if (getImageModel() === 'gpt-image-2') {
        console.log("🚀 [GPT-image-2] 正在自适应路由调用高精度中转图像编辑...");
        try {
            const isOutpaint = !mask && forcedAspectRatio && forcedAspectRatio !== '1:1';
            
            let finalImageBase64 = null;
            let finalMaskBase64 = null;
            let dimensions = null;

            let direction = 'right';
            const lowerPrompt = prompt.toLowerCase();
            if (lowerPrompt.includes('upward') || lowerPrompt.includes('向上') || lowerPrompt.includes('top') || lowerPrompt.includes('up') || lowerPrompt.includes('上')) {
                direction = 'top';
            } else if (lowerPrompt.includes('downward') || lowerPrompt.includes('向下') || lowerPrompt.includes('bottom') || lowerPrompt.includes('down') || lowerPrompt.includes('下')) {
                direction = 'bottom';
            } else if (lowerPrompt.includes('left') || lowerPrompt.includes('向左') || lowerPrompt.includes('左')) {
                direction = 'left';
            } else if (lowerPrompt.includes('right') || lowerPrompt.includes('向右') || lowerPrompt.includes('右')) {
                direction = 'right';
            }

            const prepared = await prepareNativeEdit(baseImage, mask, direction, forcedAspectRatio || '1:1', isOutpaint);
            finalImageBase64 = prepared.image;
            finalMaskBase64 = prepared.mask;

            const payload = {
                mode: "image_edit",
                image: finalImageBase64,
                prompt: prompt,
                size: `${prepared.width}x${prepared.height}`,
                quality: "auto",
                output_format: "png"
            };
            if (finalMaskBase64) {
                payload.mask = finalMaskBase64;
            }

            const response = await fetch(`${RECOGNIZE_BACKEND_URL}`, {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`GPT-image-2 接口返回 ${response.status}`);
            }

            const data = await response.json();
            console.log("📥 [GPT-image-2] 接收原始 API 响应成果:", data);

            const result = { success: true, text: '图片已使用 GPT-image-2 图像生成模型高精编辑完成。', imageData: null, mimeType: 'image/png' };

            let rawData = null;
            if (data.data && data.data[0]) {
                const item = data.data[0];
                if (item.b64_json) {
                    rawData = item.b64_json;
                } else if (item.url) {
                    rawData = item.url;
                }
            } else if (data.choices && data.choices[0] && data.choices[0].message) {
                const content = data.choices[0].message.content;
                if (content && (content.startsWith('http') || content.trim().length > 100)) {
                    rawData = content.includes(',') ? content.split(',')[1] : content;
                }
            }

            if (!rawData) {
                throw new Error("接口返回无法提取有效图像 Base64/URL 成果。");
            }

            result.imageData = rawData;

            return result;
        } catch (error) {
            console.error("⚠️ [GPT-image-2] 调用链路异常，抛出异常阻断降级:", error);
            throw error;
        }
    }

    const allParts = [];
    const isMaskedEditTask = !!mask;
    const isRemovalTask = isRemovalRequest(prompt);
    const isMaterialTask = forceMaterialTask || isMaterialRequest(prompt); 

    let finalPrompt;

    if (isMaskedEditTask) {
        const imageBase64 = await fileToBase64(baseImage);
        allParts.push({ inlineData: { data: imageBase64, mimeType: baseImage.type || 'image/png' } });
        
        const maskBase64 = mask.includes(',') ? mask.split(',')[1] : mask;
        allParts.push({ inlineData: { data: maskBase64, mimeType: 'image/png' } });

        if (isMaterialTask) {
            finalPrompt = prompts.getMaterialReplacementPrompt(prompt);
        } else if (isRemovalTask) {
            finalPrompt = prompts.INPAINT_REMOVAL_PROMPT + "\nAdditional Context: " + prompt;
        } else if (referenceImages.length > 0) {
            finalPrompt = prompts.getInpaintReferencePrompt(prompt);
        } else {
            finalPrompt = prompts.getInpaintGenericPrompt(prompt);
        }
    } else {
        const imageBase64 = await fileToBase64(baseImage);
        allParts.push({ inlineData: { data: imageBase64, mimeType: baseImage.type || 'image/png' } });
        if (isMaterialTask) {
            finalPrompt = prompts.getMasklessMaterialReplacementPrompt(prompt);
        } else if (referenceImages.length > 0) {
            finalPrompt = prompts.getReferenceStylePrompt(prompt);
        } else {
            finalPrompt = prompt;
        }
    }

    if (!isRemovalTask) {
        for (const file of referenceImages) {
            if (!file) continue;
            const base64Data = await fileToBase64(file);
            allParts.push({ inlineData: { data: base64Data, mimeType: file.type || 'image/png' } });
        }
    }

    const combinedPrompt = finalPrompt + prompts.TEXT_OUTPUT_INSTRUCTION_SUFFIX;
    allParts.push({ text: combinedPrompt });
    
    const config = {};
    if (forcedAspectRatio) {
        config.imageConfig = { aspectRatio: forcedAspectRatio };
    }

    const currentModel = getImageModel();
    let response;

    if (currentModel.startsWith('gemini')) {
        const payload = {
            contents: [{ role: "user", parts: allParts }],
            config: Object.keys(config).length > 0 ? config : undefined
        };

        const res = await fetch(RECOGNIZE_BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'gemini_lite_generate',
                model: currentModel,
                payload: payload
            })
        });
        response = await res.json();
    } else {
        response = await proxyGenerateContent({
            model: currentModel,
            contents: { parts: allParts },
            config: {
                ...(Object.keys(config).length > 0 ? config : {})
            }
        });
    }
    
    const result = { success: true, text: '', imageData: null, mimeType: null };
    
    if (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.text) {
                result.text += part.text;
            } else if (part.inlineData) {
                result.imageData = part.inlineData.data;
                result.mimeType = part.inlineData.mimeType || 'image/png';
            }
        }
    } else if (response.candidates && response.candidates[0] && response.candidates[0].parts) {
        // Fallback for slightly different structure
        for (const part of response.candidates[0].parts) {
            if (part.text) {
                result.text += part.text;
            } else if (part.inlineData) {
                result.imageData = part.inlineData.data;
                result.mimeType = part.inlineData.mimeType || 'image/png';
            }
        }
    } else {
        console.error('Gemini response structure unexpected:', response);
    }
    
    if (!result.text && !result.imageData) throw new Error("Content was blocked or the response was empty.");
    return result;
}

export async function editOrQueryImageWithGemini_Multiple(prompt, baseImage, referenceImages = [], mask = null, forcedAspectRatio = null) {
    const promises = Array(4).fill().map(() => editOrQueryImageWithGemini(prompt, baseImage, referenceImages, mask, forcedAspectRatio));
    const results = await Promise.all(promises);
    
    const combinedResult = {
        success: true,
        text: results[0].text,
        imageData: results.map(r => ({ imageData: r.imageData, mimeType: r.mimeType })),
        mimeType: results[0].mimeType
    };
    return combinedResult;
}

export async function generateSessionTitle(prompt) {
    if (!prompt || prompt.trim().length === 0) return "新对话";
    const titlePrompt = prompts.getTitleSummaryPrompt(prompt);
    const response = await proxyGenerateContent({
        model: backgroundModel,
        contents: [{ text: titlePrompt }],
        config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
    });
    const title = response.text?.trim();
    return title || prompt.substring(0, 20);
}

export async function analyzeWithAgent(agent, base64Data, mimeType) {
    try {
        let prompt = `You are a professional ${agent.role}. Analyze the provided image from your specific viewpoint. Return a JSON object with "critique", "suggestion", "x", and "y" fields. "x" and "y" are percentage coordinates (0-100) of the most relevant area. "critique" must be in Chinese. "suggestion" must be a concise, actionable English prompt for an image model.`;
        
        switch(agent.id) {
            case 'describe':
                prompt = `As an ${agent.role}, your task is to provide a purely objective description of the image.
                - Analyze: Identify the main subject, setting, composition (e.g., rule of thirds, leading lines), color palette, and lighting conditions.
                - JSON Output:
                  - "critique": (Chinese) A factual summary of the image's key visual elements. Example: "画面主体为一只猫，位于中央，采用暖色调，右侧有窗户光源。"
                  - "suggestion": (English) A descriptive prompt that could generate this image. Example: "photo of a cat sitting on a wooden floor, warm sunlight from a window on the right".
                  - "x": 50
                  - "y": 50`;
                break;
            case 'critique':
                prompt = `As an ${agent.role}, your task is to provide a professional artistic critique.
                - Analyze: Evaluate the image's strengths and weaknesses in terms of emotional impact, storytelling, technical execution (focus, exposure), and aesthetics. Be specific.
                - JSON Output:
                  - "critique": (Chinese) A concise critique identifying one key area for improvement. Example: "画面整体偏暗，主体与背景融合度过高，缺乏层次感。"
                  - "suggestion": (English) A prompt to address the critique. Example: "add dramatic backlighting to separate the subject from the background, increase contrast".
                  - "x": The x-coordinate of the main issue.
                  - "y": The y-coordinate of the main issue.`;
                break;
            case 'suggest':
                 prompt = `As a ${agent.role}, your task is to propose a concrete, creative change.
                - Analyze: Brainstorm a specific, actionable modification that would creatively alter or enhance the image. This could be a style change, adding an element, or changing the mood.
                - JSON Output:
                  - "critique": (Chinese) A brief statement of your creative idea. Example: "如果将场景变为雨夜，也许会更有故事氛围。"
                  - "suggestion": (English) A precise prompt to execute this idea. Example: "change the scene to a rainy night, add reflections on wet ground, neon lights in the background".
                  - "x": The x-coordinate of where the change would be most impactful.
                  - "y": The y-coordinate of where the change would be most impactful.`;
                break;
        }

        const fullPrompt = `${prompt}\n\nReturn ONLY raw JSON, no markdown formatting or explanations.`;
        const cleanBase64 = base64Data.includes(',') ? base64Data.split(',')[1] : base64Data;

        const response = await proxyGenerateContent({
            model: backgroundModel,
            contents: {
                parts: [
                    { inlineData: { data: cleanBase64, mimeType: mimeType || 'image/png' } },
                    { text: fullPrompt }
                ]
            },
            config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });

        const text = response.text.trim().replace(/```json|```/g, '');
        const data = JSON.parse(text);
        
        return {
            agent: agent,
            ...data
        };

    } catch (e) {
        console.error("Agent analysis failed:", e);
        return null;
    }
}

export async function getSmartSuggestions(base64ImageData) {
    try {
        const prompt = `你是一位创意助手。根据这张图片，为用户提供三个简洁、可操作、且富有创意的修改或增强建议。这些建议应该是用户会直接输入的指令短语。例如：“把天空变成暴风雨的样子”、“在水上加一只小船”、“将季节改为秋天”。请仅返回一个JSON字符串数组，格式为 \`["建议一", "建议二", "建议三"]\`。不要在你的回复中包含 "JSON" 字样或代码块标记。`;
        const cleanBase64 = base64ImageData.includes(',') ? base64ImageData.split(',')[1] : base64ImageData;
        
        const response = await proxyGenerateContent({
            model: backgroundModel,
            contents: {
                parts: [
                    { inlineData: { data: cleanBase64, mimeType: 'image/png' } },
                    { text: prompt }
                ]
            },
            config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });
        
        const jsonString = response.text.trim();
        const suggestions = JSON.parse(jsonString);

        if (Array.isArray(suggestions) && suggestions.every(s => typeof s === 'string')) {
            return suggestions.map(s => ({ label: s, prompt: s }));
        }
        return null;
    } catch (error) {
        console.error("获取智能建议时出错:", error);
        return null;
    }
}

export async function classifyImageCategory(base64Image, mimeType) {
    try {
        const cleanBase64 = base64Image.includes(',') ? base64Image.split(',')[1] : base64Image;
        const classifierResponse = await proxyGenerateContent({
            model: backgroundModel,
            contents: {
                parts: [
                    { inlineData: { data: cleanBase64, mimeType: mimeType || 'image/png' } },
                    { text: "Analyze this image and return ONLY one of these three words: 'CHARACTER' (for person/creature/statue), 'PRODUCT' (for small objects/electronics), or 'SPACE' (for building/interior/landscape). Do not return anything else." }
                ]
            },
            config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });
        return classifierResponse.text.trim().toUpperCase();
    } catch (e) {
        console.error("Image classification failed:", e);
        return 'PRODUCT'; // Default fallback
    }
}

export async function generateRelitImage(prompt, base64Image) {
    try {
        if (!base64Image) throw new Error("Base image is required for relighting.");
        
        let mimeType = 'image/png';
        let dataUrl = base64Image;

        if (base64Image.includes(',')) {
            const parts = base64Image.split(',');
            const match = parts[0].match(/:(.*?);/);
            if (match) {
                mimeType = match[1];
            }
        } else {
            dataUrl = `data:${mimeType};base64,${base64Image}`;
        }

        // Compress image to ensure it's within Gemini's limits (max 2048)
        const file = await dataURLToFile(dataUrl, "temp_image");
        const compressedFile = await compressImage(file, 2048, 0.9);
        const compressedBase64 = await fileToBase64(compressedFile);
        
        mimeType = compressedFile.type || mimeType;

        const finalPrompt = prompts.getRelightPrompt(prompt || "Relight this image with dramatic lighting");

        if (getImageModel() === 'gpt-image-2') {
            const res = await editOrQueryImageWithGemini(finalPrompt, compressedFile, [], null, '1:1');
            return { imageData: res.imageData, mimeType: 'image/png' };
        }

        const response = await proxyGenerateContent({
            model: getImageModel(),
            contents: { parts: [
                { inlineData: { data: compressedBase64, mimeType: mimeType } },
                { text: finalPrompt }
            ]}
        });
        
        if (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) {
            for(const part of response.candidates[0].content.parts) {
                if(part.inlineData) {
                    return { imageData: part.inlineData.data, mimeType: part.inlineData.mimeType || 'image/jpeg' };
                }
            }
        }
        
        // If no image part found, log the response for debugging
        console.warn("Relighting response did not contain an image part:", response);
    } catch (e) {
        console.error("Relighting failed:", e);
        // Re-throw the error so it can be caught by the node executor with more context
        throw e;
    }
    return null;
}

export async function generatePreciseEditImage(contentsParts, aspectRatio = '1:1') {
    try {
        if (getImageModel() === 'gpt-image-2') {
            let baseImageBase64 = null;
            let maskBase64 = null;
            let mimeType = 'image/png';
            let prompt = "";
            let inlineDataIndex = 0;
            for (const part of contentsParts) {
                if (part.inlineData) {
                    if (inlineDataIndex === 0) {
                        baseImageBase64 = part.inlineData.data;
                        mimeType = part.inlineData.mimeType || 'image/png';
                    } else if (inlineDataIndex === 1) {
                        maskBase64 = part.inlineData.data;
                    }
                    inlineDataIndex++;
                }
                if (part.text) prompt += part.text + " ";
            }
            
            // Clean up Gemini SYSTEM INSTRUCTION for generic APIs
            let cleanPrompt = prompt.trim();
            const instructionMatch = cleanPrompt.match(/instruction:\s*"([^"]+)"/);
            if (instructionMatch) {
                cleanPrompt = instructionMatch[1];
            } else {
                // Fallback attempt to strip boilerplate if regex failed
                const finalCommandMatch = cleanPrompt.match(/FINAL COMMAND:(.*?)WARNING:/s);
                if (finalCommandMatch) {
                    cleanPrompt = finalCommandMatch[1].trim();
                }
            }

            const baseFile = await dataURLToFile(`data:${mimeType};base64,${baseImageBase64}`, 'base.png');
            let maskFile = null;
            if (maskBase64) maskFile = await dataURLToFile(`data:image/png;base64,${maskBase64}`, 'mask.png');
            
            const res = await editOrQueryImageWithGemini(cleanPrompt, baseFile, [], maskFile, aspectRatio);
            return res.imageData;
        }

        const currentModel = getImageModel();
        let editResponse;

        if (currentModel.startsWith('gemini')) {
            const payload = {
                contents: [{ role: "user", parts: contentsParts }],
                config: { imageConfig: { aspectRatio: aspectRatio } }
            };

            const res = await fetch(RECOGNIZE_BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: 'gemini_lite_generate',
                    model: currentModel,
                    payload: payload
                })
            });
            editResponse = await res.json();
        } else {
            editResponse = await proxyGenerateContent({
                model: currentModel,
                contents: { parts: contentsParts },
                config: {
                    imageConfig: {
                        aspectRatio: aspectRatio
                    }
                }
            });
        }

        if (editResponse.candidates?.[0]?.content?.parts) {
            for (const part of editResponse.candidates[0].content.parts) {
                if (part.inlineData) { return part.inlineData.data; }
            }
        }
    } catch (e) {
        console.error("Precise edit failed:", e);
    }
    return null;
}

export async function generateVisualSearch(croppedBase64) {
    try {
        const cleanBase64 = croppedBase64.includes(',') ? croppedBase64.split(',')[1] : croppedBase64;
        const searchPrompt = `
            Analyze this image content specifically.
            1. Identify the furniture/object name, style, material, and likely brand (if famous).
            2. Generate a precise search query (3-5 words) to find this product on shopping sites.
            3. Return a JSON object with:
               - "productName": Detailed name (e.g. "Eames Lounge Chair Replica Walnut")
               - "searchQuery": Optimized search keyword string (e.g. "Mid-century modern lounge chair walnut")
               - "description": Brief visual description (color, material)
               - "category": General category (e.g. "Chair", "Lamp")
            
            Also use Google Search to find real pricing or availability if possible.
            IMPORTANT: Output ONLY valid JSON.
        `;

        const response = await proxyGenerateContent({
            model: backgroundModel, 
            contents: {
                parts: [
                    { inlineData: { data: cleanBase64, mimeType: 'image/png' } },
                    { text: searchPrompt }
                ]
            },
            config: { 
                tools: [{ googleSearch: {} }],
                thinkingConfig: { thinkingLevel: getThinkingLevel() }
            }
        });
        
        let text = response.text || "{}";
        text = text.replace(/```json\s*/g, '').replace(/```\s*$/g, '').trim();
        let resultData = null;
        try {
            resultData = JSON.parse(text);
        } catch(e) {
            console.warn("JSON parse failed, falling back", e);
            resultData = {
                productName: "识别到的物体",
                searchQuery: "furniture",
                description: "未获取到详细信息"
            };
        }

        const webLinks = response.candidates?.[0]?.groundingMetadata?.groundingChunks
            ?.map(c => c.web)
            .filter(w => w && w.uri && w.title) || [];

        return { resultData, webLinks };
    } catch (e) {
        console.error("Visual search failed:", e);
        throw e;
    }
}
export async function planGraph(userRequest, availableNodes, history = []) {
    try {
        const historyContext = history.length > 0 
            ? `\nRecent Conversation History:\n${history.slice(-5).map(m => {
                if (m.type === 'image') return `${m.sender}: [Uploaded an Image]`;
                return `${m.sender}: ${m.content}`;
            }).join('\n')}`
            : "";

        const prompt = `
            You are a Workflow Architect. Your task is to translate a user's request into a multi-step AI Node Graph.
            
            Available Node Types:
            ${JSON.stringify(availableNodes, null, 2)}
            
            ${historyContext}
            
            Current User Request: "${userRequest}"
            
            Rules:
            1. Analyze the "Current User Request" in the context of the "Recent Conversation History".
            2. If the user is asking to modify an existing image (e.g., "make it brighter", "change the material", "replace this"), DO NOT use a "generate-image" node. Start the graph directly with the modification node (e.g., "material-replace", "relight"). The provided image will be automatically fed into the first node's "image" input.
            3. Break complex tasks into logical steps.
            4. Use edges to pass data (like images) between nodes.
            5. CRITICAL: If the user asks for multiple sequential edits on the same image (e.g., "do A, then do B, then do C"), you MUST chain the nodes sequentially. Node 1's output goes to Node 2's input, Node 2's output goes to Node 3's input, etc. Do NOT run them in parallel.
            6. If a node needs an image from a previous step, you MUST create an edge from the previous node's "image" output to the current node's "image" input.
            7. If the user wants to "extract" or "get" a specific part, use "layer-analysis" to find it, then "crop" to extract it.
            8. NODE SELECTION GUIDELINES:
               - Use "relight" for lighting, brightness, or shadow adjustments.
               - Use "material-replace" for changing materials, textures, or surface properties (e.g., "change wood to metal").
               - Use "inpainting" for local edits, adding/removing objects in a specific area, or redrawing a part.
               - Use "outpaint" for expanding the image boundaries. If the user doesn't specify a target ratio, explicitly set the "aspectRatio" input to a ratio different from the original (e.g., "16:9" or "9:16") to ensure expansion occurs.
               - Use "edit-image" ONLY for general style changes or edits that don't fit the above categories.
            9. Return a JSON object with "nodes" (id, type, inputs) and "edges" (from, to).
            10. The "from" and "to" in edges MUST be objects: {"from": {"nodeId": "id1", "outputId": "image"}, "to": {"nodeId": "id2", "inputId": "image"}}.
            11. Return ONLY valid JSON.
        `;

        const response = await proxyGenerateContent({
            model: getTextModel(),
            contents: [{ text: prompt }],
            config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });

        let text = response.text.trim();
        const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
            text = jsonMatch[1];
        } else {
            text = text.replace(/```json|```/g, '').trim();
            // Try to find the first { and last }
            const start = text.indexOf('{');
            const end = text.lastIndexOf('}');
            if (start !== -1 && end !== -1) {
                text = text.substring(start, end + 1);
            }
        }
        return JSON.parse(text);
    } catch (e) {
        console.error("Graph planning failed:", e);
        return null;
    }
}

import { buildSceneDocument } from '../core/scene-manager.js';

export async function analyzeImageLayers(file) {
    if (!file) throw new Error("A base image is required for layer analysis.");
    const semanticModel = getSemanticModel();
    console.log(`[SemanticLayers] model=${semanticModel}, thinking=ui`);
    const base64Data = await fileToBase64(file);
    const imageDataUrl = `data:${file.type || 'image/png'};base64,${base64Data}`;
    
    // Attempt to get image dimensions
    let imageSize = { width: null, height: null };
    try {
        const img = new Image();
        img.src = imageDataUrl;
        await new Promise((resolve, reject) => {
            img.onload = () => {
                imageSize = { width: img.width, height: img.height };
                resolve();
            };
            img.onerror = reject;
        });
    } catch (e) {
        console.warn("Could not get image dimensions", e);
    }

    const prompt = `Act as a Layered Scene Reconstruction System. Your task is to decompose this flat image into a logical editable scene graph, similar to a PSD file plus object hierarchy.
Do not just detect objects. Propose a layer structure that allows for non-destructive editing and recomposition.

Identify 4 to 14 top-level layers for normal scenes. For flat ad / poster layout mode, identify all meaningful design units, usually 22 to 50 layers, because design files require separate background shapes, panels, badges, product images, and text blocks.
Categorize them as either "foreground_asset" (objects worth extracting as independent assets) or "background_plate" (the environment, walls, floors, large surfaces).

Rules for Layer Proposal:
- First decide whether the image is a NORMAL SCENE or a FLAT AD / POSTER / MENU DESIGN.
- If it is a flat design, ignore normal interior-scene grouping instincts. Do NOT describe broad visual regions; describe editable design layers.
- Suitable for "foreground_asset": Sofas, chairs, tables, lamps, vases, people, main products.
- Suitable for "background_plate": Floor textures, large wall surfaces, ceilings.
- FLAT AD / POSTER LAYOUT MODE: If the image is a graphic advertisement, poster, menu, flyer, banner, social media ad, packaging layout, menu board, or promotional graphic, decompose it like a PSD/Figma/Canva design file instead of a real-world object scene.
- The flat ad unit of analysis is DESIGN UNITS, not physical object parts. Output the layers a designer would move, recolor, rewrite, hide, or replace independently.
- For flat ads, enumerate these units without merging them into broad sections:
  1. "ad_background" / designRole "base_background": the full-canvas poster background. Keep texture/pattern as part of this background.
  2. "shape_panel" / designRole "local_panel": each independent carrier panel/card/label strip/copy block behind text or products. Include occluded panels as their full inferred rectangle when clear.
  3. "price_badge" / designRole "price_badge": each badge/circle/pill shape behind a price.
  4. "product_food" / "product_drink" / designRole "product_image": each photo-realistic product image as a whole unit. Do not split food from plate, cup from saucer, or garnish from dish.
  5. "element_text" / text designRole: each visible text block, including logo, headline words, product names, prices, drink names, descriptions, and website/URL. Use tight bboxes for text.
  6. "decor_graphic" / designRole "decor_shape": standalone decorative dots, lines, frames, and graphic accents.
- Self-check for flat food/menu posters before returning JSON: if visible, include the full background, each large menu panel, each small label panel, each price badge shape, every product photo, and every text group. A simple menu with 2 dishes and 3 drinks commonly has 20+ layers. If your output has only 10-15 layers, you probably merged or omitted editable design units.
- Do not output composite menu sections as editable atomic layers. For example, split a food section into panel, product image, price badge, product title text, and description text.
- Repeated product cards, drink labels, price/name strips, and small rounded rectangles are separate shape_panel layers. Do NOT output one broad "drink area", "bottom drink row", "menu strip", or "product row" instead of the individual card/panel backgrounds. If three drinks each sit on a cream label panel, output three separate shape_panel layers.
- If text is too small to read exactly, still output a tight element_text layer for that paragraph area; use an empty textContent or best short approximation rather than omitting it.
- Product layers should be tight around the visible product unit only. Panel/badge/text layers may overlap product bboxes because they are separate stacked design layers.
- Price badge shape and price text are always separate layers. Label panel shape and label text are always separate layers.
- A local panel/card/badge is NOT a replacement for its text. Every visible word, number, product label, small paragraph, URL, and caption printed on top of a panel/card/badge must also be output as an "element_text" layer with a tight glyph bbox.
- When a product photo overlaps a label panel, still output the label text separately if visible. Do not omit label text just because the cream/white panel has been output.
- For flat ads, prefer a flat list of atomic layers. Do not use parent/children groups unless there is a real grouped symbol; groups are less useful than editable atomic layers here.
- EDITABLE ATOMICITY: If a bbox contains multiple nameable objects that a designer may edit separately, output them as separate child layers. Examples: a table and objects sitting on it, a wall art and vase/flowers in front of it, a shelf and decorative items, a sofa and pillows, a lamp and shade/base.
- Parent layers are only organizational/composite groups. Children are the actual editable objects for segmentation when present.
- Do not merge a support surface with objects sitting on it. Tables, bowls, vases, flowers, books, candles, sculptures, stools, wall art, lamps and rugs should be separate editable objects when visually separable.
- DO NOT create layers for: Tiny shadows, complex reflections, small highlights on glass, or inseparable textures.
- Ensure the layers, if stacked back together, would reconstruct the original scene.
- TEXT RULE FOR NORMAL PHOTOS / INTERIOR SCENES: Group all typography into exactly ONE "element_text" layer named "全局文字" unless the image is a flat ad/poster.
- TEXT RULE FOR FLAT ADS: Do NOT use one global text layer. Output separate "element_text" layers for visible typography groups. Keep text layers tight; do not use one large bbox covering unrelated text.
- In flat ads, separate text blocks when typography, meaning, or edit intent differs. If two words/numbers have clearly different font size, weight, color, baseline, line position, or container relationship, they must be separate text layers. Preserve each text layer's approximate original typography in fontStyle.
- Avoid duplicate text layers. If you output "NEW" and "DISHES & DRINKS" separately, do not also output a larger combined headline layer containing both.

Return ONLY a valid JSON array.
Each item must contain:
- "id": short unique English snake_case id
- "name": short Chinese display name
- "layerType": "foreground_asset" or "background_plate"
- "reasoning": short Chinese explanation of why this is proposed as a layer and how it relates to the background.
- "semanticType": A specific tag from this list: ["furniture_sofa", "furniture_table", "furniture_chair", "furniture_stool", "furniture_bed", "furniture_cabinet", "lighting_pendant", "lighting_floor", "lighting_table", "surface_floor", "surface_wall", "surface_ceiling", "soft_rug", "soft_curtain", "soft_pillow", "decor_plant", "decor_flower", "decor_vase", "decor_bowl", "decor_book", "decor_sculpture", "decor_art", "product_food", "product_drink", "product_packaging", "ad_background", "shape_panel", "price_badge", "cta_button", "logo_mark", "decor_graphic", "element_text", "flat_ad_layout", "composite_group", "other"]
- "designRole": One of ["base_background", "local_panel", "product_image", "price_badge", "headline_text", "body_text", "label_text", "price_text", "logo_text", "url_text", "decor_shape", "scene_object", "unknown"]. Use the flat ad roles when in flat ad mode.
- "renderMode": One of ["background_plate", "vector_shape", "raster_cutout", "text_css", "semantic_group", "deferred"]. Use "vector_shape" for panels/badges/decor shapes, "text_css" for text, "raster_cutout" only for photo-real products or real objects.
- "compositeRole": "atomic_object", "composite_group", or "background"
- "children": optional array of child layers using the same fields. Use children for tabletop arrangements, shelf decor, wall decor clusters, sofa pillows, plants with pots, and any bbox containing multiple editable objects.
- "textContent": IMPORTANT: If this layer contains text/typography, extract the EXACT raw text string here.
- "fontStyle": (If semanticType is "element_text") A brief guess at the font family and overall color (e.g., "sans-serif, #000000").
- "bbox": [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000
- "zIndex": integer, larger means visually in front (background should be 0 or 1)
- "editable": true or false
- "promptHint": short Chinese description for future AI editing

Output raw JSON only.`;

    const response = await proxyGenerateContent({
        model: semanticModel, 
        contents: {
            parts: [
                { inlineData: { data: base64Data, mimeType: (file && file.type) || 'image/png' } },
                { text: prompt }
            ]
        },
        config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
    });

    const text = response.text.trim().replace(/```json|```/g, '');
    let cleaned = [];
    
    // 对 semanticType 和 name 匹配文本类型
    const sanitizeSemanticType = (type, name = '', textContent = '') => {
        const validTypes = [
            "furniture_sofa", "furniture_table", "furniture_chair", "furniture_stool", "furniture_bed", "furniture_cabinet",
            "lighting_pendant", "lighting_floor", "lighting_table", "surface_floor", "surface_wall", "surface_ceiling",
            "soft_rug", "soft_curtain", "soft_pillow", "decor_plant", "decor_flower", "decor_vase", "decor_bowl",
            "decor_book", "decor_sculpture", "decor_art", "product_food", "product_drink", "product_packaging",
            "ad_background", "shape_panel", "price_badge", "cta_button", "logo_mark", "decor_graphic", "element_text",
            "flat_ad_layout", "composite_group", "other"
        ];
        const nameLower = String(name || '').toLowerCase();
        const typeLower = String(type || '').toLowerCase();
        const textLower = String(textContent || '').toLowerCase();
        const combined = `${typeLower} ${nameLower} ${textLower}`;
        const looksLikeTextPanel =
            combined.includes("文字背景") ||
            combined.includes("文本背景") ||
            combined.includes("文字底板") ||
            combined.includes("文本底板") ||
            combined.includes("copy block") ||
            combined.includes("text panel") ||
            combined.includes("text background");
        const explicitTextLayer =
            !looksLikeTextPanel && (
                textLower.trim().length > 0 ||
                typeLower === "element_text" ||
                combined.includes("price_text") ||
                combined.includes("headline_text") ||
                combined.includes("body_text") ||
                combined.includes("label_text") ||
                combined.includes("logo_text") ||
                combined.includes("url_text") ||
                combined.includes("文字") ||
                combined.includes("文本") ||
                combined.includes("标题") ||
                combined.includes("正文") ||
                combined.includes("说明文字") ||
                combined.includes("text layer") ||
                combined.includes("typography")
            );

        if (explicitTextLayer) return "element_text";

        if (
            combined.includes("总背景") || combined.includes("大背景") || combined.includes("整体背景") ||
            combined.includes("广告背景") || combined.includes("海报背景") || combined.includes("橙色背景") ||
            combined.includes("ad_background") || combined.includes("ad background") ||
            combined.includes("poster background") || combined.includes("base background") ||
            combined.includes("full background") || combined.includes("main background")
        ) return "ad_background";
        if (
            combined.includes("价格") || combined.includes("价签") || combined.includes("price") ||
            combined.includes("badge") || combined.includes("price_badge") ||
            /\$\s*\d+/.test(combined)
        ) return "price_badge";
        if (
            combined.includes("面板") || combined.includes("底板") || combined.includes("色块") ||
            combined.includes("背景框") || combined.includes("文字背景") || combined.includes("文本背景") ||
            combined.includes("内容背景") || combined.includes("局部背景") || combined.includes("区域背景") ||
            combined.includes("标签背景") || combined.includes("标签底板") || combined.includes("底色") ||
            combined.includes("米色背景") || combined.includes("白色背景") || combined.includes("奶油色背景") ||
            combined.includes("beige background") || combined.includes("cream background") ||
            combined.includes("white background") || combined.includes("local background") ||
            combined.includes("copy block") || combined.includes("label_panel") ||
            combined.includes("shape_panel") || combined.includes("panel") ||
            combined.includes("card") || combined.includes("shape") || combined.includes("label")
        ) return "shape_panel";
        if (combined.includes("logo") || combined.includes("标志")) return "logo_mark";
        if (combined.includes("字") || combined.includes("text") || combined.includes("文本") || combined.includes("排版")) return "element_text";

        if (validTypes.includes(type)) return type;

        if (
            nameLower.includes("食物") || nameLower.includes("食品") || nameLower.includes("菜品") ||
            nameLower.includes("炒饭") || nameLower.includes("米饭") || nameLower.includes("猪肉") ||
            nameLower.includes("烤肉") || nameLower.includes("餐盘") || nameLower.includes("food") ||
            nameLower.includes("dish") || nameLower.includes("meal") || nameLower.includes("rice") ||
            nameLower.includes("pork") || nameLower.includes("plate")
        ) return "product_food";
        if (
            nameLower.includes("饮料") || nameLower.includes("可乐") || nameLower.includes("茶") ||
            nameLower.includes("咖啡") || nameLower.includes("热巧") || nameLower.includes("杯") ||
            nameLower.includes("drink") || nameLower.includes("beverage") || nameLower.includes("cola") ||
            nameLower.includes("tea") || nameLower.includes("coffee") || nameLower.includes("choco") ||
            nameLower.includes("cup")
        ) return "product_drink";
        if (nameLower.includes("灯") || nameLower.includes("吊灯") || nameLower.includes("灯具")) return "lighting_pendant";
        if (nameLower.includes("画") || nameLower.includes("装饰画")) return "decor_art";
        if (nameLower.includes("花瓶") || nameLower.includes("vase")) return "decor_vase";
        if (nameLower.includes("花") || nameLower.includes("flower")) return "decor_flower";
        if (nameLower.includes("碗") || nameLower.includes("bowl")) return "decor_bowl";
        if (nameLower.includes("书") || nameLower.includes("book")) return "decor_book";
        if (nameLower.includes("雕塑") || nameLower.includes("摆件") || nameLower.includes("sculpture")) return "decor_sculpture";
        if (nameLower.includes("沙发")) return "furniture_sofa";
        if (nameLower.includes("桌")) return "furniture_table";
        if (nameLower.includes("凳")) return "furniture_stool";
        if (nameLower.includes("椅")) return "furniture_chair";
        if (nameLower.includes("柜")) return "furniture_cabinet";
        if (nameLower.includes("床")) return "furniture_bed";
        if (nameLower.includes("抱枕") || nameLower.includes("枕")) return "soft_pillow";
        if (nameLower.includes("地毯")) return "soft_rug";
        if (nameLower.includes("窗帘")) return "soft_curtain";
        if (nameLower.includes("植物")) return "decor_plant";
        if (nameLower.includes("墙")) return "surface_wall";
        if (nameLower.includes("地")) return "surface_floor";
        
        return "other";
    };

    const sanitizeDesignRole = (role, semanticType, name = '', textContent = '') => {
        const validRoles = [
            "base_background", "local_panel", "product_image", "price_badge",
            "headline_text", "body_text", "label_text", "price_text", "logo_text", "url_text",
            "decor_shape", "scene_object", "unknown"
        ];
        const roleLower = String(role || '').toLowerCase();
        if (validRoles.includes(roleLower)) return roleLower;

        const semanticLower = String(semanticType || '').toLowerCase();
        const combined = `${semanticLower} ${String(name || '').toLowerCase()} ${String(textContent || '').toLowerCase()}`;
        if (semanticLower === 'ad_background') return "base_background";
        if (semanticLower === 'shape_panel' || semanticLower === 'cta_button') return "local_panel";
        if (semanticLower === 'price_badge') return "price_badge";
        if (semanticLower === 'product_food' || semanticLower === 'product_drink' || semanticLower === 'product_packaging') return "product_image";
        if (semanticLower === 'logo_mark' || combined.includes("logo") || combined.includes("标志")) return "logo_text";
        if (semanticLower === 'decor_graphic') return "decor_shape";
        if (combined.includes("面板") || combined.includes("底板") || combined.includes("背景") || combined.includes("panel") || combined.includes("card")) return "local_panel";
        if (combined.includes("产品图") || combined.includes("商品图") || combined.includes("菜品") || combined.includes("饮料") || combined.includes("product image")) return "product_image";
        if (semanticLower === 'element_text') {
            if (combined.includes("价格") || combined.includes("价") || combined.includes("price") || /\$\s*\d+/.test(combined)) return "price_text";
            if (combined.includes("网站") || combined.includes("url") || combined.includes("www.") || combined.includes(".com")) return "url_text";
            if (combined.includes("标题") || combined.includes("headline") || combined.includes("new") || combined.includes("主标题")) return "headline_text";
            if (combined.includes("说明") || combined.includes("description") || combined.includes("body") || combined.includes("正文")) return "body_text";
            return "label_text";
        }
        return "scene_object";
    };

    const sanitizeRenderMode = (mode, semanticType, compositeRole, layerType) => {
        const validModes = ["background_plate", "vector_shape", "raster_cutout", "text_css", "semantic_group", "deferred"];
        const modeLower = String(mode || '').toLowerCase();
        if (validModes.includes(modeLower)) return modeLower;

        if (compositeRole === "composite_group") return "semantic_group";
        if (layerType === "background_plate" || semanticType === "ad_background") return "background_plate";
        if (semanticType === "element_text") return "text_css";
        if (["shape_panel", "price_badge", "cta_button", "logo_mark", "decor_graphic", "flat_ad_layout"].includes(semanticType)) {
            return "vector_shape";
        }
        return "raster_cutout";
    };

    const shouldDowngradeFlatGroupToPanel = (layer, semanticType, children) => {
        if (Array.isArray(children) && children.length > 0) return false;
        const text = [
            layer?.name,
            layer?.semanticType,
            layer?.designRole,
            layer?.compositeRole
        ].map(value => String(value || '').toLowerCase()).join(' ');
        const isGroupLike = semanticType === "composite_group" ||
            semanticType === "flat_ad_layout" ||
            text.includes("区域") ||
            text.includes("行") ||
            text.includes("row") ||
            text.includes("section") ||
            text.includes("area");
        const isFlatPanelLike = text.includes("饮料") ||
            text.includes("菜单") ||
            text.includes("标签") ||
            text.includes("面板") ||
            text.includes("底板") ||
            text.includes("panel") ||
            text.includes("label");
        return isGroupLike && isFlatPanelLike;
    };

    const getBboxArea = (bbox) => {
        if (!Array.isArray(bbox) || bbox.length !== 4) return 0;
        const [ymin, xmin, ymax, xmax] = bbox;
        return Math.max(0, ymax - ymin) * Math.max(0, xmax - xmin);
    };

    const getBboxIntersectionArea = (a, b) => {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
        const [ay1, ax1, ay2, ax2] = a;
        const [by1, bx1, by2, bx2] = b;
        const ymin = Math.max(ay1, by1);
        const xmin = Math.max(ax1, bx1);
        const ymax = Math.min(ay2, by2);
        const xmax = Math.min(ax2, bx2);
        return Math.max(0, ymax - ymin) * Math.max(0, xmax - xmin);
    };

    const clampBbox = (bbox) => bbox.map(value => Math.round(Math.max(0, Math.min(1000, value))));

    const hasPanelCovering = (layers, bbox) => {
        const bboxAreaValue = Math.max(1, getBboxArea(bbox));
        return layers.some(layer => {
            if (layer.semanticType !== "shape_panel" && layer.designRole !== "local_panel") return false;
            const layerArea = getBboxArea(layer.bbox);
            if (layerArea > bboxAreaValue * 2.2) return false;
            const overlap = getBboxIntersectionArea(layer.bbox, bbox) / Math.max(1, Math.min(layerArea, bboxAreaValue));
            return overlap >= 0.45;
        });
    };

    const supplementFlatAdLabelPanels = (layers) => {
        if (!Array.isArray(layers) || layers.length === 0) return layers;
        const flatSignals = layers.filter(layer => [
            "ad_background", "shape_panel", "price_badge", "product_food", "product_drink"
        ].includes(layer.semanticType)).length;
        if (flatSignals < 4) return layers;

        const textLayers = layers.filter(layer => layer.semanticType === "element_text" && Array.isArray(layer.bbox));
        const productLayers = layers.filter(layer =>
            ["product_drink", "product_packaging"].includes(layer.semanticType) &&
            Array.isArray(layer.bbox)
        );
        const additions = [];

        textLayers.forEach((textLayer, index) => {
            const textContent = String(textLayer.textContent || textLayer.name || "").trim();
            const textRole = String(textLayer.designRole || "").toLowerCase();
            if (["price_text", "headline_text", "body_text", "logo_text", "url_text"].includes(textRole)) return;
            if (/^\$?\s*\d+$/i.test(textContent)) return;
            const [ty1, tx1, ty2, tx2] = textLayer.bbox;
            const textWidth = tx2 - tx1;
            const textHeight = ty2 - ty1;
            if (textWidth <= 18 || textHeight <= 12 || textHeight > 95) return;
            const textCenterY = (ty1 + ty2) / 2;

            const nearbyProduct = productLayers.find(productLayer => {
                const [py1, px1, py2, px2] = productLayer.bbox;
                const productCenterY = (py1 + py2) / 2;
                const verticalNear = Math.abs(productCenterY - textCenterY) <= Math.max(90, textHeight * 3.5);
                const horizontalNear = px2 >= tx1 - 25 && px1 <= tx2 + 190;
                const productLikelyRight = px1 >= tx1 - 10 || px2 >= tx2;
                return verticalNear && horizontalNear && productLikelyRight;
            });
            if (!nearbyProduct) return;

            const [py1, px1, py2, px2] = nearbyProduct.bbox;
            const panelBbox = clampBbox([
                Math.min(ty1, py1 + (py2 - py1) * 0.52) - 10,
                Math.min(tx1, px1) - 18,
                Math.max(ty2, py2) + 8,
                Math.max(tx2, px2) + 14
            ]);
            const panelWidth = panelBbox[3] - panelBbox[1];
            const panelHeight = panelBbox[2] - panelBbox[0];
            if (panelWidth < 80 || panelHeight < 24 || panelWidth > 340 || panelHeight > 130) return;
            if (hasPanelCovering(layers.concat(additions), panelBbox)) return;

            additions.push({
                id: `supplement_label_panel_${index + 1}`,
                name: `${textLayer.name || textLayer.textContent || "标签"}底板`,
                category: "object",
                layerType: "foreground_asset",
                reasoning: "根据平面广告中标签文字与产品图的空间关系补全的独立承载底板。",
                semanticType: "shape_panel",
                designRole: "local_panel",
                renderMode: "vector_shape",
                compositeRole: "atomic_object",
                parentLayerId: null,
                children: [],
                textContent: "",
                fontStyle: "",
                bbox: panelBbox,
                zIndex: Math.max(1, Number(textLayer.zIndex || 1) - 1),
                editable: true,
                promptHint: "标签/产品信息底板",
                assetStatus: "idle"
            });
        });

        return additions.length > 0 ? layers.concat(additions) : layers;
    };

    try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
            // Ensure stable uniqueness of IDs and sort by zIndex
            let counts = {};
            const normalizeRawLayer = (layer, parent = null, childIndex = 0) => {
                let id = layer.id || (parent ? `${parent.id || 'group'}_child_${childIndex + 1}` : "layer");
                counts[id] = (counts[id] || 0) + 1;
                if (counts[id] > 1) id = `${id}_${counts[id]}`;
                
                const textContent = layer.textContent || "";
                const semanticType = sanitizeSemanticType(layer.semanticType, layer.name, textContent);
                const children = Array.isArray(layer.children)
                    ? layer.children.map((child, index) => normalizeRawLayer(child, { id }, index))
                    : [];
                const normalizedSemanticType = shouldDowngradeFlatGroupToPanel(layer, semanticType, children)
                    ? "shape_panel"
                    : semanticType;
                const designRole = sanitizeDesignRole(layer.designRole, normalizedSemanticType, layer.name, textContent);
                const isBaseBackground = normalizedSemanticType === "ad_background" || designRole === "base_background";
                const layerType = isBaseBackground ? "background_plate" : "foreground_asset";
                const compositeRole = layer.compositeRole || (
                    isBaseBackground
                        ? "background"
                        : (children.length > 0 ? "composite_group" : "atomic_object")
                );
                const normalizedCompositeRole = normalizedSemanticType === "shape_panel" && children.length === 0
                    ? "atomic_object"
                    : compositeRole;
                const renderMode = sanitizeRenderMode(layer.renderMode, normalizedSemanticType, normalizedCompositeRole, layerType);
                
                return {
                    id: id,
                    name: layer.name || "未命名图层",
                    category: layerType === "background_plate" ? "background" : "object",
                    layerType,
                    reasoning: layer.reasoning || "",
                    semanticType: normalizedSemanticType,
                    designRole,
                    renderMode,
                    compositeRole: normalizedCompositeRole,
                    parentLayerId: parent?.id || layer.parentLayerId || null,
                    children,
                    textContent,
                    fontStyle: layer.fontStyle || "",
                    bbox: layer.bbox || [250, 250, 750, 750],
                    zIndex: layer.zIndex || 1,
                    editable: layer.editable !== undefined ? layer.editable : compositeRole !== "composite_group",
                    promptHint: layer.promptHint || layer.name || "可编辑视觉元素",
                    assetStatus: "idle"
                };
            };
            cleaned = parsed.map((layer, index) => normalizeRawLayer(layer, null, index));
            cleaned = supplementFlatAdLabelPanels(cleaned);
            cleaned.sort((a, b) => a.zIndex - b.zIndex);
        }
    } catch (e) { console.error(e); }
    
    if (cleaned.length === 0) {
        // Fallback
        cleaned = [
            {
                id: "background_main",
                name: "背景",
                category: "background",
                layerType: "background_plate",
                reasoning: "默认背景层",
                semanticType: "surface_wall",
                bbox: [0, 0, 1000, 1000],
                zIndex: 0,
                editable: false,
                promptHint: "整体背景",
                assetStatus: "idle"
            },
            {
                id: "subject_main",
                name: "主体",
                category: "object",
                layerType: "foreground_asset",
                reasoning: "默认主体层",
                semanticType: "other",
                bbox: [250, 250, 750, 750],
                zIndex: 2,
                editable: true,
                promptHint: "主要视觉主体",
                assetStatus: "idle"
            }
        ];
    }

    const scene = buildSceneDocument(file, imageDataUrl, imageSize, cleaned);
    
    // Return the new structure, but also keep the array format for backward compatibility
    // if existing code expects an array. Actually, let's just return the array but attach the scene to it,
    // or return an object and update callers. Let's return an object.
    return { rawLayers: cleaned, scene: scene };
}

export async function identifyObjectInCrop(base64Image) {
    if (!base64Image) throw new Error("A cropped image is required.");
    let data = base64Image;
    if (base64Image.includes(',')) {
        data = base64Image.split(',')[1];
    }
    const sysPrompt = `Identify the main salient object in this image crop. Return ONLY a very short, concise name for it in Chinese (e.g., "红色包包", "人物面部", "咖啡杯"). Do not explain. Do not use punctuation. Max 6 characters if possible.`;
    const response = await proxyGenerateContent({
        model: backgroundModel, 
        contents: {
            parts: [
                { inlineData: { data: data, mimeType: 'image/png' } },
                { text: sysPrompt }
            ]
        }
    });
    return response.text.trim().replace(/[*`\n]/g, '');
}

export async function analyzeTargetBoundingBox(image, prompt) {
    if (!image) throw new Error("A base image is required for layer analysis.");
    const base64Data = await fileToBase64(image);
    const sysPrompt = `Analyze this image and find the bounding box of the main object mentioned in this request: "${prompt}".
    Return a JSON object with:
    - "name": Short name of the identified object.
    - "bbox": [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000.
    Example: {"name":"沙发","bbox":[200,500,800,900]}
    If the object is not found, return null.
    Do NOT use markdown.`;

    const response = await proxyGenerateContent({
        model: backgroundModel, 
        contents: {
            parts: [
                { inlineData: { data: base64Data, mimeType: 'image/png' } },
                { text: sysPrompt }
            ]
        },
        config: { thinkingConfig: { thinkingLevel: getThinkingLevel() } }
    });

    const text = response.text.trim().replace(/```json|```/g, '');
    try {
        const parsed = JSON.parse(text);
        if (parsed && parsed.bbox) return parsed;
    } catch (e) { console.error(e); }
    
    return null;
}

export async function generateImage(prompt, aspectRatio = '1:1') {
    prompt = prompt || "生成一张图片";
    
    const imageModel = getImageModel();
    if (imageModel === 'gpt-image-2') {
        console.log("🚀 [GPT-image-2] 正在自适应路由调用高精度中转图像生成...");
        try {
            // Mapping ratios to size format supported by OpenAI-like endpoint
            let size = "1024x1024";
            if (aspectRatio === '16:9') size = "1536x1024";
            else if (aspectRatio === '9:16') size = "1024x1536";
            else if (aspectRatio === '4:3') size = "1024x768";
            else if (aspectRatio === '3:4') size = "768x1024";
            else if (aspectRatio.includes('x')) size = aspectRatio;

            const payload = {
                mode: "image_generation",
               
                prompt: prompt,
                size: size,
                quality: "auto",
                output_format: "jpeg"
            };

            const response = await fetch(`${RECOGNIZE_BACKEND_URL}`, {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`GPT-image-2 接口返回 ${response.status}`);
            }

            const data = await response.json();
            console.log("📥 [GPT-image-2] 接收原始 API 响应成果:", data);

            let imageData = null;
            if (data.data && data.data[0]) {
                const item = data.data[0];
                if (item.b64_json) {
                    imageData = item.b64_json;
                } else if (item.url) {
                    imageData = item.url;
                }
            } else if (data.choices && data.choices[0] && data.choices[0].message) {
                const content = data.choices[0].message.content;
                if (content && (content.startsWith('http') || content.trim().length > 100)) {
                    imageData = content.includes(',') ? content.split(',')[1] : content;
                }
            }

            if (!imageData) {
                throw new Error("接口返回无法提取有效图像 Base64/URL 成果。");
            }

            return { success: true, imageData: imageData, mimeType: 'image/jpeg' };
        } catch (error) {
            console.error("⚠️ [GPT-image-2] 图像生成链路异常，抛出异常阻断降级:", error);
            throw error;
        }
    }
    
    if (imageModel.includes('flash-image') || imageModel.includes('pro-image')) {
        const imageConfig = { aspectRatio: aspectRatio };
        if (imageModel.includes('3.1') || imageModel.includes('3-pro')) {
            imageConfig.imageSize = "4K";
        }
        
        if (imageModel.startsWith('gemini')) {
            const payload = {
                contents: [{ role: "user", parts: [{ text: prompt }] }],
                config: { imageConfig: imageConfig }
            };

            const response = await fetch(RECOGNIZE_BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: 'gemini_lite_generate',
                    model: imageModel,
                    payload: payload
                })
            });
            const data = await response.json();
            
            if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) {
                for (const part of data.candidates[0].content.parts) {
                    if (part.inlineData) {
                        return { success: true, imageData: part.inlineData.data, mimeType: part.inlineData.mimeType || 'image/png' };
                    }
                }
            }
            throw new Error("Failed to generate image via proxy.");
        }

        const response = await proxyGenerateContent({
            model: imageModel,
            contents: { parts: [{ text: prompt }] },
            config: {
                imageConfig: imageConfig
            }
        });
        
        if (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) {
            for (const part of response.candidates[0].content.parts) {
                if (part.inlineData) {
                    return { success: true, imageData: part.inlineData.data, mimeType: part.inlineData.mimeType || 'image/png' };
                }
            }
        }
        throw new Error("Failed to generate image with banana model.");
    }
    
    // Fallback to Imagen
    const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: prompt,
        config: {
            numberOfImages: 1,
            outputMimeType: 'image/jpeg',
            aspectRatio: aspectRatio
        }
    });
    
    if (response.generatedImages && response.generatedImages.length > 0) {
        const imageData = response.generatedImages[0].image.imageBytes;
        return { success: true, imageData, mimeType: 'image/jpeg' };
    }
    
    throw new Error("Failed to generate image with Imagen model.");
}

export async function cropImageByLayer(image, layers, layerName) {
    const { cropImageByBox } = await import("../graphics/image-processor.js");
    const { blobToBase64 } = await import("../core/utils.js");
    
    // Find the layer by name (fuzzy match)
    const layer = layers.find(l => l.name.includes(layerName) || layerName.includes(l.name)) || layers[0];
    if (!layer) throw new Error(`Layer "${layerName}" not found.`);
    
    // layer.bbox is already [ymin, xmin, ymax, xmax]
    const box = layer.bbox;
    
    const croppedBlob = await cropImageByBox(image, box);
    const base64 = await blobToBase64(croppedBlob);
    
    return { imageData: base64, mimeType: 'image/png' };
}

export async function outpaintImage(image, prompt, aspectRatio) {
    const promptToUse = prompt || "请将这张图的边界向外扩展，保持原有风格";
    return await editOrQueryImageWithGemini(promptToUse, image, [], null, aspectRatio);
}

export async function applyCameraLens(image, prompt) {
    const promptToUse = prompt || "改变透视和构图，保留原主体特征和环境氛围，必须严格执行新的视角！";
    return await editOrQueryImageWithGemini(promptToUse, image);
}

export async function generateVideo(image, prompt, aspectRatio) {
    prompt = prompt || "将这张图片转换为动态视频";
    const { generateVeoVideo } = await import("./gemini-client.js");
    const { fileToBase64 } = await import("../core/utils.js");
    const b64 = await fileToBase64(image);
    const vidBlob = await generateVeoVideo(prompt, b64, image.type || 'image/png', aspectRatio);
    return vidBlob;
}

export async function upscaleImage(image, prompt) {
    const promptToUse = prompt || "Image Restoration & Reconstruction: Redraw this low-quality image into a pristine, ultra-high-resolution (4K) masterpiece. Aggressively remove all blur, noise, and compression artifacts. CRITICAL: Do not just sharpen the existing pixels. Instead, synthesise and hallucinate missing high-frequency details (such as skin texture, hair strands, fabric patterns, and sharp edges) that are lost in the original. Re-imagine the subject with perfect focus and clarity while keeping the original subject identity, pose, and overall composition intact. The output must look like a sharp, professional commercial photograph taken with a modern high-end DSLR.";
    return await editOrQueryImageWithGemini(promptToUse, image);
}

export async function generateMultiview(image, customPrompt) {
    const { fileToBase64 } = await import("../core/utils.js");
    const base64 = await fileToBase64(image);
    const category = await classifyImageCategory(base64, image.type || 'image/png');
    
    let finalPrompt = "";
    if (category.includes('CHARACTER')) {
        finalPrompt = `Create a professional character sheet (3-view) based on this character. 
        Must include: Front view, Side view, and Back view.
        Style: Character design orthographic projection, neutral background, consistent clothing and features.`;
    } else if (category.includes('PRODUCT')) {
        finalPrompt = `Create a professional product design multi-view (3-view) based on this object.
        Must include: Front, Side, and Top views.
        Style: Product design orthographic sheet, industrial design presentation, identical material and lighting.`;
    } else {
        finalPrompt = `Create a professional architectural orthographic multi-view presentation (3-view sheet) based on this image. 
                Maintain 100% consistency in materials, lighting, textures, and spatial modeling from the original image. 
                This is NOT a drawing, blueprint, or sketch; it is a photorealistic multi-angle photorealistic visualization. 
        The final image MUST contain exactly:
        1. A Top view (俯视图) from a bird's eye perspective.
        2. A Left Elevation (左视图) showing the left side of the structure/space.
        3. A Right Elevation (右视图) showing the right side of the structure/space.
        Style: Professional architectural rendering, realistic textures, consistent spatial logic, high-fidelity visualization.`;
    }

    const promptToUse = customPrompt || finalPrompt;
    return await editOrQueryImageWithGemini(promptToUse, image);
}
