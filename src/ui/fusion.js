import { generatePreciseEditImage } from '../ai-services/skills-engine.js';
import { fileToBase64 } from '../core/utils.js';

function getClosestAspectRatio(width, height) {
    const ratios = [
        { name: '1:1', value: 1 },
        { name: '3:4', value: 3/4 },
        { name: '4:3', value: 4/3 },
        { name: '9:16', value: 9/16 },
        { name: '16:9', value: 16/9 }
    ];
    const targetRatio = width / height;
    let closest = ratios[0];
    let minDiff = Math.abs(targetRatio - closest.value);
    for (let i = 1; i < ratios.length; i++) {
        const diff = Math.abs(targetRatio - ratios[i].value);
        if (diff < minDiff) {
            minDiff = diff;
            closest = ratios[i];
        }
    }
    return closest.name;
}

export async function performDeepFusion(baseImageFile, refImageFile, maskDataUrl, prompt) {
    // 1. Get original dimensions
    const baseImg = new Image();
    baseImg.src = URL.createObjectURL(baseImageFile);
    await new Promise(r => baseImg.onload = r);
    const origW = baseImg.naturalWidth;
    const origH = baseImg.naturalHeight;
    
    const aspectRatioStr = getClosestAspectRatio(origW, origH);
    
    // 2. Call AI Inpainting with reference background
    const systemProtocol = `
    You are an elite AI image editor specializing in ultra-precise, localized modifications.
    CRITICAL RULES:
    1. **THE QUARANTINE ZONE**: You are ONLY permitted to alter the pixels corresponding to the WHITE area in Image 1. 
    2. **ABSOLUTE FREEZE OUTSIDE**: Every single pixel corresponding to the BLACK area in Image 1 MUST remain 100% identical to Image 0. If you alter the background, lighting, or any object in the black zone, the operation is considered a catastrophic failure.
    3. **SEAMLESS INTEGRATION**: The new content inside the white zone must blend flawlessly with the frozen pixels outside the zone. Match the original lighting, shadows, and grain perfectly.
    4. **NO HALLUCINATIONS**: Do not invent new objects or structures that were not requested.
    `;

    const finalCommandAction = `Perform a high-precision redraw strictly confined to the masked area. The user request is: "${prompt}"`;

    const finalEditPrompt = `
    [SYSTEM INSTRUCTION]
    ${systemProtocol}
    
    FINAL COMMAND: ${finalCommandAction} 
    WARNING: You will be penalized if ANY pixel corresponding to the BLACK area in Image 1 is changed. 
    MANDATORY: Return ONLY the clean, final image.
    `;

    const base64Original = await fileToBase64(baseImageFile);
    const base64Mask = maskDataUrl.includes(',') ? maskDataUrl.split(',')[1] : maskDataUrl;
    const base64Ref = await fileToBase64(refImageFile);

    const contentsParts = [
        { inlineData: { data: base64Original, mimeType: baseImageFile.type || 'image/png' } },
        { inlineData: { data: base64Mask, mimeType: 'image/png' } },
        { inlineData: { data: base64Ref, mimeType: refImageFile.type || 'image/png' } },
        { text: finalEditPrompt }
    ];

    const resultImageData = await generatePreciseEditImage(contentsParts, aspectRatioStr);
    
    if (!resultImageData) {
        throw new Error("AI Deep Fusion failed");
    }

    // 3. Restore exact original dimensions
    const resultImg = new Image();
    resultImg.src = `data:image/png;base64,${resultImageData}`;
    await new Promise(r => resultImg.onload = r);

    const canvas = document.createElement('canvas');
    canvas.width = origW;
    canvas.height = origH;
    const ctx = canvas.getContext('2d');

    // Calculate object-fit: cover dimensions to perfectly align the center
    const imgRatio = resultImg.naturalWidth / resultImg.naturalHeight;
    const canvasRatio = origW / origH;
    
    let drawW, drawH, drawX, drawY;
    if (imgRatio > canvasRatio) {
        drawH = origH;
        drawW = origH * imgRatio;
        drawX = (origW - drawW) / 2;
        drawY = 0;
    } else {
        drawW = origW;
        drawH = origW / imgRatio;
        drawX = 0;
        drawY = (origH - drawH) / 2;
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(resultImg, drawX, drawY, drawW, drawH);

    return canvas.toDataURL('image/png');
}
