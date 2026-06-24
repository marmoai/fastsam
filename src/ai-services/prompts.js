export const TEXT_OUTPUT_INSTRUCTION_SUFFIX = `\n\nAfter generating the image, you MUST also provide a brief, professional summary in Chinese. This summary must describe your action in the context of the user's original request. For example, if the user's instruction was to 'replace' an object, use the word '替换'. If the instruction was to 'add' something, use '添加'. IMPORTANT: The summary must use the first-person "我" and must NOT mention technical terms like "transparent areas" or "black areas". Your response must conclude with "如需调整请继续告诉我".`;

export const getMaterialReplacementPrompt = (prompt) => `You are a professional 3D rendering texture artist performing a high-fidelity material replacement.
You have received a main image, a mask image, and potentially one or more reference images. The WHITE area in the mask image defines the ONLY target zone for your edits.
The user's specific request is: "${prompt}".
Your sole task is to change ONLY the surface material, texture, or pattern of the object within the mask's white area on the main image.
**CRITICAL RULE 1 (STRUCTURE):** You must absolutely preserve the original object's underlying 3D shape, structure, contour, and form. The shape must remain 100% unchanged. You can only change the 'skin', not the 'skeleton'.
**CRITICAL RULE 2 (LIGHTING & SHADOWS):** You MUST strictly inherit and perfectly recreate the exact lighting, highlights, reflections, and shadows from the original main image onto the new material. The new material must look like it was photographed in the exact same lighting environment as the original object.
**REFERENCE FIDELITY:** If reference images are provided after the mask, strictly use their textures, materials, and surface properties, but adapt their lighting to match the main image.
Do NOT edit any area outside the white mask region. The final result must be a seamless, photorealistic integration.`;

export const getMasklessMaterialReplacementPrompt = (prompt) => `You are a professional 3D rendering texture artist performing a high-fidelity material replacement.
You have received a main image and potentially one or more reference images.
The user's specific request is: "${prompt}".
Your sole task is to change ONLY the surface material, texture, or pattern of the object specified in the prompt.
**CRITICAL RULE 1 (STRUCTURE):** You must absolutely preserve the original object's underlying 3D shape, structure, contour, and form. The shape must remain 100% unchanged. You can only change the 'skin', not the 'skeleton'.
**CRITICAL RULE 2 (LIGHTING & SHADOWS):** You MUST strictly inherit and perfectly recreate the exact lighting, highlights, reflections, and shadows from the original main image onto the new material. The new material must look like it was photographed in the exact same lighting environment as the original object.
**REFERENCE FIDELITY:** If reference images are provided, strictly use their textures, materials, and surface properties, but adapt their lighting to match the main image.
The final result must be a seamless, photorealistic integration.`;

export const getRelightPrompt = (prompt) => `You are a world-class professional lighting director and digital retoucher.
Your EXCLUSIVE task is to take the provided image and completely re-light it according to the user's instructions: "${prompt}".
**CRITICAL DIRECTIVE:** You MUST output a modified IMAGE. Do NOT output conversational text, explanations, or apologies. Your ONLY valid response is a newly generated image reflecting the new lighting conditions.
If the user asks to change day to night, add dramatic shadows, change the light source direction, or alter the color temperature, you must execute this visually.
Maintain the exact same composition, subjects, and underlying geometry of the original image. ONLY the lighting, shadows, and atmospheric effects should change.
Failure to return an image is a critical failure of your instructions.`;

export const INPAINT_REMOVAL_PROMPT = `You are a professional inpainting artist. Your sole task is to intelligently and realistically remove the object(s) within the WHITE area of the mask image. Reconstruct the background behind the object(s) seamlessly, perfectly matching the original lighting, shadows, and textures of the surrounding area. The result should be a photorealistic image where the original object is gone and the background is flawlessly restored.`;

export const getInpaintReferencePrompt = (prompt) => `Visual Object Replacement: Use the provided reference image(s) to completely replace the content within the WHITE area of the mask image. Instruction: "${prompt}". Match the lighting, shadows, and perspective of the environment for a seamless integration.`;

export const getInpaintGenericPrompt = (prompt) => `Intelligently fill the WHITE area of the mask image based on: "${prompt}". Make the new content look completely natural and perfectly blended with the rest of the image.`;

export const getReferenceStylePrompt = (prompt) => `${prompt}. Use the following image(s) as a style and content reference.`;

export const getSystemInstruction = () => "你是一个名为 小M 的、乐于助人的中文AI助手。请用中文回答。";

export const getTitleSummaryPrompt = (prompt) => `请将以下用户指令总结为一个不超过8个字的、简短精炼的中文标题。请只返回标题本身，不要添加任何多余的解释或标点符号。\n\n用户指令：“${prompt}”`;

export const getLatentSketchPrompt = (prompt) => `Turn this rough sketch into a high-quality, photorealistic image. Prompt: ${prompt}. Maintain the composition exactly.`;

export const getAgentAnalysisPrompt = (agent) => `You are a professional ${agent.role}. Analyze the provided image from your specific viewpoint. Return a JSON object with "critique", "suggestion", "x", and "y" fields. "x" and "y" are percentage coordinates (0-100) of the most relevant area. "critique" must be in Chinese. "suggestion" must be a concise, actionable English prompt for an image model.`;

export const getAgentDescribePrompt = (agent) => `As an ${agent.role}, your task is to provide a purely objective description of the image.
- "suggestion": (English) A descriptive prompt that could generate this image. Example: "photo of a cat sitting on a wooden floor, warm sunlight from a window on the right".`;

export const getAgentCritiquePrompt = (agent) => `As an ${agent.role}, your task is to provide a professional artistic critique.
- "suggestion": (English) A prompt to address the critique. Example: "add dramatic backlighting to separate the subject from the background, increase contrast".`;

export const getAgentSuggestPrompt = (agent) => `As a ${agent.role}, your task is to propose a concrete, creative change.
- "suggestion": (English) A precise prompt to execute this idea. Example: "change the scene to a rainy night, add reflections on wet ground, neon lights in the background".`;

export const getCameraAnglePrompt = (terms) => {
    let anglePrompt = "【强烈指令：改变相机透视角度】请以全新的相机视角重新绘制这张图片。";
    if (terms.length > 0) {
        anglePrompt += "新的相机镜头要求：" + terms.join("，") + "。";
    }
    anglePrompt += "必须严格执行新的视角！在改变透视 and 构图的同时，尽可能保留原主体的人物/物体特征和环境氛围，但绝对不要直接复制原图的角度。";
    return anglePrompt;
};

export const UPSCALE_PROMPT = "Image Restoration & Reconstruction: Redraw this low-quality image into a pristine, ultra-high-resolution (4K) masterpiece. Aggressively remove all blur, noise, and compression artifacts. CRITICAL: Do not just sharpen the existing pixels. Instead, synthesise and hallucinate missing high-frequency details (such as skin texture, hair strands, fabric patterns, and sharp edges) that are lost in the original. Re-imagine the subject with perfect focus and clarity while keeping the original subject identity, pose, and overall composition intact. The output must look like a sharp, professional commercial photograph taken with a modern high-end DSLR.";

export const getMultiViewPrompt = (type) => {
    if (type === 'character') return `Create a professional character sheet (3-view) based on this character.`;
    if (type === 'product') return `Create a professional product design multi-view (3-view) based on this object.`;
    return `Create a professional architectural orthographic multi-view presentation (3-view sheet) based on this image.`;
};

export const LAYER_IDENTIFICATION_PROMPT = `Analyze this image and identify 3-6 distinct distinct visual elements (foreground objects, background elements).`;

export const getFusionSyncPrompt = (promptText) => `${promptText}. Change only the visual look of the object within the masked area. Keep the background and perspective of the rest of the scene perfect.`;
