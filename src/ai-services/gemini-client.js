import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";
import { state } from "../core/state.js";
import { RECOGNIZE_BACKEND_URL } from "../core/utils.js";

const API_KEY = "AIzaSyAuqFL8VJ1pQS0ZvVwQEwY4RIwH-wJ7qa4";
export const ai = new GoogleGenAI({ apiKey: API_KEY });

export async function proxyGenerateContent(request) {
    const model = request.model;
    if (model && model.startsWith('gemini')) {
        let normalizedContents = request.contents;
        
        if (typeof normalizedContents === 'string') {
            normalizedContents = [{ role: 'user', parts: [{ text: normalizedContents }] }];
        } else if (Array.isArray(normalizedContents)) {
            normalizedContents = normalizedContents.map(c => {
                if (c.role && c.parts) return c;
                if (c.parts) return { role: 'user', parts: c.parts };
                if (c.text) return { role: 'user', parts: [{ text: c.text }] };
                if (c.inlineData) return { role: 'user', parts: [{ inlineData: c.inlineData }] };
                return c;
            });
        } else if (typeof normalizedContents === 'object' && normalizedContents.parts) {
            normalizedContents = [{ role: 'user', parts: normalizedContents.parts }];
        }

        const payload = {
            contents: normalizedContents,
            systemInstruction: request.config?.systemInstruction || request.systemInstruction,
            config: request.config
        };

        const res = await fetch(RECOGNIZE_BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'gemini_lite_generate',
                model: model,
                payload: payload
            })
        });
        
        if (!res.ok) {
            throw new Error(`Proxy error: ${res.status}`);
        }
        const data = await res.json();
        
        // 确保代理返回的 API 报错能够被捕获，而不是后续引发未定义属性异常
        if (data && data.error) {
            const errorMsg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
            throw new Error(`Proxy API Error: ${errorMsg}`);
        }

        // 统一接口：REST 接口返回的是 RAW JSON，缺少 SDK 生成的 .text 属性/getter
        // 我们在此将 candidates[0].content.parts[0].text 统一映射挂载
        if (data && !Object.prototype.hasOwnProperty.call(data, 'text')) {
            let extractedText = "";
            if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) {
                for (const part of data.candidates[0].content.parts) {
                    if (part.text) {
                        extractedText += part.text;
                    }
                }
            }
            Object.defineProperty(data, 'text', {
                get() {
                    return extractedText;
                },
                configurable: true,
                enumerable: true
            });
        }
        return data;
    } else {
        return await ai.models.generateContent(request);
    }
}


export const MODEL_SUITES = {
    'pro': {
        text: 'gemini-3.1-pro-preview',
        image: 'gemini-3-pro-image-preview',
        name: 'Gemini 3.1 Pro 套装 (最强逻辑/顶级生图)'
    },
    'flash': {
        text: 'gemini-3-flash-preview',
        image: 'gemini-3.1-flash-image-preview',
        name: 'Gemini 3.1 Flash 套装 (全能/极速生图)'
    },
    'lite': {
        text: 'gemini-3.1-flash-lite',
        image: 'gemini-2.5-flash-image',
        name: 'Gemini Flash Lite 套装 (日常/经典生图)'
    },
    'gpt-image-2': {
        text: 'gemini-3.1-pro-preview',
        image: 'gpt-image-2',
        name: 'GPT-image-2 (中转高精生成与编辑)'
    },
    'qwen-cleanplate': {
        text: 'gemini-3.1-pro-preview',
        image: 'gpt-image-2',
        name: 'Qwen Clean Plate (仅背景净化)'
    }
};

export const getTextModel = () => {
    return MODEL_SUITES[state.selectedModel]?.text || MODEL_SUITES['flash'].text;
};

export const getImageModel = () => {
    return MODEL_SUITES[state.selectedModel]?.image || MODEL_SUITES['flash'].image;
};

export const backgroundModel = 'gemini-3.1-flash-lite';

const getThinkingLevel = () => {
    return state.thinkingLevel === 'MEDIUM' ? ThinkingLevel.HIGH : ThinkingLevel.LOW;
};

const editImageTool = {
    name: "edit_image",
    description: "编辑工作台上的某张图片。当用户要求修改、调整、重绘某张图片时调用此工具。",
    parameters: {
        type: Type.OBJECT,
        properties: {
            itemId: {
                type: Type.STRING,
                description: "要编辑的图片的ID（从工作台状态中获取）"
            },
            prompt: {
                type: Type.STRING,
                description: "编辑指令（例如：'把猫变成狗'、'放大'）"
            }
        },
        required: ["itemId", "prompt"]
    }
};

const manipulateItemTool = {
    name: "manipulate_item",
    description: "操作工作台上的元素，如移动、缩放、删除等。",
    parameters: {
        type: Type.OBJECT,
        properties: {
            itemId: {
                type: Type.STRING,
                description: "要操作的元素的ID"
            },
            action: {
                type: Type.STRING,
                description: "操作类型：'move', 'resize', 'delete'"
            },
            value: {
                type: Type.OBJECT,
                description: "操作的具体值，例如移动的坐标 {\"x\": 100, \"y\": 100}，缩放的大小 {\"width\": 200, \"height\": 200}"
            }
        },
        required: ["itemId", "action"]
    }
};

export async function generateVisualSearch(base64Image) {
    let response;
    if (backgroundModel.startsWith('gemini')) {
        const payload = {
            contents: [{ role: "user", parts: [
                { inlineData: { data: base64Image, mimeType: 'image/png' } },
                { text: 'Analyze this image and identify the main product. Provide a product name, a short description, and a search query for finding similar products. Return the result in JSON format.' }
            ]}],
            config: {
                tools: [{ googleSearch: {} }],
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        productName: { type: Type.STRING },
                        description: { type: Type.STRING },
                        searchQuery: { type: Type.STRING }
                    },
                    required: ["productName", "description", "searchQuery"]
                },
                thinkingConfig: { thinkingLevel: getThinkingLevel() }
            }
        };
        const res = await fetch(RECOGNIZE_BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'gemini_lite_generate', model: backgroundModel, payload })
        });
        response = await res.json();
    } else {
        response = await ai.models.generateContent({
            model: backgroundModel,
            contents: { parts: [
                { inlineData: { data: base64Image, mimeType: 'image/png' } },
                { text: 'Analyze this image and identify the main product. Provide a product name, a short description, and a search query for finding similar products. Return the result in JSON format.' }
            ]},
            config: {
                tools: [{ googleSearch: {} }],
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        productName: { type: Type.STRING },
                        description: { type: Type.STRING },
                        searchQuery: { type: Type.STRING }
                    },
                    required: ["productName", "description", "searchQuery"]
                },
                thinkingConfig: { thinkingLevel: getThinkingLevel() }
            }
        });
    }

    let resultText = "";
    if (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.text) { resultText += part.text; }
        }
    } else if (response.text) {
        resultText = response.text;
    }

    const resultData = JSON.parse(resultText);
    const webLinks = response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.web).filter(w => w?.uri) || [];
    
    return { resultData, webLinks };
}

export async function generateTextWithSearch(prompt) {
    let response;
    const userMessage = { role: "user", parts: [{ text: `请为接下来的图像生成任务，详细描述一下“${prompt}”。请专注于视觉细节，例如颜色、形状、纹理、构图和氛围。` }] };
    
    if (backgroundModel.startsWith('gemini')) {
        const payload = {
            contents: [userMessage],
            config: { tools: [{ googleSearch: {} }], thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        };
        const res = await fetch(RECOGNIZE_BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'gemini_lite_generate', model: backgroundModel, payload })
        });
        response = await res.json();
    } else {
        response = await ai.models.generateContent({
           model: backgroundModel,
           contents: userMessage.parts[0].text,
           config: { tools: [{googleSearch: {}}], thinkingConfig: { thinkingLevel: getThinkingLevel() } },
        });
    }
    
    let resultText = "";
    if (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.text) { resultText += part.text; }
        }
    } else if (response.text) {
        resultText = response.text;
    }
    
    const sources = response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.web).filter(w => w?.uri) || [];
    return { success: true, text: resultText, sources };
}

export async function generateLatentImage(prompt, base64Sketch, seed) {
    const currentModel = getImageModel();
    let response;

    if (currentModel.startsWith('gemini')) {
        const payload = {
            contents: [{ role: "user", parts: [
                { inlineData: { data: base64Sketch, mimeType: 'image/jpeg' } },
                { text: `Turn this rough sketch into a high-quality, photorealistic image. Prompt: ${prompt}. Maintain the composition exactly.` }
            ]}],
            config: { seed, thinkingConfig: { thinkingLevel: getThinkingLevel() } }
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
        response = await ai.models.generateContent({
            model: currentModel,
            contents: { parts: [
                { inlineData: { data: base64Sketch, mimeType: 'image/jpeg' } },
                { text: `Turn this rough sketch into a high-quality, photorealistic image. Prompt: ${prompt}. Maintain the composition exactly.` }
            ]},
            config: { seed, thinkingConfig: { thinkingLevel: getThinkingLevel() } }
        });
    }
    
    let imageData = null;
    if (response.candidates && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
                imageData = part.inlineData.data;
                break;
            }
        }
    }
    if (!imageData) {
        console.log("No image data in response. Full response:", response);
    }
    return imageData;
}

export async function createChatSession(history, systemInstruction) {
    return ai.chats.create({
        model: getTextModel(),
        history: history,
        config: { 
            systemInstruction,
            tools: [{ functionDeclarations: [editImageTool, manipulateItemTool] }],
            thinkingConfig: { thinkingLevel: getThinkingLevel() }
        },
    });
}

export async function generateTextWithGemini(prompt, sessionId, history, canvasState) {
    const formattedHistory = history
        .slice(0, -1)
        .filter(msg => msg.content && (msg.type === 'text' || msg.type === 'bot-rich' || (msg.sender === 'user' && msg.content.trim() !== '')))
        .map(msg => ({
            role: msg.sender === 'user' ? 'user' : 'model',
            parts: [{ text: msg.content }]
        }));
    
    const systemInstruction = `你是一个名为 小M 的、乐于助人的中文AI助手。
你现在可以感知用户的工作台(Canvas)状态。请根据用户的问题和工作台状态进行回答。
如果用户提到“左边”、“上面”等方位词，请参考工作台状态中的 position (left, top) 和 size (width, height) 来理解。
如果用户要求修改工作台上的某张图片，请调用 edit_image 工具。
如果用户要求移动、缩放或删除工作台上的元素，请调用 manipulate_item 工具。
请用中文回答。`;

    let finalPrompt = prompt;
    if (canvasState) {
        finalPrompt = `[系统提示：当前工作台状态更新如下(JSON格式)：\n${JSON.stringify(canvasState)}]\n\n用户输入: ${prompt}`;
    }

    const textModel = getTextModel();

    if (textModel.startsWith('gemini')) {
        const payload = {
            contents: [
                ...formattedHistory,
                { role: "user", parts: [{ text: finalPrompt }] }
            ],
            systemInstruction: {
                parts: [{ text: systemInstruction }]
            },
            tools: [{ functionDeclarations: [editImageTool, manipulateItemTool] }]
        };

        const res = await fetch(RECOGNIZE_BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'gemini_lite_generate',
                model: textModel,
                payload: payload
            })
        });

        const data = await res.json();
        
        let responseText = "";
        let functionCalls = [];

        if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) {
            for (const part of data.candidates[0].content.parts) {
                if (part.text) responseText += part.text;
                if (part.functionCall) functionCalls.push(part.functionCall);
            }
        }

        if (functionCalls.length > 0) {
            return {
                success: true,
                text: responseText || "正在执行操作...",
                functionCalls: functionCalls.map(call => ({
                    name: call.name,
                    args: call.args
                }))
            };
        }
        
        return { success: true, text: responseText };
    }

    const chat = await createChatSession(formattedHistory, systemInstruction);
    
    const response = await chat.sendMessage({ message: finalPrompt });
    
    if (response.functionCalls && response.functionCalls.length > 0) {
        let responseText = "正在执行操作...";
        try {
            if (response.text) responseText = response.text;
        } catch (e) {}
        return { 
            success: true, 
            text: responseText, 
            functionCalls: response.functionCalls.map(call => ({
                name: call.name,
                args: call.args
            }))
        };
    }
    
    return { success: true, text: response.text };
}

const chatSessionsCache = new Map();

export function clearChatSession(sessionId) {
    chatSessionsCache.delete(sessionId);
}

export async function generateVeoVideo(prompt, base64Image, mimeType, aspectRatio) {
    let operation = await ai.models.generateVideos({
        model: 'veo-3.1-generate-preview',
        prompt: prompt,
        image: {
            imageBytes: base64Image,
            mimeType: mimeType
        },
        config: {
            numberOfVideos: 1,
            resolution: '1080p',
            aspectRatio: aspectRatio
        }
    });

    while (!operation.done) {
        await new Promise(r => setTimeout(r, 5000));
        operation = await ai.operations.getVideosOperation({ operation: operation });
    }

    const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
    if (videoUri) {
        const vidRes = await fetch(`${videoUri}&key=${API_KEY}`);
        return await vidRes.blob();
    }
    return null;
}
