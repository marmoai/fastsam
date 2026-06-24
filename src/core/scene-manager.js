import { fileToBase64 } from './utils.js';

export function createSceneId() {
    return `scene_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function buildSceneDocument(file, imageDataUrl, imageSize, rawLayers) {
    const now = Date.now();

    const layers = rawLayers.map((layer) => ({
        id: layer.id,
        name: layer.name,
        category: layer.category,
        semanticType: layer.semanticType, // 添加这一行
        designRole: layer.designRole || "unknown",
        renderMode: layer.renderMode || null,
        compositeRole: layer.compositeRole || null,
        parentLayerId: layer.parentLayerId || null,
        textContent: layer.textContent || "",
        fontStyle: layer.fontStyle || "",
        promptHint: layer.promptHint || "",
        bbox: layer.bbox,
        zIndex: layer.zIndex,
        visible: true,
        locked: layer.category === "background",
        editable: layer.editable,
        assetStatus: layer.assetStatus || "idle",

        // Future asset placeholders
        maskUrl: null,
        cutoutUrl: null,
        cleanPlateUrl: null,

        transform: {
            x: 0,
            y: 0,
            scale: 1,
            rotation: 0,
        },

        semantic: {
            promptHint: layer.promptHint,
            designRole: layer.designRole || "unknown",
            renderMode: layer.renderMode || null,
            categoryConfidence: undefined,
        },

        style: {
            color: null,
            material: null,
        },

        provenance: {
            source: "original",
            parentLayerId: null,
        },

        history: [],
    }));

    return {
        id: createSceneId(),
        createdAt: now,
        updatedAt: now,
        originalImageName: file.name,
        originalImageMimeType: file.type || "image/png",
        originalImageDataUrl: imageDataUrl,

        canvas: {
            width: imageSize.width,
            height: imageSize.height,
        },

        layers,
        history: [],
        version: 1,
    };
}

export function applyEditAction(scene, action) {
    const nextScene = JSON.parse(JSON.stringify(scene)); // Deep clone
    nextScene.updatedAt = Date.now();
    nextScene.version += 1;
    nextScene.history.push(action);

    const layer = nextScene.layers.find((l) => l.id === action.layerId);
    if (!layer) return nextScene;

    switch (action.type) {
        case "move":
            layer.transform.x += action.dx;
            layer.transform.y += action.dy;
            pushLayerHistory(layer, "move", { dx: action.dx, dy: action.dy });
            break;
        case "scale":
            layer.transform.scale = Math.max(0.05, Math.min(10, action.scale));
            pushLayerHistory(layer, "scale", { scale: action.scale });
            break;
        case "rotate":
            layer.transform.rotation = action.rotation;
            pushLayerHistory(layer, "rotate", { rotation: action.rotation });
            break;
        case "delete":
            layer.visible = false;
            pushLayerHistory(layer, "delete", {});
            break;
        case "toggle_visibility":
            layer.visible = action.visible;
            pushLayerHistory(layer, "toggle_visibility", { visible: action.visible });
            break;
        case "recolor":
            layer.style.color = action.color;
            pushLayerHistory(layer, "recolor", { color: action.color });
            break;
        case "replace":
            layer.semantic.promptHint = action.prompt;
            layer.provenance.source = "edited";
            pushLayerHistory(layer, "replace", { prompt: action.prompt });
            break;
    }

    return nextScene;
}

function pushLayerHistory(layer, action, params) {
    layer.history.push({
        action,
        timestamp: Date.now(),
        params,
    });
}

export function reorderLayers(scene, orderedLayerIds) {
    const nextScene = JSON.parse(JSON.stringify(scene));
    const layerMap = new Map(nextScene.layers.map((l) => [l.id, l]));
    const reordered = [];

    for (const id of orderedLayerIds) {
        const layer = layerMap.get(id);
        if (layer) reordered.push(layer);
    }

    for (const layer of nextScene.layers) {
        if (!orderedLayerIds.includes(layer.id)) {
            reordered.push(layer);
        }
    }

    reordered.forEach((layer, index) => {
        layer.zIndex = index;
    });

    nextScene.layers = reordered;
    nextScene.updatedAt = Date.now();
    nextScene.version += 1;

    return nextScene;
}

export function findTopmostLayerAtPoint(scene, point) {
    const visibleLayers = scene.layers
        .filter((l) => l.visible)
        .sort((a, b) => b.zIndex - a.zIndex);

    for (const layer of visibleLayers) {
        const [ymin, xmin, ymax, xmax] = layer.bbox;
        if (point.x >= xmin && point.x <= xmax && point.y >= ymin && point.y <= ymax) {
            return layer;
        }
    }

    return null;
}
