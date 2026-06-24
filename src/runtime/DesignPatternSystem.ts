export interface DesignPattern {
    id: string;
    name: string;
    intent: string;
    description: string;
    emoji: string;
    apply: (dispatcher: any, targetUid: string) => void;
}

export const DESIGN_PATTERNS: DesignPattern[] = [
    {
        id: 'make_appetizing',
        name: '提升食物吸引力',
        intent: '提升食物吸引力 (Make Appetizing)',
        description: '增加暖色调和饱和度，适合美食',
        emoji: '🍔',
        apply: (dispatcher, targetUid) => {
            dispatcher.dispatch({
                type: 'UPDATE_FUSION',
                intent: '提升食物吸引力 (Make Appetizing)',
                payload: {
                    uid: targetUid,
                    fusionProperties: {
                        warmth: 25,
                        saturation: 115,
                        contrast: 105
                    }
                }
            });
        }
    },
    {
        id: 'cyberpunk_style',
        name: '增强赛博朋克氛围',
        intent: '增强赛博朋克氛围 (Cyberpunk Style)',
        description: '冷色调、高对比亮度环境',
        emoji: '🌌',
        apply: (dispatcher, targetUid) => {
            dispatcher.dispatch({
                type: 'UPDATE_FUSION',
                intent: '增强赛博朋克氛围 (Cyberpunk Style)',
                payload: {
                    uid: targetUid,
                    fusionProperties: {
                        warmth: -35,
                        contrast: 125,
                        saturation: 110,
                        blur: 2
                    }
                }
            });
        }
    },
    {
        id: 'increase_depth',
        name: '增强画面质感与深度',
        intent: '增强画面质感与深度 (Increase Depth)',
        description: '加深阴影，提高对比度',
        emoji: '🎬',
        apply: (dispatcher, targetUid) => {
            dispatcher.dispatch({
                type: 'UPDATE_FUSION',
                intent: '增强画面质感与深度 (Increase Depth)',
                payload: {
                    uid: targetUid,
                    fusionProperties: {
                        brightness: 85,
                        contrast: 125
                    }
                }
            });
        }
    },
    {
        id: 'highlight_subject',
        name: '突出画面主体',
        intent: '突出画面主体 (Highlight Subject)',
        description: '缩放主体以提高视觉比重',
        emoji: '🎯',
        apply: (dispatcher, targetUid) => {
            dispatcher.dispatch({
                type: 'UPDATE_TRANSFORM',
                intent: '突出画面主体 (Highlight Subject)',
                payload: {
                    uid: targetUid,
                    scaleX: 1.18,
                    scaleY: 1.18
                }
            });
        }
    },
    {
        id: 'faded_cinematic',
        name: '褪色电影感',
        intent: '褪色电影感 (Faded Cinematic)',
        description: '降低饱和度，模拟经典胶片',
        emoji: '🎞',
        apply: (dispatcher, targetUid) => {
            dispatcher.dispatch({
                type: 'UPDATE_FUSION',
                intent: '褪色电影感 (Faded Cinematic)',
                payload: {
                    uid: targetUid,
                    fusionProperties: {
                        saturation: 45,
                        contrast: 115,
                        warmth: 10
                    }
                }
            });
        }
    }
];
