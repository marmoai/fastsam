// Removed import to avoid circular dependencies

export class IntentEngine {
    static infer(action: any): string | undefined {
        // 1. Explicit user intent from input
        if (action.intent) return action.intent;
        if (action.meta?.intent) return action.meta.intent;

        // 2. Rule based implicit inference
        if (action.type === 'UPDATE_FUSION') {
            const props = action.payload?.fusionProperties;
            if (!props) return undefined;

            // Default values based on UI
            const b = props.brightness !== undefined ? props.brightness : 100;
            const c = props.contrast !== undefined ? props.contrast : 100;
            const s = props.saturation !== undefined ? props.saturation : 100;
            const w = props.warmth !== undefined ? props.warmth : 0;
            const blur = props.blur !== undefined ? props.blur : 0;

            // Design Semantics Rules (High Quality Rules)
            
            // 🍔 Make Appetizing
            if (w > 20 && s > 110) return "提升食物吸引力 (Make Appetizing)";
            
            // 🌌 Cyberpunk Style
            if (w < -20 && c > 110) return "增强赛博朋克氛围 (Cyberpunk Style)";
            
            // 🎬 Increase Depth (Darker but higher contrast)
            if (b < 90 && c > 120) return "增强画面质感与深度 (Increase Depth)";
            
            // 💡 Increase Visibility
            if (b > 110 && c <= 100) return "提亮画面可见度 (Increase Visibility)";
            
            // 🌫 Soften Atmosphere
            if (blur > 4) return "柔化主体外氛围 (Soften Atmosphere)";
            
            // 💥 Boost Impact
            if (c > 130) return "强化视觉冲击力 (Boost Impact)";
            
            // 🎞 Faded Cinematic
            if (s < 50 && c > 110) return "褪色电影感 (Faded Cinematic)";
            
            // ☀️ Warm Up
            if (w > 30) return "暖化整体色调 (Warm Up)";
            
            // ❄️ Cool Down
            if (w < -30) return "冷却整体色调 (Cool Down)";
        }

        if (action.type === 'UPDATE_TRANSFORM') {
            const props = action.payload;
            // 🎯 Highlight Subject (Scale up significantly)
            if (props && (props.scaleX > 1.15 || props.scaleY > 1.15)) {
                return "突出画面主体 (Highlight Subject)";
            }
        }

        return undefined;
    }
}
