export const SNAP_THRESHOLD = 8;
export const ZOOM_MIN = 0.01;
export const ZOOM_MAX = 5;
export const ZOOM_STEP = 0.05;
export const RESIZE_RADIUS = 5;
export const OUTPAINT_RADIUS = 12;
export const CANVAS_CENTER = 50000;
export const DEFAULT_ZOOM = 0.2;
export const RATIO_MAP = { '1:1': 1, '4:3': 4/3, '3:4': 3/4, '16:9': 16/9, '9:16': 9/16 };
export const ATMOSPHERE_OPTS = {
    'sun': { label: '暖阳', icon: 'fa-sun', prompt: 'warm sunset lighting', color: [1.0, 0.9, 0.6] },
    'neon': { label: '霓虹', icon: 'fa-bolt', prompt: 'cyberpunk neon lighting', color: [0.0, 1.0, 0.8] },
    'spotlight': { label: '聚光灯', icon: 'fa-bullseye', prompt: 'hard spotlight', color: [1.0, 1.0, 1.0] },
    'mist': { label: '迷雾', icon: 'fa-smog', prompt: 'foggy atmospheric depth', color: [0.8, 0.8, 0.9] },
    'rain': { label: '雨夜', icon: 'fa-cloud-rain', prompt: 'rainy night atmosphere', color: [0.4, 0.5, 0.7] },
    'snow': { label: '飞雪', icon: 'fa-snowflake', prompt: 'snowy winter atmosphere', color: [0.9, 0.95, 1.0] }
};
