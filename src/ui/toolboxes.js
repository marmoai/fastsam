import { state } from '../core/state.js';
import { addMessage } from './chat-panel.js';
import { editOrQueryImageWithGemini, classifyImageCategory } from '../ai-services/skills-engine.js';
import { dataURLToFile, fileToDataURL, getProxiedUrl } from '../core/utils.js';
import { addImageToWorkbench, deleteWorkbenchItem } from './workbench-core.js';
import { openMaskEditor } from '../graphics/mask-drawer.js';
import { showLayerManagerModal, openCameraAngleModal, showVideoPromptModal, getLayerState } from './modals.js';
import { startPreciseEditMode } from './fusion-editor.js';
import { generateWanMotionPreview, pickWanImageSize } from '../ai-services/siliconflow-video.js';
import { exportCurrentSceneImage } from './workbench/layer-assets.js';
import { uploadImageToOSS } from '../services/ossService.js';
import { triggerMagicLayers } from './layer-manager.js';
import { buildSemanticLayerViews, applySemanticLayerViewsToItem, refreshMotionReadyViewsForItem } from '../services/semantic-layer-views.js';
import { restoreTextContainerShapes } from './text-container-restore.js';

export function setupToolboxEvents(context) {
    const {
        workbenchToolbox,
        workbenchGrid,
        workbenchZoomContainer,
        pushSelectedToChat
    } = context;

    const bindToolboxBtn = (selector, handler) => {
        const btn = workbenchToolbox.querySelector(selector);
        if (btn) {
            btn.addEventListener('click', handler);
        }
    };

    const replayMagicMotionPreview = (containerId) => {
        if (!containerId || typeof document === 'undefined') return;
        const root = document.getElementById(containerId);
        if (!root) return;
        if (root.querySelector('.mmp-canvas-stage')) {
            if (window.startMagicMotionCanvasPreview) {
                window.startMagicMotionCanvasPreview(containerId);
            }
            return;
        }
        root.classList.remove('mmp-playing');
        root.querySelectorAll('.mmp-anim').forEach(el => {
            el.style.animation = 'none';
        });
        void root.offsetWidth;
        root.querySelectorAll('.mmp-anim').forEach(el => {
            el.style.animation = '';
        });
        root.classList.add('mmp-playing');
    };

    if (typeof window !== 'undefined') {
        window.replayMagicMotionPreview = replayMagicMotionPreview;
    }

    const escapeHtml = (value) => String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');

    const decodeHtmlEntities = (value) => {
        let source = String(value ?? '');
        if (!source) return '';
        // Workbench note HTML can be escaped more than once. Decode until it
        // stops changing so an ampersand never reaches the canvas as `amp;`.
        for (let pass = 0; pass < 3; pass += 1) {
            const decoded = typeof document === 'undefined'
                ? source
                    .replace(/&amp;/g, '&')
                    .replace(/&lt;/g, '<')
                    .replace(/&gt;/g, '>')
                    .replace(/&quot;/g, '"')
                    .replace(/&#39;/g, "'")
                : (() => {
                    const textarea = document.createElement('textarea');
                    textarea.innerHTML = source;
                    return textarea.value;
                })();
            if (decoded === source) break;
            source = decoded;
        }
        return source;
    };

    const serializeJsonForHtmlScript = (value) => JSON.stringify(value)
        .replace(/</g, '\\u003c')
        .replace(/>/g, '\\u003e')
        .replace(/&/g, '\\u0026')
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029');

    const svgDataUrlFromMarkup = (svgMarkup) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgMarkup)}`;

    const clamp01 = (value) => Math.min(1, Math.max(0, Number(value) || 0));
    const lerp = (start, end, progress) => start + (end - start) * progress;
    const lerpValue = (from, to, progress) => {
        if (Array.isArray(from) && Array.isArray(to)) {
            return from.map((value, index) => lerp(Number(value) || 0, Number(to[index]) || 0, progress));
        }
        return lerp(Number(from) || 0, Number(to) || 0, progress);
    };

    const applyMotionEase = (name, value) => {
        const t = clamp01(value);
        switch (name) {
            case 'easeOutCubic':
                return 1 - Math.pow(1 - t, 3);
            case 'easeInOutCubic':
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            case 'easeOutQuint':
                return 1 - Math.pow(1 - t, 5);
            case 'easeOutBack': {
                const c1 = 1.70158;
                const c3 = c1 + 1;
                return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
            }
            case 'easeOutExpo':
                return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
            case 'linear':
            default:
                return t;
        }
    };

    const sampleTrack = (track = [], progress = 0) => {
        if (!Array.isArray(track) || track.length === 0) return null;
        if (track.length === 1) return track[0]?.v ?? null;

        const t = clamp01(progress);
        if (t <= Number(track[0]?.t || 0)) return track[0]?.v ?? null;

        for (let index = 1; index < track.length; index += 1) {
            const prev = track[index - 1];
            const next = track[index];
            const startT = Number(prev?.t || 0);
            const endT = Number(next?.t || 0);
            if (t > endT) continue;
            const segmentProgress = endT <= startT ? 1 : (t - startT) / (endT - startT);
            return lerpValue(prev?.v, next?.v, applyMotionEase(next?.ease || 'linear', segmentProgress));
        }

        return track[track.length - 1]?.v ?? null;
    };

    const buildMotionTrack = (points) => points
        .map(point => ({
            t: clamp01(point.t),
            v: point.v,
            ease: point.ease || 'linear'
        }))
        .sort((a, b) => a.t - b.t);

    const parseCssSizePx = (value, fallback = 0) => {
        if (typeof value === 'number' && Number.isFinite(value)) return value;
        const match = String(value || '').match(/-?\d+(\.\d+)?/);
        return match ? Number(match[0]) : fallback;
    };

    const isVisibleMotionColor = (value) => {
        if (!value || typeof value !== 'string') return false;
        const normalized = value.trim().toLowerCase();
        return normalized !== 'transparent' &&
            normalized !== 'rgba(0, 0, 0, 0)' &&
            normalized !== 'rgba(0,0,0,0)';
    };

    const stripHtmlToText = (value) => String(value || '')
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<\/div>/gi, '\n')
        .replace(/<[^>]+>/g, '')
        .replace(/&nbsp;/gi, ' ')
        .replace(/&amp;/gi, '&')
        .replace(/&lt;/gi, '<')
        .replace(/&gt;/gi, '>')
        .replace(/&quot;/gi, '"')
        .replace(/&#39;/gi, "'")
        .replace(/\n{3,}/g, '\n\n')
        .trim();

    const extractInlineStyleValue = (styleText, propertyNames = []) => {
        const source = String(styleText || '');
        for (const propertyName of propertyNames) {
            const pattern = new RegExp(`${propertyName}\\s*:\\s*([^;]+)`, 'i');
            const match = source.match(pattern);
            if (match?.[1]) return match[1].trim();
        }
        return '';
    };

    const resolveMotionPreset = (entryMotion = '', role = 'unknown') => {
        const presetName = String(entryMotion || '');
        const base = {
            from: [0, 0],
            overshoot: [0, 0],
            settle: [0, 0],
            scaleFrom: [0.92, 0.92],
            scaleOvershoot: [1.03, 1.03],
            scaleSettle: [0.992, 0.992],
            rotateFrom: 0,
            rotateOvershoot: 0,
            rotateSettle: 0,
            blurFrom: role === 'text' ? 8 : 10,
            glowFrom: role === 'product' ? 0.26 : 0.16,
            glowPeak: role === 'product' ? 0.42 : 0.24,
            revealOrigin: role === 'panel' ? 'bottom' : 'center',
            revealFrom: role === 'panel' ? 0 : 0.2
        };

        const overrides = {
            'hero-product-focus': { from: [0, 0.54], overshoot: [0, -0.08], settle: [0, 0.02], scaleFrom: [0.42, 0.42], scaleOvershoot: [1.14, 1.14], rotateFrom: -10, rotateOvershoot: 2, blurFrom: 16, glowPeak: 0.48 },
            'hero-product-lift': { from: [0, 0.62], overshoot: [0, -0.1], settle: [0, 0.02], scaleFrom: [0.38, 0.38], scaleOvershoot: [1.16, 1.16], rotateFrom: -8, rotateOvershoot: 1, blurFrom: 18, glowPeak: 0.5 },
            'hero-product-swing-left': { from: [-1.2, 0.38], overshoot: [0.08, -0.08], settle: [-0.03, 0.02], scaleFrom: [0.34, 0.34], scaleOvershoot: [1.16, 1.16], rotateFrom: -28, rotateOvershoot: 6, blurFrom: 16, glowPeak: 0.52 },
            'hero-product-swing-right': { from: [1.2, 0.38], overshoot: [-0.08, -0.08], settle: [0.03, 0.02], scaleFrom: [0.34, 0.34], scaleOvershoot: [1.16, 1.16], rotateFrom: 28, rotateOvershoot: -6, blurFrom: 16, glowPeak: 0.52 },
            'support-product-left': { from: [-0.44, 0.22], overshoot: [0.04, -0.03], settle: [0, 0.01], scaleFrom: [0.82, 0.82], scaleOvershoot: [1.05, 1.05], rotateFrom: -4, rotateOvershoot: 1, blurFrom: 8, glowPeak: 0.28 },
            'support-product-right': { from: [0.44, 0.22], overshoot: [-0.04, -0.03], settle: [0, 0.01], scaleFrom: [0.82, 0.82], scaleOvershoot: [1.05, 1.05], rotateFrom: 4, rotateOvershoot: -1, blurFrom: 8, glowPeak: 0.28 },
            'support-product-skim-left': { from: [-0.62, 0.1], overshoot: [0.03, -0.02], settle: [0, 0.01], scaleFrom: [0.74, 0.74], scaleOvershoot: [1.06, 1.06], rotateFrom: -10, rotateOvershoot: 1, blurFrom: 10, glowPeak: 0.3 },
            'support-product-skim-right': { from: [0.62, 0.1], overshoot: [-0.03, -0.02], settle: [0, 0.01], scaleFrom: [0.74, 0.74], scaleOvershoot: [1.06, 1.06], rotateFrom: 10, rotateOvershoot: -1, blurFrom: 10, glowPeak: 0.3 },
            'support-product-rise-center': { from: [0, 0.42], overshoot: [0, -0.04], settle: [0, 0.01], scaleFrom: [0.72, 0.72], scaleOvershoot: [1.08, 1.08], blurFrom: 10, glowPeak: 0.28 },
            'support-product-sweep-up': { from: [0, 0.68], overshoot: [0, -0.08], settle: [0, 0.02], scaleFrom: [0.48, 0.48], scaleOvershoot: [1.12, 1.12], rotateFrom: -12, rotateOvershoot: 2, blurFrom: 14, glowPeak: 0.34 },
            'accent-product-pop-left': { from: [-0.3, 0.24], overshoot: [0.02, -0.02], settle: [0, 0.01], scaleFrom: [0.58, 0.58], scaleOvershoot: [1.12, 1.12], rotateFrom: -12, rotateOvershoot: 1, blurFrom: 8 },
            'accent-product-pop-center': { from: [0, 0.28], overshoot: [0, -0.03], settle: [0, 0.01], scaleFrom: [0.56, 0.56], scaleOvershoot: [1.14, 1.14], blurFrom: 8 },
            'accent-product-pop-right': { from: [0.3, 0.24], overshoot: [-0.02, -0.02], settle: [0, 0.01], scaleFrom: [0.58, 0.58], scaleOvershoot: [1.12, 1.12], rotateFrom: 12, rotateOvershoot: -1, blurFrom: 8 },
            'panel-reveal-primary': { from: [0, 0.16], overshoot: [0, -0.01], scaleFrom: [1, 0.72], scaleOvershoot: [1, 1.03], scaleSettle: [1, 1], blurFrom: 6, revealOrigin: 'bottom', revealFrom: 0 },
            'panel-reveal-secondary': { from: [0.18, 0.02], overshoot: [-0.02, 0], scaleFrom: [0.88, 0.88], scaleOvershoot: [1.03, 1.03], rotateFrom: 4, rotateOvershoot: -1, blurFrom: 5, revealOrigin: 'right', revealFrom: 0 },
            'panel-reveal-tertiary': { from: [0, 0.22], overshoot: [0, -0.01], scaleFrom: [0.74, 0.82], scaleOvershoot: [1.03, 1.02], rotateFrom: -3, blurFrom: 6, revealOrigin: 'center-y', revealFrom: 0 },
            'panel-unfold-center': { from: [0, 0.1], overshoot: [0, -0.01], scaleFrom: [0.18, 0.78], scaleOvershoot: [1.04, 1.02], blurFrom: 7, revealOrigin: 'center-x', revealFrom: 0 },
            'panel-sweep-left': { from: [-0.22, 0.04], overshoot: [0.01, 0], scaleFrom: [0.86, 0.9], scaleOvershoot: [1.02, 1.02], rotateFrom: -4, blurFrom: 5, revealOrigin: 'left', revealFrom: 0 },
            'panel-sweep-right': { from: [0.22, 0.04], overshoot: [-0.01, 0], scaleFrom: [0.86, 0.9], scaleOvershoot: [1.02, 1.02], rotateFrom: 4, blurFrom: 5, revealOrigin: 'right', revealFrom: 0 },
            'panel-pop-tilt': { from: [0, 0.32], overshoot: [0, -0.03], scaleFrom: [0.3, 0.3], scaleOvershoot: [1.16, 1.16], rotateFrom: -18, rotateOvershoot: 5, blurFrom: 8, revealOrigin: 'center', revealFrom: 0.08 },
            'headline-establish': { from: [0, 0.24], overshoot: [0, -0.03], scaleFrom: [0.8, 0.8], scaleOvershoot: [1.06, 1.06], rotateFrom: -6, rotateOvershoot: 1, blurFrom: 10, glowPeak: 0.22, revealFrom: 0.1 },
            'headline-slam-left': { from: [-0.64, 0.3], overshoot: [0.04, -0.02], scaleFrom: [0.66, 0.66], scaleOvershoot: [1.08, 1.08], rotateFrom: -14, rotateOvershoot: 2, blurFrom: 12, glowPeak: 0.24, revealFrom: 0.08 },
            'headline-slam-right': { from: [0.64, 0.3], overshoot: [-0.04, -0.02], scaleFrom: [0.66, 0.66], scaleOvershoot: [1.08, 1.08], rotateFrom: 14, rotateOvershoot: -2, blurFrom: 12, glowPeak: 0.24, revealFrom: 0.08 },
            'copy-settle': { from: [-0.22, 0.03], overshoot: [0.01, 0], scaleFrom: [0.9, 0.9], scaleOvershoot: [1.02, 1.02], blurFrom: 6, glowPeak: 0.16, revealFrom: 0.15 },
            'copy-slide-left': { from: [-0.34, 0.02], overshoot: [0.02, 0], scaleFrom: [0.92, 0.92], scaleOvershoot: [1.02, 1.02], rotateFrom: -2, blurFrom: 6, glowPeak: 0.16, revealFrom: 0.12 },
            'copy-slide-right': { from: [0.34, 0.02], overshoot: [-0.02, 0], scaleFrom: [0.92, 0.92], scaleOvershoot: [1.02, 1.02], rotateFrom: 2, blurFrom: 6, glowPeak: 0.16, revealFrom: 0.12 },
            'text-roll-in': { from: [0, 0.34], overshoot: [0, -0.03], scaleFrom: [0.94, 0.34], scaleOvershoot: [1.04, 1.08], rotateFrom: -10, rotateOvershoot: 2, blurFrom: 10, glowPeak: 0.2, revealFrom: 0.06 },
            'text-bounce-in': { from: [0, 0.28], overshoot: [0, -0.03], scaleFrom: [0.16, 0.16], scaleOvershoot: [1.24, 1.24], rotateFrom: -16, rotateOvershoot: 5, blurFrom: 8, glowPeak: 0.2, revealFrom: 0.08 },
            'caption-rise': { from: [0, 0.24], overshoot: [0, -0.02], scaleFrom: [0.92, 0.92], scaleOvershoot: [1.03, 1.03], blurFrom: 6, glowPeak: 0.16, revealFrom: 0.12 },
            'logo-establish': { from: [0, -0.08], overshoot: [0, 0], scaleFrom: [0.92, 0.92], scaleOvershoot: [1.02, 1.02], blurFrom: 4, glowPeak: 0.18, revealFrom: 0.18 },
            'logo-soft-fade': { from: [0, -0.05], overshoot: [0, 0], scaleFrom: [0.96, 0.96], scaleOvershoot: [1.01, 1.01], blurFrom: 4, glowPeak: 0.16, revealFrom: 0.24 },
            'logo-drop-in': { from: [0, -0.64], overshoot: [0, 0.03], scaleFrom: [0.42, 0.42], scaleOvershoot: [1.1, 1.1], rotateFrom: -12, rotateOvershoot: 1, blurFrom: 8, glowPeak: 0.18, revealFrom: 0.08 },
            'price-emphasis': { from: [0, 0.08], overshoot: [0, -0.01], scaleFrom: [0.56, 0.56], scaleOvershoot: [1.14, 1.14], rotateFrom: -8, rotateOvershoot: 1, blurFrom: 6, glowPeak: 0.34, revealFrom: 0.08 },
            'price-emphasis-late': { from: [0, 0.12], overshoot: [0, -0.01], scaleFrom: [0.48, 0.48], scaleOvershoot: [1.16, 1.16], rotateFrom: -10, rotateOvershoot: 1, blurFrom: 6, glowPeak: 0.38, revealFrom: 0.06 },
            'price-stamp-pop': { from: [0, 0.18], overshoot: [0, -0.02], scaleFrom: [0.18, 0.18], scaleOvershoot: [1.22, 1.22], rotateFrom: -22, rotateOvershoot: 2, blurFrom: 8, glowPeak: 0.4, revealFrom: 0.04 },
            'price-flip-pop': { from: [0, 0.18], overshoot: [0, -0.02], scaleFrom: [0.32, 0.32], scaleOvershoot: [1.18, 1.18], rotateFrom: -18, rotateOvershoot: 2, blurFrom: 8, glowPeak: 0.4, revealFrom: 0.04 },
            'decor-orbit-soft': { from: [-0.22, -0.18], overshoot: [0.02, 0.01], scaleFrom: [0.82, 0.82], scaleOvershoot: [1.03, 1.03], rotateFrom: -10, rotateOvershoot: 1, blurFrom: 6, glowPeak: 0.28 },
            'decor-flyby-left': { from: [-1.2, -0.1], overshoot: [0.06, 0.02], scaleFrom: [0.7, 0.7], scaleOvershoot: [1.08, 1.08], rotateFrom: -24, rotateOvershoot: 4, blurFrom: 12, glowPeak: 0.32 },
            'decor-flyby-right': { from: [1.2, -0.1], overshoot: [-0.06, 0.02], scaleFrom: [0.7, 0.7], scaleOvershoot: [1.08, 1.08], rotateFrom: 24, rotateOvershoot: -4, blurFrom: 12, glowPeak: 0.32 },
            'decor-glint-early': { from: [0, -0.08], overshoot: [0, 0], scaleFrom: [0.94, 0.94], scaleOvershoot: [1.02, 1.02], blurFrom: 5, glowPeak: 0.32, revealFrom: 0.24 },
            'decor-glint-late': { from: [0, -0.06], overshoot: [0, 0], scaleFrom: [0.96, 0.96], scaleOvershoot: [1.01, 1.01], blurFrom: 4, glowPeak: 0.3, revealFrom: 0.24 },
            'soft-settle': { from: [0, 0.06], overshoot: [0, 0], scaleFrom: [0.94, 0.94], scaleOvershoot: [1.01, 1.01], blurFrom: 4, glowPeak: 0.14, revealFrom: 0.2 }
        };

        return {
            ...base,
            ...(overrides[presetName] || {})
        };
    };

    const getLayerStageWindow = (entry) => {
        const role = String(entry?.role || 'unknown');
        const slot = Number(entry?.sequenceSlot || 0);
        const areaRatio = Math.max(0, getLayerAreaScore(entry?.bbox)) / 1000000;
        const centerY = getLayerCenterY(entry?.bbox) / 1000;
        const isBottom = centerY >= 0.68;
        const heroRank = Number.isFinite(Number(entry?.heroRank)) ? Number(entry.heroRank) : -1;
        const isFeaturedProduct = role === 'product' && heroRank >= 0 && heroRank <= 1;

        if (role === 'panel') {
            const panelSlot = Math.min(slot, 2);
            return {
                start: 0.04 + panelSlot * 0.026,
                peak: 0.14 + panelSlot * 0.032,
                settle: 0.26 + panelSlot * 0.034,
                release: 0.94
            };
        }

        if (role === 'product') {
            if (heroRank === 0) {
                return { start: 0.24, peak: 0.38, settle: 0.54, release: 0.92 };
            }
            if (heroRank === 1) {
                return { start: 0.32, peak: 0.46, settle: 0.6, release: 0.92 };
            }
            if (isBottom || areaRatio <= 0.05) {
                return {
                    start: 0.5 + Math.min(slot, 3) * 0.022,
                    peak: 0.62 + Math.min(slot, 3) * 0.022,
                    settle: 0.72 + Math.min(slot, 3) * 0.016,
                    release: 0.92
                };
            }
            return {
                start: 0.42 + Math.min(slot, 3) * 0.03,
                peak: 0.56 + Math.min(slot, 3) * 0.03,
                settle: 0.68 + Math.min(slot, 3) * 0.02,
                release: 0.92
            };
        }

        if (role === 'text') {
            return {
                start: 0.62 + Math.min(slot, 4) * 0.016,
                peak: 0.74 + Math.min(slot, 4) * 0.016,
                settle: 0.84 + Math.min(slot, 4) * 0.012,
                release: 0.95
            };
        }

        if (role === 'logo') {
            return {
                start: 0.58 + Math.min(slot, 2) * 0.014,
                peak: 0.7 + Math.min(slot, 2) * 0.014,
                settle: 0.8 + Math.min(slot, 2) * 0.01,
                release: 0.94
            };
        }

        if (role === 'price') {
            return {
                start: 0.74 + Math.min(slot, 4) * 0.012,
                peak: 0.84 + Math.min(slot, 4) * 0.012,
                settle: 0.9 + Math.min(slot, 4) * 0.008,
                release: 0.95
            };
        }

        if (role === 'decoration') {
            return {
                start: 0.68 + Math.min(slot, 3) * 0.018,
                peak: 0.8 + Math.min(slot, 3) * 0.018,
                settle: 0.9 + Math.min(slot, 3) * 0.012,
                release: 0.95
            };
        }

        return {
            start: isFeaturedProduct ? 0.32 : 0.3,
            peak: isFeaturedProduct ? 0.46 : 0.46,
            settle: isFeaturedProduct ? 0.62 : 0.6,
            release: 0.92
        };
    };

    const buildMotionTracksForLayer = (entry, recipeDurationMs) => {
        const preset = resolveMotionPreset(entry?.entryMotion, entry?.role);
        const { start: rawStart, peak: rawPeak, settle: rawSettle, release: rawRelease } = getLayerStageWindow(entry);

        let start = rawStart;
        let peak = rawPeak;
        let settle = rawSettle;
        let release = rawRelease;

        start = clamp01(start);
        peak = clamp01(Math.max(start + 0.02, peak));
        settle = clamp01(Math.max(peak + 0.03, settle));
        release = clamp01(Math.max(settle + 0.08, release));
        const revealEase = entry?.role === 'panel' ? 'easeOutCubic' : 'easeOutQuint';

        return {
            position: buildMotionTrack([
                { t: 0, v: preset.from },
                { t: start, v: preset.from },
                { t: peak, v: preset.overshoot, ease: 'easeOutBack' },
                { t: settle, v: preset.settle, ease: 'easeInOutCubic' },
                { t: release, v: [0, 0], ease: 'easeOutCubic' },
                { t: 1, v: [0, 0] }
            ]),
            scale: buildMotionTrack([
                { t: 0, v: preset.scaleFrom },
                { t: start, v: preset.scaleFrom },
                { t: peak, v: preset.scaleOvershoot, ease: 'easeOutBack' },
                { t: settle, v: preset.scaleSettle || [0.992, 0.992], ease: 'easeInOutCubic' },
                { t: release, v: [1, 1], ease: 'easeOutCubic' },
                { t: 1, v: [1, 1] }
            ]),
            rotate: buildMotionTrack([
                { t: 0, v: preset.rotateFrom },
                { t: start, v: preset.rotateFrom },
                { t: peak, v: preset.rotateOvershoot, ease: 'easeOutBack' },
                { t: settle, v: preset.rotateSettle || 0, ease: 'easeInOutCubic' },
                { t: release, v: 0, ease: 'easeOutCubic' },
                { t: 1, v: 0 }
            ]),
            opacity: buildMotionTrack([
                { t: 0, v: 0 },
                { t: Math.max(0, start - 0.012), v: 0 },
                { t: Math.max(start + 0.03, peak - 0.04), v: entry?.role === 'panel' ? 0.92 : 0.86, ease: 'easeOutCubic' },
                { t: peak, v: 1, ease: 'easeOutCubic' },
                { t: 1, v: 1 }
            ]),
            blur: buildMotionTrack([
                { t: 0, v: preset.blurFrom },
                { t: start, v: preset.blurFrom },
                { t: peak, v: 0.8, ease: 'easeOutCubic' },
                { t: settle, v: 0, ease: 'easeInOutCubic' },
                { t: 1, v: 0 }
            ]),
            glow: buildMotionTrack([
                { t: 0, v: preset.glowFrom || 0.14 },
                { t: start, v: preset.glowFrom || 0.14 },
                { t: peak, v: preset.glowPeak || 0.24, ease: 'easeOutCubic' },
                { t: settle, v: 0.08, ease: 'easeInOutCubic' },
                { t: 1, v: 0 }
            ]),
            reveal: buildMotionTrack([
                { t: 0, v: preset.revealFrom ?? 0.08 },
                { t: start, v: preset.revealFrom ?? 0.08 },
                { t: peak, v: 0.92, ease: revealEase },
                { t: settle, v: 1, ease: 'easeOutCubic' },
                { t: 1, v: 1 }
            ]),
            revealOrigin: preset.revealOrigin || 'center'
        };
    };

    const sanitizeMotionTextStyle = (styleText = '', options = {}) => {
        const preserveRuntimeLayout = options?.preserveRuntimeLayout === true;
        let sanitized = String(styleText || '')
            .replace(/(^|;)\s*right\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*bottom\s*:[^;]*/gi, '');

        if (!preserveRuntimeLayout) {
            sanitized = sanitized
                .replace(/(^|;)\s*overflow\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*min-width\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*min-height\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*max-width\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*max-height\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*word-break\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*overflow-wrap\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*left\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*top\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*width\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*height\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*position\s*:[^;]*/gi, '')
                .replace(/(^|;)\s*transform\s*:[^;]*/gi, '');
        }

        return sanitized;
    };

    const buildComputedStyleText = (el, propertyNames = []) => {
        if (typeof window === 'undefined' || !el || !Array.isArray(propertyNames) || propertyNames.length === 0) {
            return '';
        }

        const computed = window.getComputedStyle(el);
        return propertyNames.map((propertyName) => {
            const value = computed.getPropertyValue(propertyName);
            if (!value) return '';
            return `${propertyName}:${value.trim()}`;
        }).filter(Boolean).join(';');
    };

    const getMotionTextLayoutDiagnostics = (contentEl) => {
        if (typeof window === 'undefined' || !contentEl) {
            return {
                renderedLineCount: 1,
                explicitLineCount: 1,
                isAutoWrappedText: false,
                hasExplicitBreaks: false
            };
        }

        const rawText = String(contentEl.textContent || '');
        const htmlText = String(contentEl.innerHTML || '');
        const hasExplicitBreaks = /<br\s*\/?>/i.test(htmlText) || rawText.includes('\n');
        const explicitLineCount = Math.max(1, rawText.split('\n').length);
        const computed = window.getComputedStyle(contentEl);
        const lineHeightPx = parseCssSizePx(computed.lineHeight, 0);
        const measuredHeight = Math.max(
            contentEl.scrollHeight || 0,
            contentEl.offsetHeight || 0,
            Math.round(contentEl.getBoundingClientRect?.().height || 0)
        );
        const renderedLineCount = lineHeightPx > 0
            ? Math.max(1, Math.round(measuredHeight / Math.max(1, lineHeightPx)))
            : explicitLineCount;
        const isAutoWrappedText = !hasExplicitBreaks && renderedLineCount > 1;

        return {
            renderedLineCount,
            explicitLineCount,
            isAutoWrappedText,
            hasExplicitBreaks
        };
    };

    const measureNaturalSingleLineTextWidth = (contentEl) => {
        if (typeof document === 'undefined' || !contentEl) return 0;
        const clone = contentEl.cloneNode(true);
        clone.style.position = 'absolute';
        clone.style.visibility = 'hidden';
        clone.style.pointerEvents = 'none';
        clone.style.left = '-99999px';
        clone.style.top = '-99999px';
        clone.style.width = 'auto';
        clone.style.maxWidth = 'none';
        clone.style.minWidth = '0';
        clone.style.height = 'auto';
        clone.style.whiteSpace = 'nowrap';
        clone.style.transform = 'none';
        clone.style.webkitTransform = 'none';
        clone.style.overflow = 'visible';
        clone.style.display = 'inline-block';
        document.body.appendChild(clone);
        const width = Math.max(
            clone.scrollWidth || 0,
            clone.offsetWidth || 0,
            Math.round(clone.getBoundingClientRect?.().width || 0)
        );
        clone.remove();
        return width;
    };

    const resolveAutoWrappedTextDiagnostics = (childItem, contentEl) => {
        const base = getMotionTextLayoutDiagnostics(contentEl);
        if (typeof window === 'undefined' || !childItem?.el || !contentEl) {
            return {
                ...base,
                wrapperWidthPx: 0,
                naturalSingleLineWidthPx: 0
            };
        }

        const wrapperWidthPx = Math.max(
            childItem.el.clientWidth || 0,
            parseWorkbenchCssPx(childItem.el.style.width),
            Math.round(childItem.el.getBoundingClientRect?.().width || 0)
        );
        const naturalSingleLineWidthPx = measureNaturalSingleLineTextWidth(contentEl);
        const computed = window.getComputedStyle(contentEl);
        const canWrap = computed.whiteSpace !== 'nowrap';
        const inferredAutoWrap = !base.hasExplicitBreaks &&
            canWrap &&
            wrapperWidthPx > 0 &&
            naturalSingleLineWidthPx > wrapperWidthPx * 1.08;
        const lineHeightPx = parseCssSizePx(computed.lineHeight, 0);
        const inferredRenderedLines = inferredAutoWrap && lineHeightPx > 0
            ? Math.max(
                2,
                Math.min(
                    6,
                    Math.round(
                        Math.max(
                            contentEl.scrollHeight || 0,
                            contentEl.offsetHeight || 0,
                            Math.round(contentEl.getBoundingClientRect?.().height || 0)
                        ) / Math.max(1, lineHeightPx)
                    )
                )
            )
            : base.renderedLineCount;

        return {
            ...base,
            renderedLineCount: inferredAutoWrap ? inferredRenderedLines : base.renderedLineCount,
            isAutoWrappedText: inferredAutoWrap || base.isAutoWrappedText,
            wrapperWidthPx,
            naturalSingleLineWidthPx
        };
    };

    const buildExactTextSnapshotMarkup = ({
        pixelWidth,
        pixelHeight,
        safePadding,
        contentHtml,
        styleText,
        wrapperStyle = '',
        contentFillMode = 'natural',
        preserveRuntimeLayout = false
    }) => {
        const snapshotWidth = pixelWidth + safePadding * 2;
        const snapshotHeight = pixelHeight + safePadding * 2;
        const exactStyle = sanitizeMotionTextStyle(styleText, { preserveRuntimeLayout });
        const fillStyle = contentFillMode === 'fill'
            ? 'display:block;width:100%;max-width:100%;height:auto;overflow:visible;box-sizing:border-box;'
            : 'box-sizing:border-box;overflow:visible;';
        return `
            <svg xmlns="http://www.w3.org/2000/svg" width="${snapshotWidth}" height="${snapshotHeight}" viewBox="0 0 ${snapshotWidth} ${snapshotHeight}">
                <foreignObject x="0" y="0" width="100%" height="100%">
                    <div xmlns="http://www.w3.org/1999/xhtml" style="width:${snapshotWidth}px;height:${snapshotHeight}px;padding:${safePadding}px;overflow:visible;box-sizing:border-box;background:transparent;">
                        <div style="position:relative;width:${pixelWidth}px;height:${pixelHeight}px;overflow:visible;box-sizing:border-box;${wrapperStyle}">
                            <div style="${fillStyle}${exactStyle}">
                                ${contentHtml}
                            </div>
                        </div>
                    </div>
                </foreignObject>
            </svg>
        `;
    };

    const buildExactTextFragmentBlockMarkup = ({
        left,
        top,
        width,
        height,
        contentHtml,
        styleText,
        preserveRuntimeLayout = true
    }) => {
        const exactStyle = sanitizeMotionTextStyle(styleText, { preserveRuntimeLayout });
        return `
            <div style="position:absolute;left:${left}px;top:${top}px;width:${width}px;height:${height}px;display:flex;align-items:center;justify-content:center;overflow:visible;box-sizing:border-box;">
                <div style="box-sizing:border-box;overflow:visible;${exactStyle}">
                    ${contentHtml}
                </div>
            </div>
        `;
    };

    const buildTextLayerSnapshotDataUrl = (entry, pixelWidth, pixelHeight) => {
        const groupFragments = Array.isArray(entry?.textFragments) ? entry.textFragments.filter(Boolean) : [];
        const isExactRuntimeText = /text_note_(child|group)/.test(String(entry?.runtimeType || ''));
        if (groupFragments.length > 1 && Array.isArray(entry?.bbox) && entry.bbox.length === 4) {
            const safePadding = Math.max(10, Math.round(Math.min(pixelWidth, pixelHeight) * 0.08));
            const snapshotWidth = pixelWidth + safePadding * 2;
            const snapshotHeight = pixelHeight + safePadding * 2;
            const [groupY1, groupX1, groupY2, groupX2] = entry.bbox.map(Number);
            const groupNormWidth = Math.max(1, groupX2 - groupX1);
            const groupNormHeight = Math.max(1, groupY2 - groupY1);
            const fragmentMarkup = groupFragments.map((fragment) => {
                const fragmentBbox = Array.isArray(fragment?.bbox) && fragment.bbox.length === 4 ? fragment.bbox.map(Number) : null;
                if (!fragmentBbox) return '';
                const [y1, x1, y2, x2] = fragmentBbox;
                const left = ((x1 - groupX1) / groupNormWidth) * pixelWidth;
                const top = ((y1 - groupY1) / groupNormHeight) * pixelHeight;
                const width = Math.max(1, ((x2 - x1) / groupNormWidth) * pixelWidth);
                const height = Math.max(1, ((y2 - y1) / groupNormHeight) * pixelHeight);
                const textHtml = String(fragment?.textHtml || '').trim();
                const textContent = textHtml || escapeHtml(String(fragment?.textContent || '')).replace(/\n/g, '<br>');
                if (!textContent) return '';

                return buildExactTextFragmentBlockMarkup({
                    left,
                    top,
                    width,
                    height,
                    contentHtml: textContent,
                    styleText: fragment?.textStyleText || '',
                    preserveRuntimeLayout: true
                });
            }).filter(Boolean).join('');

            if (fragmentMarkup) {
                const svgMarkup = `
                    <svg xmlns="http://www.w3.org/2000/svg" width="${snapshotWidth}" height="${snapshotHeight}" viewBox="0 0 ${snapshotWidth} ${snapshotHeight}">
                        <foreignObject x="0" y="0" width="100%" height="100%">
                            <div xmlns="http://www.w3.org/1999/xhtml" style="width:${snapshotWidth}px;height:${snapshotHeight}px;padding:${safePadding}px;overflow:visible;box-sizing:border-box;background:transparent;position:relative;">
                                <div style="position:relative;width:${pixelWidth}px;height:${pixelHeight}px;overflow:visible;box-sizing:border-box;">
                                    ${fragmentMarkup}
                                </div>
                            </div>
                        </foreignObject>
                    </svg>
                `;
                return svgDataUrlFromMarkup(svgMarkup);
            }
        }

        const textHtml = String(entry?.textHtml || '').trim();
        const textContent = textHtml || escapeHtml(String(entry?.text || '')).replace(/\n/g, '<br>');
        if (!textContent) return '';

        const safePadding = Math.max(
            10,
            Math.round(
                Math.min(pixelWidth, pixelHeight) * (
                    entry?.isAutoWrappedText || Number(entry?.renderedLineCount || 1) > 1
                        ? 0.18
                        : 0.08
                )
            )
        );
        if (isExactRuntimeText && entry?.textStyleText) {
            return svgDataUrlFromMarkup(buildExactTextSnapshotMarkup({
                pixelWidth,
                pixelHeight,
                safePadding,
                contentHtml: textContent,
                styleText: entry.textSnapshotStyleText || entry.textStyleText,
                wrapperStyle: entry.textSnapshotWrapperStyleText || '',
                contentFillMode: entry?.isAutoWrappedText ? 'fill' : 'natural',
                preserveRuntimeLayout: true
            }));
        }

        const snapshotWidth = pixelWidth + safePadding * 2;
        const snapshotHeight = pixelHeight + safePadding * 2;
        const textPlain = stripHtmlToText(entry?.textHtml || entry?.text || '');
        const textLines = textPlain.split('\n').map(line => line.trim()).filter(Boolean);
        const longestLineLength = Math.max(1, ...textLines.map(line => line.length || 0));
        const lineCount = Math.max(1, textLines.length);
        const parsedLineHeight = Math.max(0.82, parseCssSizePx(entry?.lineHeight, 0.96));
        const baseFontSizePx = parseCssSizePx(entry?.fontSize, Math.max(10, Math.min(72, Math.round(pixelHeight * 0.52))));
        const avgCharWidthFactor = /impact|arial black|oswald|bebas/i.test(String(entry?.fontFamily || ''))
            ? 0.68
            : 0.6;
        const fitWidthFontSizePx = Math.max(8, (pixelWidth * 0.96) / Math.max(1, longestLineLength * avgCharWidthFactor));
        const fitHeightFontSizePx = Math.max(8, (pixelHeight * 0.94) / Math.max(1, lineCount * parsedLineHeight));
        const resolvedFontSizePx = Math.max(8, Math.min(baseFontSizePx, fitWidthFontSizePx, fitHeightFontSizePx));
        const fontSize = `${resolvedFontSizePx}px`;
        const color = entry?.fontColor || '#ffffff';
        const fontFamily = entry?.fontFamily || 'Impact, Arial Black, sans-serif';
        const fontWeight = entry?.fontWeight || '800';
        const fontStyle = entry?.fontStyle || 'normal';
        const lineHeight = entry?.lineHeight || String(parsedLineHeight);
        const letterSpacing = entry?.letterSpacing || '0';
        const textAlign = entry?.textAlign || 'center';
        const textShadow = entry?.textShadow || '0 1px 2px rgba(0,0,0,0.18)';
        const stroke = entry?.WebkitTextStroke ? `-webkit-text-stroke:${entry.WebkitTextStroke};` : '';
        const whiteSpace = entry?.whiteSpace || 'pre-wrap';
        const visualTransform = entry?.textVisualTransform ? `transform:${entry.textVisualTransform};transform-origin:center center;` : '';
        const styleText = sanitizeMotionTextStyle(entry?.textStyleText || '');

        const svgMarkup = `
            <svg xmlns="http://www.w3.org/2000/svg" width="${snapshotWidth}" height="${snapshotHeight}" viewBox="0 0 ${snapshotWidth} ${snapshotHeight}">
                <foreignObject x="0" y="0" width="100%" height="100%">
                    <div xmlns="http://www.w3.org/1999/xhtml" style="width:${snapshotWidth}px;height:${snapshotHeight}px;padding:${safePadding}px;display:flex;align-items:center;justify-content:center;overflow:visible;box-sizing:border-box;background:transparent;">
                        <div style="width:${pixelWidth}px;height:${pixelHeight}px;display:flex;align-items:center;justify-content:center;text-align:${textAlign};color:${color};font-family:${fontFamily};font-weight:${fontWeight};font-style:${fontStyle};font-size:${fontSize};line-height:${lineHeight};letter-spacing:${letterSpacing};white-space:${whiteSpace};text-shadow:${textShadow};overflow:visible;box-sizing:border-box;${stroke}${visualTransform}${styleText}">
                            ${textContent}
                        </div>
                    </div>
                </foreignObject>
            </svg>
        `;

        return svgDataUrlFromMarkup(svgMarkup);
    };

    const buildShapeLayerSnapshotDataUrl = (entry, pixelWidth, pixelHeight) => {
        const shapeStyleText = String(entry?.shapeStyleText || '')
            .replace(/(^|;)\s*left\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*top\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*width\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*height\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*z-index\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*position\s*:[^;]*/gi, '');
        const clipPath = entry?.clipPath ? `clip-path:${entry.clipPath};-webkit-clip-path:${entry.clipPath};` : '';
        const svgMarkup = `
            <svg xmlns="http://www.w3.org/2000/svg" width="${pixelWidth}" height="${pixelHeight}" viewBox="0 0 ${pixelWidth} ${pixelHeight}">
                <foreignObject x="0" y="0" width="100%" height="100%">
                    <div xmlns="http://www.w3.org/1999/xhtml" style="width:${pixelWidth}px;height:${pixelHeight}px;box-sizing:border-box;overflow:hidden;background:transparent;">
                        <div style="width:100%;height:100%;box-sizing:border-box;${clipPath}${shapeStyleText}"></div>
                    </div>
                </foreignObject>
            </svg>
        `;
        return svgDataUrlFromMarkup(svgMarkup);
    };

    const getMotionStageSize = (size, fallbackWidth = 900, fallbackHeight = 1200) => {
        const width = Number(size?.width) || fallbackWidth;
        const height = Number(size?.height) || fallbackHeight;
        const longest = Math.max(width, height);
        if (longest <= 960) {
            return { width, height };
        }
        const ratio = 960 / longest;
        return {
            width: Math.round(width * ratio),
            height: Math.round(height * ratio)
        };
    };

    const buildCanvasMotionLayerPayload = (entry, recipeDurationMs, fallbackSliceSourceUrl, stageSize = null) => {
        const bbox = Array.isArray(entry?.bbox) && entry.bbox.length === 4 ? entry.bbox.map(Number) : [0, 0, 1000, 1000];
        const pixelWidth = Math.max(8, Math.round((((bbox[3] - bbox[1]) / 1000) * (stageSize?.width || 1000))));
        const pixelHeight = Math.max(8, Math.round((((bbox[2] - bbox[0]) / 1000) * (stageSize?.height || 1000))));
        const semanticType = String(entry?.semanticType || '').toLowerCase();
        const designRole = String(entry?.designRole || '').toLowerCase();
        const centerY = getLayerCenterY(bbox);
        const areaScore = getLayerAreaScore(bbox);
        const isSemanticVectorPanel =
            entry?.renderMode === 'vector_shape' &&
            (
                entry?.role === 'panel' ||
                semanticType === 'shape_panel' ||
                semanticType === 'price_badge' ||
                designRole === 'local_panel' ||
                designRole === 'price_badge'
            );
        const isRuntimeText =
            ['text', 'price', 'logo'].includes(entry?.role) &&
            /text_note_(child|group)/i.test(String(entry?.runtimeType || ''));
        // Runtime text must stay on the canvas path. SVG foreignObject
        // snapshots are browser-dependent and were the reason otherwise
        // valid NEW/COLA layers disappeared while multiline text was being
        // repaired. Semantic text that is not a runtime note keeps its
        // existing snapshot fallback.
        const preferCanvasText = isRuntimeText;
        const textSnapshotUrl = ['text', 'price', 'logo'].includes(entry?.role)
            ? (preferCanvasText ? '' : buildTextLayerSnapshotDataUrl(entry, pixelWidth, pixelHeight))
            : '';
        const shapeSnapshotUrl = entry?.runtimeType === 'shape_node'
            ? buildShapeLayerSnapshotDataUrl(entry, pixelWidth, pixelHeight)
            : '';
        const resolvedImageUrl = entry?.imageUrl
            ? (getProxiedUrl(entry.imageUrl) || entry.imageUrl)
            : (textSnapshotUrl || shapeSnapshotUrl || '');
        const kind = resolvedImageUrl
            ? 'image'
            : (entry?.role === 'text' || entry?.role === 'price' || entry?.role === 'logo')
                ? 'text'
            : (entry?.runtimeType === 'shape_node' || isSemanticVectorPanel)
                ? 'shape'
                    : fallbackSliceSourceUrl
                        ? 'slice'
                        : 'unknown';

        const inferredPanelClipPath = isSemanticVectorPanel && !String(entry?.clipPath || '').trim()
            ? (
                areaScore >= 90000 && centerY < 470
                    ? 'polygon(0% 10%, 100% 10%, 100% 72%, 0% 100%)'
                    : areaScore >= 90000
                        ? 'polygon(0% 0%, 100% 18%, 100% 100%, 0% 100%)'
                        : ''
            )
            : '';
        const inferredPanelStyleText = isSemanticVectorPanel && !String(entry?.shapeStyleText || '').trim()
            ? (
                semanticType === 'price_badge' || designRole === 'price_badge' || entry?.shapeType === 'ellipse'
                    ? `background:${entry?.fillColor || '#f59e0b'};border-radius:999px;`
                    : getLayerAreaScore(bbox) <= 50000
                        ? `background:${entry?.fillColor || 'rgba(255,248,235,0.98)'};border-radius:${entry?.borderRadius || '18px'};`
                        : `background:${entry?.fillColor || 'rgba(255,255,255,0.96)'};border-radius:${entry?.borderRadius || '22px'};`
            )
            : '';

        // Recipe text has already been normalized from the Magic Layers
        // semantic layer. Runtime notes may have DOM HTML containing only the
        // numeric part (for example `8`) while the atomic layer text is `$8`.
        // Do not let that HTML overwrite the authoritative recipe value.
        const textSource = isRuntimeText
            ? (entry?.text || entry?.textHtml || '')
            : (entry?.textHtml || entry?.text || '');
        const textValue = decodeHtmlEntities(stripHtmlToText(textSource));
        const textSnapshotPaddingPx = entry?.role === 'price'
            ? 0
            : ['text', 'price', 'logo'].includes(entry?.role)
            ? Math.max(
                10,
                Math.round(
                    Math.min(pixelWidth, pixelHeight) * (
                        entry?.isAutoWrappedText || Number(entry?.renderedLineCount || 1) > 1
                            ? 0.18
                            : 0.08
                    )
                )
            )
            : 0;
        const roleRenderBoost = entry?.role === 'logo'
            ? 4000
            : entry?.role === 'price'
                ? 3600
                : entry?.role === 'text'
                    ? 3400
                    : entry?.role === 'product'
                        ? 2200
                        : entry?.role === 'decoration'
                            ? 1800
                            : entry?.role === 'panel'
                                ? 1200
                                : 1000;
        return {
            id: entry?.id || `motion-layer-${Math.random().toString(36).slice(2, 8)}`,
            name: entry?.name || 'unnamed',
            role: entry?.role || 'unknown',
            kind,
            bbox,
            zIndex: Number(entry?.zIndex || 0),
            renderOrder: roleRenderBoost + Number(entry?.zIndex || 0),
            imageUrl: resolvedImageUrl,
            sliceSourceUrl: kind === 'slice' ? (getProxiedUrl(fallbackSliceSourceUrl) || fallbackSliceSourceUrl) : '',
            text: textValue,
            textAlign: entry?.textAlign || 'center',
            textColor: isVisibleMotionColor(entry?.fontColor)
                ? entry.fontColor
                : (isVisibleMotionColor(entry?.css?.color) ? entry.css.color : '#ffffff'),
            fontFamily: entry?.fontFamily || entry?.css?.fontFamily || 'Impact, Arial Black, sans-serif',
            fontWeight: entry?.fontWeight || entry?.css?.fontWeight || '800',
            fontStyle: entry?.fontStyle || entry?.css?.fontStyle || 'normal',
            fontSize: entry?.fontSize || '',
            lineHeight: parseCssSizePx(entry?.lineHeight, 1.0),
            letterSpacing: parseCssSizePx(entry?.letterSpacing, 0),
            textShadow: entry?.textShadow || '',
            textStroke: entry?.WebkitTextStroke || '',
            shapeStyleText: entry?.shapeStyleText || inferredPanelStyleText || '',
            clipPath: entry?.clipPath || inferredPanelClipPath || '',
            snapshotPaddingPx: textSnapshotPaddingPx,
            renderedLineCount: Number(entry?.renderedLineCount || 1),
            explicitLineCount: Number(entry?.explicitLineCount || 1),
            isAutoWrappedText: entry?.isAutoWrappedText === true,
            tracks: buildMotionTracksForLayer(entry, recipeDurationMs)
        };
    };

    const motionPreviewImageCache = new Map();

    const isUploadableMotionRasterUrl = (src) => {
        if (typeof src !== 'string') return false;
        if (src.startsWith('blob:')) return true;
        return /^data:image\/(?:png|jpe?g|webp|gif|avif);base64,/i.test(src);
    };

    const loadMotionPreviewImage = (src) => {
        if (!src) return Promise.resolve(null);
        if (motionPreviewImageCache.has(src)) return motionPreviewImageCache.get(src);
        const promise = new Promise(resolve => {
            const image = new Image();
            // Local data/blob snapshots already belong to this document. A
            // crossOrigin attribute on them can make SVG/text snapshots fail
            // to load even though remote OSS images need anonymous CORS.
            if (!src.startsWith('data:') && !src.startsWith('blob:')) {
                image.crossOrigin = 'anonymous';
            }
            image.onload = () => resolve(image);
            image.onerror = () => resolve(null);
            image.src = src;
        });
        motionPreviewImageCache.set(src, promise);
        return promise;
    };

    const drawRoundedRectPath = (ctx, x, y, width, height, radius = 0) => {
        const r = Math.max(0, Math.min(Number(radius) || 0, width / 2, height / 2));
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + width - r, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + r);
        ctx.lineTo(x + width, y + height - r);
        ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
        ctx.lineTo(x + r, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    };

    const applyRevealClip = (ctx, width, height, progress, origin = 'center') => {
        const reveal = clamp01(progress);
        if (reveal >= 0.999) return;
        const w = Math.max(1, width);
        const h = Math.max(1, height);
        if (origin === 'left') {
            ctx.beginPath();
            ctx.rect(0, 0, w * reveal, h);
            ctx.clip();
            return;
        }
        if (origin === 'right') {
            ctx.beginPath();
            ctx.rect(w * (1 - reveal), 0, w * reveal, h);
            ctx.clip();
            return;
        }
        if (origin === 'top') {
            ctx.beginPath();
            ctx.rect(0, 0, w, h * reveal);
            ctx.clip();
            return;
        }
        if (origin === 'bottom') {
            ctx.beginPath();
            ctx.rect(0, h * (1 - reveal), w, h * reveal);
            ctx.clip();
            return;
        }
        if (origin === 'center-x') {
            const clipWidth = w * reveal;
            ctx.beginPath();
            ctx.rect((w - clipWidth) / 2, 0, clipWidth, h);
            ctx.clip();
            return;
        }
        if (origin === 'center-y') {
            const clipHeight = h * reveal;
            ctx.beginPath();
            ctx.rect(0, (h - clipHeight) / 2, w, clipHeight);
            ctx.clip();
            return;
        }
        const clipWidth = w * reveal;
        const clipHeight = h * reveal;
        ctx.beginPath();
        ctx.rect((w - clipWidth) / 2, (h - clipHeight) / 2, clipWidth, clipHeight);
        ctx.clip();
    };

    const applyLayerClipPath = (ctx, clipPath, width, height) => {
        const source = String(clipPath || '').trim();
        if (!source) return false;

        const polygonMatch = source.match(/^polygon\((.+)\)$/i);
        if (polygonMatch?.[1]) {
            const pointPairs = polygonMatch[1]
                .split(/\s*,\s*/)
                .map(pair => pair.trim())
                .map(pair => {
                    const coords = pair.split(/\s+/).filter(Boolean);
                    if (coords.length < 2) return null;
                    return coords.slice(0, 2);
                })
                .filter(Boolean)
                .map(([xValue, yValue]) => {
                    const resolve = (value, size) => {
                        if (/%$/.test(value)) return (parseFloat(value) / 100) * size;
                        return parseFloat(value);
                    };
                    return [resolve(xValue, width), resolve(yValue, height)];
                })
                .filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1]));

            if (pointPairs.length >= 3) {
                ctx.beginPath();
                pointPairs.forEach(([x, y], index) => {
                    if (index === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.closePath();
                ctx.clip();
                return true;
            }
        }

        const insetMatch = source.match(/^inset\((.+)\)$/i);
        if (insetMatch?.[1]) {
            const parts = insetMatch[1].split(/\s+/).filter(Boolean);
            const resolveInset = (value, size) => {
                if (/%$/.test(value)) return (parseFloat(value) / 100) * size;
                return parseFloat(value);
            };
            const top = resolveInset(parts[0] || '0', height);
            const right = resolveInset(parts[1] || parts[0] || '0', width);
            const bottom = resolveInset(parts[2] || parts[0] || '0', height);
            const left = resolveInset(parts[3] || parts[1] || parts[0] || '0', width);
            ctx.beginPath();
            ctx.rect(left, top, Math.max(0, width - left - right), Math.max(0, height - top - bottom));
            ctx.clip();
            return true;
        }

        return false;
    };

    const parseTextStroke = (value) => {
        const source = String(value || '').trim();
        if (!source) return { width: 0, color: '' };
        const widthMatch = source.match(/(-?\d+(\.\d+)?)px/i);
        const width = widthMatch ? Number(widthMatch[1]) : 0;
        const color = source.replace(widthMatch?.[0] || '', '').trim();
        return { width, color };
    };

    const drawCanvasTextLayer = (ctx, layer, width, height) => {
        let fontSize = Math.max(8, parseCssSizePx(layer?.fontSize, Math.round(height * 0.48)));
        const fontFamily = String(layer?.fontFamily || 'Impact, Arial Black, sans-serif');
        const fontWeight = String(layer?.fontWeight || '800');
        const fontStyle = String(layer?.fontStyle || 'normal');
        const letterSpacing = Number(layer?.letterSpacing || 0);
        const lines = String(layer?.text || '').split('\n').filter(Boolean);
        if (lines.length === 0) return;

        const setFont = () => {
            ctx.font = `${fontStyle} ${fontWeight} ${fontSize}px ${fontFamily}`;
        };
        setFont();
        const maxLineWidth = Math.max(...lines.map(line => {
            const baseWidth = ctx.measureText(line).width;
            return baseWidth + Math.max(0, Array.from(line).length - 1) * letterSpacing;
        }));
        const widthScale = maxLineWidth > width * 0.94 ? (width * 0.94) / maxLineWidth : 1;
        if (widthScale < 1) fontSize *= Math.max(0.45, widthScale);

        const parsedLineHeightPx = parseCssSizePx(layer?.lineHeight, fontSize * 0.96);
        let lineHeightPx = Math.max(fontSize * 0.78, parsedLineHeightPx || (fontSize * 0.96));
        const heightScale = lines.length * lineHeightPx > height * 0.94
            ? (height * 0.94) / (lines.length * lineHeightPx)
            : 1;
        if (heightScale < 1) {
            fontSize *= Math.max(0.45, heightScale);
            lineHeightPx *= Math.max(0.45, heightScale);
        }

        ctx.fillStyle = layer?.textColor || '#ffffff';
        ctx.textAlign = layer?.textAlign === 'left' ? 'left' : layer?.textAlign === 'right' ? 'right' : 'center';
        ctx.textBaseline = 'middle';
        setFont();

        const shadowValue = String(layer?.textShadow || '');
        if (shadowValue) {
            const shadowColor = shadowValue.match(/rgba?\([^)]+\)|#[0-9a-fA-F]+/);
            const blurMatch = shadowValue.match(/(?:^|\s)(-?\d+(\.\d+)?)px(?:\s+(-?\d+(\.\d+)?)px)?(?:\s+(-?\d+(\.\d+)?)px)?/);
            ctx.shadowColor = shadowColor?.[0] || 'rgba(0,0,0,0.22)';
            ctx.shadowBlur = blurMatch ? Math.max(0, Number(blurMatch[5] || blurMatch[1] || 0)) : 4;
            ctx.shadowOffsetX = blurMatch ? Number(blurMatch[1] || 0) : 0;
            ctx.shadowOffsetY = blurMatch ? Number(blurMatch[3] || 0) : 1;
        }

        const stroke = parseTextStroke(layer?.textStroke);
        const totalHeight = lines.length * lineHeightPx;
        const startY = (height - totalHeight) / 2 + (lineHeightPx / 2);
        const anchorX = ctx.textAlign === 'left' ? 0 : ctx.textAlign === 'right' ? width : width / 2;

        lines.forEach((line, index) => {
            const y = startY + index * lineHeightPx;
            if (stroke.width > 0 && stroke.color) {
                ctx.lineWidth = stroke.width;
                ctx.strokeStyle = stroke.color;
                ctx.strokeText(line, anchorX, y);
            }

            if (letterSpacing !== 0) {
                const chars = Array.from(line);
                let cursorX = anchorX;
                if (ctx.textAlign === 'center') {
                    const lineWidth = chars.reduce((sum, char) => sum + ctx.measureText(char).width + letterSpacing, 0) - letterSpacing;
                    cursorX = (width - lineWidth) / 2;
                    ctx.textAlign = 'left';
                } else if (ctx.textAlign === 'right') {
                    const lineWidth = chars.reduce((sum, char) => sum + ctx.measureText(char).width + letterSpacing, 0) - letterSpacing;
                    cursorX = width - lineWidth;
                    ctx.textAlign = 'left';
                }
                chars.forEach(char => {
                    ctx.fillText(char, cursorX, y);
                    cursorX += ctx.measureText(char).width + letterSpacing;
                });
                ctx.textAlign = layer?.textAlign === 'left' ? 'left' : layer?.textAlign === 'right' ? 'right' : 'center';
            } else {
                ctx.fillText(line, anchorX, y);
            }
        });

        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
    };

    const drawCanvasShapeLayer = (ctx, layer, width, height) => {
        const styleText = String(layer?.shapeStyleText || '');
        const fill = extractInlineStyleValue(styleText, ['background', 'background-color']) || 'rgba(255,255,255,0.92)';
        const borderRadius = parseCssSizePx(extractInlineStyleValue(styleText, ['border-radius']), Math.min(width, height) * 0.12);
        const borderText = extractInlineStyleValue(styleText, ['border']);
        drawRoundedRectPath(ctx, 0, 0, width, height, borderRadius);
        ctx.fillStyle = fill;
        ctx.fill();
        if (borderText) {
            const widthMatch = borderText.match(/(-?\d+(\.\d+)?)px/);
            const colorMatch = borderText.match(/rgba?\([^)]+\)|#[0-9a-fA-F]+/);
            ctx.lineWidth = widthMatch ? Number(widthMatch[1]) : 1;
            ctx.strokeStyle = colorMatch?.[0] || 'rgba(255,255,255,0.4)';
            ctx.stroke();
        }
    };

    const drawCanvasImageLayer = (ctx, image, width, height, crop = null) => {
        if (!image) return;
        if (crop) {
            ctx.drawImage(
                image,
                crop.sx,
                crop.sy,
                crop.sw,
                crop.sh,
                0,
                0,
                width,
                height
            );
            return;
        }
        ctx.drawImage(image, 0, 0, width, height);
    };

    const drawMotionLayerFrame = (ctx, layer, progress, assets, stageWidth, stageHeight) => {
        const bbox = Array.isArray(layer?.bbox) && layer.bbox.length === 4 ? layer.bbox.map(Number) : [0, 0, 1000, 1000];
        const left = (bbox[1] / 1000) * stageWidth;
        const top = (bbox[0] / 1000) * stageHeight;
        const width = ((bbox[3] - bbox[1]) / 1000) * stageWidth;
        const height = ((bbox[2] - bbox[0]) / 1000) * stageHeight;
        if (width <= 0 || height <= 0) return;

        const opacity = Number(sampleTrack(layer?.tracks?.opacity, progress) ?? 1);
        if (opacity <= 0.001) return;

        const position = sampleTrack(layer?.tracks?.position, progress) || [0, 0];
        const scale = sampleTrack(layer?.tracks?.scale, progress) || [1, 1];
        const rotation = Number(sampleTrack(layer?.tracks?.rotate, progress) ?? 0);
        const blurPx = Math.max(0, Number(sampleTrack(layer?.tracks?.blur, progress) ?? 0));
        const glow = Math.max(0, Number(sampleTrack(layer?.tracks?.glow, progress) ?? 0));
        const reveal = Number(sampleTrack(layer?.tracks?.reveal, progress) ?? 1);
        const shouldClipReveal = layer?.role === 'panel' || layer?.role === 'decoration';
        const snapshotPaddingPx = Math.max(0, Number(layer?.snapshotPaddingPx || 0));
        const drawLeft = left - snapshotPaddingPx;
        const drawTop = top - snapshotPaddingPx;
        const drawWidth = width + snapshotPaddingPx * 2;
        const drawHeight = height + snapshotPaddingPx * 2;

        ctx.save();
        ctx.globalAlpha = opacity;
        const effectiveBlurPx = layer?.role === 'text'
            ? 0
            : layer?.role === 'panel'
                ? Math.min(2, blurPx)
                : Math.min(6, blurPx);
        const effectiveGlow = layer?.role === 'text' ? Math.min(glow, 0.08) : glow;
        ctx.filter = effectiveBlurPx > 0.1 ? `blur(${effectiveBlurPx}px)` : 'none';
        ctx.shadowColor = effectiveGlow > 0 ? `rgba(255,255,255,${Math.min(0.58, effectiveGlow)})` : 'transparent';
        ctx.shadowBlur = effectiveGlow > 0 ? Math.max(width, height) * 0.14 * effectiveGlow : 0;

        const centerX = drawLeft + drawWidth / 2 + position[0] * width;
        const centerY = drawTop + drawHeight / 2 + position[1] * height;
        ctx.translate(centerX, centerY);
        ctx.rotate((rotation * Math.PI) / 180);
        ctx.scale(scale[0] || 1, scale[1] || 1);
        ctx.translate(-drawWidth / 2, -drawHeight / 2);
        if (snapshotPaddingPx <= 0) {
            applyLayerClipPath(ctx, layer?.clipPath, drawWidth, drawHeight);
        }
        if (shouldClipReveal) {
            applyRevealClip(ctx, drawWidth, drawHeight, reveal, layer?.tracks?.revealOrigin || 'center');
        }

        if (layer.kind === 'image') {
            drawCanvasImageLayer(ctx, assets?.[layer.id], drawWidth, drawHeight);
        } else if (layer.kind === 'slice') {
            const sourceImage = assets?.[`${layer.id}:slice`];
            if (sourceImage) {
                const sx = (bbox[1] / 1000) * sourceImage.width;
                const sy = (bbox[0] / 1000) * sourceImage.height;
                const sw = ((bbox[3] - bbox[1]) / 1000) * sourceImage.width;
                const sh = ((bbox[2] - bbox[0]) / 1000) * sourceImage.height;
                drawCanvasImageLayer(ctx, sourceImage, drawWidth, drawHeight, { sx, sy, sw, sh });
            }
        } else if (layer.kind === 'text') {
            drawCanvasTextLayer(ctx, layer, drawWidth, drawHeight);
        } else if (layer.kind === 'shape') {
            drawCanvasShapeLayer(ctx, layer, drawWidth, drawHeight);
        }

        ctx.restore();
    };

    const drawMotionPreviewFrame = (ctx, payload, assets, progress) => {
        const stageWidth = payload?.stage?.width || ctx.canvas.width;
        const stageHeight = payload?.stage?.height || ctx.canvas.height;
        ctx.clearRect(0, 0, stageWidth, stageHeight);

        const plateImage = assets?.__plate;
        if (plateImage) {
            ctx.save();
            ctx.globalAlpha = 1;
            ctx.drawImage(plateImage, 0, 0, stageWidth, stageHeight);
            ctx.restore();
        } else {
            ctx.save();
            ctx.fillStyle = '#f97316';
            ctx.fillRect(0, 0, stageWidth, stageHeight);
            ctx.restore();
        }

        const layers = Array.isArray(payload?.layers)
            ? [...payload.layers].sort((a, b) => Number(a?.renderOrder ?? a?.zIndex ?? 0) - Number(b?.renderOrder ?? b?.zIndex ?? 0))
            : [];
        layers.forEach(layer => drawMotionLayerFrame(ctx, layer, progress, assets, stageWidth, stageHeight));

        // The final frame is the compositor result itself. Do not fade the
        // source/original image over it: Magic Layers can intentionally differ
        // from that source after extraction and layout normalization.
    };

    const startMagicMotionCanvasPreview = async (containerId) => {
        if (!containerId || typeof document === 'undefined') return;
        const root = document.getElementById(containerId);
        if (!root) return;

        const canvas = root.querySelector('.mmp-canvas-stage');
        const payloadNode = root.querySelector('.mmp-payload');
        if (!canvas || !payloadNode) return;

        if (root.__mmpRaf) {
            cancelAnimationFrame(root.__mmpRaf);
            root.__mmpRaf = null;
        }

        let payload = null;
        try {
            payload = JSON.parse(payloadNode.textContent || '{}');
        } catch (error) {
            console.error('Failed to parse Magic Motion canvas payload:', error);
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const stageWidth = Number(payload?.stage?.width || 900);
        const stageHeight = Number(payload?.stage?.height || 1200);
        canvas.width = stageWidth;
        canvas.height = stageHeight;

        const assets = {};
        assets.__plate = await loadMotionPreviewImage(payload?.plateUrl || '');
        await Promise.all((payload?.layers || []).map(async (layer) => {
            if (layer?.kind === 'image' && layer?.imageUrl) {
                assets[layer.id] = await loadMotionPreviewImage(layer.imageUrl);
            } else if (layer?.kind === 'slice' && layer?.sliceSourceUrl) {
                assets[`${layer.id}:slice`] = await loadMotionPreviewImage(layer.sliceSourceUrl);
            }
        }));

        const durationMs = Math.max(1000, Number(payload?.durationMs || 7600));
        const startAt = performance.now();
        const renderFrame = (now) => {
            const progress = clamp01((now - startAt) / durationMs);
            drawMotionPreviewFrame(ctx, payload, assets, progress);
            if (progress < 1) {
                root.__mmpRaf = requestAnimationFrame(renderFrame);
                return;
            }
            root.__mmpRaf = null;
        };

        drawMotionPreviewFrame(ctx, payload, assets, 0);
        root.__mmpRaf = requestAnimationFrame(renderFrame);
    };

    if (typeof window !== 'undefined') {
        window.startMagicMotionCanvasPreview = startMagicMotionCanvasPreview;
    }

    const hasPanelLikeKeywords = (text = '') => /panel|card|frame|tag|badge|background|bg|backdrop|block|banner|tile|底板|底色|背景|卡片|标签框|面板|色块|价签框|标题框|框/.test(text);
    const hasProductLikeKeywords = (text = '') => /product|food|drink|dish|meal|rice|pork|cola|tea|coffee|choco|beverage|cup|bottle|plate|商品|产品|菜品|食物|饮料|烤肉|炒饭|可乐|热茶|咖啡|巧克力|茶|饮品|杯|瓶/.test(text);
    const hasDecorationLikeKeywords = (text = '') => /decor|shape|graphic|sticker|particle|spark|light|flare|ornament|accent|highlight|装饰|光效|粒子|贴纸|图形|高光|星芒|阴影/.test(text);

    const getMotionLayerRoleByName = (name = '') => {
        const text = String(name || '').toLowerCase();
        if (/logo|brand|品牌/.test(text)) return 'logo';
        if (/price|价格|价签|¥|\$|￥/.test(text)) return 'price';
        // Edited price notes can retain only their current numeric content
        // after the original layer name is no longer available.
        if (/^\s*\d+(?:\.\d+)?\s*$/.test(text)) return 'price';
        if (/text|文字|标题|文案|label|字/.test(text)) return 'text';
        if (hasPanelLikeKeywords(text)) return 'panel';
        if (hasProductLikeKeywords(text)) return 'product';
        if (hasDecorationLikeKeywords(text)) return 'decoration';
        return 'unknown';
    };

    const clampMotionValue = (value, min, max) => Math.min(Math.max(Number(value) || 0, min), max);

    const mergeMotionBboxes = (bboxes = []) => {
        const valid = bboxes.filter(bbox => Array.isArray(bbox) && bbox.length === 4);
        if (valid.length === 0) return null;
        return [
            Math.min(...valid.map(bbox => Number(bbox[0]) || 0)),
            Math.min(...valid.map(bbox => Number(bbox[1]) || 0)),
            Math.max(...valid.map(bbox => Number(bbox[2]) || 0)),
            Math.max(...valid.map(bbox => Number(bbox[3]) || 0))
        ];
    };

    const parseWorkbenchCssPx = (value) => {
        if (typeof value === 'number') return value;
        const parsed = Number.parseFloat(String(value || '').replace('px', '').trim());
        return Number.isFinite(parsed) ? parsed : 0;
    };

    const getWorkbenchChildBbox = (childItem, parentItem, options = {}) => {
        const preferOriginal = options?.preferOriginal === true;
        if (preferOriginal && Array.isArray(childItem?.originalBbox) && childItem.originalBbox.length === 4) {
            return childItem.originalBbox.map(value => clampMotionValue(value, 0, 1000));
        }

        const childEl = childItem?.el;
        const parentEl = parentItem?.el;
        if (!childEl || !parentEl) return null;

        // Use rendered rectangles when available. CSS left/top can describe a
        // pre-transform layout box and was consistently shifting text left in
        // the compositor even though Workbench displayed it correctly.
        if (typeof childEl.getBoundingClientRect === 'function' &&
            typeof parentEl.getBoundingClientRect === 'function') {
            const parentRect = parentEl.getBoundingClientRect();
            const childRect = childEl.getBoundingClientRect();
            if (parentRect.width > 0 && parentRect.height > 0 && childRect.width > 0 && childRect.height > 0) {
                return [
                    clampMotionValue(((childRect.top - parentRect.top) / parentRect.height) * 1000, 0, 1000),
                    clampMotionValue(((childRect.left - parentRect.left) / parentRect.width) * 1000, 0, 1000),
                    clampMotionValue(((childRect.bottom - parentRect.top) / parentRect.height) * 1000, 0, 1000),
                    clampMotionValue(((childRect.right - parentRect.left) / parentRect.width) * 1000, 0, 1000)
                ];
            }
        }

        const parentLeft = parseWorkbenchCssPx(parentEl.style.left);
        const parentTop = parseWorkbenchCssPx(parentEl.style.top);
        const parentWidth = Math.max(1, parseWorkbenchCssPx(parentEl.style.width));
        const parentHeight = Math.max(1, parseWorkbenchCssPx(parentEl.style.height));
        const childLeft = parseWorkbenchCssPx(childEl.style.left);
        const childTop = parseWorkbenchCssPx(childEl.style.top);
        const childWidth = Math.max(1, parseWorkbenchCssPx(childEl.style.width));
        const childHeight = Math.max(1, parseWorkbenchCssPx(childEl.style.height));

        return [
            clampMotionValue(((childTop - parentTop) / parentHeight) * 1000, 0, 1000),
            clampMotionValue(((childLeft - parentLeft) / parentWidth) * 1000, 0, 1000),
            clampMotionValue((((childTop - parentTop) + childHeight) / parentHeight) * 1000, 0, 1000),
            clampMotionValue((((childLeft - parentLeft) + childWidth) / parentWidth) * 1000, 0, 1000)
        ];
    };

    const getWorkbenchTextContentBbox = (childItem, parentItem) => {
        const contentEl = childItem?.el?.querySelector?.('.note-content');
        const parentEl = parentItem?.el;
        if (!contentEl || !parentEl || typeof contentEl.getBoundingClientRect !== 'function' || typeof parentEl.getBoundingClientRect !== 'function') {
            return null;
        }

        const parentRect = parentEl.getBoundingClientRect();
        const contentRect = contentEl.getBoundingClientRect();
        const parentWidth = Math.max(1, parentRect.width);
        const parentHeight = Math.max(1, parentRect.height);

        return [
            clampMotionValue(((contentRect.top - parentRect.top) / parentHeight) * 1000, 0, 1000),
            clampMotionValue(((contentRect.left - parentRect.left) / parentWidth) * 1000, 0, 1000),
            clampMotionValue(((contentRect.bottom - parentRect.top) / parentHeight) * 1000, 0, 1000),
            clampMotionValue(((contentRect.right - parentRect.left) / parentWidth) * 1000, 0, 1000)
        ];
    };

    const hasMotionParentLink = (candidate, parentId) => {
        if (!candidate || !parentId) return false;
        if ([candidate.parentId, candidate.sourceParentId, candidate.metadata?.parentId].some(value => value === parentId)) return true;
        return Array.isArray(candidate?.genealogy?.parents) && candidate.genealogy.parents.includes(parentId);
    };

    const getMotionChildOriginalBbox = (candidate) => {
        const bbox = candidate?.originalBbox || candidate?.metadata?.originalBbox;
        return Array.isArray(bbox) && bbox.length === 4 ? bbox : null;
    };

    const getMotionChildMediaUrl = (candidate) => candidate?.dataUrl || candidate?.cutoutUrl ||
        candidate?.previewUrl || candidate?.originalDataUrl || candidate?.sourceImage ||
        candidate?.metadata?.cutoutUrl || candidate?.metadata?.previewUrl || candidate?.metadata?.sourceImage || '';

    const isExtractedMotionWorkbenchItem = (candidate, parentId) => {
        if (!hasMotionParentLink(candidate, parentId)) return false;
        if (!getMotionChildOriginalBbox(candidate)) return false;
        const extractedType = ['layer-explode', 'layer-extract', 'isolated-edit', 'extraction'].includes(
            candidate.type || candidate.metadata?.type
        );
        if (extractedType) return true;

        // Runtime hydration can normalize an extracted asset's type to its semantic
        // type (for example, "product"). Parent/bbox/media still identify it as a
        // real child asset, so do not fall back to the coarse semantic bbox slice.
        const hasMedia = !!getMotionChildMediaUrl(candidate);
        const isNonVisualWorkbenchNode = ['text-note', 'group-label', 'shape', 'atmosphere'].includes(candidate.type || candidate.metadata?.type);
        return hasMedia && !isNonVisualWorkbenchNode;
    };

    const isTextNoteMotionWorkbenchItem = (candidate, parentId) => {
        if (!hasMotionParentLink(candidate, parentId) || candidate.type !== 'text-note') return false;
        return !!candidate?.el?.querySelector?.('.note-content');
    };

    const isShapeMotionWorkbenchItem = (candidate, parentId) => {
        if (!hasMotionParentLink(candidate, parentId) || candidate.type !== 'shape') return false;
        return !!candidate?.el;
    };

    const buildExtractedMotionLayer = (childId, childItem) => {
        const name = childItem.layerName || childItem.label || childItem.name || `图层 ${childId}`;
        const role = getMotionLayerRoleByName(name);
        return {
            id: `motion-child-${childId}`,
            name,
            category: role === 'background' ? 'background' : 'object',
            semanticType: role === 'product' ? 'product_food' : role === 'panel' ? 'shape_panel' : role === 'decoration' ? 'decor_graphic' : 'motion_child_asset',
            designRole: role === 'price' ? 'price_text' : role === 'logo' ? 'brand_logo' : role === 'panel' ? 'local_panel' : role === 'product' ? 'product_image' : role,
            renderMode: 'cutout_asset',
            runtimeType: 'motion_child_asset',
            extractionProfile: 'motion_ready_extracted_asset',
            compositeRole: 'atomic_object',
            childLayerIds: [],
            bbox: getMotionChildOriginalBbox(childItem),
            zIndex: Number.parseInt(childItem?.el?.style?.zIndex || childItem?.zIndex || childItem?.transform?.zIndex || '0', 10) || 0,
            cutoutUrl: getMotionChildMediaUrl(childItem),
            previewUrl: childItem.previewUrl || getMotionChildMediaUrl(childItem),
            motionRole: role,
            sourceWorkbenchItemId: childId,
            sourceParentId: childItem.parentId || childItem.sourceParentId || childItem.metadata?.parentId
        };
    };

    const buildTextNoteMotionLayer = (childId, childItem, parentItem) => {
        const wrapperBbox = getWorkbenchChildBbox(childItem, parentItem);
        const originalBbox = Array.isArray(childItem?.originalBbox) && childItem.originalBbox.length === 4
            ? childItem.originalBbox.map(value => clampMotionValue(value, 0, 1000))
            : null;

        const contentEl = childItem?.el?.querySelector?.('.note-content');
        const textContent = String(contentEl?.innerText || contentEl?.textContent || '').trim();
        if (!textContent) return null;

        // Use the live note content together with the persisted name. This
        // keeps an edited numeric price an independent price atom even when
        // its old name was replaced during Workbench hydration.
        const nameBase = `${textContent} ${childItem.layerName || childItem.label || childItem.name || ''}`.trim() || `文字 ${childId}`;
        const role = getMotionLayerRoleByName(nameBase);
        const inlineStyle = contentEl?.style?.cssText || '';
        const visualTransform = contentEl?.dataset?.textVisualTransform || '';
        const sourceTextLayerId = childItem?.sourceTextLayerId || null;
        const textGroupKey = String(sourceTextLayerId || childItem.layerName || childId);
        const textLayoutDiagnostics = resolveAutoWrappedTextDiagnostics(childItem, contentEl);
        const contentBbox = getWorkbenchTextContentBbox(childItem, parentItem);
        // Single-line notes have a reliable semantic bbox. Their DOM content
        // rect may be the full note wrapper, which makes titles/labels shrink
        // or appear to enter from the wrong place in canvas playback. Keep the
        // DOM bbox for wrapped text, where it carries the actual line layout.
        const isSingleLineText = Number(textLayoutDiagnostics.renderedLineCount || 1) <= 1 &&
            textLayoutDiagnostics.hasExplicitBreaks !== true;
        // The content rectangle is the visual text position inside the note;
        // using the wrapper rectangle centers text in the container and can
        // shift right-aligned/fit-to-box text left in the compositor.
        // Text notes are CSS layout objects, not raster cutouts. Their wrapper
        // rectangle is the Workbench's authoritative position; the exported
        // original bbox describes the source recognition region and can shift
        // text left when used as the canvas text box.
        const layoutBbox = wrapperBbox || contentBbox || originalBbox;
        const bbox = layoutBbox || originalBbox;
        if (!Array.isArray(bbox) || bbox.length !== 4) return null;
        const contentComputedStyleText = buildComputedStyleText(contentEl, [
            'display',
            'width',
            'height',
            'text-align',
            'color',
            'font-family',
            'font-weight',
            'font-style',
            'font-size',
            'line-height',
            'letter-spacing',
            'white-space',
            'word-break',
            'overflow-wrap',
            'text-shadow',
            '-webkit-text-stroke',
            '-webkit-text-fill-color',
            'background',
            'background-image',
            'background-size',
            'background-position',
            'background-repeat',
            'background-clip',
            '-webkit-background-clip',
            'box-sizing',
            'min-width',
            'min-height',
            'padding',
            'margin'
        ]);
        const wrapperComputedStyleText = buildComputedStyleText(childItem?.el, [
            'display',
            'align-items',
            'justify-content',
            'box-sizing',
            'overflow',
            'background',
            'background-color'
        ]);
        const computedTextFillColor = extractInlineStyleValue(contentComputedStyleText, ['-webkit-text-fill-color']);
        const computedFontColor = extractInlineStyleValue(contentComputedStyleText, ['color']);
        const computedTextShadow = extractInlineStyleValue(contentComputedStyleText, ['text-shadow']);
        const computedFontFamily = extractInlineStyleValue(contentComputedStyleText, ['font-family']);
        const computedFontWeight = extractInlineStyleValue(contentComputedStyleText, ['font-weight']);
        const computedFontStyle = extractInlineStyleValue(contentComputedStyleText, ['font-style']);
        const computedFontSize = extractInlineStyleValue(contentComputedStyleText, ['font-size']);
        const computedLineHeight = extractInlineStyleValue(contentComputedStyleText, ['line-height']);
        const computedLetterSpacing = extractInlineStyleValue(contentComputedStyleText, ['letter-spacing']);
        const computedTextAlign = extractInlineStyleValue(contentComputedStyleText, ['text-align']);
        const computedTextStroke = extractInlineStyleValue(contentComputedStyleText, ['-webkit-text-stroke', 'text-stroke']);
        const computedWhiteSpace = extractInlineStyleValue(contentComputedStyleText, ['white-space']);

        const cssFontColor = childItem?.css?.WebkitTextFillColor ||
            childItem?.css?.webkitTextFillColor ||
            childItem?.css?.['-webkit-text-fill-color'] ||
            childItem?.css?.color ||
            '';
        const resolvedFontColor = [
            contentEl?.style?.webkitTextFillColor,
            contentEl?.style?.color,
            computedTextFillColor,
            computedFontColor,
            cssFontColor
        ].find(isVisibleMotionColor) || '';

        return {
            id: `motion-text-${childId}`,
            name: childItem.layerName || `文字: ${textContent.slice(0, 24)}`,
            category: 'object',
            semanticType: role === 'logo' ? 'logo_mark' : 'element_text',
            designRole: role === 'price' ? 'price_text' : role === 'logo' ? 'brand_logo' : 'label_text',
            renderMode: 'text_css',
            runtimeType: 'text_note_child',
            extractionProfile: 'motion_ready_text_asset',
            compositeRole: 'atomic_object',
            childLayerIds: [],
            bbox,
            layoutBbox: Array.isArray(layoutBbox) ? layoutBbox : bbox,
            wrapperBbox: Array.isArray(wrapperBbox) ? wrapperBbox : null,
            contentBbox: Array.isArray(contentBbox) ? contentBbox : null,
            originalBbox,
            zIndex: Number.parseInt(childItem?.el?.style?.zIndex || childItem?.zIndex || '0', 10) || 0,
            textContent,
            textHtml: contentEl?.innerHTML || '',
            textStyleText: inlineStyle,
            textSnapshotStyleText: contentComputedStyleText || inlineStyle,
            textSnapshotWrapperStyleText: wrapperComputedStyleText || '',
            textVisualTransform: visualTransform,
            renderedLineCount: Number(textLayoutDiagnostics.renderedLineCount || 1),
            explicitLineCount: Number(textLayoutDiagnostics.explicitLineCount || 1),
            isAutoWrappedText: textLayoutDiagnostics.isAutoWrappedText === true,
            hasExplicitBreaks: textLayoutDiagnostics.hasExplicitBreaks === true,
            wrapperWidthPx: Number(textLayoutDiagnostics.wrapperWidthPx || 0),
            naturalSingleLineWidthPx: Number(textLayoutDiagnostics.naturalSingleLineWidthPx || 0),
            fontColor: resolvedFontColor,
            fontFamily: contentEl?.style?.fontFamily || computedFontFamily || '',
            fontWeight: contentEl?.style?.fontWeight || computedFontWeight || '',
            fontStyle: contentEl?.style?.fontStyle || computedFontStyle || '',
            fontSize: contentEl?.style?.fontSize || computedFontSize || '',
            lineHeight: contentEl?.style?.lineHeight || computedLineHeight || '',
            letterSpacing: contentEl?.style?.letterSpacing || computedLetterSpacing || '',
            textAlign: contentEl?.style?.textAlign || computedTextAlign || '',
            textShadow: contentEl?.style?.textShadow || computedTextShadow || '',
            WebkitTextStroke: contentEl?.style?.webkitTextStroke || contentEl?.style?.textStroke || computedTextStroke || '',
            whiteSpace: contentEl?.style?.whiteSpace || computedWhiteSpace || '',
            css: childItem?.css || null,
            motionRole: role,
            sourceWorkbenchItemId: childId,
            sourceParentId: childItem.parentId,
            sourceTextLayerId,
            textGroupKey
        };
    };

    const buildTextNoteGroupMotionLayer = (groupKey, layers = []) => {
        const validLayers = layers.filter(Boolean);
        if (validLayers.length === 0) return null;
        if (validLayers.length === 1) return validLayers[0];

        const bbox = mergeMotionBboxes(validLayers.map(layer => layer.layoutBbox || layer.bbox));
        if (!bbox) return validLayers[0];

        const prioritizedRole = validLayers.some(layer => layer.motionRole === 'logo')
            ? 'logo'
            : validLayers.some(layer => layer.motionRole === 'price')
                ? 'price'
                : 'text';
        const sortedLayers = [...validLayers].sort((a, b) => {
            const topDelta = (Number(a?.bbox?.[0]) || 0) - (Number(b?.bbox?.[0]) || 0);
            if (Math.abs(topDelta) > 8) return topDelta;
            return (Number(a?.bbox?.[1]) || 0) - (Number(b?.bbox?.[1]) || 0);
        });

        return {
            id: `motion-text-group-${groupKey}`,
            name: sortedLayers[0]?.name || `文字组 ${groupKey}`,
            category: 'object',
            semanticType: prioritizedRole === 'logo' ? 'logo_mark' : 'element_text',
            designRole: prioritizedRole === 'price' ? 'price_text' : prioritizedRole === 'logo' ? 'brand_logo' : 'label_text',
            renderMode: 'text_group_snapshot',
            runtimeType: 'text_note_group',
            extractionProfile: 'motion_ready_text_group_asset',
            compositeRole: 'atomic_object',
            childLayerIds: sortedLayers.map(layer => layer.sourceWorkbenchItemId).filter(Boolean),
            bbox,
            zIndex: Math.max(...sortedLayers.map(layer => Number(layer?.zIndex || 0))),
            motionRole: prioritizedRole,
            sourceWorkbenchItemId: sortedLayers[0]?.sourceWorkbenchItemId || null,
            sourceParentId: sortedLayers[0]?.sourceParentId || null,
            sourceTextLayerId: sortedLayers[0]?.sourceTextLayerId || null,
            textContent: sortedLayers.map(layer => String(layer?.textContent || '').trim()).filter(Boolean).join('\n'),
            textFragments: sortedLayers.map(layer => ({
                bbox: Array.isArray(layer?.layoutBbox || layer?.bbox) ? (layer.layoutBbox || layer.bbox).map(Number) : null,
                textContent: layer?.textContent || '',
                textHtml: layer?.textHtml || '',
                textStyleText: layer?.textStyleText || '',
                textVisualTransform: layer?.textVisualTransform || '',
                fontColor: layer?.fontColor || '',
                fontFamily: layer?.fontFamily || '',
                fontWeight: layer?.fontWeight || '',
                fontStyle: layer?.fontStyle || '',
                fontSize: layer?.fontSize || '',
                lineHeight: layer?.lineHeight || '',
                letterSpacing: layer?.letterSpacing || '',
                textAlign: layer?.textAlign || '',
                textShadow: layer?.textShadow || '',
                WebkitTextStroke: layer?.WebkitTextStroke || '',
                whiteSpace: layer?.whiteSpace || ''
            }))
        };
    };

    const buildShapeMotionLayer = (childId, childItem, parentItem) => {
        const bbox = getWorkbenchChildBbox(childItem, parentItem);
        if (!Array.isArray(bbox) || bbox.length !== 4) return null;

        const layerName = childItem.layerName || childItem.label || childItem.name || `形状 ${childId}`;
        const role = getMotionLayerRoleByName(layerName);
        return {
            id: `motion-shape-${childId}`,
            name: layerName,
            category: 'object',
            semanticType: role === 'price' ? 'price_badge' : 'shape_panel',
            designRole: role === 'price' ? 'price_badge' : role === 'decoration' ? 'decor_shape' : 'local_panel',
            renderMode: 'vector_shape',
            runtimeType: 'shape_node',
            extractionProfile: 'motion_ready_shape_asset',
            compositeRole: 'atomic_object',
            childLayerIds: [],
            bbox: Array.isArray(childItem?.originalBbox) && childItem.originalBbox.length === 4
                ? childItem.originalBbox.map(value => clampMotionValue(value, 0, 1000))
                : bbox,
            zIndex: Number.parseInt(childItem?.el?.style?.zIndex || childItem?.zIndex || '0', 10) || 0,
            shapeStyleText: childItem?.el?.style?.cssText || '',
            clipPath: childItem?.clipPath || childItem?.el?.style?.clipPath || '',
            shapeType: childItem?.shapeType || childItem?.el?.dataset?.shapeType || 'rect',
            motionRole: role === 'unknown' ? 'panel' : role,
            sourceWorkbenchItemId: childId,
            sourceParentId: childItem.parentId
        };
    };

    const findMotionLayerStateIndex = (item, layer) => {
        const sceneLayers = item?.scene?.layers?.length ? item.scene.layers : item?.layers;
        if (!Array.isArray(sceneLayers) || !layer) return null;

        const byId = sceneLayers.findIndex(candidate =>
            candidate?.id === layer?.id ||
            candidate?.id === layer?.cleanPlateLayerId ||
            candidate?.cleanPlateLayerId === layer?.id ||
            candidate?.sourceTextLayerId === layer?.id
        );
        if (byId >= 0) return byId;

        const byName = sceneLayers.findIndex(candidate => candidate?.name && layer?.name && candidate.name === layer.name);
        return byName >= 0 ? byName : null;
    };

    const getMotionEntryPreferenceScore = (entry) => {
        const runtimeType = String(entry?.layer?.runtimeType || '').toLowerCase();
        const source = String(entry?.source || '');
        if (runtimeType === 'text_note_group') return 95;
        if (runtimeType === 'text_note_child') return 90;
        if (runtimeType === 'shape_node') return 85;
        if (runtimeType === 'motion_child_asset') return 80;
        if (source === 'semantic') return 50;
        return 40;
    };

    const isSemanticMotionSource = (source = '') => /^semantic(?:_|$)/i.test(String(source || ''));

    const normalizeMotionText = (value = '') => String(value || '')
        .replace(/<[^>]*>/g, ' ')
        .replace(/&(?:amp|ndash|mdash|nbsp);/gi, ' ')
        .replace(/[^a-z0-9$¥￥]+/gi, ' ')
        .trim()
        .toLowerCase();

    const getMotionEntryText = (entry) => normalizeMotionText(
        entry?.layer?.textContent || entry?.layer?.content || entry?.layer?.text || entry?.layer?.name || ''
    );

    const getMotionEntryPriceNumber = (entry) => getPriceNumber(getMotionEntryText(entry));

    const isNearbyMotionEntry = (leftEntry, rightEntry, xTolerance = 150, yTolerance = 110) => {
        const leftBbox = leftEntry?.layer?.bbox;
        const rightBbox = rightEntry?.layer?.bbox;
        if (!Array.isArray(leftBbox) || !Array.isArray(rightBbox)) return false;
        return Math.abs(getLayerCenterX(leftBbox) - getLayerCenterX(rightBbox)) <= xTolerance &&
            Math.abs(getLayerCenterY(leftBbox) - getLayerCenterY(rightBbox)) <= yTolerance;
    };

    const hasNearbyStandaloneCurrencyEntry = (entry, entries = []) => entries.some(candidate =>
        candidate !== entry &&
        isStandaloneCurrencySymbol(candidate?.layer) &&
        isNearbyMotionEntry(entry, candidate)
    );

    const isSemanticTextCoveredByWorkbenchChildren = (entry, childTextEntries) => {
        if (!isSemanticMotionSource(entry?.source) || !['text', 'price', 'logo'].includes(getLayerRole(entry?.layer))) {
            return false;
        }

        const semanticText = getMotionEntryText(entry);
        if (!semanticText) return false;
        const semanticTokens = new Set(semanticText.match(/[a-z0-9$¥￥]+/gi) || []);
        const semanticBbox = entry?.layer?.bbox;
        if (!Array.isArray(semanticBbox) || semanticBbox.length !== 4) return false;

        const nearbyChildren = childTextEntries.filter(child => {
            const childBbox = child?.layer?.bbox;
            if (!Array.isArray(childBbox) || childBbox.length !== 4) return false;
            const overlap = getBboxOverlapRatio(semanticBbox, childBbox);
            const centerInside =
                getLayerCenterX(childBbox) >= Number(semanticBbox[1]) - 55 &&
                getLayerCenterX(childBbox) <= Number(semanticBbox[3]) + 55 &&
                getLayerCenterY(childBbox) >= Number(semanticBbox[0]) - 55 &&
                getLayerCenterY(childBbox) <= Number(semanticBbox[2]) + 55;
            return overlap >= 0.18 || centerInside;
        });
        if (nearbyChildren.length === 0) return false;

        // Magic Layers can contain a semantic "$8" carrier plus three real
        // Workbench atoms: "$", "8", and the surrounding panel. Once both
        // atomic text pieces exist, the semantic combined price is a duplicate
        // and must not be rendered on top of them.
        const semanticPriceNumber = getPriceNumber(semanticText);
        if (/[$¥￥]/.test(semanticText) && semanticPriceNumber) {
            const numericChild = nearbyChildren.find(child =>
                getLayerRole(child?.layer) === 'price' &&
                !isStandaloneCurrencySymbol(child?.layer) &&
                getMotionEntryPriceNumber(child) === semanticPriceNumber
            );
            const standaloneCurrencyChild = nearbyChildren.some(child =>
                isStandaloneCurrencySymbol(child?.layer)
            );
            if (numericChild && standaloneCurrencyChild) return true;
        }

        const exactMatch = nearbyChildren.some(child => {
            const childText = getMotionEntryText(child);
            return childText && childText === semanticText;
        });
        if (exactMatch) return true;

        // A semantic group such as "DISHES & DRINKS" is commonly represented
        // by several real child notes (DISHES, &, DRINKS). Treat the group as
        // covered once its meaningful tokens are represented by nearby child
        // notes, rather than rendering the group a second time.
        const childTokens = new Set(
            nearbyChildren.flatMap(child => getMotionEntryText(child).match(/[a-z0-9$¥￥]+/gi) || [])
        );
        if (semanticTokens.size >= 2) {
            const represented = [...semanticTokens].filter(token => childTokens.has(token)).length;
            return represented / semanticTokens.size >= 0.66;
        }

        return false;
    };

    const isSemanticPriceCoveredByAtomicChildren = (entry, childTextEntries) => {
        if (!isSemanticMotionSource(entry?.source) || getLayerRole(entry?.layer) !== 'price') {
            return false;
        }
        const semanticText = getMotionEntryText(entry);
        const semanticNumber = getPriceNumber(semanticText);
        if (!semanticNumber || !/[$¥￥]/.test(semanticText)) return false;
        const semanticBbox = entry?.layer?.bbox;
        if (!Array.isArray(semanticBbox) || semanticBbox.length !== 4) return false;

        const nearby = childTextEntries.filter(child => {
            const childBbox = child?.layer?.bbox;
            if (!Array.isArray(childBbox) || childBbox.length !== 4) return false;
            return getBboxOverlapRatio(semanticBbox, childBbox) >= 0.08 ||
                (Math.abs(getLayerCenterX(semanticBbox) - getLayerCenterX(childBbox)) <= 150 &&
                    Math.abs(getLayerCenterY(semanticBbox) - getLayerCenterY(childBbox)) <= 130);
        });
        const hasNumberAtom = nearby.some(child =>
            getLayerRole(child?.layer) === 'price' &&
            !isStandaloneCurrencySymbol(child?.layer) &&
            getMotionEntryPriceNumber(child) === semanticNumber
        );
        const hasCurrencyAtom = nearby.some(child => isStandaloneCurrencySymbol(child?.layer));
        return hasNumberAtom && hasCurrencyAtom;
    };

    const dedupeMotionEntryPool = (entries = []) => {
        const kept = [];
        for (const entry of entries) {
            const duplicate = kept.find(existing => {
                // Overlapping panel/text/product layers are intentional in a
                // composition. Only compare entries within the same role.
                if (getLayerRole(existing.layer) !== getLayerRole(entry.layer)) return false;
                const sameName = existing.layer?.name && entry.layer?.name && existing.layer.name === entry.layer.name;
                const sameBbox = Array.isArray(existing.layer?.bbox) &&
                    Array.isArray(entry.layer?.bbox) &&
                    existing.layer.bbox.join(',') === entry.layer.bbox.join(',');
                const overlap = Array.isArray(existing.layer?.bbox) &&
                    Array.isArray(entry.layer?.bbox)
                    ? getBboxOverlapRatio(existing.layer.bbox, entry.layer.bbox)
                    : 0;
                const existingArea = getLayerAreaScore(existing.layer?.bbox);
                const entryArea = getLayerAreaScore(entry.layer?.bbox);
                const areaRatio = Math.min(existingArea, entryArea) / Math.max(1, Math.max(existingArea, entryArea));
                const sameRuntimeType = String(existing.layer?.runtimeType || '') === String(entry.layer?.runtimeType || '');
                const bothWorkbenchTextNotes =
                    entry?.source === 'workbench_child' &&
                    existing?.source === 'workbench_child' &&
                    /text_note_(child|group)/i.test(String(entry.layer?.runtimeType || '')) &&
                    /text_note_(child|group)/i.test(String(existing.layer?.runtimeType || ''));

                if (sameBbox) return true;
                if (bothWorkbenchTextNotes) {
                    return sameName && overlap >= 0.94 && sameRuntimeType;
                }
                return sameName || (overlap >= 0.985 && areaRatio >= 0.72);
            });

            if (!duplicate) {
                kept.push(entry);
                continue;
            }

            const duplicateScore = getMotionEntryPreferenceScore(duplicate);
            const entryScore = getMotionEntryPreferenceScore(entry);
            const duplicateHasImage = !!(duplicate.layer?.cutoutUrl || duplicate.layer?.previewUrl);
            const entryHasImage = !!(entry.layer?.cutoutUrl || entry.layer?.previewUrl);
            if (entryScore > duplicateScore || (!duplicateHasImage && entryHasImage && entryScore >= duplicateScore)) {
                const replaceIndex = kept.indexOf(duplicate);
                if (replaceIndex >= 0) kept.splice(replaceIndex, 1, entry);
            }
        }
        return kept;
    };

    const filterSemanticTextLayersCoveredByChildAssets = (entries = []) => {
        const childTextEntries = entries.filter(entry =>
            entry?.source === 'workbench_child' &&
            ['text', 'price', 'logo'].includes(getLayerRole(entry.layer)) &&
            Array.isArray(entry?.layer?.bbox) &&
            entry.layer.bbox.length === 4
        );

        if (childTextEntries.length === 0) return entries;

        return entries.filter(entry => {
            if (!isSemanticMotionSource(entry?.source)) return true;

            const role = getLayerRole(entry.layer);
            if (!['text', 'price', 'logo'].includes(role)) return true;
            if (!Array.isArray(entry?.layer?.bbox) || entry.layer.bbox.length !== 4) return true;

            if (isSemanticTextCoveredByWorkbenchChildren(entry, childTextEntries)) return false;
            if (isSemanticPriceCoveredByAtomicChildren(entry, childTextEntries)) return false;

            const coveredByChildAsset = childTextEntries.some(childEntry => {
                const overlap = getBboxOverlapRatio(entry.layer.bbox, childEntry.layer.bbox);
                if (role === 'logo') {
                    return overlap >= 0.72;
                }
                return overlap >= 0.58;
            });

            return !coveredByChildAsset;
        });
    };

    const filterCoarseCompositeMotionEntries = (entries = []) => {
        const childAtomicEntries = entries.filter(entry =>
            entry?.source === 'workbench_child' &&
            Array.isArray(entry?.layer?.bbox) &&
            entry.layer.bbox.length === 4 &&
            getLayerRole(entry.layer) !== 'background'
        );

        if (childAtomicEntries.length === 0) return entries;

        return entries.filter(entry => {
            const bbox = entry?.layer?.bbox;
            if (!Array.isArray(bbox) || bbox.length !== 4) return true;

            const role = getLayerRole(entry.layer);
            if (role === 'background') return true;

            const containingChildren = childAtomicEntries.filter(childEntry => {
                if (childEntry === entry) return false;
                const childBbox = childEntry?.layer?.bbox;
                if (!Array.isArray(childBbox) || childBbox.length !== 4) return false;
                const overlap = getBboxOverlapRatio(bbox, childBbox);
                const parentArea = Math.max(1, getLayerAreaScore(bbox));
                const childArea = Math.max(1, getLayerAreaScore(childBbox));
                const areaRatio = childArea / parentArea;
                return overlap >= 0.76 && areaRatio <= 0.92;
            });

            if (containingChildren.length === 0) return true;

            const childRoles = new Set(containingChildren.map(child => getLayerRole(child.layer)));
            const childAreaRatio = containingChildren.reduce((sum, child) => (
                sum + (getLayerAreaScore(child.layer?.bbox) / Math.max(1, getLayerAreaScore(bbox)))
            ), 0);
            const hasTextChild = containingChildren.some(child => ['text', 'price', 'logo'].includes(getLayerRole(child.layer)));
            const hasPanelChild = containingChildren.some(child => getLayerRole(child.layer) === 'panel');
            const hasProductChild = containingChildren.some(child => getLayerRole(child.layer) === 'product');
            const imageLikeChildren = containingChildren.filter(child =>
                !!(child?.layer?.cutoutUrl || child?.layer?.previewUrl)
            );
            const imageLikeProductChildren = imageLikeChildren.filter(child => getLayerRole(child.layer) === 'product');
            const imageLikePanelChildren = imageLikeChildren.filter(child => getLayerRole(child.layer) === 'panel');
            const runtimeType = String(entry?.layer?.runtimeType || '').toLowerCase();
            const semanticType = String(entry?.layer?.semanticType || '').toLowerCase();
            const designRole = String(entry?.layer?.designRole || '').toLowerCase();
            const renderMode = String(entry?.layer?.renderMode || '').toLowerCase();
            const isCarrierPanel =
                role === 'panel' &&
                (
                    semanticType === 'shape_panel' ||
                    semanticType === 'price_badge' ||
                    designRole === 'local_panel' ||
                    designRole === 'price_badge' ||
                    renderMode === 'vector_shape'
                );
            const isCarrierProduct =
                role === 'product' &&
                (
                    semanticType === 'product_food' ||
                    semanticType === 'product_drink' ||
                    semanticType === 'product_packaging' ||
                    designRole === 'product_image' ||
                    renderMode === 'raster_cutout' ||
                    !!entry?.layer?.cutoutUrl
                );

            if (isSemanticMotionSource(entry?.source)) {
                const lacksAtomicImageReplacement = imageLikeChildren.length === 0;
                const lacksAtomicProductReplacement = imageLikeProductChildren.length === 0;
                const lacksAtomicPanelReplacement = imageLikePanelChildren.length === 0;

                // Keep semantic fallback slices when no extracted atomic asset exists for that region.
                // This restores the earlier panel/card presence that disappeared after coarse filtering tightened.
                if (role === 'panel' && (lacksAtomicPanelReplacement || lacksAtomicImageReplacement)) {
                    return true;
                }
                if (role === 'product' && (lacksAtomicProductReplacement || lacksAtomicImageReplacement)) {
                    return true;
                }

                if (isCarrierPanel) {
                    if (hasPanelChild && childAreaRatio >= 0.46) return false;
                    return true;
                }
                if (isCarrierProduct) {
                    if (hasProductChild && childAreaRatio >= 0.58) return false;
                    return true;
                }
                if (role === 'panel' && hasPanelChild) return false;
                if (role !== 'product' && containingChildren.length >= 2 && (childRoles.size >= 2 || childAreaRatio >= 0.34)) return false;
                if (role === 'product' && hasProductChild && hasTextChild) return false;
                return true;
            }

            if (entry?.source === 'workbench_child' && runtimeType === 'motion_child_asset') {
                if (role === 'panel' && (!!entry?.layer?.cutoutUrl || !!entry?.layer?.clipPath || isCarrierPanel)) {
                    if (hasPanelChild && childAreaRatio >= 0.52) return false;
                    return true;
                }
                if (role === 'product' && (!!entry?.layer?.cutoutUrl || isCarrierProduct)) {
                    if (hasProductChild && childAreaRatio >= 0.62) return false;
                    return true;
                }
                if (role !== 'product' && containingChildren.length >= 2 && (hasTextChild || hasPanelChild) && childAreaRatio >= 0.22) return false;
                if (role !== 'product' && childRoles.size >= 2 && childAreaRatio >= 0.3) return false;
                if (role === 'product' && !entry?.layer?.cutoutUrl && hasTextChild && hasPanelChild && childAreaRatio >= 0.7) return false;
            }

            return true;
        });
    };

    const mergeFallbackMotionEntries = (entries = [], fallbackEntries = [], sourceOverride = '') => {
        const merged = [...entries];

        fallbackEntries.forEach((fallbackEntry) => {
            if (!fallbackEntry?.layer) return;
            const fallbackRole = getLayerRole(fallbackEntry.layer);
            const fallbackBbox = fallbackEntry.layer?.bbox;
            const duplicate = merged.some(existing => {
                const existingRole = getLayerRole(existing?.layer);
                if (existingRole !== fallbackRole) return false;
                const existingBbox = existing?.layer?.bbox;
                const sameId = String(existing?.layer?.id || '') && String(existing?.layer?.id || '') === String(fallbackEntry?.layer?.id || '');
                if (sameId) return true;
                const existingArea = getLayerAreaScore(existingBbox);
                const fallbackArea = getLayerAreaScore(fallbackBbox);
                const areaRatio = Math.min(existingArea, fallbackArea) / Math.max(1, Math.max(existingArea, fallbackArea));
                // A small badge/panel can sit inside a large card. That is a
                // real nested layer, not a duplicate of the card.
                return getBboxOverlapRatio(existingBbox, fallbackBbox) >= 0.9 && areaRatio >= 0.72;
            });
            if (!duplicate) {
                merged.push({
                    ...fallbackEntry,
                    source: sourceOverride || fallbackEntry.source
                });
            }
        });

        return merged;
    };

    const recoverMissingSemanticCarrierEntries = (entries = [], semanticEntries = []) => {
        const recovered = [...entries];
        const currentPanels = recovered.filter(entry => getLayerRole(entry?.layer) === 'panel');
        const currentProducts = recovered
            .filter(entry => getLayerRole(entry?.layer) === 'product')
            .sort((a, b) => getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox));
        const currentLargeProducts = currentProducts.filter(entry => getLayerAreaScore(entry?.layer?.bbox) >= 42000);
        const semanticPanels = semanticEntries
            .filter(entry => getLayerRole(entry?.layer) === 'panel')
            .filter(entry => {
                const bbox = entry?.layer?.bbox;
                const area = getLayerAreaScore(bbox);
                return area >= 4000 && area <= 260000;
            })
            .sort((a, b) => {
                const areaDelta = getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox);
                if (Math.abs(areaDelta) > 1) return areaDelta;
                return getLayerCenterY(a?.layer?.bbox) - getLayerCenterY(b?.layer?.bbox);
            });
        const semanticProducts = semanticEntries
            .filter(entry => getLayerRole(entry?.layer) === 'product')
            .filter(entry => {
                const area = getLayerAreaScore(entry?.layer?.bbox);
                const centerY = getLayerCenterY(entry?.layer?.bbox);
                return area >= 30000 && centerY <= 760;
            })
            .sort((a, b) => {
                const topBandBias = (getLayerCenterY(a?.layer?.bbox) < 700 ? 0 : 1) - (getLayerCenterY(b?.layer?.bbox) < 700 ? 0 : 1);
                if (topBandBias !== 0) return topBandBias;
                return getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox);
            })
            .slice(0, 4);

        const missingSemanticPanels = semanticPanels.filter((panelEntry) => {
            const panelBbox = panelEntry?.layer?.bbox;
            const panelArea = getLayerAreaScore(panelBbox);
            const panelCenterY = getLayerCenterY(panelBbox);
            const panelCenterX = getLayerCenterX(panelBbox);
            return !currentPanels.some((currentEntry) => {
                const currentBbox = currentEntry?.layer?.bbox;
                const overlap = getBboxOverlapRatio(currentBbox, panelBbox);
                const currentArea = getLayerAreaScore(currentBbox);
                const areaRatio = panelArea > 0 ? Math.min(currentArea, panelArea) / panelArea : 0;
                const sameBand = Math.abs(getLayerCenterY(currentBbox) - panelCenterY) <= 72;
                const sameColumn = Math.abs(getLayerCenterX(currentBbox) - panelCenterX) <= 96;
                return overlap >= 0.72 || (sameBand && sameColumn && areaRatio >= 0.68);
            });
        });

        if (currentPanels.length === 0) {
            const withPanels = mergeFallbackMotionEntries(
                recovered,
                semanticPanels,
                'semantic_recovered_panel'
            );
            return mergeFallbackMotionEntries(
                withPanels,
                currentLargeProducts.length < 2
                    ? semanticProducts
                    : [],
                'semantic_recovered_product'
            );
        }

        if (missingSemanticPanels.length > 0) {
            return mergeFallbackMotionEntries(recovered, missingSemanticPanels, 'semantic_recovered_panel');
        }

        if (currentLargeProducts.length < 2) {
            return mergeFallbackMotionEntries(recovered, semanticProducts, 'semantic_recovered_product');
        }

        return recovered;
    };

    const getMotionLayerEntries = (item, itemId) => {
        const motionReadyLayers = item?.semanticViews?.motionReadyLayers;
        const fallbackMotionReadyLayers = item?.motionReadyLayers;
        const editableSceneLayers = item?.semanticViews?.editableSceneLayers;
        const cleanPlateLayers = item?.semanticViews?.cleanPlateLayers;
        const sceneLayers = item?.scene?.layers?.length ? item.scene.layers : item?.layers;
        const semanticPools = [
            { layers: Array.isArray(motionReadyLayers) ? motionReadyLayers : [], source: 'semantic_motion_ready' },
            { layers: Array.isArray(fallbackMotionReadyLayers) ? fallbackMotionReadyLayers : [], source: 'semantic_motion_ready_fallback' },
            { layers: Array.isArray(editableSceneLayers) ? editableSceneLayers : [], source: 'semantic_editable' },
            { layers: Array.isArray(cleanPlateLayers) ? cleanPlateLayers : [], source: 'semantic_clean_plate' },
            { layers: Array.isArray(sceneLayers) ? sceneLayers : [], source: 'semantic_scene' }
        ];

        const semanticEntries = semanticPools.flatMap(({ layers, source }) => layers.map(layer => ({
            layer,
            layerStateIndex: findMotionLayerStateIndex(item, layer),
            source
        })));

        const parentItem = itemId ? state.workbenchItems.get(itemId) : null;
        const childEntries = itemId && parentItem
            ? (() => {
                const childLayerPool = [];

                const childCandidates = new Map(Array.from(state.workbenchItems.entries()));
                const runtimeRegistry = window.mvrRuntime?.getCurrentWorkspace?.()?.currentState?.assetRegistry;
                runtimeRegistry?.getAll?.().forEach(asset => {
                    if (!asset?.uid) return;
                    const existing = childCandidates.get(asset.uid);
                    if (!existing) {
                        childCandidates.set(asset.uid, asset);
                        return;
                    }
                    childCandidates.set(asset.uid, {
                        ...asset,
                        ...existing,
                        parentId: existing.parentId || asset.parentId,
                        sourceParentId: existing.sourceParentId || asset.sourceParentId,
                        originalBbox: getMotionChildOriginalBbox(existing) || getMotionChildOriginalBbox(asset),
                        sourceImage: existing.sourceImage || asset.sourceImage,
                        originalDataUrl: existing.originalDataUrl || asset.originalDataUrl,
                        cutoutUrl: existing.cutoutUrl || asset.cutoutUrl,
                        previewUrl: existing.previewUrl || asset.previewUrl,
                        metadata: { ...(asset.metadata || {}), ...(existing.metadata || {}) }
                    });
                });

                childCandidates.forEach((childItem, childId) => {
                    if (isExtractedMotionWorkbenchItem(childItem, itemId)) {
                        childLayerPool.push({
                            layer: buildExtractedMotionLayer(childId, childItem),
                            layerStateIndex: null,
                            source: 'workbench_child'
                        });
                        return;
                    }

                    if (isTextNoteMotionWorkbenchItem(childItem, itemId)) {
                        const layer = buildTextNoteMotionLayer(childId, childItem, parentItem);
                        if (!layer) return;
                        childLayerPool.push({
                            layer,
                            layerStateIndex: null,
                            source: 'workbench_child'
                        });
                        return;
                    }

                    if (isShapeMotionWorkbenchItem(childItem, itemId)) {
                        const layer = buildShapeMotionLayer(childId, childItem, parentItem);
                        if (!layer) return;
                        childLayerPool.push({
                            layer,
                            layerStateIndex: null,
                            source: 'workbench_child'
                        });
                    }
                });

                return childLayerPool.filter(Boolean);
            })()
            : [];

        const filteredEntries = filterCoarseCompositeMotionEntries(
            filterSemanticTextLayersCoveredByChildAssets(
                dedupeMotionEntryPool([...semanticEntries, ...childEntries])
            )
        );
        return recoverMissingSemanticCarrierEntries(filteredEntries, semanticEntries);
    };

    const getMotionLayers = (item, itemId) => getMotionLayerEntries(item, itemId).map(entry => entry.layer);

    const hasMagicMotionLayers = (item, itemId) => getMotionLayers(item, itemId).some(layer => layer?.category !== 'background');

    const getLayerImageUrl = (layer) => {
        const activeVersion = layer?.versions?.find(v => v.id === layer.activeVersionId);
        return activeVersion?.cutoutUrl || activeVersion?.previewUrl ||
            layer?.cutoutUrl || layer?.previewUrl ||
            layer?.mask?.cutoutUrl || layer?.mask?.visibleMaskUrl ||
            layer?.maskUrl || null;
    };

    const getMotionStableBbox = (layer) => {
        const runtimeType = String(layer?.runtimeType || '').toLowerCase();
        if (/text_note_(child|group)/i.test(runtimeType)) {
            const textLayoutBbox = layer?.wrapperBbox || layer?.contentBbox || layer?.layoutBbox;
            if (Array.isArray(textLayoutBbox) && textLayoutBbox.length === 4) {
                return textLayoutBbox.map(value => clampMotionValue(value, 0, 1000));
            }
        }
        const originalBbox = layer?.originalBbox || layer?.metadata?.originalBbox;
        if (Array.isArray(originalBbox) && originalBbox.length === 4) {
            return originalBbox.map(value => clampMotionValue(value, 0, 1000));
        }
        return Array.isArray(layer?.bbox) && layer.bbox.length === 4
            ? layer.bbox.map(value => clampMotionValue(value, 0, 1000))
            : layer?.bbox;
    };

    const getLayerText = (layer) => {
        const explicit = String(layer?.textContent || layer?.content || '').trim();
        if (explicit) return explicit;
        return String(layer?.name || '')
            .replace(/^文字[:：]\s*/i, '')
            .replace(/^Text:\s*/i, '')
            .trim();
    };

    const getRenderableLayerText = (layer) => {
        const explicit = String(layer?.textContent || layer?.content || '').trim();
        if (!explicit) return '';
        if (/text_note_(child|group)/i.test(String(layer?.runtimeType || ''))) {
            return explicit;
        }
        if (/lorem ipsum|consectetur|adipiscing|tempor incididunt|sed do eiusmod/i.test(explicit)) {
            return '';
        }
        return explicit;
    };

    const isStandaloneCurrencySymbol = (layer) => {
        const value = String(
            layer?.textContent || layer?.content || layer?.text || layer?.name || layer?.label || ''
        )
            .replace(/^文字[:：]\s*/i, '')
            .replace(/^Text:\s*/i, '')
            .trim();
        return /^[\$¥￥]$/.test(value);
    };

    const getPriceNumber = (value = '') => {
        const match = String(value || '').match(/(?:\$|¥|￥)?\s*(\d+(?:\.\d+)?)/);
        return match ? match[1] : '';
    };

    const getMotionPriceText = (layer, semanticPriceEntries = [], standaloneCurrencyEntries = []) => {
        const text = getRenderableLayerText(layer) || getLayerText(layer);
        if (!text) return text;
        if (/[$¥￥]/.test(text)) return text;

        // A standalone currency note is a separate Magic Layers atom. When it
        // is present beside this numeric Workbench note, never synthesize a
        // second currency prefix from semantic price metadata.
        const isRuntimeTextNote = /text_note_(child|group)/i.test(String(layer?.runtimeType || ''));
        if (isRuntimeTextNote) {
            // Currency is its own Magic Layers atom. Runtime numeric notes
            // must never inherit a "$" from a stale semantic carrier; doing
            // so would draw two symbols or place the symbol over the number.
            return text;
        }

        const number = getPriceNumber(text);
        if (!number) return text;

        // Hydrated price children sometimes keep only the numeric text in
        // textContent while layerName still contains the original currency
        // symbol. Prefer that same-child source before cross-pool matching.
        const ownSources = [layer?.name, layer?.label, layer?.layerName, layer?.originalText];
        const ownCurrencyText = ownSources.find(value => {
            const candidate = String(value || '');
            return /[$¥￥]/.test(candidate) && getPriceNumber(candidate) === number;
        });
        if (ownCurrencyText) {
            const currencyMatch = String(ownCurrencyText).match(/[$¥￥]\s*\d+(?:\.\d+)?/);
            if (currencyMatch) return currencyMatch[0].replace(/\s+/g, '');
        }

        const layerBbox = layer?.bbox;
        const matchingSemanticPrice = semanticPriceEntries.find(candidate => {
            const candidateLayer = candidate?.layer;
            if (getLayerRole(candidateLayer) !== 'price') return false;
            const candidateText = getRenderableLayerText(candidateLayer) || getLayerText(candidateLayer);
            if (!/[$¥￥]/.test(candidateText) || getPriceNumber(candidateText) !== number) return false;
            if (!Array.isArray(layerBbox) || !Array.isArray(candidateLayer?.bbox)) return true;
            return getBboxOverlapRatio(layerBbox, candidateLayer.bbox) >= 0.12 ||
                (Math.abs(getLayerCenterX(layerBbox) - getLayerCenterX(candidateLayer.bbox)) <= 90 &&
                    Math.abs(getLayerCenterY(layerBbox) - getLayerCenterY(candidateLayer.bbox)) <= 90);
        });

        if (matchingSemanticPrice) {
            return getRenderableLayerText(matchingSemanticPrice.layer) || getLayerText(matchingSemanticPrice.layer);
        }
        return text;
    };

    const isTextLikeLayer = (layer) => {
        const name = String(layer?.name || '').toLowerCase();
        const semanticType = String(layer?.semanticType || '').toLowerCase();
        const designRole = String(layer?.designRole || '').toLowerCase();
        const renderMode = String(layer?.renderMode || '').toLowerCase();
        const runtimeType = String(layer?.runtimeType || '').toLowerCase();
        const text = String(layer?.textContent || '').toLowerCase();

        return (
            isStandaloneCurrencySymbol(layer) ||
            renderMode === 'text_css' ||
            runtimeType === 'text_node' ||
            semanticType === 'element_text' ||
            designRole.includes('text') ||
            name.includes('文字') ||
            name.includes('text') ||
            name.includes('logo') ||
            name.includes('price') ||
            name.includes('价格') ||
            /^\$?\s*\d+/.test(text || getLayerText(layer))
        );
    };

    const getLayerRole = (layer) => {
        if (!layer) return 'unknown';
        if (layer.category === 'background') return 'background';

        const combined = [
            layer?.name,
            layer?.semanticType,
            layer?.designRole,
            layer?.renderMode,
            layer?.runtimeType,
            layer?.extractionProfile
        ].map(value => String(value || '').toLowerCase()).join(' ');

        // A price badge is a visual carrier, not the price text itself. Check
        // carrier metadata before the generic price/text keyword classifier.
        const semanticType = String(layer?.semanticType || '').toLowerCase();
        const designRole = String(layer?.designRole || '').toLowerCase();
        const renderMode = String(layer?.renderMode || '').toLowerCase();
        if (
            semanticType === 'shape_panel' ||
            semanticType === 'price_badge' ||
            semanticType === 'cta_button' ||
            designRole === 'local_panel' ||
            designRole === 'price_badge' ||
            renderMode === 'vector_shape'
        ) {
            return 'panel';
        }

        if (isTextLikeLayer(layer)) {
            if (/logo|brand/.test(combined)) return 'logo';
            if (isStandaloneCurrencySymbol(layer) || /price|价格|价签|sale|discount|coupon|¥|\$|￥/.test(combined)) return 'price';
            return 'text';
        }

        if (hasPanelLikeKeywords(combined)) {
            return 'panel';
        }

        if (hasProductLikeKeywords(combined)) {
            return 'product';
        }

        if (hasDecorationLikeKeywords(combined)) {
            return 'decoration';
        }

        return getLayerImageUrl(layer) ? 'decoration' : 'unknown';
    };

    const getLayerMotionStyle = (role) => {
        if (role === 'product') return 'float';
        if (role === 'panel') return 'glide';
        if (role === 'decoration') return 'shimmer';
        return 'freeze';
    };

    const getLayerCenterY = (bbox) => {
        if (!Array.isArray(bbox) || bbox.length !== 4) return 500;
        return (Number(bbox[0]) + Number(bbox[2])) / 2;
    };

    const getLayerCenterX = (bbox) => {
        if (!Array.isArray(bbox) || bbox.length !== 4) return 500;
        return (Number(bbox[1]) + Number(bbox[3])) / 2;
    };

    const getLayerMotionTiming = (entry, slot = 0, isHero = false) => {
        const role = String(entry?.role || 'unknown');
        const centerY = getLayerCenterY(entry?.bbox);
        const centerX = getLayerCenterX(entry?.bbox);
        const area = getLayerAreaScore(entry?.bbox);
        const heroRank = Number.isFinite(Number(entry?.heroRank)) ? Number(entry.heroRank) : -1;
        const isBottom = centerY >= 700;
        const isTop = centerY < 220;
        const isLeft = centerX < 360;
        const isRight = centerX > 640;

        let delayMs = 240;
        let durationMs = 1120;
        let easing = 'cubic-bezier(0.22, 0.61, 0.36, 1)';

        if (role === 'logo') {
            delayMs = 620 + Math.min(slot, 2) * 90;
            durationMs = 980;
            easing = 'cubic-bezier(0.18, 0.9, 0.2, 1)';
        } else if (role === 'panel') {
            delayMs = isBottom ? 120 + slot * 80 : 80 + slot * 90;
            durationMs = isBottom ? 1180 : 1380;
            easing = 'cubic-bezier(0.16, 0.84, 0.24, 1)';
        } else if (role === 'text') {
            delayMs = isTop ? 1380 + slot * 80 : (isBottom ? 1520 : 1460 + slot * 90);
            durationMs = isBottom ? 980 : 1120;
            easing = 'cubic-bezier(0.18, 0.78, 0.18, 1)';
        } else if (role === 'price') {
            delayMs = isBottom ? 1760 + slot * 80 : 1680 + slot * 70;
            durationMs = 920;
            easing = 'cubic-bezier(0.16, 0.88, 0.22, 1)';
        } else if (role === 'product') {
            if (heroRank === 0 || isHero) {
                delayMs = 520;
                durationMs = 1560;
                easing = 'cubic-bezier(0.12, 0.88, 0.18, 1)';
            } else if (heroRank === 1) {
                delayMs = 760;
                durationMs = 1480;
                easing = 'cubic-bezier(0.14, 0.86, 0.18, 1)';
            } else if (area <= 50000 && isBottom) {
                delayMs = 1180 + slot * 130;
                durationMs = 1140;
                easing = 'cubic-bezier(0.16, 0.86, 0.24, 1)';
            } else if (area <= 90000) {
                delayMs = isLeft ? 980 + slot * 110 : isRight ? 1060 + slot * 110 : 940 + slot * 110;
                durationMs = 1280;
                easing = 'cubic-bezier(0.18, 0.82, 0.2, 1)';
            } else {
                delayMs = 920 + slot * 120;
                durationMs = 1380;
            }
        } else if (role === 'decoration') {
            delayMs = isBottom ? 1620 : 1540 + slot * 100;
            durationMs = 1180;
            easing = 'cubic-bezier(0.2, 0.7, 0.2, 1)';
        } else {
            delayMs = 880 + slot * 100;
            durationMs = 1080;
        }

        return {
            delayMs,
            durationMs,
            easing
        };
    };

    const buildLayerMotionStyle = (entry) => {
        const duration = Math.max(360, Number(entry?.motionDurationMs || 0));
        const delay = Math.max(0, Number(entry?.motionDelayMs || 0));
        const easing = entry?.motionEasing || 'cubic-bezier(0.22, 0.61, 0.36, 1)';
        return [
            `--mmp-layer-duration:${duration}ms`,
            `--mmp-layer-delay:${delay}ms`,
            `--mmp-layer-easing:${easing}`,
            `animation-duration:${duration}ms`,
            `animation-delay:${delay}ms`,
            `animation-timing-function:${easing}`
        ].join(';');
    };

    const buildLayerBoxStyle = (layer, totalLayers, index) => {
        const bbox = Array.isArray(layer?.bbox) && layer.bbox.length === 4 ? layer.bbox : [0, 0, 1000, 1000];
        const [ymin, xmin, ymax, xmax] = bbox.map(Number);
        const top = (ymin / 1000) * 100;
        const left = (xmin / 1000) * 100;
        const width = ((xmax - xmin) / 1000) * 100;
        const height = ((ymax - ymin) / 1000) * 100;
        const zIndex = Number.isFinite(Number(layer?.zIndex)) ? Number(layer.zIndex) + 10 : totalLayers - index + 10;
        return `left:${left}%;top:${top}%;width:${width}%;height:${height}%;z-index:${zIndex};`;
    };

    const buildFallbackSliceImageStyle = (bbox) => {
        const safeBbox = Array.isArray(bbox) && bbox.length === 4 ? bbox.map(Number) : [0, 0, 1000, 1000];
        const [, xmin, , xmax] = safeBbox;
        const [ymin, , ymax] = safeBbox;
        const boxWidth = Math.max(1, xmax - xmin);
        const boxHeight = Math.max(1, ymax - ymin);
        const innerWidth = (1000 / boxWidth) * 100;
        const innerHeight = (1000 / boxHeight) * 100;
        const offsetLeft = -((xmin / boxWidth) * 100);
        const offsetTop = -((ymin / boxHeight) * 100);
        return `position:absolute;left:${offsetLeft}%;top:${offsetTop}%;width:${innerWidth}%;height:${innerHeight}%;max-width:none;object-fit:fill;`;
    };

    const buildFallbackLayerSliceHtml = ({
        entry,
        boxStyle,
        sourceUrl
    }) => {
        const proxiedSourceUrl = getProxiedUrl(sourceUrl);
        if (!proxiedSourceUrl) return '';
        const motionStyle = buildLayerMotionStyle(entry);
        return `
            <div class="mml-layer mml-role-${entry.role} ${entry.entryMotion} mml-fallback-slice mmp-anim" style="${boxStyle};${motionStyle};" title="${escapeHtml(entry.name)}">
                <img src="${escapeHtml(proxiedSourceUrl)}" alt="${escapeHtml(entry.name)}" crossorigin="anonymous" style="${buildFallbackSliceImageStyle(entry.bbox)}">
                ${entry.role === 'decoration' ? '<span class="mml-shimmer"></span>' : ''}
            </div>
        `;
    };

    const buildShapeLayerHtml = (entry, boxStyle, motionClass = '') => {
        const shapeStyle = String(entry?.shapeStyleText || '')
            .replace(/(^|;)\s*left\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*top\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*width\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*height\s*:[^;]*/gi, '')
            .replace(/(^|;)\s*z-index\s*:[^;]*/gi, '');
        const motionStyle = buildLayerMotionStyle(entry);
        return `
            <div class="mml-layer mml-role-${entry.role} ${motionClass}" style="${boxStyle};${motionStyle};" title="${escapeHtml(entry.name)}">
                <div class="mml-shape-node" style="width:100%;height:100%;box-sizing:border-box;${shapeStyle}"></div>
            </div>
        `;
    };

    const getBboxOverlapRatio = (a, b) => {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
        const [ay1, ax1, ay2, ax2] = a.map(Number);
        const [by1, bx1, by2, bx2] = b.map(Number);
        const overlapX = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
        const overlapY = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
        const overlap = overlapX * overlapY;
        const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
        const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
        const smallest = Math.min(areaA, areaB);
        return smallest > 0 ? overlap / smallest : 0;
    };

    const getImageNaturalSize = (src) => new Promise(resolve => {
        const resolvedSrc = getProxiedUrl(src);
        if (!resolvedSrc) {
            resolve(null);
            return;
        }

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve({
            width: img.naturalWidth || img.width,
            height: img.naturalHeight || img.height
        });
        img.onerror = () => resolve(null);
        img.src = resolvedSrc;
    });

    const isAtomicMotionLayer = (entry) => {
        if (!entry || entry.role === 'background') {
            return false;
        }
        const runtimeType = String(entry.runtimeType || '').toLowerCase();
        const compositeRole = String(entry.compositeRole || '').toLowerCase();
        const renderMode = String(entry.renderMode || '').toLowerCase();
        const areaRatio = Math.max(0, Number(entry.areaScore || 0)) / 1000000;

        if (runtimeType === 'background_master' || runtimeType === 'semantic_group' || compositeRole === 'composite_group') return false;
        if (Array.isArray(entry.childLayerIds) && entry.childLayerIds.length > 0) return false;
        if (renderMode === 'semantic_group' || renderMode === 'background_plate') return false;
        if (areaRatio > 0.94) return false;

        return true;
    };

    const shouldRenderMotionLayer = (entry) => {
        if (!isAtomicMotionLayer(entry)) return false;
        const isRuntimeTextNote = /text_note_(child|group)/i.test(String(entry?.runtimeType || ''));
        if (
            ['text', 'price', 'logo'].includes(entry.role) &&
            (
                !String(entry.text || '').trim() ||
                (
                    !isRuntimeTextNote &&
                    /补全|lorem ipsum/i.test(String(entry.name || ''))
                )
            )
        ) {
            return false;
        }
        if (entry.imageUrl) return true;
        if (['text', 'price', 'logo'].includes(entry.role) && !!String(entry.text || '').trim()) return true;
        return Array.isArray(entry.bbox) && entry.bbox.length === 4;
    };

    const getRenderableMotionEntryPreferenceScore = (entry) => {
        const runtimeType = String(entry?.runtimeType || '').toLowerCase();
        const source = String(entry?.source || '').toLowerCase();
        const role = String(entry?.role || '').toLowerCase();
        const hasImage = !!entry?.imageUrl;
        const hasClipPath = !!String(entry?.clipPath || '').trim();
        const hasShapeStyle = !!String(entry?.shapeStyleText || '').trim();
        const areaScore = Math.max(0, Number(entry?.areaScore || getLayerAreaScore(entry?.bbox) || 0));

        let score = 0;
        if (runtimeType === 'text_note_group') score += 120;
        else if (runtimeType === 'text_note_child') score += 110;
        else if (runtimeType === 'shape_node') score += 105;
        else if (runtimeType === 'motion_child_asset') score += 100;
        else if (source === 'workbench_child') score += 90;
        else if (source === 'semantic') score += 50;
        else score += 40;

        if (hasImage) score += 20;
        if (hasClipPath) score += 10;
        if (hasShapeStyle) score += 8;
        if (role === 'panel') score += 14;
        if (role === 'product') score += 12;
        if (role === 'text' || role === 'price' || role === 'logo') score += 6;
        score += Math.min(24, areaScore / 12000);
        return score;
    };

    const dedupeMotionLayerEntries = (entries) => {
        const kept = [];
        for (const entry of entries) {
            const duplicate = kept.find(existing => {
                if (existing.role !== entry.role) return false;
                const overlap = getBboxOverlapRatio(existing.bbox, entry.bbox);
                const sameName = String(existing?.name || '').trim() && String(existing?.name || '').trim() === String(entry?.name || '').trim();
                const sameRuntimeType = String(existing?.runtimeType || '').toLowerCase() === String(entry?.runtimeType || '').toLowerCase();
                const sameText = String(existing?.text || '').trim() === String(entry?.text || '').trim();
                const exactBbox = Array.isArray(existing?.bbox) &&
                    Array.isArray(entry?.bbox) &&
                    existing.bbox.join(',') === entry.bbox.join(',');

                if (['text', 'price', 'logo'].includes(entry.role)) {
                    // Runtime text children are independent visual layers. A
                    // shared text_note_child runtime type is not an identity.
                    // Only remove an actual duplicate with the same text/name.
                    if (sameRuntimeType && (existing?.sourceWorkbenchItemId || entry?.sourceWorkbenchItemId)) {
                        return overlap >= 0.94 && sameText && (sameName || exactBbox);
                    }
                    return overlap >= 0.94 && (sameName || sameText || exactBbox);
                }

                return exactBbox || (overlap >= 0.988 && sameName);
            });

            if (!duplicate) {
                kept.push(entry);
                continue;
            }

            const existingScore = getRenderableMotionEntryPreferenceScore(duplicate);
            const nextScore = getRenderableMotionEntryPreferenceScore(entry);
            if (nextScore > existingScore) {
                const replaceIndex = kept.indexOf(duplicate);
                if (replaceIndex >= 0) kept.splice(replaceIndex, 1, entry);
            }
        }

        return kept.sort((a, b) => Number(a?.zIndex || 0) - Number(b?.zIndex || 0));
    };

    const formatMotionExcludedLayerDebugLines = (allLayers = [], renderableLayers = []) => {
        if (!Array.isArray(allLayers) || allLayers.length === 0) return [];
        const renderableIds = new Set(renderableLayers.map(entry => String(entry?.id || '')));
        const excluded = allLayers.filter(entry => !renderableIds.has(String(entry?.id || '')));
        if (excluded.length === 0) return [];

        return excluded.map((entry, index) => {
            const bbox = Array.isArray(entry?.bbox) && entry.bbox.length === 4
                ? entry.bbox.map(value => Math.round(Number(value))).join(',')
                : 'no-bbox';
            const role = entry?.role || 'unknown';
            const text = String(entry?.text || '').trim();
            const likelyDuplicateText = ['text', 'price', 'logo'].includes(role) &&
                (text.length <= 3 || /\$\s*\d+/.test(text));
            const reason = likelyDuplicateText
                ? 'duplicate_or_fragment_text'
                : entry?.imageUrl || role === 'panel'
                    ? 'render_filter'
                    : 'missing_render_source';
            return `- ${index + 1}. ${entry?.name || 'unnamed'} | role=${role} | reason=${reason} | runtime=${entry?.runtimeType || 'unknown'} | source=${entry?.source || 'recipe'} | bbox=${bbox}`;
        });
    };

    const filterMotionArtifactEntries = (entries = []) => {
        const textEntries = entries.filter(entry => ['text', 'price', 'logo'].includes(entry.role));
        const dominantPanels = entries.filter(entry =>
            entry?.role === 'panel' &&
            getLayerAreaScore(entry?.bbox) >= 70000
        );
        if (textEntries.length === 0) return entries;

        return entries.filter(entry => {
            if (!['text', 'price', 'logo'].includes(entry.role)) return true;
            const entryText = String(entry.text || '').trim();
            const entryName = String(entry.name || '').trim();
            const bbox = entry.bbox;
            const isProtectedWorkbenchText =
                ['text', 'price', 'logo'].includes(entry.role) &&
                entry?.source === 'workbench_child' &&
                /text_note_(child|group)/i.test(String(entry?.runtimeType || ''));
            const isStandaloneCurrencyLayer =
                entry.role === 'price' &&
                /^[\$¥￥]$/.test(entryText) &&
                (entry?.source === 'workbench_child' || /text_note_(child|group)/i.test(String(entry?.runtimeType || '')));

            // These are already independent note layers from Workbench. Their
            // short names (NEW, &, COLA, etc.) must not be mistaken for OCR
            // fragments or removed because they sit inside a panel.
            if (isProtectedWorkbenchText || isStandaloneCurrencyLayer) return true;

            const area = getLayerAreaScore(bbox);
            const generatedBySupplement = /补全/i.test(entryName) || /补全/i.test(entryText);
            const tooTiny = area > 0 && area < 5000;
            const tooShort = entryText.length > 0 && entryText.length <= 3;

            const coveredByLargerSibling = textEntries.some(other => {
                if (other === entry) return false;
                const otherText = String(other.text || '').trim();
                if (!otherText) return false;
                if (otherText === entryText) return true;
                if (entryText && otherText.includes(entryText) && otherText.length >= entryText.length + 2) {
                    const overlap = getBboxOverlapRatio(other.bbox, bbox);
                    const sameBand = Math.abs(getLayerCenterY(other.bbox) - getLayerCenterY(bbox)) <= 90;
                    return overlap >= 0.28 || sameBand;
                }
                return false;
            });

            const isSingleCharacterFragment =
                entryText.length === 1 &&
                /^[A-Z0-9$¥￥]$/i.test(entryText) &&
                entry.role !== 'logo';
            const isShortUpperFragment =
                entry.role === 'text' &&
                entryText.length > 0 &&
                entryText.length <= 3 &&
                /^[A-Z0-9&]+$/i.test(entryText);
            const insideDominantPanel = dominantPanels.some(panel => {
                const overlap = getBboxOverlapRatio(panel?.bbox, bbox);
                const sameBand = Math.abs(getLayerCenterY(panel?.bbox) - getLayerCenterY(bbox)) <= 160;
                return overlap >= 0.4 || sameBand;
            });
            const bboxWidth = Array.isArray(bbox) ? Math.max(0, Number(bbox[3]) - Number(bbox[1])) : 0;
            const bboxHeight = Array.isArray(bbox) ? Math.max(0, Number(bbox[2]) - Number(bbox[0])) : 0;
            const narrowFragment = bboxHeight > 0 && (bboxWidth / bboxHeight) < 0.72;

            if (coveredByLargerSibling && (tooTiny || tooShort || generatedBySupplement || isSingleCharacterFragment)) {
                return false;
            }

            if (isSingleCharacterFragment && entry.role === 'price' && /^[\$¥￥]$/.test(entryText)) {
                return true;
            }

            if (isSingleCharacterFragment && tooTiny) {
                return false;
            }

            if (insideDominantPanel && (isSingleCharacterFragment || (isShortUpperFragment && narrowFragment))) {
                return false;
            }

            return true;
        });
    };

    const pickRenderableMotionLayers = (entries) => {
        const filtered = filterMotionArtifactEntries(entries.filter(shouldRenderMotionLayer));
        const deduped = dedupeMotionLayerEntries(filtered);
        const preferred = deduped.filter(entry =>
            entry.imageUrl ||
            (['text', 'price', 'logo'].includes(entry.role) && !!String(entry.text || '').trim()) ||
            ['product', 'decoration', 'panel', 'unknown'].includes(entry.role)
        );
        return preferred.length > 0 ? preferred : deduped;
    };

    const getLayerAreaScore = (bbox) => {
        const safeBbox = Array.isArray(bbox) && bbox.length === 4 ? bbox : [0, 0, 1000, 1000];
        const width = Math.max(0, Number(safeBbox[3]) - Number(safeBbox[1]));
        const height = Math.max(0, Number(safeBbox[2]) - Number(safeBbox[0]));
        return width * height;
    };

    const getVisibleMotionLayerEntries = (item, itemId) => getMotionLayerEntries(item, itemId)
        .filter(({ layer, layerStateIndex }) => {
            const layerState = Number.isInteger(layerStateIndex) ? getLayerState(itemId, layerStateIndex) : null;
            const isVisible = layerState ? layerState.visible !== false : layer?.visible !== false;
            const isHeldSemanticFallback =
                layerState?.visible === false &&
                layer?.quality?.runtimeAction === 'hold';
            return isVisible || isHeldSemanticFallback;
        });

    const getMotionFocusProductLayers = (item, itemId) => {
        const visibleEntries = getVisibleMotionLayerEntries(item, itemId);
        const selectedProducts = visibleEntries
            .filter(({ layer, layerStateIndex }) => {
                const layerState = Number.isInteger(layerStateIndex) ? getLayerState(itemId, layerStateIndex) : null;
                return layerState?.selected && getLayerRole(layer) === 'product';
            })
            .sort((a, b) => getLayerAreaScore(b.layer?.bbox) - getLayerAreaScore(a.layer?.bbox))
            .map(({ layer }) => layer);

        if (selectedProducts.length > 0) {
            return selectedProducts;
        }

        return visibleEntries
            .filter(({ layer }) => getLayerRole(layer) === 'product')
            .sort((a, b) => getLayerAreaScore(b.layer?.bbox) - getLayerAreaScore(a.layer?.bbox))
            .map(({ layer }) => layer);
    };

    const collectSemanticRoleStats = (item) => {
        const pools = [
            ...(Array.isArray(item?.semanticViews?.motionReadyLayers) ? item.semanticViews.motionReadyLayers : []),
            ...(Array.isArray(item?.semanticViews?.editableSceneLayers) ? item.semanticViews.editableSceneLayers : []),
            ...(Array.isArray(item?.semanticViews?.cleanPlateLayers) ? item.semanticViews.cleanPlateLayers : []),
            ...(Array.isArray(item?.scene?.layers) ? item.scene.layers : []),
            ...(Array.isArray(item?.layers) ? item.layers : [])
        ];

        const unique = new Map();
        pools.forEach((layer, index) => {
            if (!layer) return;
            const key = String(layer?.id || `${layer?.name || 'layer'}_${index}`);
            if (!unique.has(key)) unique.set(key, layer);
        });

        const layers = [...unique.values()];
        const panelLayers = layers.filter(layer => getLayerRole(layer) === 'panel');
        const productLayers = layers.filter(layer => getLayerRole(layer) === 'product');
        const largeProductLayers = productLayers.filter(layer => getLayerAreaScore(layer?.bbox) >= 70000);
        return {
            panelCount: panelLayers.length,
            productCount: productLayers.length,
            largeProductCount: largeProductLayers.length
        };
    };

    const collectMotionEntryRoleStats = (item, itemId) => {
        const entries = getMotionLayerEntries(item, itemId);
        const panelEntries = entries.filter(entry => getLayerRole(entry?.layer) === 'panel');
        const productEntries = entries
            .filter(entry => getLayerRole(entry?.layer) === 'product')
            .sort((a, b) => getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox));

        return {
            panelCount: panelEntries.length,
            productCount: productEntries.length,
            largeProductCount: productEntries.filter(entry => getLayerAreaScore(entry?.layer?.bbox) >= 70000).length
        };
    };

    const ensureMotionSemanticCoverage = async (item, itemId) => {
        if (!item) return item;

        refreshMotionReadyViewsForItem(item);
        const textLinesForPanelRestore = getMotionLayerEntries(item, itemId)
            .filter(entry => ['text', 'price', 'logo'].includes(getLayerRole(entry?.layer)))
            .map(entry => ({
                textContent: getRenderableLayerText(entry?.layer) || getLayerText(entry?.layer) || '',
                bbox: Array.isArray(entry?.layer?.bbox) ? entry.layer.bbox : null,
                fontStyle: entry?.layer?.fontStyle || '',
                css: entry?.layer?.css || null
            }))
            .filter(line => String(line?.textContent || '').trim() && Array.isArray(line?.bbox) && line.bbox.length === 4);

        const semanticStatsBeforeRestore = collectSemanticRoleStats(item);
        if (semanticStatsBeforeRestore.panelCount > 0 && textLinesForPanelRestore.length > 0) {
            try {
                await restoreTextContainerShapes({
                    item,
                    itemId,
                    textLines: textLinesForPanelRestore,
                    baseX: parseFloat(item?.el?.style?.left) || 0,
                    baseY: parseFloat(item?.el?.style?.top) || 0,
                    parentWidth: parseFloat(item?.el?.style?.width) || 300,
                    parentHeight: parseFloat(item?.el?.style?.height) || 300,
                    sourceImage: item?.originalDataUrl || item?.dataUrl || item?.cleanPlateDataUrl || null,
                    zIndexBase: parseInt(item?.el?.style?.zIndex || 0, 10) + 1
                });
            } catch (error) {
                console.warn('Failed to restore text container shapes before motion runtime:', error);
            }
        }

        const initialSemanticStats = collectSemanticRoleStats(item);
        const initialMotionStats = collectMotionEntryRoleStats(item, itemId);
        const needsSemanticRebuild =
            initialMotionStats.panelCount === 0 ||
            initialMotionStats.largeProductCount < 2;
        if (!needsSemanticRebuild) return item;

        const baseImage = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
        const sceneLayers = Array.isArray(item?.scene?.layers) && item.scene.layers.length > 0
            ? item.scene.layers
            : (Array.isArray(item?.layers) ? item.layers : []);
        if (!baseImage || sceneLayers.length === 0) return item;

        try {
            const semanticViews = await buildSemanticLayerViews(baseImage, sceneLayers, { expandText: true });
            applySemanticLayerViewsToItem(item, semanticViews);
            refreshMotionReadyViewsForItem(item);
            const rebuiltSemanticStats = collectSemanticRoleStats(item);
            const rebuiltMotionStats = collectMotionEntryRoleStats(item, itemId);
            addMessage({
                sender: 'bot',
                type: 'text',
                content: `🛠️ Motion 语义层已自动刷新：panel ${initialMotionStats.panelCount} -> ${rebuiltMotionStats.panelCount}，主商品候选 ${initialMotionStats.largeProductCount} -> ${rebuiltMotionStats.largeProductCount}（语义池 panel ${initialSemanticStats.panelCount} -> ${rebuiltSemanticStats.panelCount}）`
            });
        } catch (error) {
            console.warn('Failed to rebuild semantic views before motion runtime:', error);
        }

        return item;
    };

    const getLayerEntryMotion = (role, slot = 0, isHero = false, layer = null) => {
        const centerY = getLayerCenterY(layer?.bbox);
        const centerX = getLayerCenterX(layer?.bbox);
        const areaScore = getLayerAreaScore(layer?.bbox);
        const heroRank = Number.isFinite(Number(layer?.heroRank)) ? Number(layer.heroRank) : -1;
        const width = Array.isArray(layer?.bbox) && layer.bbox.length === 4
            ? Math.max(0, Number(layer.bbox[3]) - Number(layer.bbox[1]))
            : 0;
        const height = Array.isArray(layer?.bbox) && layer.bbox.length === 4
            ? Math.max(0, Number(layer.bbox[2]) - Number(layer.bbox[0]))
            : 0;
        const isBottomAccessoryProduct = role === 'product' && centerY >= 700 && areaScore <= 50000;
        const isBottomPanel = role === 'panel' && centerY >= 700;
        const isBottomText = role === 'text' && centerY >= 720;
        const isBottomPrice = role === 'price' && centerY >= 700;
        const isLeft = centerX < 360;
        const isRight = centerX > 640;
        const isCenter = !isLeft && !isRight;
        const isLargeTitle = role === 'text' && areaScore >= 52000;
        const isCompactLabel = role === 'text' && areaScore <= 18000;
        const isMultiLineText = role === 'text' && Number(layer?.renderedLineCount || 1) > 1;
        const isAutoWrappedText = role === 'text' && layer?.isAutoWrappedText === true;
        const isLargePanel = role === 'panel' && areaScore >= 90000;
        const isMediumPanel = role === 'panel' && areaScore >= 30000 && areaScore < 90000;
        const isTallProduct = role === 'product' && height > width * 0.88;
        const isCompactProduct = role === 'product' && areaScore <= 90000;
        const isUpperPrice = role === 'price' && centerY < 520;
        const isTopLogo = role === 'logo' && centerY < 220;
        const isEarlyDecor = role === 'decoration' && centerY < 500;

        if (role === 'product') {
            if (heroRank === 0 || isHero) {
                if (isTallProduct) return 'hero-product-lift';
                if (isLeft) return 'hero-product-swing-left';
                if (isRight) return 'hero-product-swing-right';
                return 'hero-product-focus';
            }
            if (heroRank === 1) {
                if (isLeft) return 'hero-product-focus';
                if (isRight) return 'hero-product-lift';
                return 'support-product-sweep-up';
            }
            if (isBottomAccessoryProduct) {
                if (slot % 3 === 0) return 'accent-product-pop-left';
                if (slot % 3 === 1) return 'accent-product-pop-center';
                return 'accent-product-pop-right';
            }
            if (areaScore > 120000 && isCenter) return 'support-product-sweep-up';
            if (isCompactProduct && isLeft) return 'support-product-skim-left';
            if (isCompactProduct && isRight) return 'support-product-skim-right';
            if (isCenter) return 'support-product-rise-center';
            return slot % 2 === 0 ? 'support-product-left' : 'support-product-right';
        }
        if (role === 'panel') {
            if (isBottomPanel) return 'panel-reveal-tertiary';
            if (isLargePanel && isCenter) return 'panel-unfold-center';
            if (isMediumPanel && isLeft) return 'panel-sweep-left';
            if (isMediumPanel && isRight) return 'panel-sweep-right';
            if (areaScore < 24000) return 'panel-pop-tilt';
            return slot === 0 ? 'panel-reveal-primary' : 'panel-reveal-secondary';
        }
        if (role === 'price') {
            if (isBottomPrice) return 'price-emphasis-late';
            if (isUpperPrice) return 'price-stamp-pop';
            if (areaScore < 14000) return 'price-flip-pop';
            return 'price-emphasis';
        }
        if (role === 'logo') return isTopLogo ? 'logo-establish' : 'logo-drop-in';
        if (role === 'text') {
            if (isAutoWrappedText) {
                return isLargeTitle ? 'headline-establish' : 'copy-settle';
            }
            if (isMultiLineText) {
                return isLargeTitle ? 'headline-establish' : 'caption-rise';
            }
            if (isBottomText) return 'caption-rise';
            if (isLargeTitle && isLeft) return 'headline-slam-left';
            if (isLargeTitle && (isRight || isCenter)) return 'headline-slam-right';
            if (isCompactLabel && isRight) return 'copy-slide-right';
            if (isCompactLabel && isLeft) return 'copy-slide-left';
            if (areaScore >= 22000 && width > height * 1.5) return 'text-roll-in';
            if (areaScore <= 12000) return 'text-bounce-in';
            return slot === 0 ? 'headline-establish' : 'copy-settle';
        }
        if (role === 'decoration') {
            if (isEarlyDecor && isLeft) return 'decor-orbit-soft';
            if (isRight) return 'decor-flyby-right';
            if (isLeft) return 'decor-flyby-left';
            return slot % 2 === 0 ? 'decor-glint-early' : 'decor-glint-late';
        }
        return 'soft-settle';
    };

    const getLayerSequenceLabel = (role, slot = 0, isHero = false) => {
        if (role === 'logo') return '品牌露出';
        if (role === 'text') return slot === 0 ? '标题建立' : '文案收束';
        if (role === 'panel') return '版式铺垫';
        if (role === 'product') return isHero ? '主商品上场' : '辅商品跟进';
        if (role === 'price') return '价格点题';
        if (role === 'decoration') return '光效收尾';
        return '轻微收束';
    };

    const buildRecipeSummary = (recipeLayers) => {
        const sequenceOrder = [
            '版式铺垫',
            '品牌露出',
            '标题建立',
            '主商品上场',
            '辅商品跟进',
            '价格点题',
            '光效收尾'
        ];
        const layerLabels = new Set(recipeLayers.map(entry => entry.sequenceLabel).filter(Boolean));
        const presentStages = sequenceOrder.filter(label => layerLabels.has(label));
        return ['底板起势', ...presentStages, '最终收敛为稳定图层合成'].join(' -> ');
    };

    const formatMotionLayerDebugLines = (layers = []) => {
        if (!Array.isArray(layers) || layers.length === 0) {
            return ['- 无可用图层'];
        }

        return layers.map((entry, index) => {
            const bbox = Array.isArray(entry?.bbox) && entry.bbox.length === 4
                ? entry.bbox.map(value => Math.round(Number(value))).join(',')
                : 'no-bbox';
            const flags = [
                `role=${entry?.role || 'unknown'}`,
                `cutout=${entry?.imageUrl ? 'yes' : 'no'}`,
                `runtime=${entry?.runtimeType || 'unknown'}`,
                `render=${entry?.renderMode || 'unknown'}`,
                `source=${entry?.source || 'recipe'}`,
                `bbox=${bbox}`,
                `motion=${entry?.entryMotion || 'none'}`,
                `seq=${entry?.sequenceLabel || 'none'}`,
                `delay=${Math.round(Number(entry?.motionDelayMs || 0))}`,
                `dur=${Math.round(Number(entry?.motionDurationMs || 0))}`,
                `order=${Math.round(Number(entry?.renderOrder ?? entry?.zIndex ?? 0))}`
            ];
            if (Number.isFinite(Number(entry?.heroRank)) && Number(entry?.heroRank) >= 0) {
                flags.push(`heroRank=${Number(entry.heroRank)}`);
            }
            if (['text', 'price', 'logo'].includes(entry?.role)) {
                flags.push(`text=${String(entry?.text || '').replace(/\s+/g, ' ').slice(0, 32) || '(empty)'}`);
                flags.push(`lines=${Number(entry?.renderedLineCount || 1)}`);
                flags.push(`explicit=${Number(entry?.explicitLineCount || 1)}`);
                flags.push(`wrap=${entry?.isAutoWrappedText ? 'auto' : (entry?.hasExplicitBreaks ? 'explicit' : 'single')}`);
                if (entry?.isAutoWrappedText) {
                    flags.push(`wrapW=${Math.round(Number(entry?.wrapperWidthPx || 0))}`);
                    flags.push(`naturalW=${Math.round(Number(entry?.naturalSingleLineWidthPx || 0))}`);
                }
            }
            return `- ${index + 1}. ${entry?.name || 'unnamed'} | ${flags.join(' | ')}`;
        });
    };

    const formatSemanticPoolDebugLines = (item) => {
        const pools = [
            ...(Array.isArray(item?.semanticViews?.motionReadyLayers) ? item.semanticViews.motionReadyLayers.map(layer => ({ layer, pool: 'motionReady' })) : []),
            ...(Array.isArray(item?.semanticViews?.editableSceneLayers) ? item.semanticViews.editableSceneLayers.map(layer => ({ layer, pool: 'editable' })) : []),
            ...(Array.isArray(item?.semanticViews?.cleanPlateLayers) ? item.semanticViews.cleanPlateLayers.map(layer => ({ layer, pool: 'cleanPlate' })) : []),
            ...(Array.isArray(item?.scene?.layers) ? item.scene.layers.map(layer => ({ layer, pool: 'scene' })) : []),
            ...(Array.isArray(item?.layers) ? item.layers.map(layer => ({ layer, pool: 'layers' })) : [])
        ];

        const seen = new Set();
        const unique = pools.filter(({ layer, pool }, index) => {
            const key = `${pool}:${String(layer?.id || layer?.name || index)}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });

        const interesting = unique.filter(({ layer }) => {
            const role = getLayerRole(layer);
            return role === 'panel' || role === 'product';
        });

        if (interesting.length === 0) return ['- 无 panel/product 候选'];

        return interesting
            .sort((a, b) => {
                const roleA = getLayerRole(a.layer);
                const roleB = getLayerRole(b.layer);
                if (roleA !== roleB) return roleA.localeCompare(roleB);
                return getLayerAreaScore(b.layer?.bbox) - getLayerAreaScore(a.layer?.bbox);
            })
            .map(({ layer, pool }, index) => {
                const role = getLayerRole(layer);
                const bbox = Array.isArray(layer?.bbox) && layer.bbox.length === 4
                    ? layer.bbox.map(value => Math.round(Number(value))).join(',')
                    : 'no-bbox';
                return `- ${index + 1}. ${layer?.name || 'unnamed'} | role=${role} | pool=${pool} | runtime=${layer?.runtimeType || 'unknown'} | render=${layer?.renderMode || 'unknown'} | bbox=${bbox}`;
            });
    };

    const formatForcedCoverageDebugLines = (layers = []) => {
        const forced = (Array.isArray(layers) ? layers : []).filter(entry =>
            /^semantic_forced_/.test(String(entry?.source || ''))
        );
        if (forced.length === 0) return ['- 无强制补层'];
        return forced.map((entry, index) => {
            const bbox = Array.isArray(entry?.layer?.bbox) && entry.layer.bbox.length === 4
                ? entry.layer.bbox.map(value => Math.round(Number(value))).join(',')
                : 'no-bbox';
            return `- ${index + 1}. ${entry?.layer?.name || 'unnamed'} | role=${getLayerRole(entry?.layer)} | source=${entry?.source} | runtime=${entry?.layer?.runtimeType || 'unknown'} | render=${entry?.layer?.renderMode || 'unknown'} | bbox=${bbox}`;
        });
    };

    const getSemanticPoolEntriesForMotion = (item, itemId) => {
        const pools = [
            { layers: Array.isArray(item?.semanticViews?.motionReadyLayers) ? item.semanticViews.motionReadyLayers : [], source: 'semantic_pool_motionReady' },
            { layers: Array.isArray(item?.semanticViews?.editableSceneLayers) ? item.semanticViews.editableSceneLayers : [], source: 'semantic_pool_editable' },
            { layers: Array.isArray(item?.semanticViews?.cleanPlateLayers) ? item.semanticViews.cleanPlateLayers : [], source: 'semantic_pool_cleanPlate' },
            { layers: Array.isArray(item?.scene?.layers) ? item.scene.layers : [], source: 'semantic_pool_scene' },
            { layers: Array.isArray(item?.layers) ? item.layers : [], source: 'semantic_pool_layers' }
        ];

        const pooled = pools.flatMap(({ layers, source }) => layers.map(layer => ({
            layer,
            layerStateIndex: findMotionLayerStateIndex(item, layer),
            source
        })));

        return dedupeMotionEntryPool(pooled);
    };

    const ensureRecipeSemanticCoverage = (visibleEntries = [], item, itemId) => {
        const semanticPoolEntries = getSemanticPoolEntriesForMotion(item, itemId);
        let nextEntries = [...visibleEntries];
        const semanticPanels = semanticPoolEntries
            .filter(entry => getLayerRole(entry?.layer) === 'panel')
            .filter(entry => {
                const area = getLayerAreaScore(entry?.layer?.bbox);
                return area >= 4000 && area <= 260000;
            })
            .sort((a, b) => {
                const areaDelta = getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox);
                if (Math.abs(areaDelta) > 1) return areaDelta;
                return getLayerCenterY(a?.layer?.bbox) - getLayerCenterY(b?.layer?.bbox);
            })
            .map(entry => ({ ...entry, source: 'semantic_forced_panel' }));

        nextEntries = mergeFallbackMotionEntries(nextEntries, semanticPanels, 'semantic_forced_panel');

        const semanticHeroProducts = semanticPoolEntries
            .filter(entry => getLayerRole(entry?.layer) === 'product')
            .filter(entry => {
                const area = getLayerAreaScore(entry?.layer?.bbox);
                const centerY = getLayerCenterY(entry?.layer?.bbox);
                return area >= 25000 && centerY <= 700;
            })
            .sort((a, b) => {
                const topBias = getLayerCenterY(a?.layer?.bbox) - getLayerCenterY(b?.layer?.bbox);
                if (Math.abs(topBias) > 8) return topBias;
                return getLayerAreaScore(b?.layer?.bbox) - getLayerAreaScore(a?.layer?.bbox);
            })
            .slice(0, 4)
            .map(entry => ({ ...entry, source: 'semantic_forced_product' }));

        nextEntries = mergeFallbackMotionEntries(nextEntries, semanticHeroProducts, 'semantic_forced_product');

        return nextEntries;
    };

    const buildLayeredMotionPromptPrefix = (recipe) => {
        const layers = Array.isArray(recipe?.layers) ? recipe.layers : [];
        const heroProducts = layers
            .filter(entry => entry.role === 'product' && entry.isHero)
            .map(entry => entry.name)
            .filter(Boolean)
            .slice(0, 1);
        const supportProducts = layers
            .filter(entry => entry.role === 'product' && !entry.isHero)
            .map(entry => entry.name)
            .filter(Boolean)
            .slice(0, 3);
        const hasPrice = layers.some(entry => entry.role === 'price');
        const hasLogo = layers.some(entry => entry.role === 'logo');
        const hasText = layers.some(entry => entry.role === 'text');
        const hasPanel = layers.some(entry => entry.role === 'panel');
        const hasDecoration = layers.some(entry => entry.role === 'decoration');

        const stageNotes = [
            'This image already has separated Magic Layers and should be animated like a polished ad composition, not a generic image-to-video shot.',
            'First frame should feel like the clean stage/base plate. Middle section should reveal the layered composition with controlled motion. Final frame must resolve exactly back to the original design layout.'
        ];

        if (heroProducts.length > 0) {
            stageNotes.push(`Primary focus product: ${heroProducts.join(', ')}. Let it make the strongest entrance with mild depth and parallax.`);
        }
        if (supportProducts.length > 0) {
            stageNotes.push(`Supporting product layers: ${supportProducts.join(', ')}. Let them follow slightly later with lighter motion than the hero product.`);
        }
        if (hasPanel) {
            stageNotes.push('Panel and card layers should act as layout support only, with subtle reveal motion and no large displacement.');
        }
        if (hasText || hasPrice || hasLogo) {
            stageNotes.push('All text, price, logo, and brand information must remain crisp, readable, and locked to their design positions with no re-layout or text mutation.');
        }
        if (hasPrice) {
            stageNotes.push('Price emphasis should happen near the end as a restrained ad beat, then settle back perfectly.');
        }
        if (hasDecoration) {
            stageNotes.push('Decorative light or accent layers may add one soft highlight sweep near the end.');
        }

        stageNotes.push('Avoid morphing, duplicated elements, drifting typography, or layout destruction. Motion should stay mild, premium, and composition-preserving.');
        return stageNotes.join('\n');
    };

    const buildTextLayerHtml = (layer, boxStyle, role, motionClass = '') => {
        const text = getRenderableLayerText(layer);
        if (!text) return '';

        const bbox = Array.isArray(layer?.bbox) && layer.bbox.length === 4 ? layer.bbox : [0, 0, 1000, 1000];
        const heightPct = Math.max(1, Number(bbox[2]) - Number(bbox[0])) / 1000;
        const fontSize = layer?.fontSize || `${Math.max(9, Math.min(56, Math.round(320 * heightPct * 0.52)))}px`;
        const color = layer?.fontColor || layer?.css?.color || '#ffffff';
        const weight = layer?.fontWeight || layer?.css?.fontWeight || '800';
        const family = layer?.fontFamily || layer?.css?.fontFamily || 'Impact, Arial Black, sans-serif';
        const styleText = String(layer?.textStyleText || '');
        const visualTransform = layer?.textVisualTransform ? `transform:${layer.textVisualTransform};transform-origin:center center;` : '';
        const className = role === 'price' ? 'mml-text-price' : role === 'logo' ? 'mml-text-logo' : 'mml-text-frozen';
        const textHtml = layer?.textHtml || escapeHtml(text).replace(/\n/g, '<br>');
        const motionStyle = buildLayerMotionStyle(layer);

        return `
            <div class="mml-layer ${className} ${motionClass}" style="${boxStyle};${motionStyle};">
                <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;box-sizing:border-box;overflow:hidden;">
                    <div class="mml-text-node" style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;text-align:${escapeHtml(layer?.textAlign || 'center')};color:${escapeHtml(color)};font-family:${escapeHtml(family)};font-weight:${escapeHtml(weight)};font-size:${escapeHtml(fontSize)};line-height:${escapeHtml(layer?.lineHeight || '0.92')};letter-spacing:${escapeHtml(layer?.letterSpacing || '0')};font-style:${escapeHtml(layer?.fontStyle || 'normal')};white-space:${escapeHtml(layer?.whiteSpace || 'pre-wrap')};text-shadow:${escapeHtml(layer?.textShadow || '0 1px 2px rgba(0,0,0,0.18)')};${layer?.WebkitTextStroke ? `-webkit-text-stroke:${escapeHtml(layer.WebkitTextStroke)};` : ''}${visualTransform}${styleText}">
                        ${textHtml}
                    </div>
                </div>
            </div>
        `;
    };

    const isStandaloneCurrencyMotionEntry = (entry) => /^[\$¥￥]$/.test(String(entry?.text || '').trim());

    const normalizeMotionCurrencyAtomLayout = (entries = []) => {
        const currencyEntries = entries.filter(entry =>
            entry?.role === 'price' &&
            isStandaloneCurrencyMotionEntry(entry) &&
            Array.isArray(entry?.bbox) &&
            /text_note_(child|group)/i.test(String(entry?.runtimeType || ''))
        );
        const numericEntries = entries.filter(entry =>
            entry?.role === 'price' &&
            !isStandaloneCurrencyMotionEntry(entry) &&
            /^\s*\d+(?:\.\d+)?\s*$/.test(String(entry?.text || '')) &&
            Array.isArray(entry?.bbox) &&
            /text_note_(child|group)/i.test(String(entry?.runtimeType || ''))
        );
        const usedNumbers = new Set();

        currencyEntries.forEach(currency => {
            const currencyBbox = currency.bbox.map(Number);
            const currencyCenterY = getLayerCenterY(currencyBbox);
            const currencyCenterX = getLayerCenterX(currencyBbox);
            const candidate = numericEntries
                .filter(entry => !usedNumbers.has(entry))
                .map(entry => ({
                    entry,
                    bbox: entry.bbox.map(Number),
                    yDistance: Math.abs(getLayerCenterY(entry.bbox) - currencyCenterY),
                    xDistance: Math.abs(getLayerCenterX(entry.bbox) - currencyCenterX)
                }))
                .filter(candidate => candidate.yDistance <= 120)
                .sort((a, b) => (a.yDistance - b.yDistance) || (a.xDistance - b.xDistance))[0];
            if (!candidate) return;

            usedNumbers.add(candidate.entry);
            const numberBbox = candidate.bbox;
            const currencyRight = Number(currencyBbox[3]);
            const numberLeft = Number(numberBbox[1]);
            const numberWidth = Math.max(1, Number(numberBbox[3]) - Number(numberBbox[1]));
            const gap = Math.max(8, Math.round(Math.min(
                Number(currencyBbox[2]) - Number(currencyBbox[0]),
                Number(numberBbox[2]) - Number(numberBbox[0])
            ) * 0.12));

            // The Workbench stores these as independent atoms. If hydration
            // or text fitting makes the numeric box overlap the currency box,
            // preserve both widths and move only the number to the right.
            if (numberLeft < currencyRight + gap) {
                const nextLeft = currencyRight + gap;
                candidate.entry.bbox = [
                    numberBbox[0],
                    nextLeft,
                    numberBbox[2],
                    nextLeft + numberWidth
                ];
                candidate.entry.layoutBbox = candidate.entry.bbox;
            }
        });

        return entries;
    };

    const buildAutoMotionRecipe = (item, itemId, prompt) => {
        const visibleLayers = ensureRecipeSemanticCoverage(
            getVisibleMotionLayerEntries(item, itemId),
            item,
            itemId
        );

        const productEntries = visibleLayers
            .map(({ layer, layerStateIndex }) => ({
                layer,
                index: Number.isInteger(layerStateIndex) ? layerStateIndex : null,
                role: getLayerRole(layer),
                areaScore: getLayerAreaScore(layer?.bbox)
            }))
            .filter(entry => entry.role === 'product')
            .sort((a, b) => b.areaScore - a.areaScore);

        const featuredProductRanks = new Map();
        productEntries.slice(0, 2).forEach((entry, rank) => {
            const productKey = entry?.layer?.id || entry?.index;
            if (productKey !== null && productKey !== undefined && !featuredProductRanks.has(productKey)) {
                featuredProductRanks.set(productKey, rank);
            }
        });

        const roleSlots = new Map();
        const textGroupSlots = new Map();
        const semanticPriceEntries = getSemanticPoolEntriesForMotion(item, itemId)
            .filter(entry => getLayerRole(entry?.layer) === 'price');
        const standaloneCurrencyEntries = visibleLayers.filter(entry =>
            getLayerRole(entry?.layer) === 'price' &&
            isStandaloneCurrencySymbol(entry?.layer)
        );

        const recipeLayers = visibleLayers
            .filter(({ layer }) => getLayerRole(layer) !== 'background')
            .map(({ layer, layerStateIndex, source }, index) => {
            const role = getLayerRole(layer);
            const groupKey = ['text', 'price', 'logo'].includes(role)
                ? String(layer?.textGroupKey || layer?.sourceTextLayerId || layer?.id || '')
                : '';
            let currentSlot = roleSlots.get(role) || 0;
            if (groupKey) {
                if (textGroupSlots.has(groupKey)) {
                    currentSlot = textGroupSlots.get(groupKey);
                } else {
                    textGroupSlots.set(groupKey, currentSlot);
                    roleSlots.set(role, currentSlot + 1);
                }
            } else {
                roleSlots.set(role, currentSlot + 1);
            }
            const productKey = layer?.id || (Number.isInteger(layerStateIndex) ? layerStateIndex : index);
            const heroRank = role === 'product' && featuredProductRanks.has(productKey)
                ? featuredProductRanks.get(productKey)
                : null;
            const isHero = role === 'product' && heroRank === 0;
            const isFeatured = role === 'product' && heroRank !== null;
            const motionLayer = {
                ...layer,
                role,
                heroRank,
                isHero,
                isFeatured
            };
            const entryMotion = getLayerEntryMotion(role, currentSlot, isHero, motionLayer);
            const motionTiming = getLayerMotionTiming({
                role,
                bbox: getMotionStableBbox(layer),
                heroRank,
                isHero,
                isFeatured
            }, currentSlot, isHero);
            return {
                id: layer?.id || `layer-${index}`,
                name: layer?.name || `图层 ${index + 1}`,
                role,
                motion: getLayerMotionStyle(role),
                entryMotion,
                sequenceLabel: getLayerSequenceLabel(role, currentSlot, isFeatured),
                sequenceSlot: currentSlot,
                isHero,
                isFeatured,
                heroRank,
                motionDelayMs: motionTiming.delayMs,
                motionDurationMs: motionTiming.durationMs,
                motionEasing: motionTiming.easing,
                imageUrl: getLayerImageUrl(layer),
                text: role === 'text' || role === 'logo'
                    ? getRenderableLayerText(layer)
                    : role === 'price'
                        ? getMotionPriceText(layer, semanticPriceEntries, standaloneCurrencyEntries)
                        : '',
                textHtml: layer?.textHtml || '',
                textContent: layer?.textContent || layer?.content || '',
                textFragments: Array.isArray(layer?.textFragments) ? layer.textFragments : [],
                textStyleText: layer?.textStyleText || '',
                textVisualTransform: layer?.textVisualTransform || '',
                fontColor: layer?.fontColor || '',
                fontFamily: layer?.fontFamily || '',
                fontWeight: layer?.fontWeight || '',
                fontStyle: layer?.fontStyle || '',
                fontSize: layer?.fontSize || '',
                lineHeight: layer?.lineHeight || '',
                letterSpacing: layer?.letterSpacing || '',
                textAlign: layer?.textAlign || '',
                textShadow: layer?.textShadow || '',
                WebkitTextStroke: layer?.WebkitTextStroke || '',
                whiteSpace: layer?.whiteSpace || '',
                renderedLineCount: Number(layer?.renderedLineCount || 1),
                explicitLineCount: Number(layer?.explicitLineCount || 1),
                isAutoWrappedText: layer?.isAutoWrappedText === true,
                hasExplicitBreaks: layer?.hasExplicitBreaks === true,
                wrapperWidthPx: Number(layer?.wrapperWidthPx || 0),
                naturalSingleLineWidthPx: Number(layer?.naturalSingleLineWidthPx || 0),
                shapeStyleText: layer?.shapeStyleText || '',
                clipPath: layer?.clipPath || '',
                fillColor: layer?.fillColor || '',
                shapeType: layer?.shapeType || '',
                borderRadius: layer?.borderRadius || '',
                bbox: getMotionStableBbox(layer),
                zIndex: layer?.zIndex,
                areaScore: getLayerAreaScore(getMotionStableBbox(layer)),
                index,
                runtimeType: layer?.runtimeType || '',
                compositeRole: layer?.compositeRole || '',
                renderMode: layer?.renderMode || '',
                designRole: layer?.designRole || '',
                semanticType: layer?.semanticType || '',
                childLayerIds: Array.isArray(layer?.childLayerIds) ? layer.childLayerIds : [],
                source: source || layer?.source || '',
                textGroupKey: layer?.textGroupKey || ''
            };
        });

        normalizeMotionCurrencyAtomLayout(recipeLayers);

        const motionSummary = recipeLayers
            .filter(entry => entry.role !== 'background')
            .length > 0
            ? buildRecipeSummary(recipeLayers.filter(entry => entry.role !== 'background'))
            : '从底板逐步收敛为所有图层稳定合成';

        const renderableLayers = pickRenderableMotionLayers(recipeLayers);

        return {
            id: `motion-recipe-${Date.now()}`,
            name: 'Auto Motion Recipe',
            prompt,
            durationMs: Math.min(11800, Math.max(9200, 9000 + renderableLayers.length * 180)),
            stageMotion: 'plate_to_original_settle',
            layers: recipeLayers,
            renderableLayers,
            summary: motionSummary || '从底板逐步收敛为所有图层稳定合成'
        };
    };

    const buildAutoMotionPreviewHtml = async ({ item, itemId, recipe }) => {
        const sceneDataUrl = await exportCurrentSceneImage(itemId);
        const plateCompositeUrl = item?.cleanPlateDataUrl || item?.originalDataUrl || item?.dataUrl || sceneDataUrl || '';
        const finalCompositeUrl = item?.originalDataUrl || item?.dataUrl || sceneDataUrl || plateCompositeUrl || '';
        let persistablePlateUrl = plateCompositeUrl;
        if (typeof persistablePlateUrl === 'string' && (persistablePlateUrl.startsWith('data:') || persistablePlateUrl.startsWith('blob:'))) {
            try {
                persistablePlateUrl = await uploadImageToOSS(persistablePlateUrl, {
                    sessionId: state.currentSessionId || itemId
                });
            } catch (error) {
                console.warn('Failed to upload Motion plate before persisting preview:', error);
                // Keep the local source for this in-page preview. Clearing it
                // makes the compositor fall back to a flat color and hides the
                // actual Magic Layers result. Session sync will externalize or
                // scrub the data URL separately when the service is available.
                persistablePlateUrl = plateCompositeUrl;
            }
        }
        const finalSize = await getImageNaturalSize(finalCompositeUrl);
        const containerId = `mmp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const rectWidth = parseFloat(item?.el?.style?.width) || 3;
        const rectHeight = parseFloat(item?.el?.style?.height) || 4;
        const aspectRatio = finalSize?.width && finalSize?.height
            ? `${finalSize.width} / ${finalSize.height}`
            : `${Math.max(1, rectWidth)} / ${Math.max(1, rectHeight)}`;

        const motionLayers = Array.isArray(recipe?.renderableLayers) && recipe.renderableLayers.length > 0
            ? recipe.renderableLayers
            : pickRenderableMotionLayers(recipe.layers);
        // A slice source is serialized inside the chat HTML payload. Never
        // fall back to a local data/blob URL here after the upload attempt.
        const fallbackSliceSourceUrl = persistablePlateUrl || '';
        const stageSize = getMotionStageSize(finalSize, 900, Math.round(900 * (rectHeight / Math.max(1, rectWidth))));
        const payload = {
            version: 1,
            durationMs: recipe.durationMs,
            stage: stageSize,
            plateUrl: getProxiedUrl(persistablePlateUrl) || persistablePlateUrl,
            layers: motionLayers.map(entry => buildCanvasMotionLayerPayload(entry, recipe.durationMs, fallbackSliceSourceUrl, stageSize))
        };

        // Product/panel assets can also come from a local data URL after
        // hydration. Upload those URLs before embedding the payload in chat;
        // otherwise one extracted child can recreate the same oversized HTML
        // problem even when the plate itself is remote.
        await Promise.all(payload.layers.map(async (layer) => {
            const imageUrl = layer?.imageUrl;
            // SVG snapshots are intentionally kept inline. They are used for
            // vector panels and text fallback, and the upload endpoint only
            // accepts raster image assets.
            if (!isUploadableMotionRasterUrl(imageUrl)) return;
            try {
                const uploadedUrl = await uploadImageToOSS(imageUrl, {
                    sessionId: state.currentSessionId || itemId
                });
                if (uploadedUrl && !uploadedUrl.startsWith('data:') && !uploadedUrl.startsWith('blob:')) {
                    layer.imageUrl = getProxiedUrl(uploadedUrl) || uploadedUrl;
                }
            } catch (error) {
                console.warn(`Failed to upload Motion layer asset ${layer.name || layer.id}:`, error);
                // An upload failure must not turn a real panel/product into an
                // empty timeline entry. Keep the local asset for the current
                // preview; the cloud cleanup path handles persistence later.
            }
        }));

        return {
            containerId,
            motionLayerCount: motionLayers.length,
            debugLines: formatMotionLayerDebugLines(motionLayers),
            excludedDebugLines: formatMotionExcludedLayerDebugLines(recipe.layers, motionLayers),
            html: `
            <style>
                #${containerId}.mmp-wrap { width:100%; max-width:360px; }
                #${containerId} .mmp-stage {
                    position:relative;
                    width:100%;
                    aspect-ratio:${aspectRatio};
                    overflow:hidden;
                    border-radius:16px;
                    background:#f97316;
                    box-shadow:0 10px 28px rgba(15,23,42,0.16);
                    isolation:isolate;
                }
                #${containerId} .mmp-canvas-stage {
                    width:100%;
                    height:100%;
                    display:block;
                    background:#f97316;
                }
                #${containerId} .mmp-meta {
                    margin-top:10px;
                    padding:10px 12px;
                    border-radius:12px;
                    background:#f8fafc;
                    border:1px solid rgba(148,163,184,0.25);
                    font-size:12px;
                    color:#475569;
                    line-height:1.45;
                }
                #${containerId} .mmp-meta strong { color:#0f172a; }
                #${containerId} .mmp-toolbar {
                    margin-top:8px;
                    display:flex;
                    justify-content:flex-end;
                }
                #${containerId} .mmp-replay {
                    border:none;
                    border-radius:999px;
                    background:#0f172a;
                    color:#fff;
                    font-size:12px;
                    padding:6px 10px;
                    cursor:pointer;
                }
                #${containerId} .mmp-payload {
                    display:none !important;
                }
            </style>
            <div id="${containerId}" class="mmp-wrap">
                <div class="mmp-stage">
                    <canvas class="mmp-canvas-stage" width="${stageSize.width}" height="${stageSize.height}"></canvas>
                </div>
                <div class="mmp-toolbar">
                    <button class="mmp-replay" onclick="window.startMagicMotionCanvasPreview && window.startMagicMotionCanvasPreview('${containerId}')">Replay</button>
                </div>
                <div class="mmp-meta">
                    <strong>Magic Motion Preview</strong><br>
                    首帧：底板 / 尾帧：所有图层稳定合成 · 时长：${Math.round(recipe.durationMs / 1000)}s<br>
                    图层数：${motionLayers.length}<br>
                    ${escapeHtml(recipe.summary)}<br>
                    说明：系统基于 Magic Layers 的真实图层资产自动编排广告动画，中段以 compositor timeline 播放原子图层轨道，尾帧保留所有图层稳定后的合成结果。
                </div>
                <script type="application/json" class="mmp-payload">${serializeJsonForHtmlScript(payload)}</script>
            </div>
        `
        };
    };

    const buildMotionPreviewSource = async (item, itemId) => {
        const layers = item?.scene?.layers?.length ? item.scene.layers : item?.layers;
        if (Array.isArray(layers) && layers.length > 0) {
            const sceneDataUrl = await exportCurrentSceneImage(itemId);
            if (sceneDataUrl) {
                const selectedLayerNames = layers
                    .map((layer, index) => ({ layer, index }))
                    .filter(({ layer, index }) => {
                        const layerState = getLayerState(itemId, index);
                        return layerState?.selected && layer?.category !== 'background';
                    })
                    .map(({ layer }) => layer?.name)
                    .filter(Boolean);

                return {
                    imageDataUrl: sceneDataUrl,
                    promptPrefix: selectedLayerNames.length
                        ? `This is a layered composition. Prioritize motion on: ${selectedLayerNames.join(', ')}. Keep other layers stable unless the prompt says otherwise.`
                        : 'This is a layered composition exported from Magic Layers. Animate the composite naturally while preserving the layered structure.',
                    sourceType: 'magic_layers_scene'
                };
            }
        }

        const fallbackImage = await fileToDataURL(item?.file || item?.dataUrl || item?.originalDataUrl);
        return {
            imageDataUrl: fallbackImage,
            promptPrefix: '',
            sourceType: 'single_image'
        };
    };

    const buildMotionPromptSuggestion = (item, itemId) => {
        const layers = item?.scene?.layers?.length ? item.scene.layers : item?.layers;
        if (!Array.isArray(layers) || layers.length === 0) {
            return '镜头缓慢推进，主体保持清晰稳定，加入轻微光影流动，让整张画面更像高级广告短片。';
        }

        const visibleForegroundLayers = getVisibleMotionLayerEntries(item, itemId)
            .map(({ layer }) => layer)
            .filter(layer => layer?.category !== 'background');

        const textLayers = visibleForegroundLayers.filter(layer =>
            layer?.renderMode === 'text_css' ||
            layer?.runtimeType === 'text_node' ||
            layer?.semanticType === 'element_text' ||
            String(layer?.name || '').includes('文字') ||
            String(layer?.name || '').toLowerCase().includes('text')
        );

        const focusLayers = getMotionFocusProductLayers(item, itemId).slice(0, 3);

        const focusNames = focusLayers
            .map(layer => layer?.name)
            .filter(Boolean)
            .slice(0, 3);

        const hasText = textLayers.length > 0;
        const hasBackground = layers.some(layer => layer?.category === 'background');

        if (focusNames.length > 0 && hasText) {
            return `镜头轻微推进，真实商品层 ${focusNames.join('、')} 依次形成轻微空间视差与层次展开，中段完成广告式上场，尾帧完整回到原始排版，文字、价格与 Logo 保持清晰稳定，仅加入一遍柔和高光扫过，整体像精致广告动态物料。`;
        }

        if (focusNames.length > 0) {
            return `镜头轻微推进，真实商品层 ${focusNames.join('、')} 做层次分明的广告式上场与轻微立体视差，背景只保留细微光影流动，尾帧稳定收束为所有图层的最终合成，不要剧烈变形。`;
        }

        if (hasText && hasBackground) {
            return '镜头轻微推进，真实主体做小幅空间位移与节奏展开，尾帧稳定收束为所有图层的最终合成，文字信息保持清晰不抖动，仅保留柔和光影和空气感变化。';
        }

        return '镜头轻微前移，真实主体做小幅广告式层次展开，尾帧稳定收束为所有图层的最终合成，加入柔和光影和空气感变化，让画面更像可投放的动态物料。';
    };

    bindToolboxBtn('.select', () => {
        pushSelectedToChat();
        window.hideWorkbenchToolbox();
    });
    

    bindToolboxBtn('.ai-spark', () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        
        window.hideWorkbenchToolbox();
        if (typeof window.showFloatingFusionEditor === 'function') {
            window.showFloatingFusionEditor(itemId);
        }
    });

    bindToolboxBtn('.camera-angle', () => {
        if (!state.currentActiveWorkbenchItemId) return;
        const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();
        openCameraAngleModal(item);
    });

    bindToolboxBtn('.upscale', async () => {
        if (!state.currentActiveWorkbenchItemId) return;
        const itemId = state.currentActiveWorkbenchItemId;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        const executeUpscale = async (customPrompt) => {
            const promptToUse = customPrompt || "Image Restoration & Reconstruction: Redraw this low-quality image into a pristine, ultra-high-resolution (4K) masterpiece. Aggressively remove all blur, noise, and compression artifacts. CRITICAL: Do not just sharpen the existing pixels. Instead, synthesise and hallucinate missing high-frequency details (such as skin texture, hair strands, fabric patterns, and sharp edges) that are lost in the original. Re-imagine the subject with perfect focus and clarity while keeping the original subject identity, pose, and overall composition intact. The output must look like a sharp, professional commercial photograph taken with a modern high-end DSLR.";
            
            const placeholder = document.createElement('div');
            placeholder.className = 'world-placeholder';
            placeholder.style.left = `${parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20}px`;
            placeholder.style.top = `${parseFloat(item.el.style.top)}px`;
            placeholder.style.width = `${parseFloat(item.el.style.width)}px`;
            placeholder.style.height = `${parseFloat(item.el.style.height)}px`;
            
            const inverseScale = 1 / state.workbenchZoom;
            const visualBorderWidth = 2;
            const visualFontSize = 12;
            placeholder.style.border = `${visualBorderWidth * inverseScale}px dashed #ccc`;
            
            placeholder.innerHTML = '<span><i class="fas fa-spinner fa-spin"></i> 高清增强中...</span>';
            
            const span = placeholder.querySelector('span');
            if (span) {
                span.style.fontSize = `${visualFontSize * inverseScale}px`;
            }

            workbenchGrid.appendChild(placeholder);

            const tempMsg = addMessage({ 
                sender: 'bot', 
                type: 'text', 
                visibility: 'progress',
                persist: false,
                content: '✨ 正在执行智能高清增强 (Super Upscale)...\nAI 正在重绘细节、锐化边缘并提升整体质感，请稍候...' 
            });

            try {
                const result = await editOrQueryImageWithGemini(promptToUse, item.file || item.dataUrl);
                
                if (result.success && result.imageData) {
                    const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                    const file = await dataURLToFile(imgSrc, `hd-upscaled-${Date.now()}.png`);
                    
                    const itemRect = item.el.getBoundingClientRect();
                    const containerRect = workbenchZoomContainer.getBoundingClientRect();
                    
                    if (placeholder && placeholder.parentNode) placeholder.remove();
                    
                    const newId = await addImageToWorkbench(file, '高清增强', {
                        x: parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20,
                        y: parseFloat(item.el.style.top),
                        parentId: itemId
                    });
                    
                    if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('高清增强', promptToUse, imgSrc, executeUpscale);
                    } else {
                        addMessage({ sender: 'bot', type: 'text', content: '✅ **高清增强完成！**' });
                    }
                    window.selectWorkbenchItem(newId);
                } else {
                    throw new Error("生成结果为空");
                }
            } catch (e) {
                console.error("Upscale failed:", e);
                if (placeholder && placeholder.parentNode) placeholder.remove();
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                addMessage({ sender: 'bot', type: 'text', content: `❌ 高清增强失败: ${e.message}` });
            }
        };

        await executeUpscale();
    });

    bindToolboxBtn('.box-extract', () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item || !item.el) return;

        window.hideWorkbenchToolbox();

        // Remove any existing overlay
        const existingOverlay = item.el.querySelector('.box-extract-overlay');
        if (existingOverlay) existingOverlay.remove();

        // Create overlay covering the image
        const overlay = document.createElement('div');
        overlay.className = 'box-extract-overlay';
        overlay.style.cssText = `
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            z-index: 1000; cursor: crosshair; background: rgba(0,0,0,0.1);
        `;

        let isDrawing = false;
        let startX, startY;
        let rectEl = null;

        const onMouseDown = (e) => {
            e.stopPropagation();
            e.preventDefault();
            isDrawing = true;
            const zoom = state.workbenchZoom || 1;
            const rect = overlay.getBoundingClientRect();
            startX = (e.clientX - rect.left) / zoom;
            startY = (e.clientY - rect.top) / zoom;

            if (rectEl) {
                rectEl.remove();
                rectEl = null;
            }

            rectEl = document.createElement('div');
            rectEl.style.cssText = `
                position: absolute; border: 2px dashed #00ffcc;
                background: rgba(0,255,204,0.2); pointer-events: none;
                left: ${startX}px; top: ${startY}px; width: 0; height: 0;
            `;
            overlay.appendChild(rectEl);
        };

        const onMouseMove = (e) => {
            if (!isDrawing || !rectEl) return;
            e.stopPropagation();
            e.preventDefault();
            const zoom = state.workbenchZoom || 1;
            const rect = overlay.getBoundingClientRect();
            const currentX = (e.clientX - rect.left) / zoom;
            const currentY = (e.clientY - rect.top) / zoom;

            const width = Math.abs(currentX - startX);
            const height = Math.abs(currentY - startY);
            const left = Math.min(startX, currentX);
            const top = Math.min(startY, currentY);

            rectEl.style.left = `${left}px`;
            rectEl.style.top = `${top}px`;
            rectEl.style.width = `${width}px`;
            rectEl.style.height = `${height}px`;
        };

        const onMouseUp = async (e) => {
            if (!isDrawing) return;
            e.stopPropagation();
            e.preventDefault();
            isDrawing = false;
            
            // Allow user to click again to redraw if the box is too small Let's not remove listeners, just wait for popup input.
            if (!rectEl || parseInt(rectEl.style.width) < 10 || parseInt(rectEl.style.height) < 10) {
                if (rectEl) rectEl.remove();
                return;
            }

            // Remove mouse events so we don't redraw while typing
            overlay.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);

            // Pop up a loading indicator
            const popup = document.createElement('div');
            popup.style.cssText = `
                position: absolute;
                top: ${parseInt(rectEl.style.top) + parseInt(rectEl.style.height) + 5}px;
                left: ${rectEl.style.left};
                background: rgba(0,0,0,0.8); color: #fff; border-radius: 8px; padding: 15px 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); font-size: 16px; pointer-events: auto; z-index: 1001;
            `;
            popup.innerHTML = `<i class="fas fa-spinner fa-spin"></i> 正在智能识别主体...`;
            overlay.appendChild(popup);

            try {
                const ow = overlay.clientWidth;
                const oh = overlay.clientHeight;
                const rL = parseInt(rectEl.style.left);
                const rT = parseInt(rectEl.style.top);
                const rW = parseInt(rectEl.style.width);
                const rH = parseInt(rectEl.style.height);

                // Load the image to crop
                const parentImg = new Image();
                parentImg.crossOrigin = "anonymous";
                const { getProxiedUrl } = await import('../core/utils.js');
                
                let srcToCrop = item.dataUrl || item.originalDataUrl;
                if (!srcToCrop && item.file) {
                    const { fileToDataURL } = await import('../core/utils.js');
                    srcToCrop = await fileToDataURL(item.file);
                }
                
                parentImg.src = getProxiedUrl(srcToCrop);
                await new Promise((resolve, reject) => {
                    parentImg.onload = resolve;
                    parentImg.onerror = reject;
                });

                // Calculate crop relative to natural image size
                const pxMinX = (rL / ow) * parentImg.naturalWidth;
                const pxMinY = (rT / oh) * parentImg.naturalHeight;
                const pxMaxX = ((rL + rW) / ow) * parentImg.naturalWidth;
                const pxMaxY = ((rT + rH) / oh) * parentImg.naturalHeight;

                const cropCanvas = document.createElement('canvas');
                cropCanvas.width = pxMaxX - pxMinX;
                cropCanvas.height = pxMaxY - pxMinY;
                const ctx = cropCanvas.getContext('2d');
                ctx.drawImage(parentImg, pxMinX, pxMinY, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);
                const croppedDataUrl = cropCanvas.toDataURL('image/png');

                // Identify the object
                const { identifyObjectInCrop } = await import('../ai-services/skills-engine.js');
                const objectName = await identifyObjectInCrop(croppedDataUrl);

                // Map bbox to 0-1000
                const ymin = Math.floor((rT / oh) * 1000);
                const xmin = Math.floor((rL / ow) * 1000);
                const ymax = Math.floor(((rT + rH) / oh) * 1000);
                const xmax = Math.floor(((rL + rW) / ow) * 1000);
                const bbox = [ymin, xmin, ymax, xmax];

                // Append layer
                if (!item.scene) item.scene = { layers: [] };
                if (!item.layers) item.layers = [];
                const currentLayers = item.scene.layers.length > 0 ? item.scene.layers : item.layers;

                currentLayers.unshift({
                    id: `box-layer-${Date.now()}`,
                    name: objectName || '未命名物体',
                    bbox: bbox,
                    assetStatus: 'idle'
                });

                if (item.scene && item.scene.layers) item.scene.layers = currentLayers;
                item.layers = currentLayers;
                state.workbenchItems.set(itemId, item);

                overlay.remove();

                // Open Layer Manager
                const { showLayerManagerModal, renderLayerList } = await import('./modals.js');
                showLayerManagerModal(itemId);
                
                // Re-render layers over the picture
                const { renderCanvasLayers } = await import('./workbench/layers.js');
                renderCanvasLayers(itemId);

            } catch (e) {
                console.error("Auto identify failed:", e);
                alert("识别失败：" + e.message);
                overlay.remove();
            }
        };

        overlay.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);

        item.el.appendChild(overlay);
    });

    bindToolboxBtn('.erase', () => {
        if (state.currentActiveWorkbenchItemId) {
            const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                state.isContextPreviewHidden = false;
                
                state.currentIntentLock = 'erase';
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                openMaskEditor(item.file || item.dataUrl);
                window.selectWorkbenchItem(state.currentActiveWorkbenchItemId);
            }
        }
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.precise-edit', () => {
        if (state.currentActiveWorkbenchItemId) {
            const itemId = state.currentActiveWorkbenchItemId;
            const item = state.workbenchItems.get(itemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                window.hideWorkbenchToolbox();
                startPreciseEditMode(itemId);
            }
        }
    });

    bindToolboxBtn('.material', () => {
        if (state.currentActiveWorkbenchItemId) {
            const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                state.isContextPreviewHidden = false;
                
                state.currentIntentLock = 'material';
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                openMaskEditor(item.file || item.dataUrl);
                window.selectWorkbenchItem(state.currentActiveWorkbenchItemId);
            }
        }
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.delete', () => {
        if (state.currentActiveWorkbenchItemId) deleteWorkbenchItem(state.currentActiveWorkbenchItemId);
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.download', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        let dataUrl = item.dataUrl;
        if (!dataUrl && item.file) {
            dataUrl = await window.fileToDataURL(item.file);
        } else if (!dataUrl && item.el && item.el.querySelector('img')) {
            dataUrl = item.el.querySelector('img').src;
        }

        if (dataUrl) {
            try {
                // Use getProxiedUrl to avoid CORS issues when fetching the image
                const { getProxiedUrl } = await import('../core/utils.js');
                const fetchUrl = getProxiedUrl(dataUrl);
                
                // Fetch the image to get a Blob
                const response = await fetch(fetchUrl);
                const blob = await response.blob();
                
                // Determine correct extension from blob type
                let extension = 'png';
                if (blob.type) {
                    extension = blob.type.split('/')[1] || 'png';
                    if (extension === 'jpeg') extension = 'jpg';
                }
                
                // Get base filename
                let baseName = `download-${Date.now()}`;
                if (item.file && item.file.name) {
                    const lastDot = item.file.name.lastIndexOf('.');
                    baseName = lastDot !== -1 ? item.file.name.substring(0, lastDot) : item.file.name;
                }
                const suggestedName = `${baseName}.${extension}`;

                // Standard download (might auto-download depending on browser settings)
                const blobUrl = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = blobUrl;
                a.download = suggestedName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                // Clean up the blob URL after a short delay
                setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
            } catch (e) {
                console.error('下载图片失败:', e);
                // Fallback to the old method if fetch fails (e.g., CORS issues without proxy)
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = item.file ? item.file.name : `download-${Date.now()}.png`;
                a.target = '_blank';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        } else {
            console.error('无法获取图片数据进行下载');
        }
    });

    bindToolboxBtn('.critique', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();

        if (item.critiquesData && item.critiquesData.length > 0) {
            const existingPanel = document.getElementById(`critique-panel-${itemId}`);
            if (existingPanel) {
                if (existingPanel.style.display === 'none') {
                    existingPanel.style.display = 'block';
                } else {
                    existingPanel.style.display = 'none';
                }
                return;
            }
        }

        const btn = document.querySelector('.critique');
        if(btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 辩论中...';

        try {
            const { triggerAgentDebate } = await import('./debate.js');
            await triggerAgentDebate(itemId);
            
        } catch (e) {
            console.error("Critique failed:", e);
            addMessage({ sender: 'bot', type: 'text', content: `❌ 辩论生成失败: ${e.message}` });
        } finally {
            if(btn) btn.innerHTML = '<i class="fas fa-comments"></i> 专家辩论';
        }
    });

    bindToolboxBtn('.layers', async () => {
        const itemId = state.currentActiveWorkbenchItemId; 
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();
        showLayerManagerModal(itemId, true);
    });

    bindToolboxBtn('.magic-layers', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();
        console.log('[Toolbox] Magic Layers clicked:', itemId);
        try {
            await triggerMagicLayers(itemId);
        } catch (error) {
            console.error('[Toolbox] Magic Layers failed:', error);
        }
    });

    bindToolboxBtn('.multiview', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        const executeMultiview = async (customPrompt) => {
            const placeholder = document.createElement('div');
            placeholder.className = 'world-placeholder';
            const inverseScale = 1 / state.workbenchZoom;
            placeholder.style.cssText = `
                left: ${parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20}px;
                top: ${parseFloat(item.el.style.top)}px;
                width: ${parseFloat(item.el.style.width)}px;
                height: ${parseFloat(item.el.style.height)}px;
                border: ${2 * inverseScale}px dashed #ccc;
            `;
            placeholder.innerHTML = `<span style="font-size: ${14 * inverseScale}px;"><i class="fas fa-eye fa-spin"></i> 正在智能识别场景类型...</span>`;
            workbenchGrid.appendChild(placeholder);

            let tempMsg = null;

            try {
                const base64 = await fileToBase64(item.file || item.dataUrl);
                
                const category = await classifyImageCategory(base64, item.file?.type || 'image/png');
                
                let finalPrompt = "";
                let uiMessage = "";
                let placeholderTitle = "";

                if (category.includes('CHARACTER')) {
                    uiMessage = "👤 **正在执行【角色三视图】推演**...\nAI 正在精准还原角色的正面、侧面及背面细节。";
                    placeholderTitle = "正在绘制角色三视图...";
                    finalPrompt = `Create a professional character sheet (3-view) based on this character. 
                    Must include: Front view, Side view, and Back view.
                    Style: Character design orthographic projection, neutral background, consistent clothing and features.`;
                } else if (category.includes('PRODUCT')) {
                    uiMessage = "🗿 **正在执行【物品多维视图】推演**...\nAI 正在对该物体的结构、材质进行多角度拆解。";
                    placeholderTitle = "正在生成物品结构图...";
                    finalPrompt = `Create a professional product design multi-view (3-view) based on this object.
                    Must include: Front, Side, and Top views.
                    Style: Product design orthographic sheet, industrial design presentation, identical material and lighting.`;
                } else {
                    uiMessage = "📐 **正在执行【空间/建筑多维视图】推演**...\nAI 正在解析大空间的透视逻辑，生成俯视图、左视图及右视图。";
                    placeholderTitle = "正在推演空间三向视图...";
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

                tempMsg = addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: uiMessage });
                if (placeholder && placeholder.parentNode) {
                    placeholder.querySelector('span').innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${placeholderTitle}`;
                }

                const result = await editOrQueryImageWithGemini(promptToUse, item.file || item.dataUrl);

                if (result.success && result.imageData) {
                    const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                    const file = await dataURLToFile(imgSrc, `multiview-${Date.now()}.png`);
                    
                    if (placeholder && placeholder.parentNode) placeholder.remove();
                    addImageToWorkbench(file, '多维视图', {
                        x: parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20,
                        y: parseFloat(item.el.style.top),
                        parentId: itemId
                    });
                    
                    if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('多维视图', promptToUse, imgSrc, executeMultiview);
                    } else {
                        addMessage({ sender: 'bot', type: 'text', content: '✅ **多维视图生成完成！**' });
                    }
                }
            } catch (e) {
                console.error("Multiview adaptation failed:", e);
                if (placeholder && placeholder.parentNode) placeholder.remove();
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                addMessage({ sender: 'bot', type: 'text', content: `❌ 处理失败: ${e.message}` });
            }
        };

        await executeMultiview();
    });

    bindToolboxBtn('.veo-video', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();
        await ensureMotionSemanticCoverage(item, itemId);

        const initialPrompt = buildMotionPromptSuggestion(item, itemId);

        showVideoPromptModal(async (videoPrompt) => {
            addMessage({ sender: 'user', type: 'text', content: `🎥 Motion 指令: ${videoPrompt}` });
            const hasLayeredMotion = hasMagicMotionLayers(item, itemId);
            addMessage({
                sender: 'bot',
                type: 'text',
                visibility: 'progress',
                persist: false,
                content: hasLayeredMotion
                    ? '🎬 **Magic Motion Runtime 生成中**...\n系统正在基于 Magic Layers 自动编排图层进场动画，并在 chat 中回显真实预览。'
                    : '🎬 **Magic Motion Preview 生成中**...\n正在基于当前图片生成动态预览，请耐心等待 (约 1-2 分钟)。'
            });

            try {
                if (hasLayeredMotion) {
                    const recipe = buildAutoMotionRecipe(item, itemId, videoPrompt);
                    const preview = await buildAutoMotionPreviewHtml({
                        item,
                        itemId,
                        recipe
                    });

                    addMessage({
                        sender: 'bot',
                        type: 'text',
                        visibility: 'internal',
                        persist: false,
                        content: `🧭 动态方案已生成：${recipe.summary}`
                    });

                    addMessage({
                        sender: 'bot',
                        type: 'text',
                        visibility: 'internal',
                        persist: false,
                        content: [
                            `🧪 Motion 图层诊断`,
                            `总图层: ${recipe.layers.length}`,
                            `可动图层: ${preview.motionLayerCount}`,
                            ...preview.debugLines,
                            ...(preview.excludedDebugLines.length > 0
                                ? ['排除图层:', ...preview.excludedDebugLines]
                                : [])
                        ].join('\n')
                    });

                    addMessage({
                        sender: 'bot',
                        type: 'text',
                        visibility: 'internal',
                        persist: false,
                        content: [
                            '🧬 Motion 语义池候选',
                            ...formatSemanticPoolDebugLines(item)
                        ].join('\n')
                    });

                    addMessage({
                        sender: 'bot',
                        type: 'text',
                        visibility: 'internal',
                        persist: false,
                        content: [
                            '🧩 Motion 强制补层',
                            ...formatForcedCoverageDebugLines(ensureRecipeSemanticCoverage(
                                getVisibleMotionLayerEntries(item, itemId),
                                item,
                                itemId
                            ))
                        ].join('\n')
                    });

                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('Magic Motion Runtime', videoPrompt, null, null, preview.html);
                        setTimeout(() => {
                            if (window.replayMagicMotionPreview) {
                                window.replayMagicMotionPreview(preview.containerId);
                            }
                        }, 60);
                    } else {
                        addMessage({ sender: 'bot', type: 'html', content: preview.html });
                        setTimeout(() => {
                            if (window.replayMagicMotionPreview) {
                                window.replayMagicMotionPreview(preview.containerId);
                            }
                        }, 60);
                    }
                    return;
                }

                const rect = item.el.getBoundingClientRect();
                const imageSize = pickWanImageSize(rect.width, rect.height);
                const source = await buildMotionPreviewSource(item, itemId);
                const imageDataUrl = source.imageDataUrl;
                const fullPrompt = [
                    source.promptPrefix,
                    videoPrompt
                ].filter(Boolean).join('\n\n');

                let lastStatus = '';
                const result = await generateWanMotionPreview({
                    prompt: fullPrompt,
                    image: imageDataUrl,
                    imageSize,
                    onProgress: ({ status }) => {
                        if (!status || status === lastStatus || status === 'Succeed') return;
                        lastStatus = status;
                        const statusLabelMap = {
                            InQueue: '任务已进入队列，正在等待处理…',
                            InProgress: '视频正在生成中，镜头运动和动态效果处理中…'
                        };
                        const statusLabel = statusLabelMap[status] || `当前状态: ${status}`;
                        addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: `🎞️ ${statusLabel}` });
                    }
                });

                if (result?.videoUrl) {
                    const videoHtml = `
                        <div style="margin-top: 10px; width: 100%;">
                            <video src="${result.videoUrl}" controls autoplay loop muted playsinline
                                style="width: 100%; max-width: 300px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background: #000;">
                            </video>
                            <div style="font-size: 12px; color: #666; margin-top: 8px; display:flex; justify-content:space-between; align-items:center;">
                                <span>✅ Motion Preview 生成成功</span>
                                <a href="${result.videoUrl}" target="_blank" rel="noopener noreferrer" download="magic-motion-${Date.now()}.mp4" style="color:var(--primary-color); text-decoration:none; font-weight:500;">
                                    <i class="fas fa-download"></i> 下载
                                </a>
                            </div>
                        </div>
                    `;

                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat(
                            hasLayeredMotion ? 'Magic Motion Video' : 'Magic Motion Preview',
                            videoPrompt,
                            null,
                            null,
                            videoHtml
                        );
                    } else {
                        addMessage({ sender: 'bot', type: 'html', content: videoHtml });
                    }
                }
            } catch (e) {
                console.error(e);
                addMessage({ sender: 'bot', type: 'text', content: `❌ 视频生成失败: ${e.message}` });
            }
        }, {
            initialPrompt,
            placeholder: '系统会根据当前图层预填一条 motion 建议，你也可以手动改写'
        });
    });
}
