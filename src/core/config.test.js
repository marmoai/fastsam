import { afterEach, describe, expect, it } from 'vitest';
import { isImageFusionEnabled } from './config.js';

afterEach(() => {
    delete globalThis.__marmoFeatureFlags;
});

describe('image fusion feature flag', () => {
    it('is disabled by default without removing the implementation', () => {
        expect(isImageFusionEnabled()).toBe(false);
    });

    it('can be re-enabled through the existing runtime override', () => {
        globalThis.__marmoFeatureFlags = { imageFusion: true };
        expect(isImageFusionEnabled()).toBe(true);
    });
});
