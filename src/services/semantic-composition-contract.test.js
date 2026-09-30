import { describe, expect, it } from 'vitest';
import {
    acceptsMoreAtomicReplacement,
    requiredCompositionReplacementCount,
    shouldReviewCompositionGroup
} from './semantic-composition-contract.js';

describe('semantic composition contract', () => {
    it('reviews a tiered group when one existing child is a broad envelope', () => {
        const group = {
            compositeRole: 'composite_group',
            bbox: [640, 676, 871, 1000],
            children: [
                { bbox: [719, 676, 871, 1000] },
                { bbox: [643, 866, 739, 926] },
                { bbox: [725, 871, 815, 1000] }
            ]
        };

        expect(shouldReviewCompositionGroup(group)).toBe(true);
        expect(requiredCompositionReplacementCount(group.children.length)).toBe(4);
        expect(acceptsMoreAtomicReplacement(3, 3)).toBe(false);
        expect(acceptsMoreAtomicReplacement(3, 4)).toBe(true);
    });

    it('uses identical evidence for a flat overlapping composition', () => {
        const group = {
            semanticType: 'composite_group',
            bbox: [100, 100, 500, 800],
            children: [
                { bbox: [110, 110, 490, 790] },
                { bbox: [180, 600, 280, 740] }
            ]
        };
        expect(shouldReviewCompositionGroup(group)).toBe(true);
    });

    it('does not replace a balanced existing group without ambiguity evidence', () => {
        const group = {
            compositeRole: 'composite_group',
            bbox: [100, 100, 700, 700],
            children: [
                { bbox: [110, 110, 340, 340] },
                { bbox: [360, 360, 620, 620] }
            ]
        };
        expect(shouldReviewCompositionGroup(group)).toBe(false);
    });

    it('reviews an explicitly flagged group even before children exist', () => {
        const group = {
            compositeRole: 'composite_group',
            bbox: [100, 100, 700, 700],
            compositionReview: { required: true },
            children: []
        };
        expect(shouldReviewCompositionGroup(group)).toBe(true);
        expect(acceptsMoreAtomicReplacement(0, 2)).toBe(true);
    });
});
