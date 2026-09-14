import { describe, expect, it } from 'vitest';
import {
    buildCanonicalEvidence,
    isCanonicalEvidenceAccepted
} from './completion-evidence.js';

describe('canonical completion evidence contract', () => {
    it('accepts only passed observation plus retained hidden pixels', () => {
        const evidence = buildCanonicalEvidence({
            observationAudit: {
                status: 'passed',
                observedRecall: 0.93,
                visibleIoU: 0.81,
                unoccludedDriftRatio: 0.04
            },
            reconciliation: {
                hiddenCandidatePixels: 820,
                hiddenKeptPixels: 510,
                hiddenCoverage: 0.42,
                unknownGeneratedPixels: 12,
                foregroundOwnershipEvidenceAvailable: true,
                foregroundOwnershipEvidenceStatus: 'attached',
                bboxExpansion: 0.03
            },
            occluderAudit: { overlapRatio: 0.19 }
        });

        expect(evidence).toMatchObject({
            status: 'accepted',
            hiddenKeptPixels: 510,
            failureReason: null
        });
        expect(isCanonicalEvidenceAccepted(evidence)).toBe(true);
    });

    it('holds when the observed audit fails even if a candidate has pixels', () => {
        const evidence = buildCanonicalEvidence({
            observationAudit: {
                status: 'rejected',
                reason: 'completed_target_drifted_from_original_visible_silhouette',
                observedRecall: 0.51
            },
            reconciliation: { hiddenKeptPixels: 900 }
        });

        expect(evidence.status).toBe('hold');
        expect(evidence.failureReason).toBe('completed_target_drifted_from_original_visible_silhouette');
        expect(isCanonicalEvidenceAccepted(evidence)).toBe(false);
    });

    it('holds hidden pixels when foreground ownership evidence is missing', () => {
        const evidence = buildCanonicalEvidence({
            observationAudit: { status: 'passed', observedRecall: 0.96 },
            reconciliation: {
                hiddenCandidatePixels: 900,
                hiddenKeptPixels: 0,
                foregroundContextMaskAttached: false,
                foregroundOwnershipEvidenceAvailable: false,
                ownershipEvidenceMissingPixels: 900
            }
        });

        expect(evidence.status).toBe('hold');
        expect(evidence.failureReason).toBe('foreground_ownership_evidence_missing');
        expect(isCanonicalEvidenceAccepted(evidence)).toBe(false);
    });

    it('accepts an explicitly verified empty ownership context', () => {
        const evidence = buildCanonicalEvidence({
            observationAudit: { status: 'passed', observedRecall: 0.95 },
            reconciliation: {
                hiddenKeptPixels: 64,
                hiddenCandidatePixels: 64,
                foregroundOwnershipEvidenceAvailable: true,
                foregroundOwnershipEvidenceStatus: 'empty_valid'
            }
        });

        expect(evidence.status).toBe('accepted');
        expect(isCanonicalEvidenceAccepted(evidence)).toBe(true);
    });

    it('does not release a stale accepted record with insufficient observed recall', () => {
        expect(isCanonicalEvidenceAccepted({
            status: 'accepted',
            hiddenKeptPixels: 300,
            observedRecall: 0.61
        })).toBe(false);
    });
});
