const DEFAULT_MIN_OBSERVED_RECALL = 0.78;
const DEFAULT_MIN_HIDDEN_PIXELS = 1;

function finiteNumber(value, fallback = null) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

/**
 * Build the durable evidence record used by both execution and replacement.
 * Keeping this in one module prevents a stale pre-reconciliation quality flag
 * from disagreeing with the final canonical decision in the UI.
 */
function directSecondSamEvidence(
    segmentation,
    directForegroundOwnershipEvidenceStatus = null,
    directForegroundContextMaskAttached = false
) {
    const quality = segmentation?.quality || {};
    const selected = Array.isArray(quality.debugCandidates)
        ? quality.debugCandidates.find(candidate => candidate?.selected)
        : null;
    const recoveryAudit = quality.completionRecoveryAudit || {};
    const hiddenCandidatePixels = Math.max(
        finiteNumber(recoveryAudit.allowedNewPixels, 0),
        finiteNumber(quality.completionHiddenRescuePixels, 0),
        finiteNumber(selected?.completionRecoveryPixels, 0)
    );
    const ownershipStatus = directForegroundOwnershipEvidenceStatus ||
        quality.foregroundOwnershipEvidenceStatus || 'unavailable';
    const ownershipAvailable = ownershipStatus === 'attached' || ownershipStatus === 'empty_valid';
    return {
        hiddenCandidatePixels,
        hiddenKeptPixels: hiddenCandidatePixels,
        hiddenCoverage: recoveryAudit.hiddenCoverage ?? null,
        unknownGeneratedPixels: 0,
        ownershipEvidenceMissingPixels: ownershipAvailable ? 0 : hiddenCandidatePixels,
        foregroundContextMaskAttached: directForegroundContextMaskAttached === true ||
            quality.foregroundContextMaskAttached === true,
        foregroundOwnershipEvidenceAvailable: ownershipAvailable,
        foregroundOwnershipEvidenceStatus: ownershipStatus,
        verifiedCandidateEvidence: quality.completionStructureVerified === true,
        directSecondSam: true
    };
}

export function buildCanonicalEvidence({
    observationAudit,
    reconciliation,
    occluderAudit,
    directSegmentation = null,
    directForegroundOwnershipEvidenceStatus = null,
    directForegroundContextMaskAttached = false
}) {
    const directEvidence = !reconciliation && directSegmentation
        ? directSecondSamEvidence(
            directSegmentation,
            directForegroundOwnershipEvidenceStatus,
            directForegroundContextMaskAttached
        )
        : null;
    const observedPassed = observationAudit?.status === 'passed';
    const hiddenKeptPixels = finiteNumber(
        reconciliation?.hiddenKeptPixels ?? directEvidence?.hiddenKeptPixels,
        0
    );
    const foregroundOwnershipEvidenceAvailable =
        reconciliation?.foregroundOwnershipEvidenceAvailable === true ||
        reconciliation?.foregroundContextMaskAttached === true ||
        directEvidence?.foregroundOwnershipEvidenceAvailable === true;
    const status = observedPassed &&
        foregroundOwnershipEvidenceAvailable &&
        hiddenKeptPixels > DEFAULT_MIN_HIDDEN_PIXELS - 1
        ? 'accepted'
        : 'hold';
    return {
        status,
        observedRecall: observationAudit?.observedRecall ?? null,
        visibleIoU: observationAudit?.visibleIoU ?? null,
        boundaryDrift: observationAudit?.unoccludedDriftRatio ?? null,
        hiddenCoverage: reconciliation?.hiddenCoverage ?? directEvidence?.hiddenCoverage ?? null,
        hiddenCandidatePixels: reconciliation?.hiddenCandidatePixels ?? directEvidence?.hiddenCandidatePixels ?? 0,
        allowedHiddenCandidatePixels: reconciliation?.allowedHiddenCandidatePixels ?? directEvidence?.hiddenCandidatePixels ?? 0,
        hiddenKeptPixels,
        unknownGeneratedPixels: reconciliation?.unknownGeneratedPixels ?? directEvidence?.unknownGeneratedPixels ?? 0,
        ownershipEvidenceMissingPixels: reconciliation?.ownershipEvidenceMissingPixels ?? directEvidence?.ownershipEvidenceMissingPixels ?? 0,
        foregroundContextMaskAttached: reconciliation?.foregroundContextMaskAttached === true ||
            directEvidence?.foregroundContextMaskAttached === true,
        foregroundOwnershipEvidenceAvailable,
        foregroundOwnershipEvidenceStatus: reconciliation?.foregroundOwnershipEvidenceStatus ||
            directEvidence?.foregroundOwnershipEvidenceStatus || 'unavailable',
        verifiedCandidateEvidence: reconciliation?.verifiedCandidateEvidence === true ||
            directEvidence?.verifiedCandidateEvidence === true,
        bboxExpansion: reconciliation?.bboxExpansion ?? null,
        occluderLeakage: occluderAudit?.overlapRatio ?? null,
        directSecondSam: directEvidence?.directSecondSam === true,
        failureReason: status === 'accepted'
            ? null
            : (!observedPassed
                ? (observationAudit?.reason || 'original_observation_audit_failed')
                : (!foregroundOwnershipEvidenceAvailable
                    ? 'foreground_ownership_evidence_missing'
                    : 'hidden_evidence_missing')),
        retryStrategy: null
    };
}

export function isCanonicalEvidenceAccepted(
    evidence,
    { minObservedRecall = DEFAULT_MIN_OBSERVED_RECALL, minHiddenPixels = DEFAULT_MIN_HIDDEN_PIXELS } = {}
) {
    return evidence?.status === 'accepted' &&
        evidence?.foregroundOwnershipEvidenceAvailable === true &&
        finiteNumber(evidence.hiddenKeptPixels, 0) >= minHiddenPixels &&
        finiteNumber(evidence.observedRecall, 0) >= minObservedRecall;
}

export const CANONICAL_EVIDENCE_DEFAULTS = Object.freeze({
    minObservedRecall: DEFAULT_MIN_OBSERVED_RECALL,
    minHiddenPixels: DEFAULT_MIN_HIDDEN_PIXELS
});
