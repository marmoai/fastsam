import type { ProjectWorkspace } from './WorkspaceRuntime';

export interface VerificationCheck {
    id: string;
    passed: boolean;
    message: string;
}

export interface VerificationResult {
    passed: boolean;
    checks: VerificationCheck[];
    checkedAt: number;
}

function finite(value: unknown): boolean {
    return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Pure Runtime verification. It deliberately does not inspect the DOM or
 * image pixels; those belong to the host adapter. This keeps Job decisions
 * deterministic and makes the same checks usable after a reload.
 */
export class VerificationEngine {
    static verifyWorkspace(workspace: ProjectWorkspace, assetIds: string[]): VerificationResult {
        const ids = [...new Set((assetIds || []).filter(Boolean))];
        const assets = ids.map(uid => workspace.currentState.assetRegistry.get(uid));
        const checks: VerificationCheck[] = [];
        const allAssetsExist = ids.length > 0 && assets.every(Boolean);
        const allInScene = ids.length > 0 && ids.every(uid => workspace.currentState.sceneGraph.getNodes().includes(uid));

        checks.push({
            id: 'target_assets_exist',
            passed: allAssetsExist,
            message: allAssetsExist ? 'All target assets exist in Runtime.' : 'One or more target assets are missing from Runtime.'
        });
        checks.push({
            id: 'target_assets_in_scene',
            passed: allInScene,
            message: allInScene ? 'All target assets are in the scene graph.' : 'One or more target assets are not in the scene graph.'
        });

        const referencesAvailable = allAssetsExist && assets.every(asset => Boolean(asset?.sourceImage));
        checks.push({
            id: 'asset_references_available',
            passed: referencesAvailable,
            message: referencesAvailable ? 'All target assets have image references.' : 'One or more target assets have no image reference.'
        });

        const transformsValid = allAssetsExist && assets.every(asset => {
            const transform = asset?.transform;
            return Boolean(transform) && ['x', 'y', 'width', 'height', 'rotation', 'zIndex'].every(key => finite((transform as any)[key]));
        });
        checks.push({
            id: 'asset_transforms_valid',
            passed: transformsValid,
            message: transformsValid ? 'All target transforms are finite and serializable.' : 'One or more target transforms are invalid.'
        });

        const graphRegistryConsistent = allInScene && ids.every(uid => {
            const asset = workspace.currentState.assetRegistry.get(uid);
            return Boolean(asset && workspace.currentState.sceneGraph.getNodes().includes(asset.uid));
        });
        checks.push({
            id: 'graph_registry_consistent',
            passed: graphRegistryConsistent,
            message: graphRegistryConsistent ? 'Scene graph and Asset Registry agree.' : 'Scene graph and Asset Registry are inconsistent.'
        });

        const roundTrip = workspace.verifyRoundTrip(ids);
        checks.push(...roundTrip.checks);
        return {
            passed: checks.every(check => check.passed),
            checks,
            checkedAt: Date.now()
        };
    }
}

