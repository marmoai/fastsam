import { describe, expect, it } from 'vitest';
import { ProjectWorkspace } from './WorkspaceRuntime';
import { VerificationEngine } from './VerificationEngine';
import type { AssetEntity } from './AssetRuntime';

function asset(uid: string): AssetEntity {
    return {
        uid,
        version: 1,
        type: 'product',
        sourceImage: 'https://cdn.example.com/object.png',
        masks: [],
        variants: [],
        metadata: { createdAt: 1, updatedAt: 1, creatorId: 'test', usageCount: 0, tags: [] },
        fusionProperties: { brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, grayscale: 0, sepia: 0 },
        transform: { x: 0, y: 0, width: 100, height: 100, rotation: 0, zIndex: 1 }
    };
}

describe('VerificationEngine', () => {
    it('checks Runtime, SceneGraph, transform and round-trip invariants', () => {
        const workspace = new ProjectWorkspace('verification-project');
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: asset('asset-001') } });

        const result = VerificationEngine.verifyWorkspace(workspace, ['asset-001']);

        expect(result.passed).toBe(true);
        expect(result.checks.map(check => check.id)).toEqual(expect.arrayContaining([
            'target_assets_exist',
            'target_assets_in_scene',
            'asset_transforms_valid',
            'graph_registry_consistent',
            'workspace_round_trip_assets'
        ]));
    });

    it('fails without mutating the staged Runtime state', () => {
        const workspace = new ProjectWorkspace('verification-project');
        const invalid = asset('asset-002');
        invalid.transform.width = Number.NaN;
        workspace.dispatcher.dispatch({ type: 'ADD_ASSET', payload: { asset: invalid } });

        const result = VerificationEngine.verifyWorkspace(workspace, ['asset-002']);

        expect(result.passed).toBe(false);
        expect(workspace.currentState.assetRegistry.get('asset-002')?.transform.width).toBeNaN();
        expect(result.checks.find(check => check.id === 'asset_transforms_valid')?.passed).toBe(false);
    });
});

