import { describe, it, expect } from 'vitest';
import { 
    isRemovalRequest, 
    isMaterialRequest, 
    isImageGenerationRequest,
    getExplicitRequestedImageCount
} from './utils';

describe('utils.js core logic tests', () => {
    
    describe('isRemovalRequest', () => {
        it('should return true for removal keywords', () => {
            expect(isRemovalRequest('帮我移除背景')).toBe(true);
            expect(isRemovalRequest('删掉这个人物')).toBe(true);
            expect(isRemovalRequest('remove the cat')).toBe(true);
        });

        it('should return false for non-removal text', () => {
            expect(isRemovalRequest('画一只猫')).toBe(false);
            expect(isRemovalRequest('换个材质')).toBe(false);
        });
    });

    describe('isMaterialRequest', () => {
        it('should return true for material keywords', () => {
            expect(isMaterialRequest('换成木质纹理')).toBe(true);
            expect(isMaterialRequest('change to metal texture')).toBe(true);
        });

        it('should return false for non-material text', () => {
            expect(isMaterialRequest('生成一张海报')).toBe(false);
        });
    });

    describe('isImageGenerationRequest', () => {
        it('should return true for generation keywords', () => {
            expect(isImageGenerationRequest('生成一张未来城市的图片')).toBe(true);
            expect(isImageGenerationRequest('画一个机器人')).toBe(true);
        });

        it('should return false for simple chat', () => {
            expect(isImageGenerationRequest('你好')).toBe(false);
            expect(isImageGenerationRequest('你是谁？')).toBe(false);
        });
    });

    describe('getExplicitRequestedImageCount', () => {
        it('returns 1 by default when no explicit count is present', () => {
            expect(getExplicitRequestedImageCount('换成木质纹理')).toBe(1);
            expect(getExplicitRequestedImageCount('继续做这一版')).toBe(1);
        });

        it('extracts explicit chinese image counts correctly', () => {
            expect(getExplicitRequestedImageCount('生成两张图')).toBe(2);
            expect(getExplicitRequestedImageCount('来三张不同版本')).toBe(3);
            expect(getExplicitRequestedImageCount('重新生成4张')).toBe(4);
        });
    });
});
