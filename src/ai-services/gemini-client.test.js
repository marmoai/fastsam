import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the SDK before importing the client
const { mockGenerateContent, mockGenerateImages } = vi.hoisted(() => ({
    mockGenerateContent: vi.fn(),
    mockGenerateImages: vi.fn()
}));

vi.mock('@google/genai', () => {
    return {
        GoogleGenAI: class {
            constructor() {
                this.models = {
                    generateContent: mockGenerateContent,
                    generateImages: mockGenerateImages,
                    generateVideos: vi.fn()
                };
                this.chats = {
                    create: vi.fn()
                };
            }
        },
        Type: {
            OBJECT: 'OBJECT',
            STRING: 'STRING'
        }
    };
});

// Now import the client
import { 
    generateImageWithImagen, 
    generateTextWithSearch,
    ai 
} from './gemini-client';

describe('gemini-client.js logic tests', () => {
    
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('generateImageWithImagen', () => {
        it('should format the response correctly', async () => {
            const mockResponse = {
                generatedImages: [
                    { image: { imageBytes: 'base64data' } }
                ]
            };
            mockGenerateImages.mockResolvedValue(mockResponse);

            const result = await generateImageWithImagen('a cat');
            
            expect(result.success).toBe(true);
            expect(result.imageData[0].imageData).toBe('base64data');
            expect(result.imageData[0].mimeType).toBe('image/png');
        });

        it('should throw error if no images returned', async () => {
            mockGenerateImages.mockResolvedValue({ generatedImages: [] });
            await expect(generateImageWithImagen('a cat')).rejects.toThrow('API did not return any images.');
        });
    });

    describe('generateTextWithSearch', () => {
        it('should extract text and sources correctly', async () => {
            const mockResponse = {
                text: 'The sky is blue.',
                candidates: [{
                    groundingMetadata: {
                        groundingChunks: [
                            { web: { uri: 'https://example.com', title: 'Source 1' } }
                        ]
                    }
                }]
            };
            mockGenerateContent.mockResolvedValue(mockResponse);

            const result = await generateTextWithSearch('why is the sky blue');
            
            expect(result.success).toBe(true);
            expect(result.text).toBe('The sky is blue.');
            expect(result.sources[0].uri).toBe('https://example.com');
        });
    });
});
