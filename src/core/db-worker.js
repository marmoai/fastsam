self.onmessage = async function(e) {
    const { type, payload, id } = e.data;
    try {
        if (type === 'serializeSession') {
            // Convert base64 to blobs
            const session = payload;
            const processedSession = JSON.parse(JSON.stringify(session)); // deep copy
            
            // Helper to convert base64 to blob
            const b64toBlob = async (b64Data) => {
                const res = await fetch(b64Data);
                return await res.blob();
            };

            // Process messages
            if (processedSession.messages) {
                for (let msg of processedSession.messages) {
                    if (msg.imageData) {
                        const images = Array.isArray(msg.imageData) ? msg.imageData : [msg.imageData];
                        for (let img of images) {
                            if (img.src && img.src.startsWith('data:')) {
                                img.blob = await b64toBlob(img.src);
                                delete img.src; // Remove base64 to save space
                            }
                        }
                    }
                }
            }

            // Process workbenchState
            if (processedSession.workbenchState) {
                for (let item of processedSession.workbenchState) {
                    if (item.dataUrl && item.dataUrl.startsWith('data:')) {
                        item.blob = await b64toBlob(item.dataUrl);
                        delete item.dataUrl;
                    }
                    if (item.originalDataUrl && item.originalDataUrl.startsWith('data:')) {
                        item.originalBlob = await b64toBlob(item.originalDataUrl);
                        delete item.originalDataUrl;
                    }
                    if (item.cleanPlateDataUrl && item.cleanPlateDataUrl.startsWith('data:')) {
                        item.cleanPlateBlob = await b64toBlob(item.cleanPlateDataUrl);
                        delete item.cleanPlateDataUrl;
                    }
                    if (item.layers && Array.isArray(item.layers)) {
                        for (let layer of item.layers) {
                            if (layer.image && layer.image.startsWith('data:')) {
                                layer.blob = await b64toBlob(layer.image);
                                delete layer.image;
                            } else if (layer.image && !layer.image.startsWith('http') && layer.image.length > 1000) {
                                // sometimes it's raw base64 without data:image
                                layer.blob = await b64toBlob('data:image/png;base64,' + layer.image);
                                delete layer.image;
                            }
                            if (layer.mask && layer.mask.startsWith('data:')) {
                                layer.maskBlob = await b64toBlob(layer.mask);
                                delete layer.mask;
                            } else if (layer.mask && !layer.mask.startsWith('http') && layer.mask.length > 1000) {
                                layer.maskBlob = await b64toBlob('data:image/png;base64,' + layer.mask);
                                delete layer.mask;
                            }
                        }
                    }
                    if (item.scene && item.scene.layers) {
                        for (let layer of item.scene.layers) {
                            if (layer.image && layer.image.startsWith('data:')) {
                                layer.blob = await b64toBlob(layer.image);
                                delete layer.image;
                            } else if (layer.image && !layer.image.startsWith('http') && layer.image.length > 1000) {
                                layer.blob = await b64toBlob('data:image/png;base64,' + layer.image);
                                delete layer.image;
                            }
                            if (layer.mask && layer.mask.startsWith('data:')) {
                                layer.maskBlob = await b64toBlob(layer.mask);
                                delete layer.mask;
                            } else if (layer.mask && !layer.mask.startsWith('http') && layer.mask.length > 1000) {
                                layer.maskBlob = await b64toBlob('data:image/png;base64,' + layer.mask);
                                delete layer.mask;
                            }
                        }
                    }
                }
            }

            if (processedSession.runtimeWorkspace?.currentState?.assets) {
                for (let asset of processedSession.runtimeWorkspace.currentState.assets) {
                    if (asset.sourceImage && asset.sourceImage.startsWith('data:')) {
                        asset.sourceImageBlob = await b64toBlob(asset.sourceImage);
                        delete asset.sourceImage;
                    }
                    if (asset.originalDataUrl && asset.originalDataUrl.startsWith('data:')) {
                        asset.originalBlob = await b64toBlob(asset.originalDataUrl);
                        delete asset.originalDataUrl;
                    }
                    if (asset.cleanPlateDataUrl && asset.cleanPlateDataUrl.startsWith('data:')) {
                        asset.cleanPlateBlob = await b64toBlob(asset.cleanPlateDataUrl);
                        delete asset.cleanPlateDataUrl;
                    }

                    const handleLayers = async (layers) => {
                        if (!layers || !Array.isArray(layers)) return;
                        for (let layer of layers) {
                            if (layer.image && layer.image.startsWith('data:')) {
                                layer.blob = await b64toBlob(layer.image);
                                delete layer.image;
                            } else if (layer.image && !layer.image.startsWith('http') && layer.image.length > 1000) {
                                layer.blob = await b64toBlob('data:image/png;base64,' + layer.image);
                                delete layer.image;
                            }
                            if (layer.mask && layer.mask.startsWith('data:')) {
                                layer.maskBlob = await b64toBlob(layer.mask);
                                delete layer.mask;
                            } else if (layer.mask && !layer.mask.startsWith('http') && layer.mask.length > 1000) {
                                layer.maskBlob = await b64toBlob('data:image/png;base64,' + layer.mask);
                                delete layer.mask;
                            }
                        }
                    };

                    await handleLayers(asset.layers);
                    await handleLayers(asset.scene?.layers);
                }
            }

            self.postMessage({ id, result: processedSession });
        } else if (type === 'deserializeSession') {
            const session = payload;
            
            // Helper to convert blob to base64
            const blobToB64 = (blob) => {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
            };

            if (session.messages) {
                for (let msg of session.messages) {
                    if (msg.imageData) {
                        const images = Array.isArray(msg.imageData) ? msg.imageData : [msg.imageData];
                        for (let img of images) {
                            if (img.blob) {
                                img.src = await blobToB64(img.blob);
                                delete img.blob;
                            }
                        }
                    }
                }
            }

            if (session.workbenchState) {
                for (let item of session.workbenchState) {
                    if (item.blob) {
                        item.dataUrl = await blobToB64(item.blob);
                        delete item.blob;
                    }
                    if (item.originalBlob) {
                        item.originalDataUrl = await blobToB64(item.originalBlob);
                        delete item.originalBlob;
                    }
                    if (item.cleanPlateBlob) {
                        item.cleanPlateDataUrl = await blobToB64(item.cleanPlateBlob);
                        delete item.cleanPlateBlob;
                    }
                    if (item.layers && Array.isArray(item.layers)) {
                        for (let layer of item.layers) {
                            if (layer.blob) {
                                layer.image = await blobToB64(layer.blob);
                                delete layer.blob;
                            }
                            if (layer.maskBlob) {
                                layer.mask = await blobToB64(layer.maskBlob);
                                delete layer.maskBlob;
                            }
                        }
                    }
                    if (item.scene && item.scene.layers) {
                        for (let layer of item.scene.layers) {
                            if (layer.blob) {
                                layer.image = await blobToB64(layer.blob);
                                delete layer.blob;
                            }
                            if (layer.maskBlob) {
                                layer.mask = await blobToB64(layer.maskBlob);
                                delete layer.maskBlob;
                            }
                        }
                    }
                }
            }

            if (session.runtimeWorkspace?.currentState?.assets) {
                for (let asset of session.runtimeWorkspace.currentState.assets) {
                    if (asset.sourceImageBlob) {
                        asset.sourceImage = await blobToB64(asset.sourceImageBlob);
                        delete asset.sourceImageBlob;
                    }
                    if (asset.originalBlob) {
                        asset.originalDataUrl = await blobToB64(asset.originalBlob);
                        delete asset.originalBlob;
                    }
                    if (asset.cleanPlateBlob) {
                        asset.cleanPlateDataUrl = await blobToB64(asset.cleanPlateBlob);
                        delete asset.cleanPlateBlob;
                    }

                    const handleLayers = async (layers) => {
                        if (!layers || !Array.isArray(layers)) return;
                        for (let layer of layers) {
                            if (layer.blob) {
                                layer.image = await blobToB64(layer.blob);
                                delete layer.blob;
                            }
                            if (layer.maskBlob) {
                                layer.mask = await blobToB64(layer.maskBlob);
                                delete layer.maskBlob;
                            }
                        }
                    };

                    await handleLayers(asset.layers);
                    await handleLayers(asset.scene?.layers);
                }
            }

            self.postMessage({ id, result: session });
        } else if (type === 'serializeAsset') {
            const asset = payload;
            const processedAsset = { ...asset };
            if (processedAsset.dataUrl && processedAsset.dataUrl.startsWith('data:')) {
                const res = await fetch(processedAsset.dataUrl);
                processedAsset.blob = await res.blob();
                delete processedAsset.dataUrl;
            }
            self.postMessage({ id, result: processedAsset });
        } else if (type === 'deserializeAsset') {
            const asset = payload;
            if (asset.blob) {
                const reader = new FileReader();
                const b64 = await new Promise((resolve) => {
                    reader.onloadend = () => resolve(reader.result);
                    reader.readAsDataURL(asset.blob);
                });
                asset.dataUrl = b64;
                delete asset.blob;
            }
            self.postMessage({ id, result: asset });
        }
    } catch (err) {
        self.postMessage({ id, error: err.message });
    }
};
