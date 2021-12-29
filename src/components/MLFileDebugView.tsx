import React, { useEffect, useRef, useState } from 'react';
import arcfaceAlignmentService from 'services/machineLearning/arcfaceAlignmentService';
import arcfaceCropService from 'services/machineLearning/arcfaceCropService';
import blazemeshLandmarksService from 'services/machineLearning/blazemeshLandmarksService';
import tfjsFaceDetectionService from 'services/machineLearning/tfjsFaceDetectionService';
import { AlignedFace, DetectedFace } from 'types/machineLearning';
import { getMLSyncConfig } from 'utils/machineLearning';
import {
    getAlignedFaceBox,
    ibExtractFaceImage,
    ibExtractFaceImageUsingTransform,
} from 'utils/machineLearning/faceAlign';
import {
    ibExtractFaceImageFromCrop,
    transformToFaceCropDims,
    transformToImageDims,
} from 'utils/machineLearning/faceCrop';
import { ImageBitmapView, ImageBlobView } from './ImageViews';
import { MESH_ANNOTATIONS } from '@tensorflow-models/face-landmarks-detection/dist/mediapipe-facemesh/keypoints';

interface MLFileDebugViewProps {
    file: File;
}

const LANDMARKS_3D_VISIBLE = [];
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['silhouette']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['midwayBetweenEyes']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['noseTip']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['leftEyeIris']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['leftEyeUpper0']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['leftEyeLower0']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['rightEyeIris']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['rightEyeUpper0']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['rightEyeLower0']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['lipsUpperOuter']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['lipsLowerOuter']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['lipsUpperInner']);
LANDMARKS_3D_VISIBLE.push(...MESH_ANNOTATIONS['lipsLowerInner']);

function drawFaceDetection(face: AlignedFace, ctx: CanvasRenderingContext2D) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = 6;
    ctx.strokeRect(face.box.x, face.box.y, face.box.width, face.box.height);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    ctx.lineWidth = 10;
    const alignedBox = getAlignedFaceBox(face);
    ctx.strokeRect(
        alignedBox.x,
        alignedBox.y,
        alignedBox.width,
        alignedBox.height
    );
    ctx.restore();

    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 255, 0.8)';
    face.landmarks.forEach((l) => {
        ctx.beginPath();
        ctx.arc(l.x, l.y, 8, 0, Math.PI * 2, true);
        ctx.fill();
    });
    ctx.restore();

    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    face.landmarks3d.forEach((l, i) => {
        if (!LANDMARKS_3D_VISIBLE.includes(i)) {
            return;
        }
        ctx.beginPath();
        ctx.arc(l.x, l.y, 3, 0, Math.PI * 2, true);
        ctx.fill();
    });
    ctx.restore();
}

export default function MLFileDebugView(props: MLFileDebugViewProps) {
    // const [imageBitmap, setImageBitmap] = useState<ImageBitmap>();
    const [faces, setFaces] = useState<DetectedFace[]>();
    const [faceImages, setImages] = useState<ImageBitmap[]>();

    const canvasRef = useRef(null);

    useEffect(() => {
        let didCancel = false;
        const loadFile = async () => {
            // TODO: go through worker for these apis, to not include ml code in main bundle
            const imageBitmap = await createImageBitmap(props.file);
            const detectedFaces = await tfjsFaceDetectionService.detectFaces(
                imageBitmap
            );
            const mlSyncConfig = await getMLSyncConfig();
            const facePromises = detectedFaces.map(async (face) => {
                face.faceCrop = await arcfaceCropService.getFaceCrop(
                    imageBitmap,
                    face,
                    mlSyncConfig.faceCrop
                );

                const faceCropImage = await createImageBitmap(
                    face.faceCrop.image
                );
                const faceCrop = {
                    image: faceCropImage,
                    imageBox: face.faceCrop.imageBox,
                };
                face = transformToFaceCropDims(faceCrop, face);
                const landmarks3d =
                    await blazemeshLandmarksService.getFaceLandmarks(
                        faceCropImage,
                        face
                        // mlSyncConfig.faceLandmarks
                    );
                // console.log("coords: ", landmarks3d);

                face.landmarks3d = landmarks3d;
                face = transformToImageDims(faceCrop, face);
                return face;
            });

            const facesWithLandmarks = await Promise.all(facePromises);
            if (didCancel) return;
            setFaces(facesWithLandmarks);
            console.log('detectedFaces: ', facesWithLandmarks);

            const alignedFaces =
                arcfaceAlignmentService.getAlignedFaces(facesWithLandmarks);
            console.log('alignedFaces: ', alignedFaces);

            const canvas: HTMLCanvasElement = canvasRef.current;
            canvas.width = imageBitmap.width;
            canvas.height = imageBitmap.height;
            const ctx = canvas.getContext('2d');
            if (didCancel) return;
            ctx.drawImage(imageBitmap, 0, 0);
            alignedFaces.forEach((face) => drawFaceDetection(face, ctx));

            const faceCropPromises = alignedFaces.map((face) => {
                return ibExtractFaceImageFromCrop(face, 112);
            });
            const faceImagePromises = alignedFaces.map((face) => {
                return ibExtractFaceImage(imageBitmap, face, 112);
            });
            const faceImageTransformPromises = alignedFaces.map((face) => {
                return ibExtractFaceImageUsingTransform(imageBitmap, face, 112);
            });
            const faceImages = await Promise.all([
                ...faceCropPromises,
                ...faceImagePromises,
                ...faceImageTransformPromises,
            ]);
            if (didCancel) return;
            setImages(faceImages);
        };

        props.file && loadFile();
        return () => {
            didCancel = true;
        };
    }, [props.file]);

    return (
        <div>
            <p></p>
            {/* <ImageBitmapView image={imageBitmap}></ImageBitmapView> */}
            <canvas
                ref={canvasRef}
                style={{ display: 'inline', width: '100%' }}
            />
            <p></p>
            <div>Face Crops:</div>
            {faces?.map((face, i) => (
                <ImageBlobView
                    key={i}
                    blob={face.faceCrop?.image}></ImageBlobView>
            ))}

            <p></p>
            <div>
                Face Images using face crops, original image and using
                transform:
            </div>
            {faceImages?.map((image, i) => (
                <ImageBitmapView key={i} image={image}></ImageBitmapView>
            ))}
        </div>
    );
}
