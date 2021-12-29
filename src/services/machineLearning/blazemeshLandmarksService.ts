import * as tf from '@tensorflow/tfjs-core';
import { GraphModel, loadGraphModel } from '@tensorflow/tfjs-converter';
import {
    BLAZEMESH_INPUT_SIZE,
    DetectedFace,
    FaceLandmarksMethod,
    FaceLandmarksService,
    Landmark,
    Versioned,
} from 'types/machineLearning';
import { cropWithRotation } from 'utils/image';
import { enlargeBox, toTensor4D } from 'utils/machineLearning';
import { Point } from '../../../thirdparty/face-api/classes';
import { Dimensions } from 'types/image';
import { compose, scale, translate } from 'transformation-matrix';
import { transformPoint } from 'utils/machineLearning/transform';

class BlazeMeshLandmarksService implements FaceLandmarksService {
    public method: Versioned<FaceLandmarksMethod>;
    private blazeMeshModel: Promise<GraphModel>;

    constructor() {
        this.method = {
            value: 'BlazeMesh',
            version: 1,
        };
    }

    private async init() {
        this.blazeMeshModel = loadGraphModel('/models/blazemesh/v1/model.json');
    }

    private async getBlazeMeshModel(): Promise<GraphModel> {
        if (!this.blazeMeshModel) {
            await this.init();
        }

        return this.blazeMeshModel;
    }

    public async getFaceLandmarks(
        faceImage: ImageBitmap,
        face: DetectedFace
        // config: FaceLandmarksConfig
    ): Promise<Landmark[]> {
        // const BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [3, 2];
        // const [indexOfNose, indexOfForehead] =
        //     BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
        // const rotation = computeRotation(
        //     toCoord2d(face.landmarks[indexOfNose]),
        //     toCoord2d(face.landmarks[indexOfForehead])
        // );

        const enlargedBox = enlargeBox(face.box);
        // console.log({ enlargedBox });

        const faceSizeDimentions: Dimensions = {
            width: BLAZEMESH_INPUT_SIZE,
            height: BLAZEMESH_INPUT_SIZE,
        };
        const cropped = cropWithRotation(
            faceImage,
            enlargedBox,
            0, // -rotation,
            faceSizeDimentions,
            faceSizeDimentions
        );

        // const resized = resizeToSquare(faceImage, BLAZEMESH_INPUT_SIZE);
        const tfImage3D = tf.browser.fromPixels(cropped);
        const tf4dFloat32Image = toTensor4D(tfImage3D, 'float32').div(255);
        tf.dispose(tfImage3D);
        const blazeMeshModel = await this.getBlazeMeshModel();

        const [, , coords] = blazeMeshModel.predict(tf4dFloat32Image) as [
            tf.Tensor,
            tf.Tensor2D,
            tf.Tensor2D
        ];
        tf.dispose(tf4dFloat32Image);
        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        const rawCoords = await coordsReshaped.array();

        const transform = compose(
            translate(enlargedBox.x, enlargedBox.y),
            scale(
                enlargedBox.width / BLAZEMESH_INPUT_SIZE,
                enlargedBox.width / BLAZEMESH_INPUT_SIZE
            )
            // rotate(rotation, BLAZEMESH_INPUT_SIZE / 2, BLAZEMESH_INPUT_SIZE / 2),
        );
        const coordsRotated = rawCoords.map((rc) =>
            transformPoint(new Point(rc[0], rc[1]), transform)
        );

        return coordsRotated;
    }
}

export default new BlazeMeshLandmarksService();
