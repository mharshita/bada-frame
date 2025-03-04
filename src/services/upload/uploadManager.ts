import { getLocalFiles, setLocalFiles } from '../fileService';
import { getLocalCollections } from '../collectionService';
import { SetFiles } from 'types/gallery';
import { getDedicatedCryptoWorker } from 'utils/crypto';
import {
    sortFilesIntoCollections,
    sortFiles,
    preservePhotoswipeProps,
} from 'utils/file';
import { logError } from 'utils/sentry';
import { getMetadataJSONMapKey, parseMetadataJSON } from './metadataService';
import { getFileNameSize, segregateMetadataAndMediaFiles } from 'utils/upload';
import uploader from './uploader';
import UIService from './uiService';
import UploadService from './uploadService';
import { CustomError } from 'utils/error';
import { Collection } from 'types/collection';
import { EnteFile } from 'types/file';
import {
    FileWithCollection,
    MetadataAndFileTypeInfo,
    MetadataAndFileTypeInfoMap,
    ParsedMetadataJSON,
    ParsedMetadataJSONMap,
    ProgressUpdater,
} from 'types/upload';
import {
    UPLOAD_STAGES,
    FileUploadResults,
    MAX_FILE_SIZE_SUPPORTED,
} from 'constants/upload';
import { ComlinkWorker } from 'utils/comlink';
import { FILE_TYPE } from 'constants/file';
import uiService from './uiService';
import { getData, LS_KEYS, setData } from 'utils/storage/localStorage';
import { dedupe } from 'utils/export';
import { convertToHumanReadable } from 'utils/billing';
import { logUploadInfo } from 'utils/upload';

const MAX_CONCURRENT_UPLOADS = 4;
const FILE_UPLOAD_COMPLETED = 100;

class UploadManager {
    private cryptoWorkers = new Array<ComlinkWorker>(MAX_CONCURRENT_UPLOADS);
    private parsedMetadataJSONMap: ParsedMetadataJSONMap;
    private metadataAndFileTypeInfoMap: MetadataAndFileTypeInfoMap;
    private filesToBeUploaded: FileWithCollection[];
    private failedFiles: FileWithCollection[];
    private existingFilesCollectionWise: Map<number, EnteFile[]>;
    private existingFiles: EnteFile[];
    private setFiles: SetFiles;
    private collections: Map<number, Collection>;
    public initUploader(progressUpdater: ProgressUpdater, setFiles: SetFiles) {
        UIService.init(progressUpdater);
        this.setFiles = setFiles;
    }

    private async init(newCollections?: Collection[]) {
        this.filesToBeUploaded = [];
        this.failedFiles = [];
        this.parsedMetadataJSONMap = new Map<string, ParsedMetadataJSON>();
        this.metadataAndFileTypeInfoMap = new Map<
            number,
            MetadataAndFileTypeInfo
        >();
        this.existingFiles = await getLocalFiles();
        this.existingFilesCollectionWise = sortFilesIntoCollections(
            this.existingFiles
        );
        const collections = await getLocalCollections();
        if (newCollections) {
            collections.push(...newCollections);
        }
        this.collections = new Map(
            collections.map((collection) => [collection.id, collection])
        );
    }

    public async queueFilesForUpload(
        fileWithCollectionToBeUploaded: FileWithCollection[],
        newCreatedCollections?: Collection[]
    ) {
        try {
            await this.init(newCreatedCollections);
            logUploadInfo(
                `received ${fileWithCollectionToBeUploaded.length} files to upload`
            );
            const { metadataJSONFiles, mediaFiles } =
                segregateMetadataAndMediaFiles(fileWithCollectionToBeUploaded);
            logUploadInfo(
                `has ${metadataJSONFiles.length} metadata json files`
            );
            logUploadInfo(`has ${mediaFiles.length} media files`);
            if (metadataJSONFiles.length) {
                UIService.setUploadStage(
                    UPLOAD_STAGES.READING_GOOGLE_METADATA_FILES
                );
                await this.parseMetadataJSONFiles(metadataJSONFiles);
                UploadService.setParsedMetadataJSONMap(
                    this.parsedMetadataJSONMap
                );
            }
            if (mediaFiles.length) {
                UIService.setUploadStage(UPLOAD_STAGES.EXTRACTING_METADATA);
                await this.extractMetadataFromFiles(mediaFiles);
                UploadService.setMetadataAndFileTypeInfoMap(
                    this.metadataAndFileTypeInfoMap
                );
                UIService.setUploadStage(UPLOAD_STAGES.START);
                logUploadInfo(`clusterLivePhotoFiles called`);
                const analysedMediaFiles =
                    UploadService.clusterLivePhotoFiles(mediaFiles);
                uiService.setFilenames(
                    new Map<number, string>(
                        analysedMediaFiles.map((mediaFile) => [
                            mediaFile.localID,
                            UploadService.getAssetName(mediaFile),
                        ])
                    )
                );

                UIService.setHasLivePhoto(
                    mediaFiles.length !== analysedMediaFiles.length
                );
                logUploadInfo(
                    `got live photos: ${
                        mediaFiles.length !== analysedMediaFiles.length
                    }`
                );

                await this.uploadMediaFiles(analysedMediaFiles);
            }
            UIService.setUploadStage(UPLOAD_STAGES.FINISH);
            UIService.setPercentComplete(FILE_UPLOAD_COMPLETED);
        } catch (e) {
            logError(e, 'uploading failed with error');
            throw e;
        } finally {
            for (let i = 0; i < MAX_CONCURRENT_UPLOADS; i++) {
                this.cryptoWorkers[i]?.worker.terminate();
            }
        }
    }

    private async parseMetadataJSONFiles(metadataFiles: FileWithCollection[]) {
        try {
            logUploadInfo(`parseMetadataJSONFiles function executed `);

            UIService.reset(metadataFiles.length);
            const reader = new FileReader();
            for (const { file, collectionID } of metadataFiles) {
                try {
                    logUploadInfo(
                        `parsing metadata json file ${getFileNameSize(file)}`
                    );

                    const parsedMetadataJSONWithTitle = await parseMetadataJSON(
                        reader,
                        file
                    );
                    if (parsedMetadataJSONWithTitle) {
                        const { title, parsedMetadataJSON } =
                            parsedMetadataJSONWithTitle;
                        this.parsedMetadataJSONMap.set(
                            getMetadataJSONMapKey(collectionID, title),
                            parsedMetadataJSON && { ...parsedMetadataJSON }
                        );
                        UIService.increaseFileUploaded();
                    }
                    logUploadInfo(
                        `successfully parsed metadata json file ${getFileNameSize(
                            file
                        )}`
                    );
                } catch (e) {
                    logError(e, 'parsing failed for a file');
                    logUploadInfo(
                        `successfully parsed metadata json file ${getFileNameSize(
                            file
                        )} error: ${e.message}`
                    );
                }
            }
        } catch (e) {
            logError(e, 'error seeding MetadataMap');
            // silently ignore the error
        }
    }

    private async extractMetadataFromFiles(mediaFiles: FileWithCollection[]) {
        try {
            logUploadInfo(`extractMetadataFromFiles executed`);
            UIService.reset(mediaFiles.length);
            const reader = new FileReader();
            for (const { file, localID, collectionID } of mediaFiles) {
                try {
                    const { fileTypeInfo, metadata } = await (async () => {
                        if (file.size >= MAX_FILE_SIZE_SUPPORTED) {
                            logUploadInfo(
                                `${getFileNameSize(
                                    file
                                )} rejected  because of large size`
                            );

                            return { fileTypeInfo: null, metadata: null };
                        }
                        const fileTypeInfo = await UploadService.getFileType(
                            reader,
                            file
                        );
                        if (fileTypeInfo.fileType === FILE_TYPE.OTHERS) {
                            logUploadInfo(
                                `${getFileNameSize(
                                    file
                                )} rejected  because of unknown file format`
                            );
                            return { fileTypeInfo, metadata: null };
                        }
                        logUploadInfo(
                            ` extracting ${getFileNameSize(file)} metadata`
                        );
                        const metadata =
                            (await UploadService.extractFileMetadata(
                                file,
                                collectionID,
                                fileTypeInfo
                            )) || null;
                        return { fileTypeInfo, metadata };
                    })();

                    logUploadInfo(
                        `metadata extraction successful${getFileNameSize(
                            file
                        )} `
                    );
                    this.metadataAndFileTypeInfoMap.set(localID, {
                        fileTypeInfo: fileTypeInfo && { ...fileTypeInfo },
                        metadata: metadata && { ...metadata },
                    });
                    UIService.increaseFileUploaded();
                } catch (e) {
                    logError(e, 'metadata extraction failed for a file');
                    logUploadInfo(
                        `metadata extraction failed ${getFileNameSize(
                            file
                        )} error: ${e.message}`
                    );
                }
            }
        } catch (e) {
            logError(e, 'error extracting metadata');
            // silently ignore the error
        }
    }

    private async uploadMediaFiles(mediaFiles: FileWithCollection[]) {
        logUploadInfo(`uploadMediaFiles called`);
        this.filesToBeUploaded.push(...mediaFiles);
        UIService.reset(mediaFiles.length);

        await UploadService.setFileCount(mediaFiles.length);

        UIService.setUploadStage(UPLOAD_STAGES.UPLOADING);

        const uploadProcesses = [];
        for (
            let i = 0;
            i < MAX_CONCURRENT_UPLOADS && this.filesToBeUploaded.length > 0;
            i++
        ) {
            const cryptoWorker = getDedicatedCryptoWorker();
            if (!cryptoWorker) {
                throw Error(CustomError.FAILED_TO_LOAD_WEB_WORKER);
            }
            this.cryptoWorkers[i] = cryptoWorker;
            uploadProcesses.push(
                this.uploadNextFileInQueue(
                    await new this.cryptoWorkers[i].comlink(),
                    new FileReader()
                )
            );
        }
        await Promise.all(uploadProcesses);
    }

    private async uploadNextFileInQueue(worker: any, reader: FileReader) {
        while (this.filesToBeUploaded.length > 0) {
            const fileWithCollection = this.filesToBeUploaded.pop();
            const { collectionID } = fileWithCollection;
            const existingFilesInCollection =
                this.existingFilesCollectionWise.get(collectionID) ?? [];
            const collection = this.collections.get(collectionID);

            const { fileUploadResult, file } = await uploader(
                worker,
                reader,
                existingFilesInCollection,
                this.existingFiles,
                { ...fileWithCollection, collection }
            );
            if (fileUploadResult === FileUploadResults.UPLOADED) {
                this.existingFiles.push(file);
                this.existingFiles = sortFiles(this.existingFiles);
                await setLocalFiles(this.existingFiles);
                this.setFiles(preservePhotoswipeProps(this.existingFiles));
                if (!this.existingFilesCollectionWise.has(file.collectionID)) {
                    this.existingFilesCollectionWise.set(file.collectionID, []);
                }
                this.existingFilesCollectionWise
                    .get(file.collectionID)
                    .push(file);
            }
            if (fileUploadResult === FileUploadResults.FAILED) {
                this.failedFiles.push(fileWithCollection);
                setData(LS_KEYS.FAILED_UPLOADS, {
                    files: dedupe([
                        ...(getData(LS_KEYS.FAILED_UPLOADS)?.files ?? []),
                        ...this.failedFiles.map(
                            (file) =>
                                `${file.file.name}_${convertToHumanReadable(
                                    file.file.size
                                )}`
                        ),
                    ]),
                });
            } else if (fileUploadResult === FileUploadResults.BLOCKED) {
                this.failedFiles.push(fileWithCollection);
            }

            UIService.moveFileToResultList(
                fileWithCollection.localID,
                fileUploadResult
            );
            UploadService.reducePendingUploadCount();
        }
    }

    async retryFailedFiles() {
        await this.queueFilesForUpload(this.failedFiles);
    }
}

export default new UploadManager();
