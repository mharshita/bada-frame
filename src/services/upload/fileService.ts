import { MULTIPART_PART_SIZE, FILE_READER_CHUNK_SIZE } from 'constants/upload';
import {
    FileTypeInfo,
    FileInMemory,
    Metadata,
    B64EncryptionResult,
    EncryptedFile,
    EncryptionResult,
    FileWithMetadata,
    ParsedMetadataJSONMap,
    DataStream,
} from 'types/upload';
import { splitFilenameAndExtension } from 'utils/file';
import { logError } from 'utils/sentry';
import { getFileNameSize, logUploadInfo } from 'utils/upload';
import { encryptFiledata } from './encryptionService';
import { extractMetadata, getMetadataJSONMapKey } from './metadataService';
import { getFileStream, getUint8ArrayView } from '../readerService';
import { generateThumbnail } from './thumbnailService';

const EDITED_FILE_SUFFIX = '-edited';

export function getFileSize(file: File) {
    return file.size;
}

export function getFilename(file: File) {
    return file.name;
}

export async function readFile(
    reader: FileReader,
    fileTypeInfo: FileTypeInfo,
    rawFile: File
): Promise<FileInMemory> {
    const { thumbnail, hasStaticThumbnail } = await generateThumbnail(
        reader,
        rawFile,
        fileTypeInfo
    );
    logUploadInfo(`reading file datal${getFileNameSize(rawFile)} `);
    let filedata: Uint8Array | DataStream;
    if (rawFile.size > MULTIPART_PART_SIZE) {
        filedata = getFileStream(reader, rawFile, FILE_READER_CHUNK_SIZE);
    } else {
        filedata = await getUint8ArrayView(reader, rawFile);
    }

    logUploadInfo(`read file data successfully ${getFileNameSize(rawFile)} `);

    return {
        filedata,
        thumbnail,
        hasStaticThumbnail,
    };
}

export async function extractFileMetadata(
    parsedMetadataJSONMap: ParsedMetadataJSONMap,
    rawFile: File,
    collectionID: number,
    fileTypeInfo: FileTypeInfo
) {
    const originalName = getFileOriginalName(rawFile);
    const googleMetadata =
        parsedMetadataJSONMap.get(
            getMetadataJSONMapKey(collectionID, originalName)
        ) ?? {};
    const extractedMetadata: Metadata = await extractMetadata(
        rawFile,
        fileTypeInfo
    );

    for (const [key, value] of Object.entries(googleMetadata)) {
        if (!value) {
            continue;
        }
        extractedMetadata[key] = value;
    }
    return extractedMetadata;
}

export async function encryptFile(
    worker: any,
    file: FileWithMetadata,
    encryptionKey: string
): Promise<EncryptedFile> {
    try {
        const { key: fileKey, file: encryptedFiledata } = await encryptFiledata(
            worker,
            file.filedata
        );

        const { file: encryptedThumbnail }: EncryptionResult =
            await worker.encryptThumbnail(file.thumbnail, fileKey);
        const { file: encryptedMetadata }: EncryptionResult =
            await worker.encryptMetadata(file.metadata, fileKey);

        const encryptedKey: B64EncryptionResult = await worker.encryptToB64(
            fileKey,
            encryptionKey
        );

        const result: EncryptedFile = {
            file: {
                file: encryptedFiledata,
                thumbnail: encryptedThumbnail,
                metadata: encryptedMetadata,
                localID: file.localID,
            },
            fileKey: encryptedKey,
        };
        return result;
    } catch (e) {
        logError(e, 'Error encrypting files');
        throw e;
    }
}

/*
    Get the original file name for edited file to associate it to original file's metadataJSON file 
    as edited file doesn't have their own metadata file
*/
function getFileOriginalName(file: File) {
    let originalName: string = null;
    const [nameWithoutExtension, extension] = splitFilenameAndExtension(
        file.name
    );

    const isEditedFile = nameWithoutExtension.endsWith(EDITED_FILE_SUFFIX);
    if (isEditedFile) {
        originalName = nameWithoutExtension.slice(
            0,
            -1 * EDITED_FILE_SUFFIX.length
        );
    } else {
        originalName = nameWithoutExtension;
    }
    if (extension) {
        originalName += '.' + extension;
    }
    return originalName;
}
