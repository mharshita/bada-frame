import { getEndpoint } from 'utils/common/apiUtil';
import { getData, LS_KEYS } from 'utils/storage/localStorage';
import localForage from 'utils/storage/localForage';

import { getActualKey, getToken } from 'utils/common/key';
import CryptoWorker from 'utils/crypto';
import { SetDialogMessage } from 'components/MessageDialog';
import constants from 'utils/strings/constants';
import { getPublicKey } from './userService';
import { B64EncryptionResult } from 'utils/crypto';
import HTTPService from './HTTPService';
import { EnteFile } from 'types/file';
import { logError } from 'utils/sentry';
import { CustomError } from 'utils/error';
import { sortFiles } from 'utils/file';
import {
    Collection,
    CollectionAndItsLatestFile,
    AddToCollectionRequest,
    MoveToCollectionRequest,
    EncryptedFileKey,
    RemoveFromCollectionRequest,
    CreatePublicAccessTokenRequest,
    PublicURL,
    UpdatePublicURL,
} from 'types/collection';
import { COLLECTION_SORT_BY, CollectionType } from 'constants/collection';

const ENDPOINT = getEndpoint();
const COLLECTION_TABLE = 'collections';
const COLLECTION_UPDATION_TIME = 'collection-updation-time';

const getCollectionWithSecrets = async (
    collection: Collection,
    masterKey: string
) => {
    const worker = await new CryptoWorker();
    const userID = getData(LS_KEYS.USER).id;
    let decryptedKey: string;
    if (collection.owner.id === userID) {
        decryptedKey = await worker.decryptB64(
            collection.encryptedKey,
            collection.keyDecryptionNonce,
            masterKey
        );
    } else {
        const keyAttributes = getData(LS_KEYS.KEY_ATTRIBUTES);
        const secretKey = await worker.decryptB64(
            keyAttributes.encryptedSecretKey,
            keyAttributes.secretKeyDecryptionNonce,
            masterKey
        );
        decryptedKey = await worker.boxSealOpen(
            collection.encryptedKey,
            keyAttributes.publicKey,
            secretKey
        );
    }
    collection.name =
        collection.name ||
        (await worker.decryptToUTF8(
            collection.encryptedName,
            collection.nameDecryptionNonce,
            decryptedKey
        ));
    return {
        ...collection,
        key: decryptedKey,
    };
};

const getCollections = async (
    token: string,
    sinceTime: number,
    key: string
): Promise<Collection[]> => {
    try {
        const resp = await HTTPService.get(
            `${ENDPOINT}/collections`,
            {
                sinceTime,
            },
            { 'X-Auth-Token': token }
        );
        const promises: Promise<Collection>[] = resp.data.collections.map(
            async (collection: Collection) => {
                if (collection.isDeleted) {
                    return collection;
                }
                let collectionWithSecrets = collection;
                try {
                    collectionWithSecrets = await getCollectionWithSecrets(
                        collection,
                        key
                    );
                } catch (e) {
                    logError(e, `decryption failed for collection`, {
                        collectionID: collection.id,
                    });
                }
                return collectionWithSecrets;
            }
        );
        // only allow deleted or collection with key, filtering out collection whose decryption failed
        const collections = (await Promise.all(promises)).filter(
            (collection) => collection.isDeleted || collection.key
        );
        return collections;
    } catch (e) {
        logError(e, 'getCollections failed');
        throw e;
    }
};

export const getLocalCollections = async (): Promise<Collection[]> => {
    const collections: Collection[] =
        (await localForage.getItem(COLLECTION_TABLE)) ?? [];
    return collections;
};

export const getCollectionUpdationTime = async (): Promise<number> =>
    (await localForage.getItem<number>(COLLECTION_UPDATION_TIME)) ?? 0;

export const syncCollections = async () => {
    const localCollections = await getLocalCollections();
    const lastCollectionUpdationTime = await getCollectionUpdationTime();
    const token = getToken();
    const key = await getActualKey();
    const updatedCollections =
        (await getCollections(token, lastCollectionUpdationTime, key)) ?? [];
    if (updatedCollections.length === 0) {
        return localCollections;
    }
    const allCollectionsInstances = [
        ...localCollections,
        ...updatedCollections,
    ];
    const latestCollectionsInstances = new Map<number, Collection>();
    allCollectionsInstances.forEach((collection) => {
        if (
            !latestCollectionsInstances.has(collection.id) ||
            latestCollectionsInstances.get(collection.id).updationTime <
                collection.updationTime
        ) {
            latestCollectionsInstances.set(collection.id, collection);
        }
    });

    let collections: Collection[] = [];
    let updationTime = await localForage.getItem<number>(
        COLLECTION_UPDATION_TIME
    );
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for (const [_, collection] of latestCollectionsInstances) {
        if (!collection.isDeleted) {
            collections.push(collection);
            updationTime = Math.max(updationTime, collection.updationTime);
        }
    }
    collections = sortCollections(
        collections,
        [],
        COLLECTION_SORT_BY.MODIFICATION_TIME
    );
    await localForage.setItem(COLLECTION_TABLE, collections);
    await localForage.setItem(COLLECTION_UPDATION_TIME, updationTime);
    return collections;
};

export const getCollection = async (
    collectionID: number
): Promise<Collection> => {
    try {
        const token = getToken();
        if (!token) {
            return;
        }
        const resp = await HTTPService.get(
            `${ENDPOINT}/collections/${collectionID}`,
            null,
            { 'X-Auth-Token': token }
        );
        const key = await getActualKey();
        const collectionWithSecrets = await getCollectionWithSecrets(
            resp.data?.collection,
            key
        );
        return collectionWithSecrets;
    } catch (e) {
        logError(e, 'failed to get collection', { collectionID });
    }
};

export const getCollectionsAndTheirLatestFile = (
    collections: Collection[],
    files: EnteFile[]
): CollectionAndItsLatestFile[] => {
    const latestFile = new Map<number, EnteFile>();

    files.forEach((file) => {
        if (!latestFile.has(file.collectionID)) {
            latestFile.set(file.collectionID, file);
        }
    });
    const collectionsAndTheirLatestFile: CollectionAndItsLatestFile[] = [];

    for (const collection of collections) {
        collectionsAndTheirLatestFile.push({
            collection,
            file: latestFile.get(collection.id),
        });
    }
    return collectionsAndTheirLatestFile;
};

export const getFavItemIds = async (
    files: EnteFile[]
): Promise<Set<number>> => {
    const favCollection = await getFavCollection();
    if (!favCollection) return new Set();

    return new Set(
        files
            .filter((file) => file.collectionID === favCollection.id)
            .map((file): number => file.id)
    );
};

export const createAlbum = async (
    albumName: string,
    existingCollection?: Collection[]
) => createCollection(albumName, CollectionType.album, existingCollection);

export const createCollection = async (
    collectionName: string,
    type: CollectionType,
    existingCollections?: Collection[]
): Promise<Collection> => {
    try {
        if (!existingCollections) {
            existingCollections = await syncCollections();
        }
        for (const collection of existingCollections) {
            if (collection.name === collectionName) {
                return collection;
            }
        }
        const worker = await new CryptoWorker();
        const encryptionKey = await getActualKey();
        const token = getToken();
        const collectionKey: string = await worker.generateEncryptionKey();
        const {
            encryptedData: encryptedKey,
            nonce: keyDecryptionNonce,
        }: B64EncryptionResult = await worker.encryptToB64(
            collectionKey,
            encryptionKey
        );
        const {
            encryptedData: encryptedName,
            nonce: nameDecryptionNonce,
        }: B64EncryptionResult = await worker.encryptUTF8(
            collectionName,
            collectionKey
        );
        const newCollection: Collection = {
            id: null,
            owner: null,
            encryptedKey,
            keyDecryptionNonce,
            encryptedName,
            nameDecryptionNonce,
            type,
            attributes: {},
            sharees: null,
            updationTime: null,
            isDeleted: false,
        };
        let createdCollection: Collection = await postCollection(
            newCollection,
            token
        );
        createdCollection = await getCollectionWithSecrets(
            createdCollection,
            encryptionKey
        );
        return createdCollection;
    } catch (e) {
        logError(e, 'create collection failed');
        throw e;
    }
};

const postCollection = async (
    collectionData: Collection,
    token: string
): Promise<Collection> => {
    try {
        const response = await HTTPService.post(
            `${ENDPOINT}/collections`,
            collectionData,
            null,
            { 'X-Auth-Token': token }
        );
        return response.data.collection;
    } catch (e) {
        logError(e, 'post Collection failed ');
    }
};

export const addToFavorites = async (file: EnteFile) => {
    try {
        let favCollection = await getFavCollection();
        if (!favCollection) {
            favCollection = await createCollection(
                'Favorites',
                CollectionType.favorites
            );
            const localCollections = await getLocalCollections();
            await localForage.setItem(COLLECTION_TABLE, [
                ...localCollections,
                favCollection,
            ]);
        }
        await addToCollection(favCollection, [file]);
    } catch (e) {
        logError(e, 'failed to add to favorite');
    }
};

export const removeFromFavorites = async (file: EnteFile) => {
    try {
        const favCollection = await getFavCollection();
        if (!favCollection) {
            throw Error(CustomError.FAV_COLLECTION_MISSING);
        }
        await removeFromCollection(favCollection, [file]);
    } catch (e) {
        logError(e, 'remove from favorite failed');
    }
};

export const addToCollection = async (
    collection: Collection,
    files: EnteFile[]
) => {
    try {
        const token = getToken();
        const fileKeysEncryptedWithNewCollection =
            await encryptWithNewCollectionKey(collection, files);

        const requestBody: AddToCollectionRequest = {
            collectionID: collection.id,
            files: fileKeysEncryptedWithNewCollection,
        };
        await HTTPService.post(
            `${ENDPOINT}/collections/add-files`,
            requestBody,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'Add to collection Failed ');
        throw e;
    }
};

export const restoreToCollection = async (
    collection: Collection,
    files: EnteFile[]
) => {
    try {
        const token = getToken();
        const fileKeysEncryptedWithNewCollection =
            await encryptWithNewCollectionKey(collection, files);

        const requestBody: AddToCollectionRequest = {
            collectionID: collection.id,
            files: fileKeysEncryptedWithNewCollection,
        };
        await HTTPService.post(
            `${ENDPOINT}/collections/restore-files`,
            requestBody,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'restore to collection Failed ');
        throw e;
    }
};
export const moveToCollection = async (
    fromCollectionID: number,
    toCollection: Collection,
    files: EnteFile[]
) => {
    try {
        const token = getToken();
        const fileKeysEncryptedWithNewCollection =
            await encryptWithNewCollectionKey(toCollection, files);

        const requestBody: MoveToCollectionRequest = {
            fromCollectionID: fromCollectionID,
            toCollectionID: toCollection.id,
            files: fileKeysEncryptedWithNewCollection,
        };
        await HTTPService.post(
            `${ENDPOINT}/collections/move-files`,
            requestBody,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'move to collection Failed ');
        throw e;
    }
};

const encryptWithNewCollectionKey = async (
    newCollection: Collection,
    files: EnteFile[]
): Promise<EncryptedFileKey[]> => {
    const fileKeysEncryptedWithNewCollection: EncryptedFileKey[] = [];
    const worker = await new CryptoWorker();
    for (const file of files) {
        const newEncryptedKey: B64EncryptionResult = await worker.encryptToB64(
            file.key,
            newCollection.key
        );
        file.encryptedKey = newEncryptedKey.encryptedData;
        file.keyDecryptionNonce = newEncryptedKey.nonce;

        fileKeysEncryptedWithNewCollection.push({
            id: file.id,
            encryptedKey: file.encryptedKey,
            keyDecryptionNonce: file.keyDecryptionNonce,
        });
    }
    return fileKeysEncryptedWithNewCollection;
};
export const removeFromCollection = async (
    collection: Collection,
    files: EnteFile[]
) => {
    try {
        const token = getToken();
        const request: RemoveFromCollectionRequest = {
            collectionID: collection.id,
            fileIDs: files.map((file) => file.id),
        };

        await HTTPService.post(
            `${ENDPOINT}/collections/v2/remove-files`,
            request,
            null,
            { 'X-Auth-Token': token }
        );
    } catch (e) {
        logError(e, 'remove from collection failed ');
        throw e;
    }
};

export const deleteCollection = async (
    collectionID: number,
    syncWithRemote: () => Promise<void>,
    redirectToAll: () => void,
    setDialogMessage: SetDialogMessage
) => {
    try {
        const token = getToken();

        await HTTPService.delete(
            `${ENDPOINT}/collections/v2/${collectionID}`,
            null,
            null,
            { 'X-Auth-Token': token }
        );
        await syncWithRemote();
        redirectToAll();
    } catch (e) {
        logError(e, 'delete collection failed ');
        setDialogMessage({
            title: constants.ERROR,
            content: constants.DELETE_COLLECTION_FAILED,
            close: { variant: 'danger' },
        });
    }
};

export const renameCollection = async (
    collection: Collection,
    newCollectionName: string
) => {
    const token = getToken();
    const worker = await new CryptoWorker();
    const {
        encryptedData: encryptedName,
        nonce: nameDecryptionNonce,
    }: B64EncryptionResult = await worker.encryptUTF8(
        newCollectionName,
        collection.key
    );
    const collectionRenameRequest = {
        collectionID: collection.id,
        encryptedName,
        nameDecryptionNonce,
    };
    await HTTPService.post(
        `${ENDPOINT}/collections/rename`,
        collectionRenameRequest,
        null,
        {
            'X-Auth-Token': token,
        }
    );
};
export const shareCollection = async (
    collection: Collection,
    withUserEmail: string
) => {
    try {
        const worker = await new CryptoWorker();

        const token = getToken();
        const publicKey: string = await getPublicKey(withUserEmail);
        const encryptedKey: string = await worker.boxSeal(
            collection.key,
            publicKey
        );
        const shareCollectionRequest = {
            collectionID: collection.id,
            email: withUserEmail,
            encryptedKey,
        };
        await HTTPService.post(
            `${ENDPOINT}/collections/share`,
            shareCollectionRequest,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'share collection failed ');
        throw e;
    }
};

export const unshareCollection = async (
    collection: Collection,
    withUserEmail: string
) => {
    try {
        const token = getToken();
        const shareCollectionRequest = {
            collectionID: collection.id,
            email: withUserEmail,
        };
        await HTTPService.post(
            `${ENDPOINT}/collections/unshare`,
            shareCollectionRequest,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'unshare collection failed ');
    }
};

export const createShareableURL = async (collection: Collection) => {
    try {
        const token = getToken();
        if (!token) {
            return null;
        }
        const createPublicAccessTokenRequest: CreatePublicAccessTokenRequest = {
            collectionID: collection.id,
        };
        const resp = await HTTPService.post(
            `${ENDPOINT}/collections/share-url`,
            createPublicAccessTokenRequest,
            null,
            {
                'X-Auth-Token': token,
            }
        );
        return resp.data.result as PublicURL;
    } catch (e) {
        logError(e, 'createShareableURL failed ');
        throw e;
    }
};

export const deleteShareableURL = async (collection: Collection) => {
    try {
        const token = getToken();
        if (!token) {
            return null;
        }
        await HTTPService.delete(
            `${ENDPOINT}/collections/share-url/${collection.id}`,
            null,
            null,
            {
                'X-Auth-Token': token,
            }
        );
    } catch (e) {
        logError(e, 'deleteShareableURL failed ');
        throw e;
    }
};

export const updateShareableURL = async (
    request: UpdatePublicURL
): Promise<PublicURL> => {
    try {
        const token = getToken();
        if (!token) {
            return null;
        }
        const res = await HTTPService.put(
            `${ENDPOINT}/collections/share-url`,
            request,
            null,
            {
                'X-Auth-Token': token,
            }
        );
        return res.data.result as PublicURL;
    } catch (e) {
        logError(e, 'updateShareableURL failed ');
        throw e;
    }
};

export const getFavCollection = async () => {
    const collections = await getLocalCollections();
    for (const collection of collections) {
        if (collection.type === CollectionType.favorites) {
            return collection;
        }
    }
    return null;
};

export const getNonEmptyCollections = (
    collections: Collection[],
    files: EnteFile[]
) => {
    const nonEmptyCollectionsIds = new Set<number>();
    for (const file of files) {
        nonEmptyCollectionsIds.add(file.collectionID);
    }
    return collections.filter((collection) =>
        nonEmptyCollectionsIds.has(collection.id)
    );
};

export function sortCollections(
    collections: Collection[],
    collectionAndTheirLatestFile: CollectionAndItsLatestFile[],
    sortBy: COLLECTION_SORT_BY
) {
    return moveFavCollectionToFront(
        collections.sort((collectionA, collectionB) => {
            switch (sortBy) {
                case COLLECTION_SORT_BY.LATEST_FILE:
                    return compareCollectionsLatestFile(
                        collectionAndTheirLatestFile,
                        collectionA,
                        collectionB
                    );
                case COLLECTION_SORT_BY.MODIFICATION_TIME:
                    return collectionB.updationTime - collectionA.updationTime;
                case COLLECTION_SORT_BY.NAME:
                    return collectionA.name.localeCompare(collectionB.name);
            }
        })
    );
}

function compareCollectionsLatestFile(
    collectionAndTheirLatestFile: CollectionAndItsLatestFile[],
    collectionA: Collection,
    collectionB: Collection
) {
    if (!collectionAndTheirLatestFile?.length) {
        return 0;
    }
    const CollectionALatestFile = getCollectionLatestFile(
        collectionAndTheirLatestFile,
        collectionA
    );
    const CollectionBLatestFile = getCollectionLatestFile(
        collectionAndTheirLatestFile,
        collectionB
    );
    if (!CollectionALatestFile || !CollectionBLatestFile) {
        return 0;
    } else {
        const sortedFiles = sortFiles([
            CollectionALatestFile,
            CollectionBLatestFile,
        ]);
        if (sortedFiles[0].id !== CollectionALatestFile.id) {
            return 1;
        } else {
            return -1;
        }
    }
}

function getCollectionLatestFile(
    collectionAndTheirLatestFile: CollectionAndItsLatestFile[],
    collection: Collection
) {
    const collectionAndItsLatestFile = collectionAndTheirLatestFile.filter(
        (collectionAndItsLatestFile) =>
            collectionAndItsLatestFile.collection.id === collection.id
    );
    if (collectionAndItsLatestFile.length === 1) {
        return collectionAndItsLatestFile[0].file;
    }
}

function moveFavCollectionToFront(collections: Collection[]) {
    return collections.sort((collectionA, collectionB) =>
        collectionA.type === CollectionType.favorites
            ? -1
            : collectionB.type === CollectionType.favorites
            ? 1
            : 0
    );
}
