import {
    addToCollection,
    Collection,
    CollectionType,
    createCollection,
} from 'services/collectionService';
import { getSelectedFiles } from 'utils/file';
import { File } from 'services/fileService';
import { getData, LS_KEYS } from 'utils/storage/localStorage';
import { User } from 'services/userService';

export async function addFilesToCollection(
    setCollectionSelectorView: (value: boolean) => void,
    selected: any,
    files: File[],
    clearSelection: () => void,
    syncWithRemote: () => Promise<void>,
    setActiveCollection: (id: number) => void,
    collectionName: string,
    existingCollection: Collection
) {
    setCollectionSelectorView(false);
    let collection: Collection;
    if (!existingCollection) {
        collection = await createCollection(
            collectionName,
            CollectionType.album
        );
    } else {
        collection = existingCollection;
    }
    const selectedFiles = getSelectedFiles(selected, files);
    await addToCollection(collection, selectedFiles);
    clearSelection();
    await syncWithRemote();
    setActiveCollection(collection.id);
}

export function getSelectedCollection(collectionID: number, collections) {
    return collections.find((collection) => collection.id === collectionID);
}

export function addIsSharedProperty(collections: Collection[]) {
    const user: User = getData(LS_KEYS.USER);
    for (const collection of collections) {
        if (user.id === collection.owner.id) {
            collection.iSharedCollection = false;
        } else {
            collection.iSharedCollection = true;
        }
    }
    return collections;
}
