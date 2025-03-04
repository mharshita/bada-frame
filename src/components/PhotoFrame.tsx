import { GalleryContext } from 'pages/gallery';
import PreviewCard from './pages/gallery/PreviewCard';
import React, { useContext, useEffect, useRef, useState } from 'react';
import { Button } from 'react-bootstrap';
import { EnteFile } from 'types/file';
import styled from 'styled-components';
import DownloadManager from 'services/downloadManager';
import constants from 'utils/strings/constants';
import AutoSizer from 'react-virtualized-auto-sizer';
import PhotoSwipe from 'components/PhotoSwipe/PhotoSwipe';
import { isInsideBox, isSameDay as isSameDayAnyYear } from 'utils/search';
import { fileIsArchived, formatDateRelative } from 'utils/file';
import {
    ALL_SECTION,
    ARCHIVE_SECTION,
    TRASH_SECTION,
} from 'constants/collection';
import { isSharedFile } from 'utils/file';
import { isPlaybackPossible } from 'utils/photoFrame';
import { PhotoList } from './PhotoList';
import { SetFiles, SelectedState, Search, setSearchStats } from 'types/gallery';
import { FILE_TYPE } from 'constants/file';
import PublicCollectionDownloadManager from 'services/publicCollectionDownloadManager';
import { PublicCollectionGalleryContext } from 'utils/publicCollectionGallery';
import { useRouter } from 'next/router';

const Container = styled.div`
    display: block;
    flex: 1;
    width: 100%;
    flex-wrap: wrap;
    margin: 0 auto;
    overflow-x: hidden;
    .pswp-thumbnail {
        display: inline-block;
        cursor: pointer;
    }
`;

const EmptyScreen = styled.div`
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    flex: 1;
    color: #51cd7c;

    & > svg {
        filter: drop-shadow(3px 3px 5px rgba(45, 194, 98, 0.5));
    }
`;

const PHOTOSWIPE_HASH_SUFFIX = '&opened';

interface Props {
    files: EnteFile[];
    setFiles: SetFiles;
    syncWithRemote: () => Promise<void>;
    favItemIds: Set<number>;
    setSelected: (
        selected: SelectedState | ((selected: SelectedState) => SelectedState)
    ) => void;
    selected: SelectedState;
    isFirstLoad;
    openFileUploader;
    isInSearchMode: boolean;
    search: Search;
    setSearchStats: setSearchStats;
    deleted?: number[];
    activeCollection: number;
    isSharedCollection: boolean;
    enableDownload: boolean;
}

type SourceURL = {
    imageURL?: string;
    videoURL?: string;
};

const PhotoFrame = ({
    files,
    setFiles,
    syncWithRemote,
    favItemIds,
    setSelected,
    selected,
    isFirstLoad,
    openFileUploader,
    isInSearchMode,
    search,
    setSearchStats,
    deleted,
    activeCollection,
    isSharedCollection,
    enableDownload,
}: Props) => {
    const [open, setOpen] = useState(false);
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [fetching, setFetching] = useState<{ [k: number]: boolean }>({});
    const startTime = Date.now();
    const galleryContext = useContext(GalleryContext);
    const publicCollectionGalleryContext = useContext(
        PublicCollectionGalleryContext
    );
    const [rangeStart, setRangeStart] = useState(null);
    const [currentHover, setCurrentHover] = useState(null);
    const [isShiftKeyPressed, setIsShiftKeyPressed] = useState(false);
    const filteredDataRef = useRef([]);
    const filteredData = filteredDataRef?.current ?? [];
    const router = useRouter();
    const [isSourceLoaded, setIsSourceLoaded] = useState(false);
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Shift') {
                setIsShiftKeyPressed(true);
            }
        };
        const handleKeyUp = (e: KeyboardEvent) => {
            if (e.key === 'Shift') {
                setIsShiftKeyPressed(false);
            }
        };
        document.addEventListener('keydown', handleKeyDown, false);
        document.addEventListener('keyup', handleKeyUp, false);
        router.events.on('hashChangeComplete', (url: string) => {
            const start = url.indexOf('#');
            const hash = url.slice(start !== -1 ? start : url.length);
            const shouldPhotoSwipeBeOpened = hash.endsWith(
                PHOTOSWIPE_HASH_SUFFIX
            );
            if (shouldPhotoSwipeBeOpened) {
                setOpen(true);
            } else {
                setOpen(false);
            }
        });
        return () => {
            document.addEventListener('keydown', handleKeyDown, false);
            document.addEventListener('keyup', handleKeyUp, false);
        };
    }, []);

    useEffect(() => {
        if (isInSearchMode) {
            setSearchStats({
                resultCount: filteredData.length,
                timeTaken: (Date.now() - startTime) / 1000,
            });
        }
        if (search.fileIndex || search.fileIndex === 0) {
            const filteredDataIdx = filteredData.findIndex(
                (data) => data.dataIndex === search.fileIndex
            );
            if (filteredDataIdx || filteredDataIdx === 0) {
                onThumbnailClick(filteredDataIdx)();
            }
        }
    }, [search, filteredData]);

    const resetFetching = () => {
        setFetching({});
    };

    useEffect(() => {
        if (selected.count === 0) {
            setRangeStart(null);
        }
    }, [selected]);

    useEffect(() => {
        const idSet = new Set();
        filteredDataRef.current = files
            .map((item, index) => ({
                ...item,
                dataIndex: index,
                w: window.innerWidth,
                h: window.innerHeight,
                ...(item.deleteBy && {
                    title: constants.AUTOMATIC_BIN_DELETE_MESSAGE(
                        formatDateRelative(item.deleteBy / 1000)
                    ),
                }),
            }))
            .filter((item) => {
                if (deleted.includes(item.id)) {
                    return false;
                }
                if (
                    search.date &&
                    !isSameDayAnyYear(search.date)(
                        new Date(item.metadata.creationTime / 1000)
                    )
                ) {
                    return false;
                }
                if (
                    search.location &&
                    !isInsideBox(item.metadata, search.location)
                ) {
                    return false;
                }
                if (activeCollection === ALL_SECTION && fileIsArchived(item)) {
                    return false;
                }
                if (
                    activeCollection === ARCHIVE_SECTION &&
                    !fileIsArchived(item)
                ) {
                    return false;
                }

                if (isSharedFile(item) && !isSharedCollection) {
                    return false;
                }
                if (activeCollection === TRASH_SECTION && !item.isTrashed) {
                    return false;
                }
                if (activeCollection !== TRASH_SECTION && item.isTrashed) {
                    return false;
                }
                if (!idSet.has(item.id)) {
                    if (
                        activeCollection === ALL_SECTION ||
                        activeCollection === ARCHIVE_SECTION ||
                        activeCollection === TRASH_SECTION ||
                        activeCollection === item.collectionID
                    ) {
                        idSet.add(item.id);
                        return true;
                    }
                    return false;
                }
                return false;
            });
    }, [files, deleted, search, activeCollection]);

    useEffect(() => {
        const currentURL = new URL(window.location.href);
        const end = currentURL.hash.lastIndexOf('&');
        const hash = currentURL.hash.slice(1, end !== -1 ? end : undefined);
        if (open) {
            router.push({
                hash: hash + PHOTOSWIPE_HASH_SUFFIX,
            });
        } else {
            router.push({
                hash: hash,
            });
        }
    }, [open]);

    const updateURL = (index: number) => (url: string) => {
        const updateFile = (file: EnteFile) => {
            file = {
                ...file,
                msrc: url,
                w: window.innerWidth,
                h: window.innerHeight,
            };
            if (file.metadata.fileType === FILE_TYPE.VIDEO && !file.html) {
                file.html = `
                <div class="pswp-item-container">
                    <img src="${url}" onContextMenu="return false;"/>
                    <div class="spinner-border text-light" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                </div>
            `;
            } else if (
                file.metadata.fileType === FILE_TYPE.LIVE_PHOTO &&
                !file.html
            ) {
                file.html = `
                <div class="pswp-item-container">
                    <img src="${url}" onContextMenu="return false;"/>
                    <div class="spinner-border text-light" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                </div>
            `;
            } else if (
                file.metadata.fileType === FILE_TYPE.IMAGE &&
                !file.src
            ) {
                file.src = url;
            }
            return file;
        };
        setFiles((files) => {
            files[index] = updateFile(files[index]);
            return files;
        });
        return updateFile(files[index]);
    };

    const updateSrcURL = async (index: number, srcURL: SourceURL) => {
        const { videoURL, imageURL } = srcURL;
        const isPlayable = videoURL && (await isPlaybackPossible(videoURL));
        const updateFile = (file: EnteFile) => {
            file = {
                ...file,
                w: window.innerWidth,
                h: window.innerHeight,
            };
            if (file.metadata.fileType === FILE_TYPE.VIDEO) {
                if (isPlayable) {
                    file.html = `
            <video controls onContextMenu="return false;">
                <source src="${videoURL}" />
                Your browser does not support the video tag.
            </video>
        `;
                } else {
                    file.html = `
            <div class="pswp-item-container">
                <img src="${file.msrc}" onContextMenu="return false;"/>
                <div class="download-banner" >
                    ${constants.VIDEO_PLAYBACK_FAILED_DOWNLOAD_INSTEAD}
                    <a class="btn btn-outline-success" href=${videoURL} download="${file.metadata.title}"">Download</a>
                </div>
            </div>
            `;
                }
            } else if (file.metadata.fileType === FILE_TYPE.LIVE_PHOTO) {
                if (isPlayable) {
                    file.html = `
                <div class = 'pswp-item-container'>
                    <img id = "live-photo-image-${file.id}" src="${imageURL}" onContextMenu="return false;"/>
                    <video id = "live-photo-video-${file.id}" loop muted onContextMenu="return false;">
                        <source src="${videoURL}" />
                        Your browser does not support the video tag.
                    </video>
                </div>
                `;
                } else {
                    file.html = `
                <div class="pswp-item-container">
                    <img src="${file.msrc}" onContextMenu="return false;"/>
                    <div class="download-banner">
                        ${constants.VIDEO_PLAYBACK_FAILED_DOWNLOAD_INSTEAD}
                        <button class = "btn btn-outline-success" id = "download-btn-${file.id}">Download</button>
                    </div>
                </div>
                `;
                }
            } else {
                file.src = imageURL;
            }
            return file;
        };
        setFiles((files) => {
            files[index] = updateFile(files[index]);
            return files;
        });
        setIsSourceLoaded(true);
        return updateFile(files[index]);
    };

    const handleClose = (needUpdate) => {
        setOpen(false);
        needUpdate && syncWithRemote();
    };

    const onThumbnailClick = (index: number) => () => {
        setCurrentIndex(index);
        setOpen(true);
    };

    const handleSelect = (id: number, index?: number) => (checked: boolean) => {
        if (selected.collectionID !== activeCollection) {
            setSelected({ count: 0, collectionID: 0 });
        }
        if (typeof index !== 'undefined') {
            if (checked) {
                setRangeStart(index);
            } else {
                setRangeStart(undefined);
            }
        }

        setSelected((selected) => ({
            ...selected,
            [id]: checked,
            count:
                selected[id] === checked
                    ? selected.count
                    : checked
                    ? selected.count + 1
                    : selected.count - 1,
            collectionID: activeCollection,
        }));
    };
    const onHoverOver = (index: number) => () => {
        setCurrentHover(index);
    };

    const handleRangeSelect = (index: number) => () => {
        if (typeof rangeStart !== 'undefined' && rangeStart !== index) {
            const direction =
                (index - rangeStart) / Math.abs(index - rangeStart);
            let checked = true;
            for (
                let i = rangeStart;
                (index - i) * direction >= 0;
                i += direction
            ) {
                checked = checked && !!selected[filteredData[i].id];
            }
            for (
                let i = rangeStart;
                (index - i) * direction > 0;
                i += direction
            ) {
                handleSelect(filteredData[i].id)(!checked);
            }
            handleSelect(filteredData[index].id, index)(!checked);
        }
    };
    const getThumbnail = (files: EnteFile[], index: number) =>
        files[index] ? (
            <PreviewCard
                key={`tile-${files[index].id}-selected-${
                    selected[files[index].id] ?? false
                }`}
                file={files[index]}
                updateURL={updateURL(files[index].dataIndex)}
                onClick={onThumbnailClick(index)}
                selectable={!isSharedCollection}
                onSelect={handleSelect(files[index].id, index)}
                selected={
                    selected.collectionID === activeCollection &&
                    selected[files[index].id]
                }
                selectOnClick={selected.count > 0}
                onHover={onHoverOver(index)}
                onRangeSelect={handleRangeSelect(index)}
                isRangeSelectActive={isShiftKeyPressed && selected.count > 0}
                isInsSelectRange={
                    (index >= rangeStart && index <= currentHover) ||
                    (index >= currentHover && index <= rangeStart)
                }
            />
        ) : (
            <></>
        );

    const getSlideData = async (
        instance: any,
        index: number,
        item: EnteFile
    ) => {
        if (!item.msrc) {
            try {
                let url: string;
                if (galleryContext.thumbs.has(item.id)) {
                    url = galleryContext.thumbs.get(item.id);
                } else {
                    if (
                        publicCollectionGalleryContext.accessedThroughSharedURL
                    ) {
                        url =
                            await PublicCollectionDownloadManager.getThumbnail(
                                item,
                                publicCollectionGalleryContext.token,
                                publicCollectionGalleryContext.passwordToken
                            );
                    } else {
                        url = await DownloadManager.getThumbnail(item);
                    }
                    galleryContext.thumbs.set(item.id, url);
                }
                const newFile = updateURL(item.dataIndex)(url);
                item.msrc = newFile.msrc;
                item.html = newFile.html;
                item.src = newFile.src;
                item.w = newFile.w;
                item.h = newFile.h;

                try {
                    instance.invalidateCurrItems();
                    instance.updateSize(true);
                } catch (e) {
                    // ignore
                }
            } catch (e) {
                // no-op
            }
        }
        if (!fetching[item.dataIndex]) {
            try {
                fetching[item.dataIndex] = true;
                let urls: string[];
                if (galleryContext.files.has(item.id)) {
                    const mergedURL = galleryContext.files.get(item.id);
                    urls = mergedURL.split(',');
                } else {
                    galleryContext.startLoading();
                    if (
                        publicCollectionGalleryContext.accessedThroughSharedURL
                    ) {
                        urls = await PublicCollectionDownloadManager.getFile(
                            item,
                            publicCollectionGalleryContext.token,
                            publicCollectionGalleryContext.passwordToken,
                            true
                        );
                    } else {
                        urls = await DownloadManager.getFile(item, true);
                    }
                    galleryContext.finishLoading();
                    const mergedURL = urls.join(',');
                    galleryContext.files.set(item.id, mergedURL);
                }
                let imageURL;
                let videoURL;
                if (item.metadata.fileType === FILE_TYPE.LIVE_PHOTO) {
                    [imageURL, videoURL] = urls;
                } else if (item.metadata.fileType === FILE_TYPE.VIDEO) {
                    [videoURL] = urls;
                } else {
                    [imageURL] = urls;
                }
                setIsSourceLoaded(false);
                const newFile = await updateSrcURL(item.dataIndex, {
                    imageURL,
                    videoURL,
                });
                item.msrc = newFile.msrc;
                item.html = newFile.html;
                item.src = newFile.src;
                item.w = newFile.w;
                item.h = newFile.h;
                try {
                    instance.invalidateCurrItems();
                    instance.updateSize(true);
                } catch (e) {
                    // ignore
                }
            } catch (e) {
                // no-op
            } finally {
                fetching[item.dataIndex] = false;
            }
        }
    };

    return (
        <>
            {!isFirstLoad && files.length === 0 && !isInSearchMode ? (
                <EmptyScreen>
                    <img height={150} src="/images/gallery.png" />
                    <div style={{ color: '#a6a6a6', marginTop: '16px' }}>
                        {constants.UPLOAD_FIRST_PHOTO_DESCRIPTION}
                    </div>
                    <Button
                        variant="outline-success"
                        onClick={openFileUploader}
                        style={{
                            marginTop: '32px',
                            paddingLeft: '32px',
                            paddingRight: '32px',
                            paddingTop: '12px',
                            paddingBottom: '12px',
                            fontWeight: 900,
                        }}>
                        {constants.UPLOAD_FIRST_PHOTO}
                    </Button>
                </EmptyScreen>
            ) : (
                <Container>
                    <AutoSizer>
                        {({ height, width }) => (
                            <PhotoList
                                width={width}
                                height={height}
                                getThumbnail={getThumbnail}
                                filteredData={filteredData}
                                activeCollection={activeCollection}
                                showAppDownloadBanner={
                                    files.length < 30 && !isInSearchMode
                                }
                                resetFetching={resetFetching}
                            />
                        )}
                    </AutoSizer>
                    <PhotoSwipe
                        isOpen={open}
                        items={filteredData}
                        currentIndex={currentIndex}
                        onClose={handleClose}
                        gettingData={getSlideData}
                        favItemIds={favItemIds}
                        isSharedCollection={isSharedCollection}
                        isTrashCollection={activeCollection === TRASH_SECTION}
                        enableDownload={enableDownload}
                        isSourceLoaded={isSourceLoaded}
                    />
                </Container>
            )}
        </>
    );
};

export default PhotoFrame;
