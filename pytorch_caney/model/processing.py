import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal
from numpy import fliplr, flipud
# import xarray as xr

SEED = 42
np.random.seed(SEED)

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ----------------------------------------------------------------------------
# module processing
#
# General functions to perform standardization of images (numpy arrays).
# A couple of methods have been implemented for testing, including global and
# local standardization for neural networks input. Data manipulation stage,
# extract random patches for training and store them in numpy arrays.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# --------------------------- Normalization Functions ----------------------- #

def normalize(images, factor=65535.0) -> np.array:
    """
    Normalize numpy array in the range of [0,1]
    :param images: numpy array in the format (n,w,h,c).
    :param factor: float number to normalize images, e.g. 2^(16)-1
    :return: numpy array in the [0,1] range
    """
    return images / factor


# ------------------------ Standardization Functions ----------------------- #

def globalStandardization(images, strategy='per-batch') -> np.array:
    """
    Standardize numpy array using global standardization.
    :param images: numpy array in the format (n,w,h,c).
    :param strategy: can select between per-image or per-batch.
    :return: globally standardized numpy array
    """
    if strategy == 'per-batch':
        mean = np.mean(images)  # global mean of all images
        std = np.std(images)  # global std of all images
        for i in range(images.shape[0]):  # for each image in images
            images[i, :, :, :] = (images[i, :, :, :] - mean) / std
    elif strategy == 'per-image':
        for i in range(images.shape[0]):  # for each image in images
            mean = np.mean(images[i, :, :, :])  # image mean
            std = np.std(images[i, :, :, :])  # image std
            images[i, :, :, :] = (images[i, :, :, :] - mean) / std
    return images


def localStandardization(images, filename='normalization_data',
                         ndata=pd.DataFrame(), strategy='per-batch'
                         ) -> np.array:
    """
    Standardize numpy array using local standardization.
    :param images: numpy array in the format (n,w,h,c).
    :param filename: filename to store mean and std data.
    :param ndata: pandas df with mean and std values for each channel.
    :param strategy: can select between per-image or per-batch.
    :return: locally standardized numpy array
    """
    if not ndata.empty:  # for inference only
        for i in range(images.shape[-1]):  # for each channel in images
            # standardize all images based on given mean and std
            images[:, :, :, i] = \
                (images[:, :, :, i] - ndata['channel_mean'][i]) / \
                ndata['channel_std'][i]
        return images
    elif strategy == 'per-batch':  # for all images in batch
        f = open(filename + "_norm_data.csv", "w+")
        f.write(
            "i,channel_mean,channel_std,channel_mean_post,channel_std_post\n"
            )
        for i in range(images.shape[-1]):  # for each channel in images
            channel_mean = np.mean(images[:, :, :, i])  # mean for each channel
            channel_std = np.std(images[:, :, :, i])   # std for each channel
            images[:, :, :, i] = \
                (images[:, :, :, i] - channel_mean) / channel_std
            channel_mean_post = np.mean(images[:, :, :, i])
            channel_std_post = np.std(images[:, :, :, i])
            # write to file for each channel
            f.write('{},{},{},{},{}\n'.format(i, channel_mean, channel_std,
                                              channel_mean_post,
                                              channel_std_post
                                              )
                    )
        f.close()  # close file
    elif strategy == 'per-image':  # standardization for each image
        for i in range(images.shape[0]):  # for each image
            for j in range(images.shape[-1]):  # for each channel in images
                channel_mean = np.mean(images[i, :, :, j])
                channel_std = np.std(images[i, :, :, j])
                images[i, :, :, j] = \
                    (images[i, :, :, j] - channel_mean) / channel_std
    else:
        raise RuntimeError(f'Standardization <{strategy}> not supported')

    return images


# ------------------------ Data Preparation Functions ----------------------- #

def get_rand_patches_rand_cond(img, mask, n_patches=16000, sz=160, nclasses=6,
                               nodata_ascloud=True, method='rand'
                               ) -> np.array:
    """
    Generate training data.
    :param images: ndarray in the format (w,h,c).
    :param mask: integer ndarray with shape (x_sz, y_sz)
    :param n_patches: number of patches
    :param sz: tile size, will be used for both height and width
    :param nclasses: number of classes present in the output data
    :param nodata_ascloud: convert no-data values to cloud labels
    :param method: choose between rand, cond, cloud
             rand - select N number of random patches for each image
             cond - select N number of random patches for each image,
                    with the condition of having 1+ class per tile.
             cloud - select tiles that have clouds
    :return: two numpy array with data and labels.
    """
    if nodata_ascloud:
        # if no-data present, change to final class
        mask = mask.values  # return numpy array
        mask[mask > nclasses] = nclasses  # some no-data are 255 or other big
        mask[mask < 0] = nclasses  # some no-data are -128 or smaller negative

    patches = []  # list to store data patches
    labels = []  # list to store label patches

    for i in tqdm(range(n_patches)):

        # Generate random integers from image
        xc = random.randint(0, img.shape[0] - sz)
        yc = random.randint(0, img.shape[1] - sz)

        if method == 'cond':
            # while loop to regenerate random ints if tile has only one class
            while len(np.unique(mask[xc:(xc+sz), yc:(yc+sz)])) == 1 or \
                    6 in mask[xc:(xc+sz), yc:(yc+sz)] or \
                    img[xc:(xc+sz), yc:(yc+sz), :].values.min() < 0:
                xc = random.randint(0, img.shape[0] - sz)
                yc = random.randint(0, img.shape[1] - sz)
        elif method == 'rand':
            while 6 in mask[xc:(xc+sz), yc:(yc+sz)] or \
                    img[xc:(xc+sz), yc:(yc+sz), :].values.min() < 0:
                xc = random.randint(0, img.shape[0] - sz)
                yc = random.randint(0, img.shape[1] - sz)
        elif method == 'cloud':
            while np.count_nonzero(mask[xc:(xc+sz), yc:(yc+sz)] == 6) < 15:
                xc = random.randint(0, img.shape[0] - sz)
                yc = random.randint(0, img.shape[1] - sz)

        # Generate img and mask patches
        patch_img = img[xc:(xc + sz), yc:(yc + sz)]
        patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

        # Apply some random transformations
        random_transformation = np.random.randint(1, 7)
        if random_transformation == 1:  # flip left and right
            patch_img = fliplr(patch_img)
            patch_mask = fliplr(patch_mask)
        elif random_transformation == 2:  # reverse second dimension
            patch_img = flipud(patch_img)
            patch_mask = flipud(patch_mask)
        elif random_transformation == 3:  # rotate 90 degrees
            patch_img = np.rot90(patch_img, 1)
            patch_mask = np.rot90(patch_mask, 1)
        elif random_transformation == 4:  # rotate 180 degrees
            patch_img = np.rot90(patch_img, 2)
            patch_mask = np.rot90(patch_mask, 2)
        elif random_transformation == 5:  # rotate 270 degrees
            patch_img = np.rot90(patch_img, 3)
            patch_mask = np.rot90(patch_mask, 3)
        else:  # original image
            pass
        patches.append(patch_img)
        labels.append(patch_mask)
    return np.asarray(patches), np.asarray(labels)


def get_rand_patches_aug_augcond(img, mask, n_patches=16000, sz=256,
                                 nclasses=6, over=50, nodata_ascloud=True,
                                 nodata=-9999, method='augcond'
                                 ) -> np.array:
    """
    Generate training data.
    :param images: ndarray in the format (w,h,c).
    :param mask: integer ndarray with shape (x_sz, y_sz)
    :param n_patches: number of patches
    :param sz: tile size, will be used for both height and width
    :param nclasses: number of classes present in the output data
    :param over: number of pixels to overlap between images
    :param nodata_ascloud: convert no-data values to cloud labels
    :param method: choose between rand, cond, cloud
            aug - select N * 8 number of random patches for each
                  image after data augmentation.
            augcond - select N * 8 number of random patches for
                  each image, with the condition of having 1+ per
                  tile, after data augmentation.
    :return: two numpy array with data and labels.
    """
    mask = mask.values  # return numpy array

    if nodata_ascloud:
        # if no-data present, change to final class
        mask[mask > nclasses] = nodata  # some no-data are 255 or other big
        mask[mask < 0] = nodata  # some no-data are -128 or smaller negative

    patches = []  # list to store data patches
    labels = []  # list to store label patches

    for i in tqdm(range(n_patches)):

        # Generate random integers from image
        xc = random.randint(0, img.shape[0] - sz - sz)
        yc = random.randint(0, img.shape[1] - sz - sz)

        if method == 'augcond':
            # while loop to regenerate random ints if tile has only one class
            while len(np.unique(mask[xc:(xc + sz), yc:(yc + sz)])) == 1 or \
                    nodata in mask[xc:(xc + sz), yc:(yc + sz)] or \
                    nodata in mask[(xc + sz - over):(xc + sz + sz - over),
                                   (yc + sz - over):(yc + sz + sz - over)] or \
                    nodata in mask[(xc + sz - over):(xc + sz + sz - over),
                                   yc:(yc + sz)]:
                xc = random.randint(0, img.shape[0] - sz - sz)
                yc = random.randint(0, img.shape[1] - sz - sz)
        elif method == 'aug':
            # while loop to regenerate random ints if tile has only one class
            while nodata in mask[xc:(xc + sz), yc:(yc + sz)] or \
                  nodata in mask[(xc + sz - over):(xc + sz + sz - over),
                                 (yc + sz - over):(yc + sz + sz - over)] or \
                  nodata in mask[(xc + sz - over):(xc + sz + sz - over),
                                 yc:(yc + sz)]:
                xc = random.randint(0, img.shape[0] - sz - sz)
                yc = random.randint(0, img.shape[1] - sz - sz)

        # Generate img and mask patches
        patch_img = img[xc:(xc + sz), yc:(yc + sz)]  # original image patch
        patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]  # original mask patch

        # Apply transformations for data augmentation
        # 1. No augmentation and append to list
        patches.append(patch_img)
        labels.append(patch_mask)

        # 2. Rotate 90 and append to list
        patches.append(np.rot90(patch_img, 1))
        labels.append(np.rot90(patch_mask, 1))

        # 3. Rotate 180 and append to list
        patches.append(np.rot90(patch_img, 2))
        labels.append(np.rot90(patch_mask, 2))

        # 4. Rotate 270
        patches.append(np.rot90(patch_img, 3))
        labels.append(np.rot90(patch_mask, 3))

        # 5. Flipped up and down’
        patches.append(flipud(patch_img))
        labels.append(flipud(patch_mask))

        # 6. Flipped left and right
        patches.append(fliplr(patch_img))
        labels.append(fliplr(patch_mask))

        # 7. overlapping tiles - next tile, down
        patches.append(img[(xc + sz - over):(xc + sz + sz - over),
                           (yc + sz - over):(yc + sz + sz - over)])
        labels.append(mask[(xc + sz - over):(xc + sz + sz - over),
                           (yc + sz - over):(yc + sz + sz - over)])

        # 8. overlapping tiles - next tile, side
        patches.append(img[(xc + sz - over):(xc + sz + sz - over),
                           yc:(yc + sz)])
        labels.append(mask[(xc + sz - over):(xc + sz + sz - over),
                           yc:(yc + sz)])
    return np.asarray(patches), np.asarray(labels)


# ------------------------ Artifact Removal Functions ----------------------- #

def _2d_spline(window_size=128, power=2) -> np.array:
    """
    Window method for boundaries/edge artifacts smoothing.
    :param window_size: size of window/tile to smooth
    :param power: spline polinomial power to use
    :return: smoothing distribution numpy array
    """
    intersection = int(window_size/4)
    tria = scipy.signal.triang(window_size)
    wind_outer = (abs(2*(tria)) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(tria - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    wind = wind * wind.transpose(1, 0, 2)
    return wind


def _hann_matrix(window_size=128, power=2) -> np.array:
    logging.info("Placeholder for next release.")


# -------------------------------------------------------------------------------
# module preprocessing Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Unit Test #1 - Testing normalization distributions
    x = (np.random.randint(65536, size=(10, 128, 128, 6))).astype('float32')
    x_norm = normalize(x, factor=65535)  # apply static normalization
    assert x_norm.max() == 1.0, "Unexpected max value."
    logging.info(f"UT #1 PASS: {x_norm.mean()}, {x_norm.std()}")

    # Unit Test #2 - Testing standardization distributions
    standardized = globalStandardization(x_norm, strategy='per-batch')
    assert standardized.max() > 1.731, "Unexpected max value."
    logging.info(f"UT #2 PASS: {standardized.mean()}, {standardized.std()}")

    # Unit Test #3 - Testing standardization distributions
    standardized = globalStandardization(x_norm, strategy='per-image')
    assert standardized.max() > 1.73, "Unexpected max value."
    logging.info(f"UT #3 PASS: {standardized.mean()}, {standardized.std()}")

    # Unit Test #4 - Testing standardization distributions
    standardized = localStandardization(x_norm, filename='normalization_data',
                                        strategy='per-batch'
                                        )
    assert standardized.max() > 1.74, "Unexpected max value."
    logging.info(f"UT #4 PASS: {standardized.mean()}, {standardized.std()}")

    # Unit Test #5 - Testing standardization distributions
    standardized = localStandardization(x_norm, filename='normalization_data',
                                        strategy='per-image'
                                        )
    assert standardized.max() > 1.75, "Unexpected max value."
    logging.info(f"UT #5 PASS: {standardized.mean()}, {standardized.std()}")
