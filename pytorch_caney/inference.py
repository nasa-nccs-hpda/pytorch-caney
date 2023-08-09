import logging
import math
import numpy as np

import torch

from tiler import Tiler, Merger

from pytorch_caney.processing import normalize
from pytorch_caney.processing import global_standardization
from pytorch_caney.processing import local_standardization
from pytorch_caney.processing import standardize_image

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module inference
#
# Data segmentation and prediction functions.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------
def sliding_window_tiler_multiclass(
            xraster,
            model,
            n_classes: int,
            img_size: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            mean=None,
            std=None,
            normalize: float = 1.0,
            rescale: str = None,
            window: str = 'triang',  # 'overlap-tile'
            probability_map: bool = False
        ):
    """
    Sliding window using tiler.
    """

    tile_channels = xraster.shape[-1]  # model.layers[0].input_shape[0][-1]
    print(f'Standardizing: {standardization}')
    # n_classes = out of the output layer, output_shape

    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(img_size, img_size, tile_channels),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(img_size, img_size, n_classes),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    merger = Merger(tiler=tiler_mask, window=window)
    # xraster = normalize_image(xraster, normalize)

    # Iterate over the data in batches
    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):

        # Standardize
        batch = batch_i.copy()

        if standardization is not None:
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :], standardization, mean, std)

        input_batch = batch.astype('float32')
        input_batch_tensor = torch.from_numpy(input_batch)
        input_batch_tensor = input_batch_tensor.transpose(-1, 1)
        # input_batch_tensor = input_batch_tensor.cuda(non_blocking=True)
        with torch.no_grad():
            y_batch = model(input_batch_tensor)
        y_batch = y_batch.transpose(1, -1)  # .cpu().numpy()
        merger.add_batch(batch_id, batch_size, y_batch)

    prediction = merger.merge(unpad=True)

    if not probability_map:
        if prediction.shape[-1] > 1:
            prediction = np.argmax(prediction, axis=-1)
        else:
            prediction = np.squeeze(
                np.where(prediction > threshold, 1, 0).astype(np.int16)
            )
    else:
        prediction = np.squeeze(prediction)
    return prediction


# --------------------------- Segmentation Functions ----------------------- #

def segment(image, model='model.h5', tile_size=256, channels=6,
            norm_data=[], bsize=8):
    """
    Applies a semantic segmentation model to an image. Ideal for non-scene
    imagery. Leaves artifacts in boundaries if no post-processing is done.
    :param image: image to classify (numpy array)
    :param model: loaded model object
    :param tile_size: tile size of patches
    :param channels: number of channels
    :param norm_data: numpy array with mean and std data
    :param bsize: number of patches to predict at the same time
    return numpy array with classified mask
    """
    # Create blank array to store predicted label
    seg = np.zeros((image.shape[0], image.shape[1]))
    for i in range(0, image.shape[0], int(tile_size)):
        for j in range(0, image.shape[1], int(tile_size)):
            # If edge of tile beyond image boundary, shift it to boundary
            if i + tile_size > image.shape[0]:
                i = image.shape[0] - tile_size
            if j + tile_size > image.shape[1]:
                j = image.shape[1] - tile_size

            # Extract and normalise tile
            tile = normalize(
                image[i: i + tile_size, j: j + tile_size, :].astype(float),
                norm_data
                )
            out = model.predict(
                    tile.reshape(
                        (1, tile.shape[0], tile.shape[1], tile.shape[2])
                        ).astype(float),
                    batch_size=4
                    )
            out = out.argmax(axis=3)  # get max prediction for pixel in classes
            out = out.reshape(tile_size, tile_size)  # reshape to tile size
            seg[i: i + tile_size, j: j + tile_size] = out
    return seg


def segment_binary(image, model='model.h5', norm_data=[],
                   tile_size=256, channels=6, bsize=8
                   ):
    """
    Applies binary semantic segmentation model to an image. Ideal for non-scene
    imagery. Leaves artifacts in boundaries if no post-processing is done.
    :param image: image to classify (numpy array)
    :param model: loaded model object
    :param tile_size: tile size of patches
    :param channels: number of channels
    :param norm_data: numpy array with mean and std data
    return numpy array with classified mask
    """
    # Create blank array to store predicted label
    seg = np.zeros((image.shape[0], image.shape[1]))
    for i in range(0, image.shape[0], int(tile_size)):
        for j in range(0, image.shape[1], int(tile_size)):
            # If edge of tile beyond image boundary, shift it to boundary
            if i + tile_size > image.shape[0]:
                i = image.shape[0] - tile_size
            if j + tile_size > image.shape[1]:
                j = image.shape[1] - tile_size

            # Extract and normalise tile
            tile = normalize(
                image[i:i + tile_size, j:j + tile_size, :].astype(float),
                norm_data
                )
            out = model.predict(
                    tile.reshape(
                        (1, tile.shape[0], tile.shape[1], tile.shape[2])
                        ).astype(float),
                    batch_size=bsize
                    )
            out[out >= 0.5] = 1
            out[out < 0.5] = 0
            out = out.reshape(tile_size, tile_size)  # reshape to tile size
            seg[i:i + tile_size, j:j + tile_size] = out
    return seg


def pad_image(img, target_size):
    """
    Pad an image up to the target size.
    """
    rows_missing = target_size - img.shape[0]
    cols_missing = target_size - img.shape[1]
    padded_img = np.pad(
        img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant'
        )
    return padded_img


def predict_sliding(image, model='', stand_method='local',
                    stand_strategy='per-batch', stand_data=[],
                    tile_size=256, nclasses=6, overlap=0.25, spline=[]
                    ):
    """
    Predict on tiles of exactly the network input shape.
    This way nothing gets squeezed.
    """
    model.eval()
    stride = math.ceil(tile_size * (1 - overlap))
    tile_rows = max(
        int(math.ceil((image.shape[0] - tile_size) / stride) + 1), 1
        )  # strided convolution formula
    tile_cols = max(
        int(math.ceil((image.shape[1] - tile_size) / stride) + 1), 1
        )
    logging.info("Need %i x %i prediction tiles @ stride %i px" %
                 (tile_cols, tile_rows, stride)
                 )

    full_probs = np.zeros((image.shape[0], image.shape[1], nclasses))
    count_predictions = np.zeros((image.shape[0], image.shape[1], nclasses))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size, image.shape[1])
            y2 = min(y1 + tile_size, image.shape[0])
            x1 = max(int(x2 - tile_size), 0)
            y1 = max(int(y2 - tile_size), 0)

            img = image[y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1

            padded_img = np.expand_dims(padded_img, 0)

            if stand_method == 'local':
                imgn = local_standardization(
                    padded_img, ndata=stand_data, strategy=stand_strategy
                    )
            elif stand_method == 'global':
                imgn = global_standardization(
                    padded_img, strategy=stand_strategy
                    )
            else:
                imgn = padded_img

            imgn = imgn.astype('float32')
            imgn_tensor = torch.from_numpy(imgn)
            imgn_tensor = imgn_tensor.transpose(-1, 1)
            with torch.no_grad():
                padded_prediction = model(imgn_tensor)
            # if padded_prediction.shape[1] > 1:
            #     padded_prediction = np.argmax(padded_prediction, axis=1)
            padded_prediction = np.squeeze(padded_prediction)
            padded_prediction = padded_prediction.transpose(0, -1).numpy()
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # * spline
    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def predict_sliding_binary(image, model='model.h5', tile_size=256,
                           nclasses=6, overlap=1/3, norm_data=[]
                           ):
    """
    Predict on tiles of exactly the network input shape.
    This way nothing gets squeezed.
    """
    stride = math.ceil(tile_size * (1 - overlap))
    tile_rows = max(
        int(math.ceil((image.shape[0] - tile_size) / stride) + 1), 1
        )  # strided convolution formula
    tile_cols = max(
        int(math.ceil((image.shape[1] - tile_size) / stride) + 1), 1
        )
    logging.info("Need %i x %i prediction tiles @ stride %i px" %
                 (tile_cols, tile_rows, stride)
                 )
    full_probs = np.zeros((image.shape[0], image.shape[1], nclasses))
    count_predictions = np.zeros((image.shape[0], image.shape[1], nclasses))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size, image.shape[1])
            y2 = min(y1 + tile_size, image.shape[0])
            x1 = max(int(x2 - tile_size), 0)
            y1 = max(int(y2 - tile_size), 0)

            img = image[y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1

            imgn = normalize(padded_img, norm_data)
            imgn = imgn.astype('float32')
            padded_prediction = model.predict(np.expand_dims(imgn, 0))[0]
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction
    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    full_probs[full_probs >= 0.8] = 1
    full_probs[full_probs < 0.8] = 0
    return full_probs.reshape((image.shape[0], image.shape[1]))


def predict_windowing(x, model, stand_method='local',
                      stand_strategy='per-batch', stand_data=[],
                      patch_sz=160, n_classes=5, b_size=128, spline=[]
                      ):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(
        shape=(extended_height, extended_width, n_channels), dtype=np.float32
        )
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    patches_array = np.asarray(patches_list)

    # normalization(patches_array, ndata)

    if stand_method == 'local':  # apply local zero center standardization
        patches_array = local_standardization(
            patches_array, ndata=stand_data, strategy=stand_strategy
            )
    elif stand_method == 'global':  # apply global zero center standardization
        patches_array = global_standardization(
            patches_array, strategy=stand_strategy
            )

    # predictions:
    patches_predict = model.predict(patches_array, batch_size=b_size)
    prediction = np.zeros(
        shape=(extended_height, extended_width, n_classes), dtype=np.float32
        )
    logging.info("prediction shape: ", prediction.shape)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_horizontal
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :] * spline
    return prediction[:img_height, :img_width, :]


# -------------------------------------------------------------------------------
# module model Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Add unit tests here
