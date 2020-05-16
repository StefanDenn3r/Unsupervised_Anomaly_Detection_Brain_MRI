import numpy as np


def crop_center(img, cropx, cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    if len(img.shape) > 2:
        return img[starty:starty + cropy, startx:startx + cropx, :]
    else:
        return img[starty:starty + cropy, startx:startx + cropx]


def crop(img, y, x, height, width):
    return img[y:y + height, x:x + width]


# TP are green
# FP are orange
# FN are red
def augment_prediction_and_groundtruth_to_image(image, p, g):
    if image.ndim < 3:
        image = np.expand_dims(image, 2)
    tmp = np.repeat(image, 3, 2)
    p = np.squeeze(p.astype(bool))
    g = np.squeeze(g.astype(bool))

    tp_map = np.zeros(tmp.shape)
    tp_channel = np.multiply(p, g)
    tp_map[:, :, 1] = tp_channel
    fp_map = np.zeros(tmp.shape)
    fp_channel = np.multiply(p, np.invert(g))
    fp_map[:, :, 0] = fp_channel
    fp_map[:, :, 1] = 0.5 * fp_channel
    fn_map = np.zeros(tmp.shape)
    fn_channel = np.multiply(np.invert(p), g)
    fn_map[:, :, 0] = fn_channel
    map = tp_map + fp_map + fn_map
    mask = np.repeat(np.expand_dims(np.logical_or(np.logical_or(tp_channel, fp_channel), fn_channel), axis=2), 3, 2)

    tmp[tmp < 0] = 0
    tmp[mask] = map[mask]

    return tmp
