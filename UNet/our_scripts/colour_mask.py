import cv2
import numpy as np


class SurfType:
    SOIL = [0, 0, 0]
    BEDROCK = [1, 1, 1]
    SAND = [2, 2, 2]
    BIG_ROCK = [3, 3, 3]


class Colours:
    BLUE = [0, 0, 255]
    YELLOW = [255, 255, 0]
    PINK = [255, 0, 255]
    RED = [255, 0, 0]


def swap_color(img, original_color, new_color):
    r1, g1, b1 = original_color
    r2, g2, b2 = new_color

    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    img[:, :, :3][mask] = [r2, g2, b2]

    return img


def paint_and_merge(photo_path, mask_path, output_path, alpha=0.4):
    photo = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

    mask = swap_color(mask, SurfType.SOIL, Colours.BLUE)
    mask = swap_color(mask, SurfType.BEDROCK, Colours.PINK)
    mask = swap_color(mask, SurfType.SAND, Colours.YELLOW)
    mask = swap_color(mask, SurfType.BIG_ROCK, Colours.RED)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    out_img = np.zeros(mask.shape, dtype=mask.dtype)
    out_img[:, :, :] = alpha * mask[:, :, :] + photo[:, :, :]

    cv2.imwrite(output_path, out_img)


if __name__ == '__main__':
    mask = r"C:\Users\quatr\IT\Code\mars-surface-classification\UNet\data\masks\NLB_441350252EDR_F0250000NCAM00250M1.png"
    paint_and_merge(
        'D:\Python\kolorowanie-obrazow\photo.jpg',
        'D:\Python\kolorowanie-obrazow\mask.png',
        'merged.png'
    )
