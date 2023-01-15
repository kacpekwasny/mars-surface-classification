
import cv2

from glob import glob

import os.path as op

def cut_into_tiles(img: cv2.Mat, side_len: int) -> list:
    imgW, imgH = img.shape[:2]
    assert imgW == imgH, "Image has to be a square"

    assert imgW % side_len == 0, f"{imgW=} has to be a multiple of {side_len=}"

    pairs = [] # output list of tiles img

    for x in range(imgW // side_len):
        for y in range(imgH // side_len):
            pairs.append(img[x*side_len:(x+1)*side_len, y*side_len:(y+1)*side_len])

    return pairs


def cut_imgs_from_dir(in_dir: str, out_dir: str, side_len: int):
    for idx, p in enumerate(glob(op.join(in_dir, "*"))):
        print(idx, p)
        img = cv2.imread(p)
        tiles = cut_into_tiles(img, side_len)
        for i, tile in enumerate(tiles):
            p_out = op.join(out_dir, op.splitext(op.basename(p))[0] + f"_{i}.PNG")
            cv2.imwrite(p_out, cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))

if __name__ == "__main__":
    from sys import argv
    import os

    os.makedirs(argv[2], exist_ok=True)
    cut_imgs_from_dir(argv[1], argv[2], int(argv[3]))





