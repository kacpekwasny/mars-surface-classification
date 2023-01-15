"""
images folder in `ai4mars` dataset contains test and train images.
For the purpose of using that dataset with this UNet, we need to have those test and train images separate.
"""

from glob import glob
from shutil import move
import os.path as op


def basenames(root):
    train = glob(op.join(root, "labels", "train_gif", "N*"))
    test  = glob(op.join(root, "labels", "test",  "masked-gold-min3-100agree", "N*"))
    imgs  = glob(op.join(root, "images", "edr",   "N*"))

    # bn stand for basename
    test_bn  = [ op.basename(fp).split(".")[0][:-7] for fp in test  ]
    train_bn = [ op.basename(fp).split(".")[0][:-5] for fp in train ]

    imgs_bn  = [ op.basename(fp).split(".")[0] for fp in imgs ]

    return test_bn, train_bn, imgs_bn


if __name__ == "__main__":
    import sys
    root = sys.argv[1]

    test, train, imgs = basenames(root)
    not_in_train = []
    for i in imgs:
        if not (i in train):
            not_in_train.append(i)
    
    import os
    ln = len(not_in_train)
    print(f"{len(imgs)=}, {len(not_in_train)=}")
    
    images_out_dir = "images_not_train"
    os.makedirs(op.join(root, images_out_dir), exist_ok=True)
    for idx, i in enumerate(not_in_train, start=1):
        print(idx, "/", ln, sep="")
        bname = f"{i}.JPG"
        os.rename(op.join(root, "images", "edr", bname),
                  op.join(root, images_out_dir,  bname))


    


