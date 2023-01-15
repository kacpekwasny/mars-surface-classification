# author: jarek7410
# email: jrk@student.agh.edu.pl

import os
import re
import sys
import numpy as np
from PIL import Image
from itertools import product


def tile(filename, dir_in, dir_out, size_of_cuts):
    tileD(os.path.join(dir_in, filename), dir_out, size_of_cuts)


def tileD(filename, dir_in, dir_out, size_of_cuts):
    name, ext = os.path.splitext(filename)
    name = name[len(dir_in)::]
    img = Image.open(filename)
    w, h = img.size

    grid = product(range(0, h - h % size_of_cuts[0], size_of_cuts[0]),
                   range(0, w - w % size_of_cuts[1], size_of_cuts[1]))
    for i, j in grid:
        box = (j, i, j + size_of_cuts[0], i + size_of_cuts[1])
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        im = img.crop(box)
        path,notaname=os.path.split(out)
        try:
            os.mkdir(path)
        except: pass
        im.save(out)


# def potnij_zdjecie(zdjecie, maska, (wyjscie_x, wyjscie_y))
# -> list[para[zdjęcie, maska]]:
def potnij_zdjecie(zdjecie, maska, size_of_cuts):
    # grabing size of image and cheking if size is correct
    w1, h1 = zdjecie.size
    w2, h2 = maska.size
    if w1 != w2 or h1 != h2:
        raise Exception("size of mask and image are not compatible")

    # creating list for output
    out = list()

    # creating variable for storing grid
    grid = product(range(0, h1 - h1 % size_of_cuts[0], size_of_cuts[0]),
                   range(0, w1 - w1 % size_of_cuts[1], size_of_cuts[1]))
    # magic
    for i, j in grid:
        box = (j, i, j + size_of_cuts[0], i + size_of_cuts[1])
        zd = zdjecie.crop(box)
        ma = maska.crop(box)
        # chaging PIL.image to numpy.array, adding two pictures togther (as in requast)
        para = np.array(zd), np.array(ma)
        out.append(para)
    return out


if __name__ == "__main__":
    output = ".\\out\\"
    imput = ".\\in\\"
    projectdatabase = False
    arg = sys.argv
    if len(arg) < 4:
        print("""use:
              %s [-io] <sizeX> <sizeY> [filesName1 fileName2 ...]
              -i <dir> - folder with input images
              -o <dir> - folder for output tiles
              -d - all of files in -i <dir>""" % (arg[0].split("\\")[-1]))
        exit(1)

    flagGuard = 1
    while arg[flagGuard][0] == '-':
        ar = arg[flagGuard]
        flagGuard += 1
        if ar == '-o':
            output = arg[flagGuard] + "//"
            flagGuard += 1
        if ar == "-i":
            imput = arg[flagGuard] + "//"
            flagGuard += 1
        if ar == "-d":
            projectdatabase = True

    size = int(arg[flagGuard]), int(arg[flagGuard + 1])
    if projectdatabase:
        files = list()
        for (path, subdir, file) in os.walk(imput):
            for f in file:
                if '.JPG' in f:
                    files.append(os.path.join(path, f))
                if '.png' in f:
                    files.append(os.path.join(path, f))
        print(files)

        for i in files:
            print(i)
            try:
                os.mkdir(output)
            except:
                pass
            tileD(i,imput, output, size)
        exit(0)

    for i in arg[flagGuard + 2::]:
        print(i)
        try:
            os.mkdir(output + i)
        except:
            pass
        tile(i, imput, output + i, size)