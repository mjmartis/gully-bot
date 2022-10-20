# Finds the closest image in the given index.
#   Usage: query.py <index file> <image file>

import os
import pickle
import sys
from datetime import datetime

from common import EmbeddedFrame, find_nearest_frame, format_datapoint, init_nn_db


def main():
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} <metadata db> <nn db> <image file>')
        exit(1)

    with open(sys.argv[1], 'rb') as md_f:
        mds = pickle.load(md_f)
    nn_db = init_nn_db()
    nn_db.load(sys.argv[2])
    image = open(sys.argv[3], 'rb')

    result = find_nearest_frame(mds, nn_db, image)

    image.close()
    nn_db.unload()

    if result:
        c, s = result
        print(format_datapoint(c))
        print(f'Confidence: {s}')
    else:
        print('Not found.')


if __name__ == "__main__":
    main()
