# Finds the closest image in the given index.
#   Usage: query.py <index file> <image file>

import os
import sys

from common import EmbeddedFrame, find_nearest_frame, format_datapoint

from progress.bar import Bar


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <index file> <image file>')
        exit(1)

    db = open(sys.argv[1], 'rb')
    image = open(sys.argv[2], 'rb')
    bar = Bar('Searching',
              max=os.path.getsize(sys.argv[1]),
              suffix='%(percent)d%%')
    chosen = find_nearest_frame(db, image, bar)
    bar.finish()
    image.close()
    db.close()

    if chosen:
        print(format_datapoint(chosen))
    else:
        print('Not found.')


if __name__ == "__main__":
    main()
