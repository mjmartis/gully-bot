# Produce an index of image feature vectors for frames
# of the given images.
#  Usage: python3 database.py <output md> <output nn> <input files...>

import datetime
import logging
import os
import pathlib
import pickle
import sys

from common import embed_image, EmbeddedFrame, init_nn_db

from annoy import AnnoyIndex
import cv2
import numpy as np
from PIL import Image
from progress.bar import Bar

SAMPLE_HZ = 4.0
TREES_COUNT = 30


# Parse date and video title from full path, with stem in the
# in the format:
#  YYYY-MM-DD NAME
# Returns (name, date) tuple.
def parse_video_path(path):
    path_stem = pathlib.Path(path).stem
    sep_index = path_stem.find(' ')
    date = datetime.datetime.strptime(path_stem[:sep_index], '%Y-%m-%d')
    return (date, path_stem[sep_index + 1:])


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <output md> <output nn> <input files...>')
        exit(1)

    # Open NN index on disk.
    nn_db = init_nn_db()
    nn_db.on_disk_build(sys.argv[2])

    # Process each video.
    mds = []
    nn_db_i = 0
    for i, video_path in enumerate(sys.argv[3:]):
        # Extract metadata.
        date, title = parse_video_path(video_path)

        # Read video.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0:
            print(f'Error can\'t read FPS for "{title}".')
            continue

        frame_stride = fps / SAMPLE_HZ
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Display progress bar.
        progress_text = f'[{i+1:03}/{len(sys.argv)-3}] {title}'
        bar = Bar(f'{progress_text:30.30}',
                  max=int(frame_count / frame_stride),
                  suffix='%(percent)d%%')

        # Process frames for this video.
        for frame_index in np.arange(0.0, frame_count, frame_stride):
            bar.next()

            # Jump to and read frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret:
                print(f'Failed to read "{title}" frame {frame_index}.')
                continue

            # Massage frame into format required by TF.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedded = embed_image(Image.fromarray(frame))
            if embedded is None:
                continue

            # Append metadata to metadata database.
            metadata = EmbeddedFrame(
                title=title,
                date=date,
                length=datetime.timedelta(seconds=frame_count / fps),
                timestamp=datetime.timedelta(seconds=frame_index / fps),
                features=None)
            mds.append(metadata)

            # Add features to NN database.
            nn_db.add_item(nn_db_i, embedded)
            nn_db_i += 1

        bar.finish()

    # Write metadata.
    with open(sys.argv[1]) as md_db:
        pickle.dump(mds, md_db)

    # Construct actual NN truee.
    print('Building NN index... ', end='', flush=True)
    nn_db.build(TREES_COUNT)
    print('done.')


if __name__ == "__main__":
    main()
