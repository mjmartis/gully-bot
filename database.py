# Produce an index of image feature vectors for frames
# of the given images.
#  Usage: python3 database.py <output file> <input files...>

import datetime
import os
import pathlib
import pickle
import sys

from gully_types import EmbeddedFrame

import cv2
from progress.bar import Bar
import tensorflow as tf
import tensorflow_hub as hub

# Embedding model metadata.
MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
MODEL_INPUT_DIMS = (224, 224)

SAMPLE_HZ = 2

# Parse date and video title from full path, with stem in the
# in the format:
#  YYYY-MM-DD NAME
# Returns (name, date) tuple.
def parse_video_path(path):
    path_stem = pathlib.Path(path).stem
    sep_index = path_stem.find(' ')
    date = datetime.datetime.strptime(path_stem[:sep_index], '%Y-%m-%d')
    return (date, path_stem[sep_index+1:])

def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <output file> <input files...>')
        exit(1)

    # Load embedding model.
    embed_image = hub.KerasLayer(MODEL_DIR)

    # Open output index.
    output = open(sys.argv[1], 'ab')

    # Process each video.
    for i, video_path in enumerate(sys.argv[2:]):
        # Extract metadata.
        date, title = parse_video_path(video_path)

        # Read video.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_stride = int(fps / SAMPLE_HZ)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Display progress bar.
        progress_text = f'[{i+1:03}/{len(sys.argv)-2}] {title}'
        bar = Bar(f'{progress_text:30.30}', max=frame_count//frame_stride+1)

        # Process frames for this video.
        for frame_index in range(0, frame_count, frame_stride):
            # Jump to and read frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f'Failed to read "{title}" frame {frame_index}.')
                continue

            # Massage frame into format required by TF. 
            frame = cv2.resize(frame, MODEL_INPUT_DIMS)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_input = tf.convert_to_tensor(frame, dtype=tf.float32)
            frame_input = tf.expand_dims(frame_input , 0)

            # Embed image.
            embedded = embed_image(frame_input).numpy()[0]

            # Write binary blob.
            record = EmbeddedFrame(title=title,
                                   date=date,
                                   length=datetime.timedelta(seconds=frame_count/fps),
                                   timestamp=datetime.timedelta(seconds=frame_index/fps),
                                   features=embedded)
            pickle.dump(record, output)

            bar.next()

        bar.finish()

    output.close()

if __name__ == "__main__":
    main()
