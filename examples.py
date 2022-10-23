# Produce a database of training examples from a database of frames.
#  Usage: python3 examples.py <output> <input 1> <input 2> ...

import datetime
import random
import sys

from common import embed_image, EmbeddedFrame, parse_video_path

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from progress.bar import Bar
from scipy import spatial

BATCH_SIZE = 32
SAMPLE_HZ = 4.0
EXAMPLES_PER_FRAME = 5

# Accepts a "positive" array of embedded frames from one video and a "negative"
# array of embedded frames from a different video. Produces random training
# triplets of an "anchor", a "close" frame from the same video, and a "far"
# frame from the other video.
def choose_triplets(ps, ns):
    anchors = np.concatenate([ps for _ in range(EXAMPLES_PER_FRAME)])

    sample_count = ps.shape[0] * EXAMPLES_PER_FRAME
    positives = ps[np.random.choice(ps.shape[0], sample_count)]
    negatives = ns[np.random.choice(ns.shape[0], sample_count)]

    return zip(anchors, positives, negatives)


# Embeds a sample of video frames and runs the specified function on the result.
# TODO: merge with code in database.py.
def embed_video_frames(video_path, process_fn, bar):
    date, title = parse_video_path(video_path)

    # Read video.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        print(f'Error can\'t read FPS for "{title}".')
        return

    frame_stride = fps / SAMPLE_HZ
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process frames for this video.
    for frame_index in np.arange(0.0, frame_count, frame_stride):
        if bar:
            bar.next(frame_stride / frame_count)

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

        # Pass data to processer function.
        record = EmbeddedFrame(
            title=title,
            date=date,
            length=datetime.timedelta(seconds=frame_count / fps),
            timestamp=datetime.timedelta(seconds=frame_index / fps),
            features=embedded)
        process_fn(record)


# Converts a float list into a TF feature.
def float_list_feature(vs):
    return tf.train.Feature(float_list=tf.train.FloatList(value=vs))


# Converts an embedded frame triplet into a TF example.
def create_example(a, p, n):
    feature = {
        'a': float_list_feature(a),
        'p': float_list_feature(p),
        'n': float_list_feature(n),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# For each frame of each video, write a series of triplet examples to disk.
def generate_examples(video_paths, output_file):
    # Frames from the last video.
    prev = None

    # Write triplets to disk as we generate them.
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, video_path in enumerate(video_paths):
            _, title = parse_video_path(video_path)

            # Display progress bar.
            progress_text = f'[{i+1:03}/{len(video_paths)}] {title}'
            bar = Bar(f'{progress_text:30.30}', max=1.0, suffix='%(percent)d%%')

            # Store all feature vectors for this video.
            cur = []
            process = lambda r: cur.append(r.features)
            embed_video_frames(video_path, process, bar)

            bar.finish()

            # For the first iteration, just store the previous video.
            if prev:
                for trip in choose_triplets(np.array(cur), np.array(prev)):
                    writer.write(create_example(*trip).SerializeToString())

            prev = cur


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <output> <input 1> <input 2> ...')
        exit(1)

    generate_examples(sys.argv[2:], sys.argv[1])


if __name__ == "__main__":
    main()
