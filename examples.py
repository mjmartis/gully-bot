# Produce a database of training examples from a database of frames.
#  Usage: python3 examples.py <output> <input 1> <input 2> ...

import datetime
import random
import sys

from common import EmbeddedFrame, parse_video_path, embed_video_frames

import tensorflow as tf
import numpy as np
from progress.bar import Bar

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
            bar = Bar(f'{progress_text:40.40}', max=1.0, suffix='%(percent)d%%')

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
