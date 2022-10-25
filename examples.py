# Produce a database of training examples from a list of videos.
#  Usage: python3 examples.py <output prefix> <input 1> <input 2> ...

import datetime
import random
import sys

from common import EmbeddedFrame, embed_video_frames, parse_video_path, progress_bar, RECORD_COMPRESSION

import tensorflow as tf
import numpy as np

EXAMPLES_PER_FRAME = 5
SHARDS_COUNT = 10
SHARDS_DIGITS = len(str(SHARDS_COUNT))
WRITER_OPTS = 'ZLIB'


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
def generate_examples(video_paths, output_prefix):
    # Create a writer for each output shard.
    out_paths = [
        '{p}_{i:0{w}}-{c}.tfrec'.format(p=output_prefix,
                                        i=i,
                                        w=SHARDS_DIGITS,
                                        c=SHARDS_COUNT)
        for i in range(1, SHARDS_COUNT + 1)
    ]
    writers = [tf.io.TFRecordWriter(p, RECORD_COMPRESSION) for p in out_paths]

    # Frames from the last video.
    prev = None

    # Write triplets to disk as we generate them.
    example_index = 0
    for i, video_path in enumerate(video_paths):
        _, title = parse_video_path(video_path)

        # Display progress bar.
        bar = progress_bar(f'[{i+1:03}/{len(video_paths)}] {title}')

        # Store all feature vectors for this video.
        cur = []
        process = lambda r: cur.append(r.features)
        embed_video_frames(video_path, process, bar)

        bar.finish()

        # For the first iteration, just store the previous video.
        if prev:
            for trip in choose_triplets(np.array(cur), np.array(prev)):
                # Cycle through shards per-example.
                writers[example_index % SHARDS_COUNT].write(
                    create_example(*trip).SerializeToString())
                example_index += 1

        prev = cur

    for w in writers:
        w.close()

    with open(f'{output_prefix}_count.txt', 'w') as count_file:
        count_file.write(f'{example_index}\n')


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <output prefix> <input 1> <input 2> ...')
        exit(1)

    generate_examples(sys.argv[2:], sys.argv[1])


if __name__ == "__main__":
    main()
