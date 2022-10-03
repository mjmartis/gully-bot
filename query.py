# Finds the closest image in the given index.
#   Usage: query.py <index file> <image file>

import datetime
import logging
import os
import pickle
import sys

from gully_types import EmbeddedFrame

import numpy as np
from scipy import spatial

# Silence chatty TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import tensorflow_hub as hub

# Embedding model metadata.
MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
MODEL_INPUT_DIMS = (224, 224)

BATCH_SIZE = 32

CANDIDATES_SIZE = 5
CLOSE_TIME = datetime.timedelta(seconds=5.5)
MAX_DISTANCE = 0.15

# Returns the "min:sec" formatted version of the given
# second duration.
def format_duration(s):
    mins, secs = divmod(s, 60)
    return f'{mins}:{secs:02}'

# Returns a string representing the given datapoint.
def format_datapoint(d, include_ts):
    date_str = d.date.strftime('%-m %b %Y')

    if include_ts:
        length_str = format_duration(d.length.seconds)
        timestamp_str = format_duration(d.timestamp.seconds)
        return f'"{d.title}" ({date_str}) [{timestamp_str}/{length_str}]'

    return f'"{d.title}" ({date_str})'

def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <index file> <image file>')
        exit(1)

    # Load embedding model.
    embed_image = hub.KerasLayer(MODEL_DIR)

    # Load image into tensor.
    image_path = sys.argv[2]
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=MODEL_INPUT_DIMS)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_tensor = tf.expand_dims(image_array, 0)

    # Run embedding.
    query_features = embed_image(image_tensor).numpy()[0]

    # A running tally of the closest n datapoints we've seen. Stored
    # as (dist, datapoint) for sorting purposes.
    closest = []

    with open(sys.argv[1], 'rb') as database:
        # Keep going until we hit the end of the file.
        while True:
            # Collect a batch of datapoints so we can take advantage
            # of vectorised distance calculations.
            batch = []
            for _ in range(BATCH_SIZE):
                head = database.peek(1)
                if not head:
                    break

                batch.append(pickle.load(database))

            if not batch:
                break

            # Find distances across the whole batch.
            ds = spatial.distance.cdist([query_features],
                                        [r.features for r in batch],
                                        'cosine')[0]

            # Calculate the closest n from this batch.
            if len(batch) > CANDIDATES_SIZE:
                part = np.argpartition(ds, CANDIDATES_SIZE)[:-CANDIDATES_SIZE]
                batch_closest = [(ds[i], batch[i]) for i in part]

            # Merge this batch with the closest seen so far.
            closest = sorted(closest + batch_closest)[:CANDIDATES_SIZE]

            head = database.peek(1)

    ## Debug output.
    #for d, c in closest:
    #    print(f'{format_datapoint(c, True)} <d {d}>.')

    # Our closest datapoints in chronological order.
    candidates = sorted([c for _,c in closest], key=lambda c: c.timestamp)

    # All results must be from the same video.
    if any(c.title != candidates[0].title for c in candidates):
        print('Not found.')
        exit()

    chosen_dist, chosen = closest[0]

    # If most of the candidates are far away from our selection
    # chronologically, then don't offer timestamp.
    half_count = len(candidates) // 2
    far = lambda c1, c2: abs(c1.timestamp - c2.timestamp) > CLOSE_TIME
    timestamp_bad = sum(far(c, chosen) for c in candidates) > half_count

    # Selection must be moderately close.
    if chosen_dist > MAX_DISTANCE:
        print('Not found.')
        exit()

    print(format_datapoint(chosen, not timestamp_bad))

if __name__ == "__main__":
    main()
