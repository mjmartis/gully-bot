# Logic and definitions used from multiple files.

from collections import namedtuple
import datetime
import logging
import os
import pickle
import sys

import numpy as np
from PIL import Image
from progress.bar import Bar
from scipy import spatial

# Silence chatty TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import tensorflow_hub as hub

# A patch after migrating type location. TODO: update db and remove this.
import common
sys.modules['gully_types'] = common

# Embedding model metadata.
MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
MODEL_INPUT_DIMS = (224, 224)
EMBED_IMAGE = hub.KerasLayer(MODEL_DIR)

IMG_MIN_RATIO = 1.30
IMG_MAX_RATIO = 2.30

BATCH_SIZE = 32

CANDIDATES_SIZE = 5
CLOSE_TIME = datetime.timedelta(seconds=5.5)
MAX_DISTANCE = 0.125
MAX_OUTLIERS = 1

EmbeddedFrame = namedtuple('EmbeddedFrame',
                           ['title', 'date', 'length', 'timestamp', 'features'])


# Returns the "min:sec" formatted version of the given
# second duration.
def format_duration(s):
    mins, secs = divmod(s, 60)
    return f'{mins}:{secs:02}'


# Returns a string representing the given datapoint.
def format_datapoint(d):
    date_str = d.date.strftime('%-m %b %Y')

    if d.timestamp:
        length_str = format_duration(d.length.seconds)
        timestamp_str = format_duration(d.timestamp.seconds)
        return f'"{d.title}" ({date_str}) [{timestamp_str}/{length_str}]'

    return f'"{d.title}" ({date_str})'


# Accepts a database and image contents and returns the best guess of the frame
# that the image represents (or None if there is no close match). Optionally
# accepts a progress bar to increment as the serach progresses.
def find_nearest_frame(db, image_bytes, bar):
    # Convert image into TF model input.
    image = Image.open(image_bytes)
    image_w, image_h = image.size

    # Can't scale the image if it would mutate it too much.
    if not (IMG_MIN_RATIO < image_w / image_h < IMG_MAX_RATIO):
        return None

    formatted_image = image.convert('RGB') \
                           .resize(MODEL_INPUT_DIMS, Image.NEAREST)
    image_array = tf.keras.utils.img_to_array(formatted_image)
    image_tensor = tf.expand_dims(image_array, 0)

    # Run embedding.
    query_features = EMBED_IMAGE(image_tensor).numpy()[0]

    # A running tally of the closest n datapoints we've seen. Stored
    # as (dist, datapoint) for sorting purposes.
    closest = []

    # Keep going until we hit the end of the file.
    db.seek(0)
    db_progress = 0
    while True:
        # Collect a batch of datapoints so we can take advantage
        # of vectorised distance calculations.
        batch = []
        for _ in range(BATCH_SIZE):
            head = db.peek(1)
            if not head:
                break

            batch.append(pickle.load(db))

        # Advance progress bar.
        if bar:
            bar.next(db.tell() - db_progress)
        db_progress = db.tell()

        if not batch:
            break

        # Find distances across the whole batch.
        ds = spatial.distance.cdist([query_features],
                                    [r.features for r in batch], 'cosine')[0]

        # Calculate the closest n from this batch.
        if len(batch) > CANDIDATES_SIZE:
            ids = np.argpartition(ds, CANDIDATES_SIZE)[:-CANDIDATES_SIZE]
        else:
            ids = range(len(batch))
        batch_closest = [(ds[i], batch[i]) for i in ids]

        # Merge this batch with the closest seen so far.
        closest = sorted(closest + batch_closest)[:CANDIDATES_SIZE]

        head = db.peek(1)

    db.seek(0)

    ## Debug output.
    #print()
    #for d, c in closest:
    #    print(f'{format_datapoint(c)} <d {d}>')

    # Choose closest frame.
    chosen_dist, chosen = closest[0]
    candidates = [c for _, c in closest]
    candidates_len = len(candidates)

    # Selection must be moderately close.
    if chosen_dist > MAX_DISTANCE:
        return None

    # Can only have limited outlier titles.
    candidates = [c for c in candidates if c.title == chosen.title]
    if len(candidates) < candidates_len - MAX_OUTLIERS:
        return None

    # If most of the candidates are far away from our selection
    # chronologically, then don't offer timestamp.
    half_count = len(candidates) // 2
    far = lambda c1, c2: abs(c1.timestamp - c2.timestamp) > CLOSE_TIME
    no_ts = sum(far(c, chosen) for c in candidates) > half_count

    return EmbeddedFrame(title=chosen.title,
                         date=chosen.date,
                         length=chosen.length,
                         timestamp=None if no_ts else chosen.timestamp,
                         features=None)
