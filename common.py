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

CLOSE_BUF_SIZE = 10
CLOSE_TIME = datetime.timedelta(seconds=5.5)
CLOSE_TIME_FRACTION = 3
CLOSE_DISTANCE = 0.14
MAX_OUTLIERS = 3

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


# Accepts a sorted list and a maximum delta, and returns the (inclusive)
# indices of a range that contains the most elements without their difference
# exceeding the delta.
def max_close_window(ts, delta):
    e = len(ts) - 1
    u, v = 0, 0
    best_u, best_v = 0, 0
    while v < e or ts[v] - ts[u] > delta:
        while ts[v] - ts[u] > delta:
            u += 1

        if v - u > best_v - best_u:
            best_u, best_v = u, v

        if v < e:
            v += 1

    return (best_u, best_v)


# Accepts a PIL image and returns the image in a tensor of the format needed by
# the emebdding model (or None if the image isn't suitable for input).
def image_to_tensor(image):
    image_w, image_h = image.size

    # Can't scale the image if it would mutate it too much.
    if not (IMG_MIN_RATIO < image_w / image_h < IMG_MAX_RATIO):
        return None

    formatted_image = image.convert('RGB') \
                           .resize(MODEL_INPUT_DIMS, Image.NEAREST)
    image_array = tf.keras.utils.img_to_array(formatted_image)
    return tf.expand_dims(image_array, 0)


# Accepts a database and image contents and returns the best guess of the frame
# that the image represents (or None if there is no close match). Optionally
# accepts a progress bar to increment as the serach progresses.
def find_nearest_frame(db, image_bytes, bar):
    # Convert image into TF model input.
    image = Image.open(image_bytes)
    image_tensor = image_to_tensor(image)
    if image_tensor is None:
        return None

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
        if len(batch) > CLOSE_BUF_SIZE:
            ids = np.argpartition(ds, CLOSE_BUF_SIZE)[:CLOSE_BUF_SIZE]
        else:
            ids = range(len(batch))
        batch_closest = [(ds[i], batch[i]) for i in ids]

        # Merge this batch with the closest seen so far.
        closest = sorted(closest + batch_closest,
                         key=lambda t: t[0])[:CLOSE_BUF_SIZE]

        head = db.peek(1)

    db.seek(0)

    ## Debug output.
    #print()
    #for d, c in closest:
    #    print(f'{format_datapoint(c)} <d {d}>')

    # Only consider close datapoints.
    close_enough = [t for t in closest if t[0] <= CLOSE_DISTANCE]
    close_enough.sort()
    if not close_enough:
        return None

    # Calculate frequency of individual titles.
    freq = {}
    for t in close_enough:
        title = t[1].title
        freq[title] = freq.get(title, []) + [t]

    # A title is better if it is more frequent than the others, or else if its
    # closest distance is less than the others.
    better = lambda t: (len(t[1]), -t[1][0][0])
    chosen_pts = max(freq.items(), key=better)[1]
    chosen_frames = [c for (_, c) in chosen_pts]

    # Only a small number of titles may deviate from our modal title.
    if len(chosen_frames) < len(close_enough) - MAX_OUTLIERS:
        return None

    # Return the median close timestamp if many of the timestamps are close.
    chosen_by_ts = sorted(chosen_frames, key=lambda c: c.timestamp)
    chosen_ts = None
    dense_ts_u, dense_ts_v = max_close_window(
        [c.timestamp for c in chosen_by_ts], CLOSE_TIME)
    if 1 + dense_ts_v - dense_ts_u > len(chosen_pts) // CLOSE_TIME_FRACTION:
        chosen_ts = chosen_by_ts[(dense_ts_u + dense_ts_v) // 2].timestamp

    chosen_frame = chosen_frames[0]
    return EmbeddedFrame(title=chosen_frame.title,
                         date=chosen_frame.date,
                         length=chosen_frame.length,
                         timestamp=chosen_ts,
                         features=None)
