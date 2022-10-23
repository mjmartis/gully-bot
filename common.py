# Logic and definitions used from multiple files.

from collections import namedtuple
import datetime
import logging
import os
import pathlib
import pickle
import sys

from annoy import AnnoyIndex
import cv2
import numpy as np
from PIL import Image
from progress.bar import Bar

# Silence chatty TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import tensorflow_hub as hub

# Embedding model metadata.
_MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
_MODEL_INPUT_DIMS = (224, 224)
_EMBED_IMAGE = hub.KerasLayer(_MODEL_DIR)

_IMG_BORDER_THRESHOLD = 10
_IMG_MIN_WIDTH = 224
_IMG_MIN_HEIGHT = 224
_IMG_MIN_RATIO = 1.30
_IMG_MAX_RATIO = 2.30

_CLOSE_BUF_SIZE = 10
_CLOSE_TIME = datetime.timedelta(seconds=5.5)
_CLOSE_TIME_FRACTION = 3
_CLOSE_DISTANCE = 0.40
_MAX_OUTLIERS = 3

_SQUASH_EXP_FACTOR = 5.0

_FEATURE_VECTOR_LEN = 1792
_DIST_METRIC = 'angular'

_SAMPLE_HZ = 4.0

EmbeddedFrame = namedtuple('EmbeddedFrame',
                           ['title', 'date', 'length', 'timestamp', 'features'])


# Returns the "min:sec" formatted version of the given
# duration.
def _format_duration(d):
    mins, secs = divmod(d.seconds, 60)
    return f'{mins}:{secs:02}'


# Accepts a sorted list and a delta value, and returns the (inclusive) start
# and end indicies of a maximal sublist whose range does not exceed the delta.
def _max_close_window(ts, delta):
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


# Searches the database for the datapoints nearest to the given query.
def _find_nearest_frames(mds, nn_db, query):
    # Get metadata indices of closest points. Stored as:
    #   [index list, distance list].
    query_results = nn_db.get_nns_by_vector(query,
                                            _CLOSE_BUF_SIZE,
                                            include_distances=True)

    # Reconstruct records of closest points. Stored as (dist, datapoint) for
    # sorting purposes.
    return sorted((d, mds[i]) for i, d in zip(*query_results))


# Accepts a list of candidate datapoints and synthesises them into one frame
# selection.
def _choose_nearest_frame(close_pts):
    # Calculate frequency of individual titles.
    freq = {}
    for t in close_pts:
        title = t[1].title
        freq[title] = freq.get(title, []) + [t]

    # A title is better if it is more frequent than the others, or else if its
    # closest distance is less than the others.
    chosen_pts = max(freq.items(), key=lambda t: (len(t[1]), -t[1][0][0]))[1]
    chosen_frames = [c for (_, c) in chosen_pts]
    chosen_frames.sort(key=lambda c: c.timestamp)

    # Only a small number of titles may deviate from our modal title.
    outlier_limit = min((len(close_pts) + 1) // 2, _MAX_OUTLIERS)
    if len(chosen_frames) < len(close_pts) - outlier_limit:
        return None

    # Return the median close timestamp if many of the timestamps are close.
    chosen_ts = None
    dense_ts_u, dense_ts_v = _max_close_window(
        [c.timestamp for c in chosen_frames], _CLOSE_TIME)
    if 1 + dense_ts_v - dense_ts_u > len(chosen_pts) // _CLOSE_TIME_FRACTION:
        chosen_ts = chosen_frames[(dense_ts_u + dense_ts_v) // 2].timestamp

    chosen_frame = chosen_frames[0]
    return EmbeddedFrame(title=chosen_frame.title,
                         date=chosen_frame.date,
                         length=chosen_frame.length,
                         timestamp=chosen_ts,
                         features=None)


# Converts a candidate title into an ad-hoc confidence score.
def _score_title(close_pts, title):
    # Squash distances into scores in (0, 1).
    scores = [1 - d / _CLOSE_DISTANCE for d, c in close_pts if c.title == title]

    # Pull scores towards 1.
    return (sum(scores) / _CLOSE_BUF_SIZE)**(1 / _SQUASH_EXP_FACTOR)


# Accepts a PIL image and returns a version of the image with any black border
# removed. A pixel is considered "black" if all of its channels have magnitude
# less than a small threshold.
def _crop_image_border(image):
    y_light, x_light, _ = (np.asarray(image) > _IMG_BORDER_THRESHOLD).nonzero()

    # All black.
    if len(y_light) == 0:
        return Image.new('RGB', (0, 0))

    return image.crop(
        (np.min(x_light), np.min(y_light), np.max(x_light), np.max(y_light)))


# Accepts a PIL image and returns the image in a tensor of the format needed by
# the emebdding model (or None if the image isn't suitable for input).
def _image_to_tensor(image):
    cropped_image = _crop_image_border(image.convert('RGB'))
    image_w, image_h = cropped_image.size

    # Can't scale the image if it would mutate it too much.
    if image_w < _IMG_MIN_WIDTH or image_h < _IMG_MIN_HEIGHT or \
       not (_IMG_MIN_RATIO < image_w / image_h < _IMG_MAX_RATIO):
        return None

    formatted_image = cropped_image.resize(_MODEL_INPUT_DIMS, Image.NEAREST)
    image_array = tf.keras.utils.img_to_array(formatted_image)
    return tf.expand_dims(image_array, 0)


# Accepts a PIL image and returns its corresponding vector in the embedding
# space, if it is viable to embed.
def embed_image(image):
    image_tensor = _image_to_tensor(image)
    if image_tensor is None:
        return None

    return _EMBED_IMAGE(image_tensor).numpy()[0]


# Returns a string representing the given datapoint.
def format_datapoint(d):
    date_str = d.date.strftime('%-m %b %Y')

    if d.timestamp:
        length_str = _format_duration(d.length)
        timestamp_str = _format_duration(d.timestamp)
        return f'**{d.title}** [{timestamp_str}/{length_str}] ({date_str})'

    return f'**{d.title}** ({date_str})'


# Accepts a database and image contents and returns either:
#  a) a guess of the frame that the image most closely resembles and an ad-hoc
#     confidence value for the match, or
#  b) None, if there is no close match.
def find_nearest_frame(mds, nn_db, image_bytes):
    # Generate query datapoint.
    query_features = embed_image(Image.open(image_bytes))
    if query_features is None:
        return None

    # Search the database.
    closest = _find_nearest_frames(mds, nn_db, query_features)

    ## Debug output.
    #print()
    #for d, c in closest:
    #    print(f'{format_datapoint(c)} <d {d}>')

    # Only consider close datapoints.
    close_enough = [t for t in closest if t[0] <= _CLOSE_DISTANCE]
    close_enough.sort()
    if not close_enough:
        return None

    # Synthesise close points into one selection.
    chosen = _choose_nearest_frame(close_enough)
    if not chosen:
        return None

    # Include a confidence score.
    return chosen, _score_title(close_enough, chosen.title)


# Initializes a NN database with the parameters needed for our embedding.
def init_nn_db():
    return AnnoyIndex(_FEATURE_VECTOR_LEN, _DIST_METRIC)


# Parse date and video title from full path, with stem in the
# in the format:
#  YYYY-MM-DD NAME
# Returns (name, date) tuple.
def parse_video_path(path):
    path_stem = pathlib.Path(path).stem
    sep_index = path_stem.find(' ')
    date = datetime.datetime.strptime(path_stem[:sep_index], '%Y-%m-%d')
    return (date, path_stem[sep_index + 1:])


# Embeds a sample of video frames and runs the specified function on the result.
def embed_video_frames(video_path, process_fn, bar):
    date, title = parse_video_path(video_path)

    # Read video.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        print()
        print(f'Error can\'t read FPS for "{title}".')
        return

    frame_stride = fps / _SAMPLE_HZ
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process frames for this video.
    for frame_index in np.arange(0.0, frame_count, frame_stride):
        if bar:
            bar.next(frame_stride / frame_count)

        # Jump to and read frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            print()
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


# Print a pretty progress bar with max value 1.0.
def progress_bar(title):
    bar = Bar(f'{title:40.40}', max=1.0, suffix='%(percent)d%%')
    bar.bar_prefix = ' ['
    bar.bar_suffix = '] '
    return bar
