# Finds the closest image in the given index.
#   Usage: query.py <index file> <image file>

import pickle
import sys

from gully_types import EmbeddedFrame

from scipy import spatial
import tensorflow as tf
import tensorflow_hub as hub

# Embedding model metadata.
MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
MODEL_INPUT_DIMS = (224, 224)

# Print out the "min:sec" formatted version of the given
# second duration.
def format_duration(s):
    mins, secs = divmod(s, 60)
    return f'{mins}:{secs:02}'

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

    # Find closest datapoint.
    closest_dist, closest_datapoint = sys.float_info.max, None
    with open(sys.argv[1], 'rb') as database:
        # Keep going until we hit the end of the file.
        head = database.peek(1)
        while head:
            datapoint = pickle.load(database)

            d = spatial.distance.cosine(query_features, datapoint.features)

            if d < closest_dist:
                closest_dist = d
                closest_datapoint = datapoint

            head = database.peek(1)

    date_str = closest_datapoint.date.strftime('%-m %b %Y')
    length_str = format_duration(closest_datapoint.length.seconds)
    timestamp_str = format_duration(closest_datapoint.timestamp.seconds)
    print(f'"{closest_datapoint.title}" ({date_str}) [{timestamp_str}/{length_str}]')

if __name__ == "__main__":
    main()
