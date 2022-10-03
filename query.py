# Finds the closest image in the given index.
#   Usage: query.py <index file> <image file>

import pickle
from scipy import spatial
import sys
import tensorflow as tf
import tensorflow_hub as hub

# Embedding model metadata.
MODEL_DIR = 'imagenet_mobilenet_v2_140_224_feature_vector'
MODEL_INPUT_DIMS = (224, 224)

# Print out the "min:sec" formatted version of the given
# millisecond duration.
def format_duration_ms(ms):
    mins, mod_ms = divmod(ms, 60 * 1000)
    return f'{mins}:{mod_ms//1000:02}'

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
    query = embed_image(image_tensor)

    # Find closest datapoint.
    closest_dist, closest_line = sys.float_info.max, None
    with open(sys.argv[1], 'rb') as index_file:
        # Keep going until we hit the end of the file.
        try:
            while True:
                # Line format: (title, date, timestamp in ms, features)
                index_line = pickle.load(index_file)
                datapoint = index_line[-1]

                d = spatial.distance.cosine(query.numpy()[0], datapoint)

                if d < closest_dist:
                    closest_dist = d
                    closest_line = index_line[:-1]
        except EOFError:
            pass

    title, date, ts_ms = closest_line
    print(f'"{title}" ({date}) [{format_duration_ms(ts_ms)}]')

if __name__ == "__main__":
    main()
