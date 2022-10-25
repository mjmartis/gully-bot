# Accepts a database of Mobilenet feature triplets and trains an embedding model.
#   Usage: python3 train.py <output model> <input count file> <input record paths...>

import sys

from common import FEATURE_VECTOR_LEN, RECORD_COMPRESSION

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard

# Format of examples to read in.
EXAMPLE_SCHEMA = {
    'a': tf.io.FixedLenFeature([FEATURE_VECTOR_LEN], dtype=tf.float32),
    'p': tf.io.FixedLenFeature([FEATURE_VECTOR_LEN], dtype=tf.float32),
    'n': tf.io.FixedLenFeature([FEATURE_VECTOR_LEN], dtype=tf.float32),
}

# Dataset parameters.
FS_READ_BUFFER_BYTES = 1024 * 50  # 50MB.
PARALLEL_READ_COUNT = 3
SHUFFLE_BUFFER_SIZE = 1024
EXAMPLE_BATCH_SIZE = 32
PREFETCH_SIZE = 8
TRAIN_PROP = 0.8
EPOCHS = 10
LEARNING_RATE = 0.00003
TENSORBOARD_LOGDIR = './logs'

# Model parameters.
LAYER_SIZE = 48


# From: https://keras.io/examples/vision/siamese_network/.
class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, a, p, n):
        ap_distance = tf.reduce_sum(tf.square(a - p), -1)
        an_distance = tf.reduce_sum(tf.square(a - n), -1)
        return (ap_distance, an_distance)


# From: https://keras.io/examples/vision/siamese_network/.
class SiameseModel(Model):

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights))

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


# Parse an example from previously serialised bytes.
def parse_example(record_bytes):
    return tf.io.parse_single_example(record_bytes, EXAMPLE_SCHEMA)


# Preprocess a list of TFRecord databases into training and validation datasets.
def prepare_datasets(dataset_paths, count_file_path):
    # Parse number of examples from a separate file.
    with open(count_file_path, 'r') as count_file:
        example_count = int(count_file.readline()[:-1])

    # Process dataset itself.
    ds = tf.data.TFRecordDataset(
        dataset_paths,
        compression_type=RECORD_COMPRESSION,
        buffer_size=FS_READ_BUFFER_BYTES,
        num_parallel_reads=PARALLEL_READ_COUNT).shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE).map(parse_example)

    # Split into train and validation sets.
    count = round(example_count * TRAIN_PROP)
    return [
        d.batch(EXAMPLE_BATCH_SIZE,
                drop_remainder=False).prefetch(PREFETCH_SIZE)
        for d in [ds.take(count), ds.skip(count)]
    ]


# The Keras layer that embeds the output of Mobilenet into its own Euclidean
# vector.
def reembedding_layer():
    return Sequential([
        layers.Dense(LAYER_SIZE, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(LAYER_SIZE, activation='relu'),
    ])


# The model that embeds a triplet of Mobilenet features and outputs a delta
# between the positive and negative distances.
def siamese_network():
    inputs = [
        layers.Input(shape=(FEATURE_VECTOR_LEN,), name=n)
        for n in ['a', 'p', 'n']
    ]
    dists = DistanceLayer()(*map(reembedding_layer(), inputs))
    return Model(inputs=inputs, outputs=dists)


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <output model> <input count file> ' +
              '<input record paths...>')
        exit(1)

    train_ds, val_ds = prepare_datasets(sys.argv[3:], sys.argv[2])
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGDIR,
                                                    histogram_freq=1)
    siamese_model = SiameseModel(siamese_network())
    siamese_model.compile(optimizer=optimizers.Adam(LEARNING_RATE))
    siamese_model.fit(train_ds,
                      epochs=EPOCHS,
                      validation_data=val_ds,
                      callbacks=[tensorboard_cb])
    siamese_model.save(sys.argv[1])


if __name__ == "__main__":
    main()
