# Produce an index of image feature vectors for frames
# of the given images.
#  Usage: python3 database.py <output md> <output nn> <input files...>

import pickle
import sys

from common import EmbeddedFrame, init_nn_db, embed_video_frames, parse_video_path, progress_bar

TREES_COUNT = 30


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <output md> <output nn> <input files...>')
        exit(1)

    # Open NN index on disk.
    nn_db = init_nn_db()
    nn_db.on_disk_build(sys.argv[2])

    # Process each video.
    mds = []
    nn_db_i = [0]

    # Save separate metadata and features for each frame.
    def process(record):
        # Append metadata to metadata database.
        metadata = EmbeddedFrame(title=record.title,
                                 date=record.date,
                                 length=record.length,
                                 timestamp=record.timestamp,
                                 features=None)
        mds.append(metadata)

        # Add features to NN database.
        nn_db.add_item(nn_db_i[0], record.features)
        nn_db_i[0] += 1

    for i, video_path in enumerate(sys.argv[3:]):
        # Extract metadata.
        _, title = parse_video_path(video_path)

        # Display progress bar.
        bar = progress_bar(f'[{i+1:03}/{len(sys.argv)-3}] {title}')

        # Write metadata and NN data.
        embed_video_frames(video_path, process, bar)

        bar.finish()

    # Write metadata.
    with open(sys.argv[1], 'wb') as md_db:
        pickle.dump(mds, md_db)

    # Construct actual NN truee.
    print('Building NN index... ', end='', flush=True)
    nn_db.build(TREES_COUNT)
    print('done.')


if __name__ == "__main__":
    main()
