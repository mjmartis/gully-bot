# Converts a pickle database into a NN index and metadata database.
#  Usage: python3 nn_database.py <input db> <output metadata db> <output nn db>

import os
import pickle
import sys

from common import EmbeddedFrame, init_nn_db

from progress.bar import Bar

TREES_COUNT = 30


# Decomposes a monolithic linear database into a list of metadata and a a NN
# data structure. The provided NN database is populated with the data points,
# and their metadata are returned. Also accepts a progress bar to increment.
def gen_nn_database(in_db, nn_db, bar):
    # Store metadata in memory.
    mds = []

    # Keep going until we hit the end of the file.
    in_db_progress = 0
    in_db_i = 0
    while True:
        head = in_db.peek(1)
        if not head:
            break

        # Read full record.
        f = pickle.load(in_db)
        f_metadata = EmbeddedFrame(title=f.title,
                                   date=f.date,
                                   length=f.length,
                                   timestamp=f.timestamp,
                                   features=None)

        # Decompose into a list of metadata and a vector that references its
        # metadata index.
        mds.append(f_metadata)
        nn_db.add_item(in_db_i, f.features)

        # Advance progress bar.
        if bar:
            bar.next(in_db.tell() - in_db_progress)
        in_db_progress = in_db.tell()

        in_db_i += 1


def main():
    if len(sys.argv) != 4:
        print(
            f'Usage: {sys.argv[0]} <input db> <output metadata db> <output nn index>'
        )
        exit(1)

    in_db = open(sys.argv[1], 'rb')

    nn_db = init_nn_db()

    bar = Bar('Adding points',
              max=os.path.getsize(sys.argv[1]),
              suffix='%(percent)d%%')

    mds = gen_nn_database(in_db, nn_db, bar)

    bar.finish()

    with open(sys.argv[2], 'wb') as md_f:
        pickle.dump(mds, md_f)

    print('Building NN index... ', end='', flush=True)
    nn_db.build(TREES_COUNT)
    nn_db.save(sys.argv[3])
    print('done.')

    in_db.close()


if __name__ == "__main__":
    main()
