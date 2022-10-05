# Reddit bot.

from datetime import datetime
import io
import json
import requests
import sys
import os
from os import path
from urllib.parse import urlparse

from common import EmbeddedFrame, find_nearest_frame, format_datapoint

import praw
from progress.bar import Bar

SUBREDDIT = 'jakeandamir'
IMAGE_HOSTNAME = 'i.redd.it'


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <auth json> <db>')
        exit()

    # Authenticate Reddit bot.
    with open(sys.argv[1]) as f:
        reddit = praw.Reddit(**json.load(f))
    subreddit = reddit.subreddit(SUBREDDIT)

    # Load database.
    db_size = os.path.getsize(sys.argv[2])
    db = open(sys.argv[2], 'rb')

    # Watch incoming posts.
    for sub in subreddit.top(time_filter="all"):
        #for sub in subreddit.stream.submissions(skip_existing=True):
        # Need submission to include a link.
        if sub.is_self:
            continue

        # Need link to be a Reddit-hosted image.
        url_info = urlparse(sub.url)
        if url_info.hostname != IMAGE_HOSTNAME:
            continue

        print(f'Image link: "{sub.url}"')
        image = io.BytesIO(requests.get(sub.url).content)
        bar = Bar('Searching', max=db_size, suffix='%(percent)d%%')
        chosen = find_nearest_frame(db, image, bar)
        bar.finish()
        image.close()

        # TODO: construct and post comment.
        if chosen:
            print(format_datapoint(chosen))
        else:
            print('Not found.')
        print('-------')

        input()


if __name__ == "__main__":
    main()
