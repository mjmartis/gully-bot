# Reddit bot that replies to posts that include a matching frame.

import datetime
from http import HTTPStatus
import io
import json
import mimetypes
import requests
import sys
import os
from os import path
from urllib.parse import urlparse

from common import EmbeddedFrame, find_nearest_frame, format_datapoint

import praw
from progress.bar import Bar

#SUBREDDIT = 'jakeandamir'
SUBREDDIT = 'gullybot'
HOST_SUFFICES = ['.imgur.com', '.redd.it']

REQUEST_TIMEOUT = datetime.timedelta(seconds=6)
RESPONSE_LIMIT_BYTES = 4 * 1024 * 1024  # 4MB.
STREAM_CHUNK_BYTES = 1024  # 1KB.

PROJECT_URL = 'https://github.com/mjmartis/gully-bot'
POST_TEXT = \
    'If you have to ask which episode... *then you don\'t know*. ' + \
    'This looks like a frame from {}.' + \
    '\n\n' + \
    '^(Confidence score: {:.02}. For more info, check out ' + \
    'my [source code]({}).)'


# Attempts to download the given file, bailing out if it is too large or the
# download takes too long. Displays a progress bar.
#
# Based on: https://stackoverflow.com/a/22347526.
def maybe_download(url):
    start_ts = datetime.datetime.now()

    # Try to fetch file.
    try:
        r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT.seconds)
        r.raise_for_status()
    except HTTPError as e:
        print(f'Request failed for "{url}" with code ' + \
              f'{e.response.status_code}.')
        return None
    except:
        return None

    # Bail out if file is too large.
    total_size_bytes = int(r.headers.get('Content-Length'))
    if total_size_bytes > RESPONSE_LIMIT_BYTES:
        print(f'File size limit exceeded for "{url}".')
        return None

    # Display download progress bar.
    bar = Bar('Downloading', max=total_size_bytes, suffix='%(percent)d%%')

    # Construct an in-memory file from stream data.
    data = io.BytesIO()
    stream_size_bytes = 0

    # Iteratively populate file data, bailing out if data is received too
    # slowly or is too large.
    try:
        for chunk in r.iter_content(STREAM_CHUNK_BYTES):
            bar.next(len(chunk))

            # Download too slow.
            if datetime.datetime.now() - start_ts > REQUEST_TIMEOUT:
                raise (ValueError(f'Download too slow for "{url}".'))

            # File too large.
            stream_size_bytes += len(chunk)
            if stream_size_bytes > RESPONSE_LIMIT_BYTES:
                raise (ValueError(f'Download limit exceeded for "{url}".'))

            data.write(chunk)
    except ValueError as e:
        data.close()
        r.close()

        print()
        print(e.args[0])
        return None
    finally:
        bar.finish()

    return data


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
    #for sub in subreddit.top(time_filter="year"):
    for sub in subreddit.stream.submissions(skip_existing=True):
        # Need submission to include a link.
        if sub.is_self:
            continue

        # Need link to be from imgur or Reddit images.
        netloc = urlparse(sub.url).netloc
        is_image_host = any(netloc.endswith(h) for h in HOST_SUFFICES)
        if not is_image_host:
            continue

        # Need link to be an image.
        mimetype, _ = mimetypes.guess_type(sub.url)
        if not mimetype or not mimetype.startswith('image'):
            continue

        # Print out post information for manual analysis.
        print(f'--------\nNew post\n--------')
        print(f'Post link: "{sub.permalink}"')
        print(f'Image link: "{sub.url}"')

        # Download image if it is small enough and doesn't take too long.
        image = maybe_download(sub.url)
        if not image:
            continue

        # Perform actual frame matching.
        bar = Bar('Searching  ', max=db_size, suffix='%(percent)d%%')
        result = find_nearest_frame(db, image, bar)
        bar.finish()

        image.close()

        # TODO: construct and post comment.
        if result:
            c, s = result
            sub.reply(
                body=POST_TEXT.format(format_datapoint(c), s, PROJECT_URL))

            print('Result:')
            print('', format_datapoint(c))
            print('', f'Confidence: {s:.02}')
        else:
            print('Not found.')
        print()


if __name__ == "__main__":
    main()
