import pandas as pd
import os
import re
import sys
from collections import defaultdict
from clean_streams_data import clean_streams
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def non_alpha(df, col):

    df[col] = df[col].str.replace(
        r'''[^(]*\(|\)[^)]*''',
        ''
    )


def twitch_preprocess(pathhead='../data_test/',
                      get_streams=True) -> pd.DataFrame:

    # COMBINING FOLLOWER RANKING
    path = pathhead + 'ranked_streamers'
    files = [os.path.join(path, file) for file in os.listdir(path)]
    df = pd.concat((pd.read_csv(f) for f in files if f.endswith('csv')))

    non_alpha(df, 'Channel')

    df.index = df['Channel']
    df.drop_duplicates(subset='Channel', inplace=True)

    df.drop(['Unnamed: 12', 'Unnamed: 1', 'Unnamed: 0'], axis=1, inplace=True)

    # GETTING STREAM/GAME AVERAGES
    if get_streams:
        path = pathhead + 'streams_data'
        accepted = df['Channel'].tolist()

        files = [os.path.join(path, file) for file in os.listdir(path)]

        ptrn = re.compile(
            r'''streams_data\\(?P<channel>[a-zA-Z0-9_]+)\s\-\s(?P<type>Twitch|game)''')

        channels = set()
        channel_dict = defaultdict(list)
        all_games = []
        streamers_list = []
        channel_games = defaultdict(list)

        for file in files:

            channel = ptrn.search(file).group('channel')
            filetype = ptrn.search(file).group('type')

            if channel not in accepted:
                continue

            channel_frame = pd.read_csv(file)

            # Collecting data from stream files
            if filetype == 'Twitch':
                if channel in channels:
                    channel_dict[channel].append(channel_frame)
                else:
                    channels.add(channel)
                    channel_dict[channel].append(channel_frame)

            # Collecting data from game files
            if filetype == 'game':
                games = list(channel_frame['Game'])
                channel_games[channel] += games
                all_games += games

                try:
                    stream_time = int(
                        df[df['Channel'] == channel]['Stream time (mins)'])
                except TypeError:
                    print(channel)
                    print(df[df['Channel'] == channel])
                    sys.exit(1)

                channel_frame['Proportion'] = channel_frame['Stream time (mins)'] / \
                    stream_time

                channel_frame.drop([
                    'Stream time (mins)',
                    'Total watch time (mins)',
                    'Average viewers',
                    'Peak viewers',
                    'Unnamed: 0'], axis=1, inplace=True)

                table_tf = channel_frame.T
                table_tf.columns = table_tf.iloc[0]
                table_tf.drop(['Game'], axis=0, inplace=True)
                table_tf.index = [channel]
                table_tf = table_tf.groupby(lambda x: x, axis=1).sum()

                streamers_list.append(table_tf)

        # Cleaning data from stream files
        frames = {}
        for key, val in channel_dict.items():
            one_channel = pd.concat(val, axis=0)
            frames[key] = one_channel

        streams = []
        for channel, frame in frames.items():
            avg_gap, avg_length, avg_gain, avg_variety, day_props = clean_streams(
                frame)
            streamer = pd.DataFrame({
                'Channel': channel,
                'Monday': day_props['Monday'],
                'Tuesday': day_props['Tuesday'],
                'Wednesday': day_props['Wednesday'],
                'Thursday': day_props['Thursday'],
                'Friday': day_props['Friday'],
                'Saturday': day_props['Saturday'],
                'Sunday': day_props['Sunday'],
                'Average time between': avg_gap,
                'Average stream length': avg_length,
                'Average followers per stream': avg_gain,
                'Average games per stream': avg_variety
            }, index=range(1))
            streams.append(streamer)

        streams = pd.concat(streams, axis=0)
        streams.index = streams['Channel']
        streams.drop(['Channel'], axis=1, inplace=True)
        df = df.merge(streams, left_index=True, right_index=True)

        # Cleaning data from game files
        all_games = set(all_games)
        streamers = pd.concat(streamers_list, axis=0)
        streamers = streamers.groupby(level=0).sum()
        df = df.merge(streamers, left_index=True, right_index=True)

    return (df, all_games)


master, all_games = twitch_preprocess()
