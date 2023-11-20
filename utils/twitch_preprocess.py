import pandas as pd
import os
import re
import sys
import json
from collections import defaultdict
from clean_streams_data import clean_streams
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def non_alpha(df, col):

    df[col] = df[col].str.replace(
        r'''[^(]*\(|\)[^)]*''',
        ''
    )


def twitch_preprocess(get_streams=True) -> pd.DataFrame:
    
    pathhead = os.path.dirname(os.path.dirname(__file__))
    stats_path = pathhead + '\\data_test\\channel_stats\\'
    stats_files = [os.path.join(stats_path, file) for file in os.listdir(stats_path)]
    
    all_stats_data = pd.DataFrame.from_records(
        [json.load(open(f)) for f in stats_files]
    )

    all_stats_data.index = all_stats_data['channel']
    all_stats_data['mins_streamed'] = all_stats_data['hours_streamed'] * 60
    
    # df.drop_duplicates(subset='channel', inplace=True)

    # GETTING STREAM/GAME AVERAGES
    if get_streams:
        streams_path = pathhead + '\\data_test\\streams_data\\'

        stream_files = [os.path.join(streams_path, file) for file in os.listdir(streams_path)]

        ptrn = re.compile(
            r'''streams_data\\(?P<channel>[a-zA-Z0-9_]+)\s\-\s(?P<type>Twitch|game)''')

        channel_set = set()
        channel_dict = defaultdict(list)
        all_channel_games_data = []
        channel_games = defaultdict(list)

        for file in stream_files:

            channel = ptrn.search(file).group('channel')
            filetype = ptrn.search(file).group('type')

            channel_frame = pd.read_csv(file)

            # Collecting data from stream files
            if filetype == 'Twitch':
                if channel in channel_set:
                    channel_dict[channel].append(channel_frame)
                else:
                    channel_set.add(channel)
                    channel_dict[channel].append(channel_frame)

            # Collecting data from game files
            if filetype == 'game':
                games = list(channel_frame['Game'])
                channel_games[channel] += games

                try:
                    stream_time = int(
                        all_stats_data[all_stats_data['channel'] == channel]['mins_streamed'])
                except TypeError:
                    print(channel)
                    print(all_stats_data[all_stats_data['channel'] == channel])
                    sys.exit(1)

                channel_frame['proportion'] = channel_frame['Stream time (mins)'] / \
                    stream_time

                channel_frame.drop([
                    'Stream time (mins)',
                    'Total watch time (mins)',
                    'Average viewers',
                    'Peak viewers',
                    'Views',
                    'Views per hour',
                    'Unnamed: 0'], axis=1, inplace=True)

                '''
                The following series of transformations will change
                a single channel's games dataset from this:
                
                Game        Proportion (sum to 1)
                [game1]     [proportion1]
                [game2]     [proportion2]
                ...
                
                To this:
                
                                [game1]       [game2]       [game3]...
                [channelname]   [proportion1] [proportion2] [proportion3]
                '''
                table_tf = channel_frame.T                                
                table_tf.columns = table_tf.iloc[0]                
                table_tf.drop(['Game'], axis=0, inplace=True)                
                table_tf.index = [channel]
                table_tf = table_tf.groupby(lambda x: x, axis=1).sum()

                # Store all single-streamer transformed proportion
                # matrices for later concatenation into one matrix
                all_channel_games_data.append(table_tf)

        # Cleaning data from stream files
        all_channel_streams = {}
        for channel, channel_streams in channel_dict.items():
            all_streams_one_channel = pd.concat(channel_streams, axis=0)
            all_channel_streams[channel] = all_streams_one_channel

        additional_stats = []
        for channel, c_streams in all_channel_streams.items():
            
            avg_gap, avg_length, avg_gain, avg_variety, day_props = clean_streams(
                c_streams)
            streamer_additional_stats = pd.DataFrame({
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
            additional_stats.append(streamer_additional_stats)

        additional_stats = pd.concat(additional_stats, axis=0)
        additional_stats.index = additional_stats['Channel']
        additional_stats.drop(['Channel'], axis=1, inplace=True)
        all_stats_data = all_stats_data.merge(additional_stats, 
                                              left_index=True, 
                                              right_index=True)

        # Cleaning data from game files
        all_channel_games_data = pd.concat(all_channel_games_data, axis=0)
        all_channel_games_data = all_channel_games_data.groupby(level=0).sum()

    return all_stats_data, all_channel_games_data


all_stats_data, all_channel_games_data = twitch_preprocess()
print(all_stats_data)
print(all_channel_games_data)