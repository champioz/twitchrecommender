import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict


# Get list of all names in directory

def directory_count(path):
    ptrn = re.compile(
        r'''(?P<channel>[a-zA-Z0-9_]+)\s-\s(?P<type>Twitch|game)''')

    done = []
    for file in os.listdir(path):
        chan = ptrn.search(file).group('channel')
        done.append(chan)

    with open('./data/done.txt', 'w') as f:
        for strmr in set(done):
            f.write(f'{strmr}\n')


# Identify file downloads that were incomplete or corrupted
# Collect list of all affected channel names

def error_check(path):

    files = [os.path.join(path, file) for file in os.listdir(path)]
    ptrn = re.compile(r'''(?P<channel>[a-zA-z0-9_]+)_(?P<type>stream|game)_''')

    streamcheck = defaultdict(list)
    borked = []
    for file in files:
        chan, cat = ptrn.search(file).groups()
        strmr = pd.read_csv(file, engine='pyarrow',
                            dtype={
                                "Stream start time": str,
                                "Stream URL": str,
                                "Stream length (mins)": np.int32,
                                "Watch time (mins)": np.int32,
                                "Avg viewers": np.int32,
                                "Peak viewers": np.int32,
                                "Followers gained": np.int32,
                                "Followers per hour": np.int32,
                                "Views": np.int32,
                                "Views per hour": np.int32,
                                "Games": str
                            })
        if strmr.index.size == 0:
            borked.append(chan)
            continue
        streamcheck[chan].append(cat)

    for strmr in streamcheck.keys():
        if strmr in borked:
            continue
        if set(["stream", "game"]) != set(streamcheck[strmr]):
            borked.append(strmr)

    with open('../data/borked.txt', 'w') as log:
        for name in set(borked):
            log.write(name + '\n')


# All files for channels that were affected by at least one
# error or corruption

def error_delete(path, stats_path):

    files = [os.path.join(path, file) for file in os.listdir(path)]
    stats_files = [os.path.join(stats_path, file)
                   for file in os.listdir(stats_path)]
    ptrn = re.compile(r'''(?P<channel>[a-zA-z0-9_]+)_(?P<type>stream|game)_''')

    with open('../data/borked.txt', 'r') as log:
        borked = [line.strip() for line in log]
    print(len(files))
    for file in files:
        chan, cat = ptrn.search(file).groups()
        if chan in borked:
            os.remove(file)

    ptrn = re.compile(r'''/(?P<channel>[a-zA-Z0-9_]+)\.txt''')
    for file in stats_files:
        chan = ptrn.search(file).groups()[0]
        if chan in borked:
            os.remove(file)


# Identify download hangups that resulted in multiple copies
# of the same file; delete duplicates

def dedupe(path):

    ptrn = re.compile(r'''(?P<channel>[a-zA-z0-9_]+)\s-\s(Twitch|game)''')

    tol = 600
    strmrs = defaultdict(list)
    i = 0
    for file in os.listdir(path):

        if ptrn.search(file) is None:
            print(file)
            continue
        chan = ptrn.search(file).group('channel')
        strmrs[chan].append(((path+file), os.path.getmtime(path+file)))

    for chan, lst in strmrs.items():
        benchmark = lst[0][1]
        for pair in lst:
            if abs(pair[1] - benchmark) > tol:
                os.remove(pair[0])
