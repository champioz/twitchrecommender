import pandas as pd
import re
from datetime import datetime
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def date_convert(date: str) -> tuple:
    ptrn = re.compile(
        r'''(?P<weekday>[a-zA-z]+)\s(?P<day>[0-9]{1,2})[a-z]{2}\s(?P<month>[a-zA-Z]+)\s(?P<year>[0-9]{4})(?P<time>\s.+)''')
    weekday, day, month, year, time = ptrn.search(date).groups()
    return (
        weekday,
        datetime.strptime(
            '-'.join([day, month, year]) + time, '%d-%B-%Y %H:%M')
    )


def clean_streams(file):
    file.drop(['Stream URL', 'Unnamed: 0'], axis=1, inplace=True)
    file['intermediate'] = file.apply(
        lambda x: date_convert(x['Stream start time']), axis=1)

    file[['Weekday', 'Start Time']] = pd.DataFrame(
        file['intermediate'].tolist(), index=file.index)
    file.drop(['intermediate'], axis=1, inplace=True)

    file.sort_values(by=['Start Time'], inplace=True)

    file['Time since'] = file['Start Time'].diff(
        periods=-1).values.astype('timedelta64[m]')

    avg_gap = file['Time since'].mean()
    avg_length = file['Stream length (mins)'].mean()
    avg_gain = file['Followers gained'].mean()

    daycounts = Counter(file['Weekday'].tolist())
    total = file.shape[0]
    day_props = {}
    days = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        if day in daycounts.keys():
            day_props[day] = daycounts[day] / total
        else:
            day_props[day] = 0

    file['Games'] = file['Games'].apply(lambda x: x.split(','))
    file['Variety'] = file['Games'].apply(len)

    avg_variety = file['Variety'].mean()
    return (abs(avg_gap), avg_length, avg_gain, avg_variety, day_props)
