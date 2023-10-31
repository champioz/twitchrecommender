from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import chromedriver_autoinstaller
import string
import re
import sys
import os
import time
import json
import numpy as np
import pandas as pd

np.random.seed(13)


def gen_candidates(path):

    done = []
    for file in os.listdir('./data/channel_stats'):
        chan = file.split('.')[0]
        done.append(chan)

    print(len(done))
    all_chan = pd.read_csv(
        path, names=['User', 'Stream', 'Channel', 'Start', 'End'])['Channel'].unique()

    print(len(all_chan))
    all_chan = [chan for chan in all_chan if chan not in done]
    print(len(all_chan))
    return np.random.choice(all_chan, size=3000, replace=False)


def is_active(button):
    classes = button.get_attribute('class')
    if 'disabled' in classes.split(' '):
        return False
    else:
        return True


def strip_nonalph(txt: str):
    ptrn = re.compile('[\W_]+')
    return ptrn.sub('', txt)


def to_age(txt):
    ptrn = re.compile(
        r'''(?P<day>[0-9]{1,2})[thrdnd]{2}\s(?P<month>[a-zA-Z]+)\s(?P<year>[0-9]{4})''')

    day, month, year = ptrn.search(txt).groups()

    asdate = datetime.strptime(' '.join([day, month, year]), '%d %b %Y')
    global present

    return ((present - asdate).total_seconds()) / 60


def get_stats(driver, stats, chan):

    driver.execute_script(f"window.open('{stats}', 'new_window')")
    driver.switch_to.window(driver.window_handles[1])
    time.sleep(np.random.randint(4, 6))

    recent_stats = driver.find_elements(By.CLASS_NAME, 'InfoStatPanelTLCell')

    attributes = driver.find_elements(
        By.CLASS_NAME, 'MiddleSubHeaderItemValue')

    stat_log = {}

    stat_log['channel'] = chan
    stat_log['avg_viewers'] = int(strip_nonalph(recent_stats[0].text))
    stat_log['followers_gained'] = int(strip_nonalph(recent_stats[2].text))
    stat_log['peak_viewers'] = int(strip_nonalph(recent_stats[3].text))
    stat_log['hours_streamed'] = int(strip_nonalph(recent_stats[4].text))
    stat_log['streams'] = int(strip_nonalph(recent_stats[5].text))
    stat_log['mature'] = int(attributes[3].text == 'Yes')
    stat_log['language'] = attributes[4].text
    stat_log['account_age'] = to_age(attributes[5].text)

    driver.close()

    with open(f'./data/channel_stats/{chan}.txt', 'w') as f:
        f.write(json.dumps(stat_log))

    return stat_log


def get_one_streamer(driver, streams):
    driver.execute_script(f"window.open('{streams}', 'new_window')")
    driver.switch_to.window(driver.window_handles[1])
    time.sleep(10)

    csv_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'buttons-csv'))
    )
    csv_button.click()

    i = 0
    while is_active(driver.find_element(By.ID, 'tblControl_next')):
        if i > 15:
            break

        time.sleep(3)

        driver.find_element(By.ID, 'tblControl_next').click()

        time.sleep(7)

        csv_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'buttons-csv'))
        )
        csv_button.click()
        i += 1

    time.sleep(3)
    driver.close()
    return


# Set up working environment
os.chdir('C:\\Users\\prodi\\Documents\\_school\\isye-6740\\proj\\_working')
ERRORLOG = open('./data/LOG.txt', 'w')
COMPLETED = open('done.csv', 'w')
todo = gen_candidates('./data/100k_a.csv')


# Set up web scraper
chromedriver_autoinstaller.install()
PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(executable_path=PATH)


# Set up default open tab and visibility options
driver.get('https://sullygnome.com/channels/2019july/mostfollowers')
url = 'https://sullygnome.com/channel/'
present = datetime.strptime('1 Aug 2019', '%d %b %Y')
allstats = []
# Set visible results to 100
time.sleep(5)
driver.find_element(By.XPATH,
                    "//select[@name='tblControl_length']/option[text()='100']").click()

time.sleep(5)

# Collect each channel's stream and game history
for channel in todo:

    stats = url + channel + '/2019july'
    streams = stats + '/streams'
    games = stats + '/games'

    window_before = driver.window_handles[0]

    try:
        allstats.append(get_stats(driver, stats, channel))
        driver.switch_to.window(window_before)
    except Exception as e:
        ERRORLOG.write(f'{channel}\n')
        ERRORLOG.flush()
        time.sleep(np.random.randint(3, 6))
        continue

    try:
        get_one_streamer(driver, streams)
        driver.switch_to.window(window_before)
    except Exception as e:
        ERRORLOG.write(f'{channel}\n')
        ERRORLOG.flush()
        time.sleep(np.random.randint(3, 5))
        continue

    time.sleep(np.random.randint(3, 5))

    try:
        get_one_streamer(driver, games)
        driver.switch_to.window(window_before)
    except Exception as e:
        ERRORLOG.write(f'{channel}\n')
        ERRORLOG.flush()
        time.sleep(np.random.randint(3, 5))
        continue

    COMPLETED.write(f'{channel}\n')
    COMPLETED.flush()
    time.sleep(np.random.randint(3, 5))

driver.quit()
ERRORLOG.close()


pd.DataFrame(allstats).to_csv('./data/stats.csv')
