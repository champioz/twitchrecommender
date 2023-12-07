# A recommendation system for Twitch livestreams using community detection

_This project outlines efforts to build and evaluate a recommendation system using several different unsupervised community detection algorithms. For a longer writeup, see the [report](https://github.com/champioz/twitchrecommender/blob/main/report.pdf)._

## Problem
Twitch.tv is an entertainment livestreaming website where channels broadcast in a variety of categories -- like games, music, sports, and art. These categories present an opportunity to identify channels that are similar to one another. I want to test whether a large historical database of past broadcasts on Twitch.tv can be used to create a robust recommendation system for future viewers. 

## My solution
I gathered this historical data manually from [SullyGnome](https://sullygnome.com/) using Python and Selenium, cleaned the results into a large matrix representing the proportion of a channel's overall bradcast time spent on each category, and evaluated several different unsupervized community detection techniques to label groups of similar channels. I then trained and evaluated a recommender system using [a dataset of past user viewing habits](https://cseweb.ucsd.edu/~jmcauley/datasets.html).

## 1) Data Collection

### A) Web scraping for training data

[utils/web_scraping.py](https://github.com/champioz/twitchrecommender/blob/main/utils/web_scraping.py): A script using Selenium to record key channel metrics and store CSV files of historical, channel-specific broadcast data from [SullyGnome](https://sullygnome.com/). Selects a random subset of channels present in the evaluation data.

This data was used to detect communities of similar channels to be used for recommendation generation.

### B) Research evaluation dataset

Recommender evaluations were performed using Julian McAuley's (UCSD) "Recommender Systems and Personalization Datasets: Twitch," available [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html).

## 2) Data Preprocessing

### A) Cleaning corrupted and incomplete data

[utils/clean_unprocessed_files.py](https://github.com/champioz/twitchrecommender/blob/main/utils/clean_unprocessed_files.py): A preliminary data cleaning step that checks the directory of web-scraped data for errors or corrupted/empty files, and deletes all files associated with channels that encountered any such errors. This prevents a channel with faulty or incomplete data from being included in the cleaned data set. Channels affected are logged for attempted re-collection.

### B) Transforming thousands of historic broadcast CSVs into a high-dimensional adjacency matrix

[utils/twitch_preprocess.py](https://github.com/champioz/twitchrecommender/blob/main/utils/twitch_preprocess.py): Primary file for data cleaning. Combines all collected CSV files into a master adjacency matrix, containing a row for every channel in the data set and a column for every game or category. Each intersection indicates the proportion of total broadcast hours that channel dedicated to that game or category. Also generates a channel statistics dataset, cleaned and standardized to convert all dates to numerical time data (e.g., account creation date converted to account age in minutes).

[utils/clean_streams_data.py](https://github.com/champioz/twitchrecommender/blob/main/utils/clean_streams_data.py): A helper file that performs data normalization described above, converting dates into numerical data and removing extraneous columns.

## 3) Community Detection

[sparsify.py](https://github.com/champioz/twitchrecommender/blob/main/sparsify.py): Generates sparse matrices (stored as numpy NPZ files) split into testing and training data. Splits evaluation dataset into train/test.

[community.py](https://github.com/champioz/twitchrecommender/blob/main/community.py): Tunes and evaluates spectral clustering, the Louvain algorithm, and OPTICS methods for detecting communities on the sparse channel/game data. Generates adjacency matrices using both Euclidean and Jaccard distances and trials all three algorithms on data using each distance metric. 

## 4) Recommender System and Evaluation

[recommender.py](https://github.com/champioz/twitchrecommender/blob/main/recommender.py): Loads community data generated using all six unique combinations of clustering algorithms and distance metrics, as well as the evaluation dataset split into train/test sets. Generates recommendations based on a user's past viewing and the community of similar channels. Ranks final recommendations using a calculated score of channel quality, as well as using a weighted random distribution based on past viewing preferences for comparison. The two ranking approaches are compared in the final results, along with the community detection algorithms and distance metrics. Collects diagnostics of the full system based on accuracy (on both new and repeat channels in the user's future viewing behavior) and novelty of recommendations.

[random_baseline.py](https://github.com/champioz/twitchrecommender/blob/main/random_baseline.py): Creates and evaluates a control recommender that naively samples from the entire population of available channels to generate its recommendation candidates.



