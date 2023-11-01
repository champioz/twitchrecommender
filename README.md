# Twitch Recommender

This project outlines efforts to build and evaluate a recommendation system using several different unsupervised community detection algorithms. 
See the [report](https://github.com/champioz/twitchrecommender/blob/main/report.pdf) for a full description of the background research, methodology, and results.

_Please note that the data folder is left blank on this repository. Evaluations were performed using Julian McAuley's (UCSD) "Recommender Systems and Personalization_
_Datasets: Twitch," available [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html)._

## Utils

This folder contains four scripts used for data collection and preparation.

### web_scraping.py

A script using Selenium to save CSV files of historical, channel-specific stats from SullyGnome. Selects a random subset of channels present in the 
evaluation data.

### clean_unprocessed_files.py

A preliminary data cleaning step that checks the resulting directory for errors or corrupted files, and deletes all files associated with a channel
if they encountered any such errors. This prevents a channel with faulty or incomplete data from being included in the cleaned data set. Channels
affected are saved for attempted re-collection.

### twitch_preprocess.py

Master file for data cleaning. Combines all collected CSV files into a master adjacency matrix, containing a row for every channel in the data set and
a column for every game. The intersection indicates the proportion of total streamed hours each channel dedicated to that game. Also generates a channel
statistics dataset, cleaned and standardized to convert all dates to numerical time data (e.g., account creation date converted to account age in minutes).

### clean_streams_data.py

A helper file that performs data normalization described above, converting dates into numerical data and removing extraneous columns.

## Main files

Files used to a) train and evaluate community detection algorithms b) build and evaluate the final recommender system pipeline.

### sparsify.py

Generates sparse matrices (stored as numpy NPZ files) split into testing and training data.

### community.py 

Tunes and evaluates spectral clustering, Louvain, and OPTICS models to detect communities on the sparse channel/game data. Generates adjacency matrices
using both Euclidean and Jaccard distances and trials all three algorithms on data using each distance metric. 

### recommender.py

Loads community data generated with all six unique combinations of clustering algorithm and distance metric, as well as test/train evaluation data. Generates
recommendations based on past viewing and the community of similar channels. Ranks final recommendations using both a generated score of channel
quality, and a weighted random distribution based on past viewing preferences. The two ranking approaches are compared in the final results, along with the
community detection algorithms and distance metrics. Collects diagnostics of the full system based on accuracy (on both new and repeat channels in the user's
future viewing behavior) and novelty of recommendations.

### rank_score.py

A helper file that generates scores of a channel's quality based on its past streaming regularity and popularity growth.

### random_baseline.py

Evaluates a control recommender that naively samples from the entire population of available channels to generate its recommendation candidates.



