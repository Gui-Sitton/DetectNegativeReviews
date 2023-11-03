# DetectNegativeReviews

## Intro
The Film Junky Union, a new community for classic film enthusiasts, is developing a system to filter and categorize film reviews. The aim is to train a model to automatically detect negative reviews. You will use a dataset of movie reviews from IMDB with polarity labeling to create a model to classify reviews as positive and negative. It will need to achieve an F1 value of at least 0.85.

### Data description
The data was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng and Christopher Potts. (2011). Word learning vectors for sentiment analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

**Selected fields:**
review: the text of the review
pos: the goal, '0' for negative and '1' for positive
ds_part: 'train'/'test' for the training/testing part of the dataset, respectively
There are other fields in the dataset.

## Libraries used

import math

import re

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

from nltk.corpus import stopwords as nltk_stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression

import nltk

from nltk.corpus import stopwords as nltk_stopwords

nltk.download('stopwords')

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from catboost import CatBoostClassifier

## Conclusion

* Linear regression and LGBM were the best models, while the model 0 performed like a random model.
* Linear regression performed slightly better with the first type of text processing, normalization. With an accuracy of 88% and an f1 value of 0.88. The only difference was the APS value in the test set, from 0.95 with the first model to 0.94 with the second model using other text processing, which obtained a lower f1 value, also resulting in 0.87.
* In my reviews, all the models I tested performed as well as a random model, as I believe the dataframe was too small to work with and I didn't have enough examples, so it resulted in something like 50% accuracy for all the models.
