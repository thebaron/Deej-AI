#! /usr/bin/env python

# This follows MP3toVec logic but does so using as much parallelism as possible.


import argparse
import asyncio
import concurrent.futures
import os
import pickle
import sys
import time
import warnings

import librosa
import keras.backend as K
import numpy as np
import sqlite3
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tqdm import tqdm


def fetch_db(song):
    cursor = db.execute("SELECT tfidf FROM mp3tovecs WHERE filename=?", (song,))
    tfidf =  cursor.fetchone()[0]
    return pickle.loads(tfidf)


def fetch_pickle(song):
    return mp3tovecs[song]


db = sqlite3.connect("mp3tovecs.db")
cursor = db.execute("SELECT filename FROM mp3tovecs")
existing_files = [row[0] for row in cursor.fetchall()]
random_files = np.random.permutation(existing_files)

print(f"{len(existing_files)} songs currently in database.")

mp3tovecs_fullpath = "Pickles" + f'/mp3tovecs/mp3tovecs.p'
mp3tovecs = pickle.load(open(mp3tovecs_fullpath, 'rb'))

print(f"{len(mp3tovecs)} songs currently in pickle jar.")

for n, song in enumerate(random_files):
    # print(f"{song}....")
    if song not in mp3tovecs:
        continue
    else:
        pickle_vec = fetch_pickle(song)
        db_vec = fetch_db(song)
        # print(f"Fetching {song} from pickle.")
        mp3tovecs[song] = fetch_pickle(song)

    if np.array_equal(db_vec, pickle_vec):
        print(f"Song {song} matches.")
    else:
        breakpoint()
        print(f"Song {song} does not match.")
