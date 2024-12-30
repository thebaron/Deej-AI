#! /usr/bin/env python

# This follows MP3toVec logic but does so using parallelism and stores the
# results in a SQLite database.

import argparse
import os
import pickle
import sys
import time

# mute the librosa warning spam
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import librosa
import numpy as np
import sqlite3
import tensorflow as tf

from contextlib import closing, redirect_stderr, redirect_stdout
from multiprocessing import Lock, Pool, set_start_method
from tensorflow import keras
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tqdm.contrib.concurrent import ensure_lock
from typing import Optional


class Batch:
    """
    Batch class for calculating TF-IDF weights.
    """

    def __init__(
        self,
        batch_number: int,
        total_threads: int,
        epsilon_distance: float,
    ):
        self.batch_number = batch_number
        self.epsilon_distance = epsilon_distance
        self.total_threads = total_threads

    @staticmethod
    def starlaunch(args):
        return Batch.launch(*args)

    @staticmethod
    def launch(
        batch_number: int,
        total_threads: int,
        paths: list,
        epsilon_distance: float,
        duration: Optional[float] = None,
    ) -> tuple:
        """
        Static method to launch a batch of TF-IDF calculations.
        """
        b = Batch(batch_number, total_threads, epsilon_distance, )
        return batch_number, b.run(paths,)

    def run(
        self,
        paths: list,
    ) -> bool:
        """
        Runs the TF-IDF calculations for the batch and updates the database accordingly.
        """
        self.progress_bar = tqdm(
            position=1 + (self.batch_number - 1) % self.total_threads,
            maxinterval=1.0,
            desc=f"Batch {self.batch_number} - start",
            leave=False,
        )

        track2vecs = self.load_track2vec(paths)
        indices, vecs = self.normalize_track2vec(track2vecs)
        distances = self.cosine_distances(vecs)
        idfs = self.idf_weights(paths, indices, vecs, distances)
        tf_idfs = self.tf_weights(paths, indices, vecs, idfs, distances)

        self.progress_bar.set_description("Updating database")
        self.progress_bar.unit = "entry"
        self.progress_bar.reset(total=len(track2vecs) + len(tf_idfs))
        self.progress_bar.update(0)

        with closing(sqlite3.connect(database_path, timeout=10)) as db:
            db.execute("BEGIN TRANSACTION")
            for path in track2vecs:
                db.execute(
                    "UPDATE mp3tovecs SET track2vec=? WHERE filename=?",
                    (
                        pickle.dumps(track2vecs[path]),
                        path,
                    ),
                )
                self.progress_bar.update(1)
            for path in tf_idfs:
                db.execute(
                    "UPDATE mp3tovecs SET tfidf=? WHERE filename=?",
                    (
                        pickle.dumps(tf_idfs[path]),
                        path,
                    ),
                )
                self.progress_bar.update(1)
            db.commit()

        del track2vecs, indices, vecs, distances, idfs, tf_idfs

        self.progress_bar.close()
        return True

    # Load the audio file, convert it to a mel spectrogram, and slice it into chunks
    # that can be fed into the model. Return the melspectrogram chunks.
    def spectrum(self, path: str, duration: Optional[float] = None) -> np.ndarray:

        # audio parameters
        sr_in = 22050
        hop_length = 512
        n_fft = 2048
        n_mels = model.input_shape[1]
        slice_size = model.input_shape[2]

        try:
            # try to keep librosa as quiet as possible when it hits
            # weird files
            if quiet:
                with open(os.devnull, "w") as null_device:
                    with redirect_stdout(null_device):
                        with redirect_stderr(null_device):
                            y, sr = librosa.load(
                                path, mono=True, sr=sr_in, duration=duration
                            )
            else:
                y, sr = librosa.load(path, mono=True, sr=sr_in, duration=duration)

            if y.shape[0] < slice_size:
                return None

            # calculate slice_time based on the sr result
            slice_time = slice_size * hop_length / sr

            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=sr / 2,
            )
            x = np.ndarray(
                shape=(S.shape[1] // slice_size, n_mels, slice_size, 1),
                dtype=float,
            )
            for slice in range(S.shape[1] // slice_size):
                log_S = librosa.power_to_db(
                    S[:, slice * slice_size : (slice + 1) * slice_size],
                    ref=np.max,
                )

                # normalize db levels
                dd = np.max(log_S) - np.min(log_S)
                if dd != 0:
                    log_S = (log_S - np.min(log_S)) / dd
                x[slice, :, :, 0] = log_S
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return None

        with tf.device(device):
            p = model.predict(x, verbose=0)

        return p

    # Load the track2vec data from the database, or calculate it if it doesn't exist.
    # Returns a dictionary of track2vec data.
    def load_track2vec(self, paths: list):
        track2vecs = {}
        self.progress_bar.set_description(
            f"Batch {self.batch_number} - model data (1/5)"
        )
        self.progress_bar.reset(total=len(paths))
        self.progress_bar.unit = "track"
        with closing(sqlite3.connect(database_path, timeout=10)) as db:
            for path in paths:
                try:
                    with closing(
                        db.execute(
                            "SELECT track2vec FROM mp3tovecs WHERE filename = ?",
                            (path,),
                        )
                    ) as cursor:
                        t2v = cursor.fetchone()
                        try:
                            track2vecs[path] = pickle.loads(t2v[0])
                        except:
                            pass

                        if track2vecs.get(path) is None:
                            track2vecs[path] = self.spectrum(path)
                    self.progress_bar.update(1)
                except Exception as e:
                    track2vecs[path] = None

        return track2vecs

    # Normalize the track2vec data and split it into a set of path-mapped indices
    # which index into an array of vectors.
    def normalize_track2vec(self, track2vecs: dict):
        indices = {}
        vecs = []

        self.progress_bar.set_description(
            f"Batch {self.batch_number} - normalize data (2/5)"
        )
        self.progress_bar.reset(total=len(track2vecs))
        self.progress_bar.unit = "track"

        for path in track2vecs:
            vec = track2vecs[path]
            if vec is None:
                continue

            indices[path] = []
            for v in vec:
                indices[path].append(len(vecs))
                norm = np.linalg.norm(v)
                if norm == 0:
                    norm = 1  # if somehow we would divide by zero, just set it to 1
                vecs.append(v / norm)
            self.progress_bar.update(1)

        return indices, vecs

    # This, like MP3ToVec's implementation, takes a lot of memory and
    # needs speeding up, but at least it runs in parallel to other tasks.
    def cosine_distances(self, vecs: list):
        num_vecs = len(vecs)
        self.progress_bar.set_description(
            f"Batch {self.batch_number} - cosine distances (3/5)"
        )
        self.progress_bar.reset(total=num_vecs)
        self.progress_bar.unit = "vector"

        cos_distances = np.zeros((num_vecs, num_vecs), dtype=np.float16)
        try:
            for i, vec_i in enumerate(vecs):
                for j, vec_j in enumerate(vecs):
                    if i < j:
                        cos_distances[i, j] = 1 - np.dot(vec_i, vec_j)
                self.progress_bar.update(1)

            cos_distances = (
                cos_distances + cos_distances.T - np.diag(np.diag(cos_distances))
            )  # Make matrix symmetrical diagonally
        except KeyboardInterrupt:
            raise

        return cos_distances

    # This calculates the inverse document frequency weights for the vectors. It returns
    # a list of idf weights, one for each vector input.
    def idf_weights(
        self, paths: list, indices: dict, vecs: list, cos_distances: np.ndarray
    ):
        num_vecs = len(vecs)
        num_indices = len(indices)
        num_paths = len(paths)

        self.progress_bar.set_description(
            f"Batch {self.batch_number} - idf weights (4/5)"
        )
        self.progress_bar.reset(total=num_vecs)
        self.progress_bar.unit = "vector"

        idfs = []
        try:
            for i in range(num_vecs):
                idf = 0
                for song in paths:
                    if song not in indices:
                        self.progress_bar.update(1)
                        continue
                    for j in indices[song]:
                        if cos_distances[i, j] < self.epsilon_distance:
                            idf += 1
                            break
                idfs.append(-np.log(idf / num_paths))
                self.progress_bar.update(1)
        except KeyboardInterrupt:
            raise

        return idfs

    # This calculates the term frequency weights for the vectors. It returns a dictionary
    # of tf-idf weights, one for each song path input, provided the song path is in the
    # indices dictionary.
    def tf_weights(
        self,
        paths: list,
        indices: dict,
        vecs: list,
        idfs: list,
        cos_distances: np.ndarray,
    ):
        num_paths = len(paths)

        self.progress_bar.set_description(
            f"Batch {self.batch_number} - tf-idf weights (5/5)"
        )
        self.progress_bar.reset(total=num_paths)
        self.progress_bar.unit = "paths"

        tf_idfs = {}
        try:
            for song in paths:
                if song not in indices:
                    self.progress_bar.update(1)
                    continue
                tfidf = 0
                for i in indices[song]:
                    weight = 0
                    for j in indices[song]:
                        if cos_distances[i, j] < self.epsilon_distance:
                            weight += 1
                    tfidf += vecs[i] * weight * idfs[i]

                tf_idfs[song] = tfidf
                self.progress_bar.update(1)
        except KeyboardInterrupt:
            raise

        return tf_idfs


# Walk the directory path giving off filenames and their absolute paths
def walk_mp3s(folder):
    for dirpath, dirs, files in os.walk(folder, topdown=False):
        for filename in files:
            if filename.lower().endswith((".flac", ".mp3", ".m4a")):
                yield os.path.abspath(os.path.join(dirpath, filename))


# Print the model and database summary
def info(model_path, database_path, device):

    with tf.device(device):
        model = load_model(
            model_path,
            custom_objects={"cosine_proximity": tf.compat.v1.keras.losses.cosine_proximity},
        )
        model.make_predict_function()

    print(f"Model Summary for model at {model_path}")
    print(model.summary())

    print(f"\nDatabase Summary for database at {database_path}")
    with closing(sqlite3.connect(database_path, timeout=10.0)) as db:
        print(
            db.execute("SELECT COUNT(*) FROM mp3tovecs").fetchone()[0],
            "known audio files.",
        )
        print(
            db.execute(
                "SELECT COUNT(*) FROM mp3tovecs WHERE track2vec IS NOT NULL"
            ).fetchone()[0],
            "vectorized audio files.",
        )
        print(
            db.execute(
                "SELECT COUNT(*) FROM mp3tovecs WHERE tfidf IS NOT NULL"
            ).fetchone()[0],
            "with calculated term frequency weights.",
        )


# Used to initialize a pool with global lock and some other variables that seem pointless to pass around
def initialize_pool(
    l: Lock,
    d: str,
    model_path: str,
    q: bool,
    dev: str,
):
    global lock, database_path, model, quiet, device

    lock = l
    ensure_lock(tqdm, lock_name="mp_lock")
    database_path = d
    quiet = q
    device = dev

    with tf.device(device):
        model = load_model(
            model_path,
            custom_objects={"cosine_proximity": tf.compat.v1.keras.losses.cosine_proximity},
        )
        model.make_predict_function()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of media files to process in each batch",
    )
    parser.add_argument(
        "--epsilon-distance", type=float, default=0.001, help="Epsilon distance"
    )
    parser.add_argument(
        "--database-path", type=str, default="mp3tovecs.db", help="Path to database"
    )
    parser.add_argument(
        "--info", action="store_true", help="Prints the database and model's summary"
    )
    parser.add_argument("--media", type=str, help="Directory of music media to scan")
    parser.add_argument(
        "--model-path", type=str, default="speccy_model", help="Path to trained model"
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=None,
        help="Maximum duration of media samples used for model (seconds)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress excess tensorflow and librosa output",
    )
    parser.add_argument(
        "--tensorflow-device",
        type=str,
        default="cpu:0",
        help="Device which runs tensorflow model operations (default: cpu:0)",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=5,
        help="Number of threads to run concurrently",
    )

    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError("Must specify model file path.")

    if args.quiet:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    with closing(sqlite3.connect(args.database_path, timeout=10.0)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS mp3tovecs (filename TEXT PRIMARY KEY, track2vec BLOB DEFAULT NULL, tfidf BLOB DEFAULT NULL)"
        )

    if args.info:
        # just for parity
        info(args.model_path, args.database_path, args.tensorflow_device)
        return

    if args.media is None:
        raise ValueError("No media directory provided to be scanned.")

    with closing(tqdm(desc="Scanning media", total=0, unit="file")) as progress_bar:
        with closing(sqlite3.connect(args.database_path)) as db:
            for path in walk_mp3s(args.media):
                try:
                    db.execute(
                        "INSERT OR IGNORE INTO mp3tovecs (filename) VALUES (?)",
                        (path,),
                    )
                    progress_bar.total += 1
                    progress_bar.update(1)
                except KeyboardInterrupt:
                    raise
            db.commit()

    with closing(sqlite3.connect(args.database_path, timeout=10.0)) as db:
        with closing(
            db.execute(
                "SELECT filename FROM mp3tovecs WHERE tfidf IS NULL OR track2vec IS NULL ORDER BY RANDOM()"
            )
        ) as cursor:
            vecs_needed = [row[0] for row in cursor.fetchall()]
            num_needed = len(vecs_needed)

    lock = Lock()
    if num_needed != 0:
        batch_count = 1 + num_needed // args.batch_size

        batches = [
            (
                i + 1,
                args.thread_count,
                vecs_needed[i * args.batch_size : (i + 1) * args.batch_size],
                args.epsilon_distance,
                args.sample_duration,
            )
            for i in range(batch_count)
        ]

        if args.thread_count == 1:
            initialize_pool(
                lock,
                args.database_path,
                args.model_path,
                args.quiet,
                args.tensorflow_device,
            )
            for b in tqdm(batches, desc="Processing batches", unit="batch"):
                Batch.starlaunch(b)
        else:
            with Pool(
                processes=args.thread_count,
                initializer=initialize_pool,
                initargs=(
                    lock,
                    args.database_path,
                    args.model_path,
                    args.quiet,
                    args.tensorflow_device,
                ),
            ) as pool:
                for _ in tqdm(
                    pool.imap(Batch.starlaunch, batches, chunksize=1),
                    total=batch_count,
                    desc="Processing batches",
                    unit="batch",
                    maxinterval=1.0,
                ):
                    pass


if __name__ == "__main__":
    # make CUDA happy
    set_start_method("spawn")
    main()
