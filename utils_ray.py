import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_columns(df, columns):
    """
    Scales specified columns in the dataframe using StandardScaler.

    :param df: Pandas DataFrame.
    :param columns: List of column names to scale.
    :return: DataFrame with scaled columns.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return scaler, df


def create_windows(df, window_size=3, skip=1):
    """
    Creates windows of data for each segment.

    :param df: DataFrame with the columns 'lw_x', 'lw_y', 'lw_z', 'activity', 'segment'.
    :param window_size: Size of each window.
    :param skip: Number of rows to skip for the next window.
    :return: Tuple of numpy arrays (X, y).
    """
    X, y = [], []
    unique_segments = np.unique(df['segment'])

    # Convert required columns to NumPy arrays once, outside the loop
    lw_cols = df[['lw_x', 'lw_y', 'lw_z']].to_numpy()
    activities = df['activity'].to_numpy()

    for segment in unique_segments:
        segment_indices = np.flatnonzero(df['segment'].to_numpy() == segment)
        max_index = segment_indices[-1]

        # Use array slicing instead of DataFrame indexing in the loop
        for start in segment_indices:
            end = start + window_size
            if end <= max_index + 1:
                X.append(lw_cols[start:end])
                y.append(activities[end - 1])
                start += skip

    return np.array(X), np.array(y)

def append_segments(df):
    df['time_diff'] = df['time_ms'].diff()

    # Initialize the segment number
    segment = 0
    segments = []

    # Iterate over the time_diff column to assign segment numbers
    for diff in df['time_diff']:
        if diff > 1:
            segment += 1
        segments.append(segment)

    # Add the segment numbers to the DataFrame
    df['segment'] = segments
    return df
    