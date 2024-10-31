import jax.numpy as jnp
import pandas as pd
import numpy as np


# Decorrelate data
def zca_whitening(X):
    """
    Perform ZCA whitening on a dataset using JAX.
    :param X: The T x N dataset where T is the number of samples and N is the number of features.
    :return: ZCA whitened dataset.
    """
    # Step 1: Center the data (subtract the mean)
    X_mean = jnp.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix
    cov_matrix = jnp.cov(X_centered, rowvar=False)

    # Step 3: Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

    # Step 4: Form the ZCA whitening matrix
    epsilon = 1e-5  # Small constant to prevent division by zero
    D = jnp.diag(1.0 / jnp.sqrt(eigenvalues + epsilon))
    whitening_matrix = eigenvectors @ D @ eigenvectors.T

    # Step 5: Apply ZCA whitening
    X_zca = X_centered @ whitening_matrix

    return X_zca, whitening_matrix, X_mean



# Data reading and preparation
def generate_weather_data(delay_pred=24, split_indx=90000, decorrelation=False, path=''):
    '''
        Hourly climatological data of the Szeged, Hungary area, between 2006 and 2016.
        https://www.kaggle.com/datasets/budincsevity/szeged-weather/data
        Pressure variable has large outliers (usual values are ~1000, outliers are 0)
    '''

    # Read data
    data = pd.read_csv(str(path)+'weatherHistory.csv')
    data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Pressure (millibars)']].to_numpy().astype('float32')

    # set Wind Bearing to sine and cosine
    data = np.hstack([data[:,:3], np.sin(2*np.pi*data[:, 3]/360)[:, np.newaxis], np.cos(2*np.pi*data[:, 3]/360)[:, np.newaxis], data[:,-1][:, np.newaxis]])
    features_names = ['Temperature (C)', 'Relative humidity', 'Wind speed (km/h)', 'Sine wind bearing', 'Cosine wind bearing', 'Pressure (millibars)']

    # Remove outliers
    data[data[:, -1] == 0.0, -1] = jnp.nan
    df = pd.DataFrame(data)
    data = jnp.array(df.interpolate(method='linear', axis=0, limit_direction='both').to_numpy())

    # ZCA decorrelate data
    if decorrelation:
        data, W_zca, X_mean = zca_whitening(data)

    # Calculate mean and standard deviation
    data_mean = jnp.nanmean(data, axis=0)
    data -= data_mean
    data_std = jnp.nanstd(data, axis=0)
    data /= data_std

    # # Normalize data
    # if not decorrelation:



    # Generate labels
    features = data[:-delay_pred]
    labels = data[delay_pred:, 0]

    # Split train and test
    x_train = features[:split_indx]
    y_train = labels[:split_indx]
    x_test = features[split_indx:]
    y_test = labels[split_indx:]

    return (x_train, y_train), (x_test, y_test), features_names, data_mean, data_std
