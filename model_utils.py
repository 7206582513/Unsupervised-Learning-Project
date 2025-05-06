
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

def build_autoencoder(input_dim, encoding_dim=5):
    input_layer = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    x = Dense(16, activation='relu')(encoded)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

def process_and_cluster(filepath):
    df = pd.read_csv(filepath)
    df['balance_utilization'] = df['current_balance'] / (df['credit_limit'] + 1e-5)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    autoencoder, encoder = build_autoencoder(X_pca.shape[1])
    X_train, X_val = train_test_split(X_pca, test_size=0.2, random_state=42)

    autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val),
                    epochs=100, batch_size=32,
                    callbacks=[EarlyStopping(patience=10)], verbose=0)

    encoded_data = encoder.predict(X_pca)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(encoded_data)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(embedding)

    mask = labels != -1
    filtered_embedding = embedding[mask]
    filtered_labels = labels[mask]

    if len(set(filtered_labels)) > 1:
        score = silhouette_score(filtered_embedding, filtered_labels)
    else:
        score = -1

    plot_path = 'static/cluster_plot.png'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=filtered_embedding[:, 0], y=filtered_embedding[:, 1],
                    hue=filtered_labels, palette='viridis', s=100)
    plt.title("HDBSCAN Clusters after Autoencoder + UMAP")
    plt.xlabel("UMAP Feature 1")
    plt.ylabel("UMAP Feature 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    df_result = df.copy()
    df_result['Cluster'] = labels
    return df_result, round(score, 4), plot_path
