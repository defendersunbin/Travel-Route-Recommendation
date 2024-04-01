import pandas as pd
import numpy as np
import stellargraph as sg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, multiply, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec

def load_and_prepare_data():
    route_review = pd.read_csv('/content/tripadvisor_route_review.csv')
    route = pd.read_csv('/content/tripadvisor_route_list.csv')
    temp = route.merge(route_review, left_on='ROUTENAME', right_on='ROUTE_ID')
    return route_review, temp

def calculate_ratings(route_review):
    ratings_count = route_review.groupby('ROUTE_ID')['Rating'].count()
    average_ratings = route_review.groupby('ROUTE_ID')['Rating'].mean()
    return ratings_count, average_ratings

def define_thresholds(average_ratings):
    most_recommended_threshold = average_ratings.quantile(0.75)
    not_recommended_threshold = average_ratings.quantile(0.25)
    return most_recommended_threshold, not_recommended_threshold

def create_stellargraph(temp):
    # Simplified node and edge creation
    nodes = {node_type: temp[node_id].drop_duplicates().reset_index(drop=True).to_frame().set_index(node_id) 
             for node_type, node_id in [('user', 'USER_ID'), ('route', 'ROUTE_ID'), ('place', 'Place')]}
    edges = {
        'user_route': temp[['USER_ID', 'ROUTE_ID']].rename(columns={'USER_ID': 'source', 'ROUTE_ID': 'target'}),
        'route_place': temp[['ROUTE_ID', 'Place']].rename(columns={'ROUTE_ID': 'source', 'Place': 'target'})
    }
    g = sg.StellarDiGraph(nodes=nodes, edges=edges)
    return g

def generate_embeddings(g):
    walk_length = 50
    metapaths = [["user", "route", "place", "route", "user"], ["user", "route", "user"]]
    rw = UniformRandomMetaPathWalk(g)
    walks = rw.run(nodes=list(g.nodes()), length=walk_length, n=10, metapaths=metapaths, seed=42)
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=64, window=5, min_count=0, sg=1, epochs=5)
    return model

def main():
    route_review, temp = load_and_prepare_data()
    ratings_count, average_ratings = calculate_ratings(route_review)
    most_recommended_threshold, not_recommended_threshold = define_thresholds(average_ratings)
    g = create_stellargraph(temp)
    model = generate_embeddings(g)
    # Additional processing and model training steps can be added here

if __name__ == "__main__":
    main()