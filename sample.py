#!/usr/bin/env python
# coding: utf-8

# # 0.Sample Model 

# In[ ]:


from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from streamlit_folium import folium_static
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans 
import folium
import nltk
import nmslib
import numpy as np
import pandas as pd
import plotly.express as px
import re
import streamlit as st
import string


# # 1.Dataset Processing

# In[ ]:


# Function to load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("Information.csv")
    data = data.dropna(subset=["Research Keywords"])
    data = data.rename(columns={"Research Keywords": "Keywords",
                                "Faculty Information": "Information",
                                "Office Room": "Office"})
    return data

# Streamlit app layout
st.title("PI Collaboration Promotion")

# Load data
data = load_data()


# # 2.Textual Data Procesisng

# In[ ]:


nltk.download('wordnet')

# Function for text processing
def process_text(data_frame):
    text = data_frame["Keywords"].apply(lambda x: x.lower())
    text = text.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    text = text.apply(lambda x: re.sub(r"\d+", "num", x))

    stemmer = PorterStemmer()
    text_stem = text.apply(lambda x: stemmer.stem(x))

    lem = nltk.stem.wordnet.WordNetLemmatizer()
    text_lem = text_stem.apply(lambda x: lem.lemmatize(x))
    return text_lem

processed_text = process_text(data)


# # 3.Vectorization

# In[ ]:


# Initialize model_SBERT globally
model_SBERT = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate embeddings
@st.cache_data
def generate_embeddings(text):
    
    embeddings = model_SBERT.encode(text, convert_to_numpy=True)
    return embeddings

embeddings = generate_embeddings(processed_text.tolist())


# # 4.Optimal Number of K-Means

# **4.1.Append the best number of k of each method into a list to reach the final decision.**

# optimal_Ks=[]

# **4.2.The Silhouette score.**

# The Silhouette score is calculated by taking the average silhouette of all data points, where the silhouette of an individual data point is defined as the difference between the average distance of that point to all other points in the same cluster and the average distance of that point to all points in the nearest other clusters, divided by the greater of the two.

# num_clusters=list(range(2,51))
# silhouette_scores=[]
# 
# for k in num_clusters:
#     kmeans=KMeans(n_clusters=k,n_init=10)
#     kmeans.fit(embeddings)
#     labels=kmeans.predict(embeddings)
#     sil_score=silhouette_score(embeddings,labels)
#     print("cluster:",k,"score:",sil_score)
#     silhouette_scores.append(sil_score)
# 
# optimal_num_clusters=num_clusters[silhouette_scores.index(max(silhouette_scores))]
# print("The highest S-score of the k-clusters:",optimal_num_clusters)
# 
# optimal_Ks.append(optimal_num_clusters)

# **4.3.The Davies-Bouldin index.**

# The Davies-Bouldin index calculates a score for each value of k based on the ratio of the WCSS of the clusters to the distance between the centers of the clusters. The optimal number of clusters is the value of k that minimizes the Davies-Bouldin index.

# num_clusters=list(range(2,51))
# dav_scores=[]
# 
# for k in num_clusters:
#     kmeans=KMeans(n_clusters=k)
#     kmeans.fit(embeddings)
#     labels=kmeans.predict(embeddings)
#     dav_score=davies_bouldin_score(embeddings,labels)
#     print("cluster:",k,"value:",dav_score)
#     dav_scores.append(dav_score)
# 
# optimal_num_clusters=num_clusters[dav_scores.index(min(dav_scores))] 
# print("The optimum score of the k-clusters:",optimal_num_clusters) 
# 
# optimal_Ks.append(optimal_num_clusters)

# **4.4.The X-means algorithm.**

# The X-means algorithm uses a recursive splitting process to automatically determine the optimal number of clusters.

# xmeans_instance=xmeans(embeddings,kmeans_plusplus_initializer(embeddings,2).initialize())
# xmeans_instance.process()
# 
# clusters=xmeans_instance.get_clusters()
# print("The identified optimum cluster:",len(clusters))
# 
# optimal_Ks.append(len(clusters))

# **4.5.Dispaly the optimal number of K-means and the median of the suggested clusters.**

# print("The optimal number of K-means:",optimal_Ks)
# print("The median of the suggested clusters:",int(np.median(optimal_Ks)))

# # 5.Clustering

# In[ ]:


# Function for K-Means clustering
def perform_clustering(embeddings, k):
    k_model = KMeans(n_clusters=k, init="k-means++", max_iter=200, n_init=10)
    k_model.fit(embeddings)
    return k_model.labels_

# Determine the number of clusters
optimal_Ks = 40  
true_k = int(np.median([optimal_Ks]))  # If optimal_Ks is a list, else just use it directly

# Perform clustering
labels = perform_clustering(embeddings, true_k)

# Add labels to data
data["Labels"] = pd.Categorical(labels)


# # 6.Model Utilization

# In[ ]:


# Function to initialize and create NMSLIB index
def create_nmslib_index(embeddings):
    model_index = nmslib.init(method="hnsw", space="cosinesimil")
    model_index.addDataPointBatch(embeddings)
    model_index.createIndex({"post": 2})
    return model_index

# Create the NMSLIB index
model_index = create_nmslib_index(embeddings)


# In[ ]:


# Function to find best Principal Investigators
def find_best_PI(data, q_input, k, model_SBERT, model_index):
    best_fits = []
    if data is not None and q_input is not None:
        subset = data.copy()
        query = model_SBERT.encode([q_input], convert_to_tensor=True)
        ids, distances = model_index.knnQuery(query, k)
        
        for i, j in zip(ids, distances):
            best_fits.append({
                "Name": subset.Name.values[i],
                "Title": subset.Title.values[i],
                "Departments": subset.Departments.values[i],
                "Keywords": subset.Keywords.values[i],
                "Information": subset.Information.values[i],
                "Office": subset.Office.values[i],
                "Location": subset.Location.values[i],
                "Website": subset.Website.values[i],
                "Labels": subset.Labels.values[i],
                "Distance": j
            })
            
    return pd.DataFrame(best_fits)


# In[ ]:


# User input for keywords
keywords = st.text_input("Enter your research keywords or index keywords: <br>Blue Pinpoints: PIs doing complementary work. <br>Blue Pinpoints: PIs doing complementary work. <br>Blue Pinpoints: PIs doing complementary work.")

# Initialize variables
similar_results = pd.DataFrame()
complementary_results = pd.DataFrame()

if keywords:
    most_similar = find_best_PI(data, keywords, 10, model_SBERT, model_index)

    # Process to find most frequent label
    most_frequent_label = int(most_similar["Labels"].mode()[0])
    
    # Separate similar and complementary results
    similar_results = most_similar[most_similar["Labels"] == most_frequent_label]
    complementary_results = most_similar[most_similar["Labels"] != most_frequent_label]


# # 7.Visualization

# In[ ]:


# Initialize the Geolocator
geolocator = Nominatim(user_agent="my_geocoding_project")

# Function to geocode address
def geocode(address):
    address = f"{address}, Baltimore, MD"
    location = geolocator.geocode(address)
    
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

if 'Location' in similar_results.columns and 'Location' in complementary_results.columns:
    similar_results["latitude"], similar_results["longitude"] = zip(*similar_results["Location"].apply(geocode))
    complementary_results["latitude"], complementary_results["longitude"] = zip(*complementary_results["Location"].apply(geocode))
    
# Function to adjust coordinates
def adjust_coords(row, seen_coords):
    lat, long = row["latitude"], row["longitude"]
    while (lat, long) in seen_coords:
        lat += 0.0001
        long += 0.0001
    seen_coords.add((lat, long))
    return lat, long

if 'latitude' in similar_results.columns and 'longitude' in similar_results.columns:
    seen_coords = set()
    similar_results["latitude"], similar_results["longitude"] = zip(*similar_results.apply(adjust_coords, axis=1, args=(seen_coords,)))
    complementary_results["latitude"], complementary_results["longitude"] = zip(*complementary_results.apply(adjust_coords, axis=1, args=(seen_coords,)))


# In[ ]:


# Create the Folium map
m = folium.Map(location=[39.2904, -76.6122], zoom_start=12)

# Function to add a title to the map
def add_title_to_map(map_object, title_text, subtitle_text):
    title_html = '''
    <div style="position: fixed; 
                bottom: 10px; 
                left: 50%;
                transform: translate(-50%, 0);
                z-index: 9999;
                padding: 6px;
                border-radius: 10px;
                background-color: rgba(255,255,255,0.8);
                width: auto;">
        <h3 style="font-size:20px; text-align: center;"><b>{}</b></h3>
        <h4 style="font-size:16px; text-align: left;">{}</h4>
    </div>
    '''.format(title_text, subtitle_text.replace(". Blue", ".<br>Blue"))

    map_object.get_root().html.add_child(folium.Element(title_html))

# Add title to the map
title_text = "Closest 10 PIs"
subtitle_text = "Red: PIs doing similar work. Blue: PIs doing complementary work."
add_title_to_map(m, title_text, subtitle_text)


# In[ ]:


# Add markers for similar results
for idx, row in similar_results.iterrows():
    if not pd.isnull(row["latitude"]) and not pd.isnull(row["longitude"]):
        popup_text = """
                     <strong>Name:</strong> {name}<br>
                     <strong>Title:</strong> {title}<br>
                     <strong>Departments:</strong> {departments}<br>
                     <strong>Keywords:</strong> {keywords}<br>
                     <strong>Office:</strong> {office}<br>
                     """
        popup_text = popup_text.format(
            name=row["Name"],
            title=row["Title"],
            departments=row["Departments"],
            keywords=row["Keywords"],
            office=row["Office"]
        )
        if not pd.isnull(row["Information"]):
            popup_text += "<strong>Information:</strong> {info}<br>".format(info=row["Information"])
        if not pd.isnull(row["Website"]):
            popup_text += '<strong>Website:</strong> <a href="{website}" target="_blank">Click here</a>'.format(website=row["Website"])

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=800),
            icon=folium.Icon(color="red"),
        ).add_to(m)
        
# Add markers for complementary results
for idx, row in complementary_results.iterrows():
    if not pd.isnull(row["latitude"]) and not pd.isnull(row["longitude"]):
        popup_text = """
                     <strong>Name:</strong> {name}<br>
                     <strong>Title:</strong> {title}<br>
                     <strong>Departments:</strong> {departments}<br>
                     <strong>Keywords:</strong> {keywords}<br>
                     <strong>Office:</strong> {office}<br>
                     """
        popup_text = popup_text.format(
            name=row["Name"],
            title=row["Title"],
            departments=row["Departments"],
            keywords=row["Keywords"],
            office=row["Office"]
        )
        if not pd.isnull(row["Information"]):
            popup_text += "<strong>Information:</strong> {info}<br>".format(info=row["Information"])
        if not pd.isnull(row["Website"]):
            popup_text += '<strong>Website:</strong> <a href="{website}" target="_blank">Click here</a>'.format(website=row["Website"])

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=800),
            icon=folium.Icon(color="blue"),
        ).add_to(m)

# Display the map in Streamlit
folium_static(m)

st.markdown("""
    <h3 style="text-align: center;">Closest 10 PIs</h3>
    <h4 style="text-align: left;">Red Pinpoints: PIs doing similar work.<br>Blue Pinpoints: PIs doing complementary work.</h4>
    """, unsafe_allow_html=True)

# In[ ]:




