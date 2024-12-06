import streamlit as st
from kmeans import *


# Streamlit app layout
st.title("Smart Meters Model Hub")
st.sidebar.header("Input Parameters")

algorithm_options = ["KMeans", "ARIMA", "LightGBM"]
selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)

if selected_algorithm == "KMeans":

        # User input for year and number of clusters
        year = st.sidebar.number_input("Select Year", min_value=2000, max_value=2024, value=2013)
        num_of_clusters = st.sidebar.slider("Number of Clusters", min_value=1, max_value=10, value=5)


if st.sidebar.button("Run Model"):
        
        st.header("Clustering of Energy Consumption")

        # Perform KMeans clustering
        clusters, result = perform_kmeans(year, num_of_clusters)

        if clusters is not None:
        
            st.subheader("Cluster Aggregated Results")
            st.write(result)

            # Optional: Visualize the clustering results with a bar chart
            st.subheader("Cluster Count Distribution")
            cluster_counts = result['count']
            cluster_labels = result.index
            
            st.bar_chart(data = cluster_counts,
                         x_label="Cluster Number",
                         y_label= "Number of items in Cluster")

            st.subheader("Cluster Median Energy Distribution")
            cluster_counts = result['median']
            cluster_labels = result.index
            
            st.bar_chart(cluster_counts,
                         x_label="Cluster Number",
                         y_label= "Median Energy in Cluster")

    # Footer or additional information can be added here if needed.