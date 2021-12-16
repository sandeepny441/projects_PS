"""
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class KmeansCluster():
    def __init__(self):
        self.fig_count = 0
        pass

    def load_data(self):
        data_args = {
            'n_samples' : 200,
            'centers' : 3,
            'cluster_std' : 2.75,
            'random_state' : 42 
        }

        feaures, true_labels = make_blobs(**data_args)

        print('**** Training Data ****')
        print('total data points : %d' % len(feaures))
        print('total cluster centers : %d' % data_args['centers'])
        print('')

        # scaling the data
        scaled_features = StandardScaler().fit_transform(feaures)
        self.features = scaled_features

    def train_model(self, k=3):
        model_args = {
            'init' : 'random',
            'n_clusters' : k,
            'n_init' : 10,  # ten times centroid init performed
            'max_iter' : 300,
            'random_state' : 42
        }

        kmeans = KMeans(**model_args)
        res_fit = kmeans.fit(self.features)
        print('*** Training Model ***')
        print('Using k-means model with params')
        print(res_fit)
        print('')

        print('*** Model fit results')
        print('SSE (lower is better) %d' % kmeans.inertia_)
        print('Centers : %s' % (kmeans.cluster_centers_))
        print('total iterations used : %d' % kmeans.n_iter_)

    def perform_elbow_method(self):
        model_args = {
            'init' : 'random',
            'n_init' : 10,  # ten times centroid init performed
            'max_iter' : 300,
            'random_state' : 42
        }

        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **model_args)
            kmeans.fit(self.features)
            sse.append(kmeans.inertia_)

        # plot the sse against the no of clusters
        plt.style.use("fivethirtyeight")
        self.fig_count += 1
        plt.figure(self.fig_count)
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title('Elbow method')
        #plt.show()

    def perform_silhouette_coefficient_method(self):
        model_args = {
            'init' : 'random',
            'n_init' : 10,  # ten times centroid init performed
            'max_iter' : 300,
            'random_state' : 42
        }

        silhouette_coefficients = []
        for k in range(2, 11): # minimum 2 clusters required
            kmeans = KMeans(n_clusters=k, **model_args)
            kmeans.fit(self.features)
            score = silhouette_score(self.features, kmeans.labels_)
            silhouette_coefficients.append(score)

        # plot the sse against the no of clusters
        plt.style.use("fivethirtyeight")
        self.fig_count += 1
        plt.figure(self.fig_count)
        plt.plot(range(2, 11), silhouette_coefficients)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title('silhouette coefficients method')
        #plt.show()

if __name__ == '__main__':
    model = KmeansCluster()
    model.load_data()
    model.train_model()
    model.perform_elbow_method()
    model.perform_silhouette_coefficient_method()

    # show all plots
    plt.show()