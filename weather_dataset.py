import os
import glob
import json
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle
import inspect

def load_weather_data_with_clustering(json_dir, image_dir, image_size=128, n_clusters=5):
    """
    Load weather data from JSON files and perform K-means clustering for weather classification.
    
    Args:
        json_dir: Directory containing JSON files with weather data
        image_dir: Directory containing the actual images
        image_size: Size to resize images to
        n_clusters: Number of clusters for K-means (default 5 as in ClusteringV1.py)
    
    Returns:
        images, labels, ids, cls, metadata, kmeans_model
    """
    images = []
    labels = []
    ids = []
    cls = []
    metadata = []
    
    print('Reading weather data from JSON files...')
    
    # Find all JSON files recursively
    json_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    print(f'Found {len(json_files)} JSON files')
    
    # First pass: collect all weather data for clustering
    weather_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            
                            # Extract required fields (matching latest cloud_detection.py format)
                            image_path = entry.get('image_path', '')
                            cloud_coverage = entry.get('cloud_coverage', 0.0)
                            irradiance = entry.get('irradiance', 0.0)
                            sun_obscuration_percentage = entry.get('sun_obscuration_percentage', 0.0)  # Use sun_obscuration_percentage
                            timestamp = entry.get('now', '')
                            
                            # Skip entries without required data
                            if not image_path or cloud_coverage is None:
                                continue
                            
                            # Store data for clustering
                            weather_data.append({
                                'image_path': image_path,
                                'cloud_coverage': cloud_coverage,
                                'sun_obscuration_percentage': sun_obscuration_percentage,  # Use percentage directly
                                'irradiance': irradiance,
                                'timestamp': timestamp
                            })
                            
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {json_file}: {e}")
                            # ...existing code...
                            
        except Exception as e:
            print(f"Error reading file {json_file}: {e}")
            continue
    
    if not weather_data:
        raise ValueError("No valid data found!")
    
    # Perform K-means clustering (same as ClusteringV1.py)
    print(f'Performing K-means clustering with {n_clusters} clusters...')
    df = pd.DataFrame(weather_data)
    
    # Prepare features for clustering
    features = ['cloud_coverage', 'sun_obscuration_percentage', 'irradiance']
    X = df[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments to dataframe
    df['cluster'] = cluster_labels
    
    print(f'Clustering completed. Cluster distribution: {np.bincount(cluster_labels)}')
    
    # Second pass: load images and create labels
    for idx, row in df.iterrows():
        # Construct full image path
        full_image_path = os.path.join(image_dir, os.path.basename(row['image_path']))
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue
        
        # Load and preprocess image
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Could not read image: {full_image_path}")
            continue
            
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get cluster assignment as weather class
        weather_class = int(row['cluster'])
        
        # Create one-hot encoded label
        label = np.zeros(n_clusters)
        label[weather_class] = 1.0
        
        images.append(image)
        labels.append(label)
        ids.append(os.path.basename(row['image_path']))
        cls.append(weather_class)
        
        # Store metadata for analysis
        metadata.append({
            'image_path': row['image_path'],
            'cloud_coverage': row['cloud_coverage'],
            'irradiance': row['irradiance'],
            'sun_obscuration_percentage': row['sun_obscuration_percentage'],
            'timestamp': row['timestamp'],
            'weather_class': weather_class
        })
    
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    
    print(f'Loaded {len(images)} images with clustered weather data')
    print(f'Weather class distribution: {np.bincount(cls)}')
    
    return images, labels, ids, cls, metadata, kmeans


def create_weather_features(metadata):
    """
    Create additional features from metadata for enhanced classification.
    
    Args:
        metadata: List of dictionaries containing weather data
    
    Returns:
        features: numpy array of additional features
    """
    features = []
    
    for entry in metadata:
        # Extract time-based features
        timestamp = entry['timestamp']
        try:
            dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            hour = dt.hour
            month = dt.month
            # Create cyclical time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
        except:
            hour_sin = hour_cos = month_sin = month_cos = 0
        
        # Combine all features
        feature_vector = [
            entry['cloud_coverage'],
            entry['sun_obscuration_percentage'],
            entry['irradiance'] / 1000.0,  # Normalize irradiance
            hour_sin,
            hour_cos,
            month_sin,
            month_cos
        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def get_cluster_characteristics(metadata, kmeans_model):
    """
    Return the mean value of each feature for each cluster.
    Args:
        metadata: List of dictionaries containing weather data
        kmeans_model: Trained KMeans model
    Returns:
        cluster_means: Dictionary mapping cluster numbers to mean values of features
    """
    import pandas as pd
    df = pd.DataFrame(metadata)
    cluster_means = {}
    for cluster in sorted(df['weather_class'].unique()):
        cluster_data = df[df['weather_class'] == cluster]
        cluster_means[cluster] = {
            'mean_cloud_coverage': cluster_data['cloud_coverage'].mean(),
            'mean_irradiance': cluster_data['irradiance'].mean(),
            'mean_sun_obscuration_percentage': cluster_data['sun_obscuration_percentage'].mean(),
            'count': len(cluster_data)
        }
    return cluster_means


class WeatherDataSet(object):
    """Dataset class for weather classification with additional features."""
    
    def __init__(self, images, labels, ids, cls, features=None):
        self._num_examples = images.shape[0]
        
        # Normalize images
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._features = features
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def ids(self):
        return self._ids
    
    @property
    def cls(self):
        return self._cls
    
    @property
    def features(self):
        return self._features
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """Return the next batch of examples."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        
        batch_images = self._images[start:end]
        batch_labels = self._labels[start:end]
        batch_ids = self._ids[start:end]
        batch_cls = self._cls[start:end]
        
        if self._features is not None:
            batch_features = self._features[start:end]
            return batch_images, batch_labels, batch_ids, batch_cls, batch_features
        else:
            return batch_images, batch_labels, batch_ids, batch_cls


def read_weather_sets_with_clustering(json_dir, image_dir, image_size=128, use_features=True, n_clusters=5, ablation_features=None, ablation_no_image=False):
    """
    Read weather datasets using K-means clustering for classification.
    
    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing images
        image_size: Size to resize images
        validation_size: Fraction of data to use for validation
        use_features: Whether to include additional features
        n_clusters: Number of clusters for K-means
    
    Returns:
        DataSets object with train and validation sets, plus clustering info
    """

    class DataSets(object):
        pass

    # Ablation options
    ablation_features = None
    ablation_no_image = False

    # Detect ablation arguments from kwargs
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    if 'ablation_features' in values:
        ablation_features = values['ablation_features']
    if 'ablation_no_image' in values:
        ablation_no_image = values['ablation_no_image']

    # Instead of loading/saving images arrays, only load and split ids, features, labels, cls
    # Load metadata and cluster info
    images, labels, ids, cls, metadata, kmeans_model = load_weather_data_with_clustering(
        json_dir, image_dir, image_size if image_size else 128, n_clusters
    )
    cluster_descriptions = get_cluster_characteristics(metadata, kmeans_model)
    features = None
    if use_features:
        features = create_weather_features(metadata)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    # Shuffle data
    if features is not None:
        ids, labels, cls, features = shuffle(ids, labels, cls, features)
    else:
        ids, labels, cls = shuffle(ids, labels, cls)
    # Split into train, validation, and test sets (10% test, 20% validation, 70% train)
    ids_temp, ids_test, features_temp, features_test, labels_temp, labels_test, cls_temp, cls_test = train_test_split(
        ids, features, labels, cls, test_size=0.10, random_state=42, stratify=cls
    )
    val_size = 0.2 / 0.9
    ids_train, ids_val, features_train, features_val, labels_train, labels_val, cls_train, cls_val = train_test_split(
        ids_temp, features_temp, labels_temp, cls_temp, test_size=val_size, random_state=42, stratify=cls_temp
    )

    # Apply ablation after loading split
    if use_features and ablation_features:
        feature_indices = {
            'cloud_coverage': 0,
            'sun_obscuration_percentage': 1,
            'irradiance': 2,
            'time': [3,4,5,6]
        }
        keep_idx = list(range(features_train.shape[1]))
        for feat in ablation_features:
            idx = feature_indices[feat]
            if isinstance(idx, list):
                for i in idx:
                    keep_idx.remove(i)
            else:
                keep_idx.remove(idx)
        features_train = features_train[:, keep_idx]
        features_val = features_val[:, keep_idx]
        features_test = features_test[:, keep_idx]

    # Remove image branch if requested (handled in notebook, so just pass empty list if needed)
    if ablation_no_image:
        ids_train = []
        ids_val = []
        ids_test = []

    # Return only ids, features, labels, cls for each split
    data_sets = DataSets()
    data_sets.train = type('Split', (), {})()
    data_sets.valid = type('Split', (), {})()
    data_sets.test = type('Split', (), {})()
    data_sets.train.ids = ids_train
    data_sets.train.features = features_train
    data_sets.train.labels = labels_train
    data_sets.train.cls = cls_train
    data_sets.valid.ids = ids_val
    data_sets.valid.features = features_val
    data_sets.valid.labels = labels_val
    data_sets.valid.cls = cls_val
    data_sets.test.ids = ids_test
    data_sets.test.features = features_test
    data_sets.test.labels = labels_test
    data_sets.test.cls = cls_test
    data_sets.cluster_descriptions = cluster_descriptions
    data_sets.kmeans_model = kmeans_model
    data_sets.metadata = metadata

    return data_sets


def get_weather_classes_from_clusters(cluster_descriptions):
    """Get weather class names from cluster descriptions."""
    classes = []
    for i in range(len(cluster_descriptions)):
        if i in cluster_descriptions:
            classes.append(cluster_descriptions[i]['description'])
        else:
            classes.append(f"Cluster {i}")
    return classes


def analyze_weather_distribution(metadata, cluster_descriptions):
    """Analyze the distribution of weather classes and features."""
    df = pd.DataFrame(metadata)
    
    print("\nWeather Classification Analysis (K-means Clustering):")
    print("=" * 60)
    
    # Class distribution
    class_counts = df['weather_class'].value_counts().sort_index()
    
    print("\nWeather Class Distribution:")
    for i, count in class_counts.items():
        if i in cluster_descriptions:
            desc = cluster_descriptions[i]['description']
            percentage = (count / len(df)) * 100
            print(f"{desc}: {count} ({percentage:.1f}%)")
    
    # Feature statistics by class
    print("\nFeature Statistics by Weather Class:")
    for i in sorted(df['weather_class'].unique()):
        class_data = df[df['weather_class'] == i]
        if len(class_data) > 0 and i in cluster_descriptions:
            desc = cluster_descriptions[i]['description']
            print(f"\n{desc}:")
            print(f"  Cloud Coverage: {class_data['cloud_coverage'].mean():.3f} ± {class_data['cloud_coverage'].std():.3f}")
            print(f"  Sun Obscuration %: {class_data['sun_obscuration_percentage'].mean():.3f} ± {class_data['sun_obscuration_percentage'].std():.3f}")
            print(f"  Irradiance: {class_data['irradiance'].mean():.1f} ± {class_data['irradiance'].std():.1f}")
    
    return df


# Backward compatibility functions
def load_weather_data(json_dir, image_dir, image_size=128):
    """Backward compatibility function that uses clustering."""
    return load_weather_data_with_clustering(json_dir, image_dir, image_size)

def read_weather_sets(json_dir, image_dir, image_size=128, validation_size=0.2, use_features=True):
    """Backward compatibility function that uses clustering."""
    return read_weather_sets_with_clustering(json_dir, image_dir, image_size, validation_size, use_features)

def get_weather_classes():
    """Get default weather class names (for backward compatibility)."""
    return ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']