# # 1. Required Imports and Setup
# import os
# import glob
# import json
# import numpy as np
# import cv2
# import pandas as pd
# from datetime import datetime
# from sklearn.utils import shuffle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 2. Directory Setup
# base_dir = r"D:\Image"  # Your image directory
# test_result_dir = os.path.join(base_dir, "test_result")
# json_dir = os.path.join(test_result_dir, "json")

# print("Setting up with local paths:")
# print(f"Base directory: {base_dir}")
# print(f"Test result directory: {test_result_dir}")
# print(f"JSON directory: {json_dir}")

# print("\nSetting up directory structure...")
# os.makedirs(test_result_dir, exist_ok=True)
# os.makedirs(json_dir, exist_ok=True)

# print("\nSetting up month subdirectories...")
# for month in range(1, 13):
#     month_dir = os.path.join(json_dir, f"{month:02d}")
#     os.makedirs(month_dir, exist_ok=True)

# print("\nChecking for image files...")
# image_files = []
# json_files = []
# if json_files:
#     print("Sample JSON files:")
#     for file in json_files[:5]:
#         print(f"- {os.path.relpath(file, json_dir)}")

# if not image_files:
#     raise ValueError(f"No image files found in {base_dir}. Please ensure your images are in this directory.")
# if not json_files:
#     raise ValueError(f"No JSON files found in {json_dir}. Please run cloud_detection.py first to generate JSON files.")

 
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


def read_weather_sets_with_clustering(json_dir, image_dir, image_size=128, use_features=True, n_clusters=5):
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
    
    data_sets = DataSets()
    
    # Load all data with clustering
    images, labels, ids, cls, metadata, kmeans_model = load_weather_data_with_clustering(
        json_dir, image_dir, image_size, n_clusters
    )
    
    # Get cluster characteristics
    cluster_descriptions = get_cluster_characteristics(metadata, kmeans_model)
    
    # Create additional features if requested
    features = None
    if use_features:
        features = create_weather_features(metadata)
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Shuffle data
    if features is not None:
        images, labels, ids, cls, features = shuffle(images, labels, ids, cls, features)
    else:
        images, labels, ids, cls = shuffle(images, labels, ids, cls)
    
    # Split into train, validation, and test sets (10% test, 20% validation, 70% train)
    # First, split off 10% for test
    X_temp_images, X_test_images, X_temp_features, X_test_features, Y_temp, Y_test, ids_temp, ids_test, cls_temp, cls_test = train_test_split(
        images, features, labels, ids, cls, test_size=0.10, random_state=42, stratify=cls
    )

    # Then, split remaining 20% for validation, 70% for training
    val_size = 0.2 / 0.9  # 20% of original, from remaining 90%
    X_train_images, X_val_images, X_train_features, X_val_features, Y_train, Y_val, ids_train, ids_val, cls_train, cls_val = train_test_split(
        X_temp_images, X_temp_features, Y_temp, ids_temp, cls_temp, test_size=val_size, random_state=42, stratify=cls_temp
    )

    # Assign to DataSets
    data_sets.train = WeatherDataSet(X_train_images, Y_train, ids_train, cls_train, X_train_features)
    data_sets.valid = WeatherDataSet(X_val_images, Y_val, ids_val, cls_val, X_val_features)
    data_sets.test = WeatherDataSet(X_test_images, Y_test, ids_test, cls_test, X_test_features)
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