"""
Neural Network Customer Churn Classifier module (TensorFlow / Keras Architecture)
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features, get_feature_lists

def build_churn_nn(input_dim):
    """
    Constructs a Deep Neural Network classifier for churn prediction.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        return model
    except ImportError:
        print("[INFO] TensorFlow is optional. Scikit-learn and XGBoost pipelines are active.")
        return None

if __name__ == "__main__":
    print("Neural Network Churn Model Architecture defined.")
