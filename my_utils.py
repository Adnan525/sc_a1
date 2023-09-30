import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

def get_hyperparameters(model : keras.models.Sequential):
    hyperparameters = {
        "Learning Rate": f"{model.optimizer.learning_rate.numpy():.4f}",
        "Number of Hidden Layers": len(model.layers) - 2, # excluding input and output layer
        "Number of Neurons in Each Hidden Layer": [layer.units for layer in model.layers if isinstance(layer, keras.layers.Dense)][1:len(model.layers)-1],
        "Activation Functions": [layer.activation.__name__ for layer in model.layers if isinstance(layer, keras.layers.Dense)],
        "Loss Function": model.loss,
        "Optimizer": model.optimizer.__class__.__name__,
        "Metrics for Evaluation": model.metrics_names
    }
    
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")


def split_data(data, target, test_size=0.3, val_size=0.2):

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size, random_state=0)
    
    # split the training data into train and validation sets (80:20)
    train_data, val_data, train_target, val_target = train_test_split(train_data, train_target, test_size=val_size, random_state=0)
    
    return train_data, val_data, test_data, train_target, val_target, test_target


def get_float32_data(X_train, X_val, X_test, y_train, y_val, y_test):
    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32), y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)