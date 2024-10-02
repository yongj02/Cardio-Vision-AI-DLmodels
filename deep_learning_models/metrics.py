import numpy as np

def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / (predicted_positives + 1e-10)
    return precision

def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    possible_positives = np.sum(y_true == 1)
    recall = true_positives / (possible_positives + 1e-10)
    return recall

def f1(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + 1e-10))

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * y_pred, 0, 1)))
    specificity_val = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())

    return specificity_val