import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# The algorithm was changed around a little bit but the code still runs.
# 

# Custom accuracy metric to check if all bits match
def custom_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.reduce_all(tf.equal(y_true, y_pred_rounded), axis=1)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Input Preprocessing: Each line is 7 bits long
def input_process(line):
    return [int(bit) for bit in line.strip()]  # Convert entire 7-bit line to binary array

# Output Preprocessing: Extract the 6-bit block delta
# older function but didn't have time to update it, the hyperparameters still worked 
def output_process(line):
    return [int(bit) for bit in line.strip()[-6:]]  # Extract the last 6 bits for block delta

# Preprocess file to get sequences and outputs
def preprocess_file(file_path, print_interval=100000):
    outputs = []
    sequences = []
    total_lines = 0  # Track total lines processed

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(3, len(lines) - 1):  # Start from line 3 to get a full 4-timestep history
        input_sequence = []

        for j in range(3, -1, -1):  # Collect past 4 lines
            processed_input = input_process(lines[i - j])
            input_sequence.extend(processed_input)

        if len(input_sequence) == 28:  # 7 bits per timestep, 4 timesteps => 28 bits
            output_sequence = output_process(lines[i + 1])  # Output is the next block delta (6 bits)
            sequences.append(input_sequence)
            outputs.append(output_sequence)

    return np.array(sequences), np.array(outputs)

# LSTM Model Definition with hyperparameters
def create_lstm_model(params):
    model = Sequential([
        LSTM(params['units_1'], input_shape=(4, 7), return_sequences=True, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout']),
        LSTM(params['units_2'], return_sequences=True, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout']),
        LSTM(params['units_3'], return_sequences=False, dropout=params['dropout']),
        Dense(6, activation='sigmoid')  # Predict the next 6 bits (block delta)
    ])

    adam_optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[custom_accuracy, 'accuracy'])
    return model

# Random search to find best hyperparameters
def random_search_optimization(X_train, X_test, y_train, y_test, n_iterations=10):
    # Define the search space for hyperparameters
    search_space = {
        'units_1': [256, 128],  # First layer: either 256 or 128 units
        'units_2': [128, 64],   # Second layer: either 128 or 64 units
        'units_3': [64, 32],    # Third layer: either 64 or 32 units
        'learning_rate': [0.001, 0.0005, 0.0001],  # Tuning around 0.001
        'dropout': [0.2, 0.3, 0.4],               # Test different dropout rates
        'recurrent_dropout': [0.2, 0.3, 0.4],     # Test different recurrent dropout rates
    }

    best_accuracy = 0
    best_params = None
    best_model = None

    for iteration in range(n_iterations):
        # Randomly sample a combination of hyperparameters
        params = {k: random.choice(v) for k, v in search_space.items()}
        print(f"\nRandom Search Iteration {iteration + 1}/{n_iterations}")
        print(f"Trying parameters: {params}")

        # Create and train the model with the current set of hyperparameters
        model = create_lstm_model(params)
        callback = EarlyStopping(monitor='val_accuracy', patience=3)
        history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[callback], verbose=0)

        # Evaluate model
        loss, custom_acc, bitwise_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Validation accuracy: {bitwise_acc:.4f}, Custom accuracy (all bits correct): {custom_acc:.4f}")

        # If it's the best model so far, save it
        if bitwise_acc > best_accuracy:
            best_accuracy = bitwise_acc
            best_params = params
            best_model = model

    print("\nBest hyperparameters found:")
    print(best_params)
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    return best_model, best_params

# Main script
if __name__ == "__main__":
    file_path = '/Users/nathanielbush/Desktop/ECEN403/NeuralNetwork/654_block_200k.txt'  # Update with your dataset path
    sequences, outputs = preprocess_file(file_path)

    if sequences.size == 0 or outputs.size == 0:
        raise ValueError("No data generated from the preprocessing step. Check your input file.")

    # Reshape sequences to (num_sequences, 4, 7)
    num_sequences = len(sequences)
    sequences = sequences.reshape((num_sequences, 4, 7))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sequences, outputs, test_size=0.2, random_state=123)

    # Perform random search for hyperparameter tuning
    best_model, best_params = random_search_optimization(X_train, X_test, y_train, y_test, n_iterations=10)

    # Save the best model
    best_model.save('best_block_offset_model.h5')

    # Plotting for the best model
    history = best_model.history
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Cost')
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.legend()

    plt.show()
