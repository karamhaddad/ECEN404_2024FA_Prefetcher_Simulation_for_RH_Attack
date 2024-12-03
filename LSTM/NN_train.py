import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Custom accuracy metric to check if all bits match
# This was chosen to be the metric i would track as the only thing that matters is if the model is completely correct when it is guessing
def custom_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.reduce_all(tf.equal(y_true, y_pred_rounded), axis=1)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Input Preprocessing convert into 7 bit binary array
# Takes in Line outputs array
def input_process(line):
    return [int(bit) for bit in line.strip()]  # Convert entire 7-bit line to binary array

# Output Preprocessing as twos compliment
# Takes in tensor outputs binary array
def output_process(line):
    return [int(bit) for bit in line.strip()[-7:]]  # Extract the last 7 bits for block delta (including sign)

# Preprocess file to get sequences and outputs
# Outputs tensor
def preprocess_file(file_path, print_interval=100000):
    outputs = []
    sequences = []
    total_lines = 0  # Track total lines processed, sanity check

    with open(file_path, 'r') as file:  # read file
        lines = file.readlines()

    # This process could eb improved upon to make it faster but i dont care
    for i in range(3, len(lines) - 1):  # Start from line 3 to get a full 4-timestep history
        input_sequence = []

        for j in range(3, -1, -1):  # Collect past 4 lines
            processed_input = input_process(lines[i - j])
            input_sequence.extend(processed_input)

        if len(input_sequence) == 28:  # 4 timesptes 7 bits each, 28 = 7 * 4 dummy
            output_sequence = output_process(lines[i + 1])  # Output is the next block delta (7 bits, 2's complement)
            sequences.append(input_sequence)
            outputs.append(output_sequence)

    return np.array(sequences), np.array(outputs)

# LSTM Model Definition
# Hyperparameters were achieved through tuning
# Change how you wish if you make a better model email me at idontcare@hatemail.com
def create_lstm_model():
    model = Sequential([
        LSTM(256, input_shape=(4, 7), return_sequences=True, dropout=0.2, recurrent_dropout=0.4),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.4),
        LSTM(64, return_sequences=False, dropout=0.2),
        Dense(7, activation='sigmoid')  
    ])

    # adam the best optimizer for bitwise NN's
    # something that could be improved is the loss function, 
    # i tried ot create my own that took into account the output being used in the next 100 liens but coulnd't get it to work
    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[custom_accuracy, 'accuracy'])
    return model

# Main script
if __name__ == "__main__":
    file_path = '/Users/nathanielbush/Desktop/ECEN403/NeuralNetwork/654_block_200k.txt'  # Update with your dataset path
    sequences, outputs = preprocess_file(file_path)

    if sequences.size == 0 or outputs.size == 0:
        raise ValueError("No data created, check input file.")

    # Reshape sequences to (num_sequences, 4, 7)
    # sanity check
    num_sequences = len(sequences)
    sequences = sequences.reshape((num_sequences, 4, 7))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sequences, outputs, test_size=0.2, random_state=123)

    # Create model
    model = create_lstm_model()

    # Early stopping
    # This allows for there to be a large number of set epochs, but will stop the training if the custom accuracy decreases 3 times in a row
    callback = EarlyStopping(monitor='val_custom_accuracy', patience=3)

    # Change epoch size as you wish
    log = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_data=(X_test, y_test), callbacks=[callback])

    # Evaluation
    loss, custom_acc, bitwise_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}, Custom accuracy (all bits correct): {custom_acc}, Bitwise accuracy: {bitwise_acc}")

    # Save the model
    model.save('block_offset_v3/model')
    model.save('block_offset_v3.keras')

# Ultimately i don't think the plotting was too important but it was fun to look at

# Plotting for loss and custom accuracy
plt.figure(figsize=(12, 6))

# Plotting loss (Training and Validation)
plt.subplot(1, 2, 1)
plt.title('Cost')
plt.plot(log.history['loss'], label='Training Loss')
plt.plot(log.history['val_loss'], label='Validation Loss')
plt.legend()

# Plotting custom accuracy (Training and Validation)
plt.subplot(1, 2, 2)
plt.title('Custom Accuracy')
plt.plot(log.history['custom_accuracy'], label='Training Custom Accuracy')
plt.plot(log.history['val_custom_accuracy'], label='Validation Custom Accuracy')
plt.legend()

plt.show()