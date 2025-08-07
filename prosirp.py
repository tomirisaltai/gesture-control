import pandas as pd
from sklearn.model_selection import train_test_split

# Processing of the dataset
file_path = 'hand_landmark1.csv' 
data = pd.read_csv(file_path)
def stratified_split(data, test_size, val_size):
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for gesture_id in data['gesture_id'].unique():
        gesture_data = data[data['gesture_id'] == gesture_id]
        unique_sequences = gesture_data['sequence_id'].unique()
        # Splitting into training, validation, and test sets
        train_sequences, temp_sequences = train_test_split(
            unique_sequences, test_size=(test_size + val_size), shuffle=True, random_state=42
        )
        val_sequences, test_sequences = train_test_split(
            temp_sequences, test_size=(test_size / (test_size + val_size)), shuffle=True, random_state=42
        )
        #Filtering the original data
        train_data = pd.concat([train_data, gesture_data[gesture_data['sequence_id'].isin(train_sequences)]])
        val_data = pd.concat([val_data, gesture_data[gesture_data['sequence_id'].isin(val_sequences)]])
        test_data = pd.concat([test_data, gesture_data[gesture_data['sequence_id'].isin(test_sequences)]])
    
    return train_data, val_data, test_data

#stratified split
train_data, val_data, test_data = stratified_split(data, test_size=0.15, val_size=0.15)

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Displaying the first few rows
print("Training Data Sample:")
print(train_data.head(30)) 
print("\nValidation Data Sample:")
print(val_data.head(30)) 
print("\nTest Data Sample:")
print(test_data.head(30)) 
