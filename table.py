import pandas as pd

# Load the dataset
file_path = 'test_data.csv'
df = pd.read_csv(file_path)

# Group the data by gesture_id and calculate the required details
summary_table = df.groupby('gesture_id').agg(
    sequence_count=('sequence_id', 'nunique'),
    frame_count=('sequence_id', 'count')
).reset_index()

# Display the summary table
print("Gesture Distribution Summary")
print(summary_table.to_string(index=False))
