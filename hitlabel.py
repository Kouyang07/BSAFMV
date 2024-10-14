import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV files
def load_labels(file_path):
    return pd.read_csv(file_path)

# Validate the detected hits against ground truth with a tolerance of 5 frames
def validate_hits_with_tolerance(detected_hits_csv, ground_truth_csv, tolerance=5):
    # Load the data
    detected = load_labels(detected_hits_csv)
    ground_truth = load_labels(ground_truth_csv)

    # Ensure that both files have the 'Frame' column and the correct labels ('Hit' and 'hit')
    ground_truth.rename(columns={'Hit': 'Hit_true'}, inplace=True)
    detected.rename(columns={'hit': 'Hit_pred'}, inplace=True)

    # Merge the data on the 'Frame' column
    merged_df = pd.merge(ground_truth, detected, on='Frame')

    # Extract true labels and predictions
    y_true = merged_df['Hit_true']
    y_pred = merged_df['Hit_pred']

    # To handle tolerance, we create a new boolean array that checks if a hit in the predicted column
    # is within ±5 frames of a hit in the ground truth
    y_pred_with_tolerance = y_pred.copy()

    for index, row in merged_df.iterrows():
        if row['Hit_pred'] == 1:
            # Check if there is any true hit within the tolerance window of ±5 frames
            frame_range = range(max(0, row['Frame'] - tolerance), row['Frame'] + tolerance + 1)
            if any(ground_truth[(ground_truth['Frame'].isin(frame_range)) & (ground_truth['Hit_true'] == 1)].index):
                y_pred_with_tolerance[index] = 1
            else:
                y_pred_with_tolerance[index] = 0

    # Calculate validation metrics
    accuracy = accuracy_score(y_true, y_pred_with_tolerance)
    precision = precision_score(y_true, y_pred_with_tolerance)
    recall = recall_score(y_true, y_pred_with_tolerance)
    f1 = f1_score(y_true, y_pred_with_tolerance)

    # Print the validation results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Main function
def main(detected_hits_csv, ground_truth_csv):
    validate_hits_with_tolerance(detected_hits_csv, ground_truth_csv)

# Example usage
detected_hits_csv = 'detected_shuttle_hits_with_single_mark.csv'
ground_truth_csv = 'labeled_shuttle_hits.csv'
main(detected_hits_csv, ground_truth_csv)