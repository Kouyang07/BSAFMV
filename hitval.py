import pandas as pd

# Read the ground truth data
ground_truth = pd.read_csv('labeled_shuttle_hits.csv')

# Read the detected hits data
detected_hits = pd.read_csv('hits.csv')

# Define the tolerance (in frames)
tolerance = 5  # You can adjust this value as needed

# Extract the frames where hits occur in ground truth and detected hits
ground_truth_hits = ground_truth[ground_truth['Hit'] == 1]['Frame'].values
detected_hit_frames = detected_hits['Frame'].values

# Initialize lists to keep track of matches
true_positives = []
false_positives = []
false_negatives = []

# For each ground truth hit, check if a detected hit is within the tolerance
for gt_frame in ground_truth_hits:
    if any(abs(detected_hit_frames - gt_frame) <= tolerance):
        true_positives.append(gt_frame)
    else:
        false_negatives.append(gt_frame)

# For each detected hit, check if it does not match any ground truth hit within the tolerance
for detected_frame in detected_hit_frames:
    if not any(abs(ground_truth_hits - detected_frame) <= tolerance):
        false_positives.append(detected_frame)

# Calculate metrics
tp = len(true_positives)
fp = len(false_positives)
fn = len(false_negatives)

# Avoid division by zero
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print the results
print(f"Tolerance: ±{tolerance} frames")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1_score:.2f}")

# Optionally, save the results to a CSV file
results_df = pd.DataFrame({
    'Metric': ['True Positives', 'False Positives', 'False Negatives', 'Precision', 'Recall', 'F1-Score'],
    'Value': [tp, fp, fn, precision, recall, f1_score]
})
results_df.to_csv('validation_results.csv', index=False)

# Also, create a detailed comparison DataFrame
comparison_df = pd.DataFrame({
    'Ground_Truth_Hits': ground_truth_hits,
    'Detected_Match': ['Yes' if frame in true_positives else 'No' for frame in ground_truth_hits]
})
comparison_df.to_csv('hit_comparison.csv', index=False)