import pandas as pd
import re

def remove_mentions(text):
    """
    Remove '@' mentions from text.
    Example: '@AmericanAir hello' -> 'hello'
    """
    if pd.isna(text):  # Handle NaN values
        return text
    # This pattern matches '@' followed by word characters until space
    pattern = r'@\w+\s*'
    return re.sub(pattern, '', text).strip()

# Read the CSV file
print("Reading flight_data.csv...")
df = pd.read_csv('flight_data.csv')

# Display original columns
print("\nColumns in the dataset:", df.columns.tolist())

# Create backup of original text column
text_column = 'text'  # Change this if your text column has a different name
df['original_text'] = df[text_column]

# Process the text column
print("\nCleaning text...")
df[text_column] = df[text_column].apply(remove_mentions)

# Save processed data
output_file = 'flight_data_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"\nProcessed file saved as: {output_file}")

# Print some examples
print("\nExample results (first 5 rows):")
examples = pd.DataFrame({
    'Original': df['original_text'].head(),
    'Cleaned': df[text_column].head()
})
print(examples.to_string())

# Print statistics
total_rows = len(df)
rows_with_mentions = len(df[df['original_text'].str.contains('@', na=False)])
print(f"\nProcessing Summary:")
print(f"Total rows processed: {total_rows}")
print(f"Rows containing '@' mentions: {rows_with_mentions}")
print(f"Percentage of rows with mentions: {(rows_with_mentions/total_rows)*100:.2f}%")