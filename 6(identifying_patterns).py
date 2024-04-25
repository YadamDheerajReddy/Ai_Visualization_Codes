import csv
import re

def identify_patterns(csv_file_path, column_name):
    patterns = {}

    # Open the CSV file and use a dictionary reader
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate through each row in the specified column
        for row in reader:
            text = row[column_name]
            
            # Example pattern: finding words that start with 'pattern'
            pattern_matches = re.findall(r'\bFemale\b', text, flags=re.IGNORECASE)
            
            # Update patterns dictionary with matches
            for match in pattern_matches:
                if match in patterns:
                    patterns[match] += 1
                else:
                    patterns[match] = 1

    return patterns

# Path to the CSV file
csv_file_path = r"Social_Network _Ads.csv"  # Update with your CSV file path
column_name = 'Gender'  # Update with the actual column name in your CSV file

# Run the function and store the result
result = identify_patterns(csv_file_path, column_name)

# Display the identified patterns and their counts
print("Identified Patterns and Counts:")
for pattern, count in result.items():
    print(f"Pattern: {pattern}, Count: {count}")
