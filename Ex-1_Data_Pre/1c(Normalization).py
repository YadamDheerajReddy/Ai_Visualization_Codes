import csv

def min_max_scaling(input_file, output_file, new_min=0, new_max=1):
    # Read the CSV file and extract the column of numbers
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]

    # Find the minimum and maximum values in the data
    current_min = min(data)
    current_max = max(data)

    # Perform Min-Max scaling on each data point
    normalized_data = [(x - current_min) / (current_max - current_min) * (new_max - new_min) + new_min for x in data]

    # Write the normalized data to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in normalized_data:
            writer.writerow([value])

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "normalized_data.csv"
    min_max_scaling(input_file, output_file)
