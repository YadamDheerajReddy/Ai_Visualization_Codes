import csv
import statistics

def remove_mean(input_file, output_file):
    # Read the CSV file and extract the column of numbers
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]

    # Calculate the mean
    mean_value = statistics.mean(data)

    # Remove the mean from each data point
    mean_removed_data = [x - mean_value for x in data]

    # Write the mean-removed data to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in mean_removed_data:
            writer.writerow([value])

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "mean_removed_data.csv"
    remove_mean(input_file, output_file)
