import pandas as pd

# Define the input and output file paths
input_file = r'C:\Users\satwik\OneDrive\Desktop\project\angles.xlsx'  # Replace with the correct path to your input file
output_file = r'C:\Users\satwik\OneDrive\Desktop\project\output_angles.xlsx'  # Replace with the desired path for your output file

# Load the original Excel file
df = pd.read_excel(input_file)

# Drop the 'Serial No' column
df = df.drop(columns=['Serial No'])

# Group by the 'Pose' column and calculate the mean for the other columns
grouped = df.groupby('Pose').mean()

# Reset the index to make 'Pose' a column again
grouped.reset_index(inplace=True)

# Save the result to a new Excel file
grouped.to_excel(output_file, index=False)

print(f"Average values calculated and saved to {output_file}")
