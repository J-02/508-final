import pandas as pd

# Load CSV file into DataFrame
df = pd.read_csv('/Users/tadeozuniga/PycharmProjects/508-final/data/Alaska.csv')

# Display the first 5 rows of the DataFrame
#print(df.head())

# Print the DataFrame's information to view variables and their data types
#print(df.info())

missing_data = df.isnull().sum()

# Print the number of missing values in each column
print(missing_data)

# Filter rows where either longitude or latitude data is missing
missing_location_data = df[df['longitude'].isnull() | df['latitude'].isnull()]

# Print the rows with missing longitude or latitude
print(missing_location_data)

cleaned_df = df.dropna(subset=['longitude', 'latitude'])

# Remove columns that contain any missing data
cleaned_df = cleaned_df.dropna(axis=1)

# Check the resulting DataFrame to ensure the removal was successful
print(cleaned_df.isnull().sum())  # Check remaining missing values in all columns
print(cleaned_df.head())          # Display the first 5 rows of the cleaned DataFrame

cleaned_df.to_csv('/Users/tadeozuniga/PycharmProjects/508-final/data/Alasak_cleaned.csv', index=False)

# Print the contents of the cleaned CSV file
with open('/data/Alasak_cleaned.csv', 'r') as file:
    print(file.read())

