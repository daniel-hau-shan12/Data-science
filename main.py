#You work for the local government environment agency and have taken on a project about creating pollinator bee-friendly spaces. You can use both native and non-native plants to create these spaces and therefore need to ensure that you use the correct plants to optimize the environment for these bees.

#The team has collected data on native and non-native plants and their effects on pollinator bees. Your task will be to analyze this data and provide recommendations on which plants create an optimized environment for pollinator bees.

#You have assembled information on the plants and bees research in a file called plants_and_bees.csv. Each row represents a sample that was taken from a patch of land where the plant species were being studied.

# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into dataframe 'df'

df = pd.read_csv("farm_data_for_publication.csv")
#print(df)

#Meaning of variables and their type
df.info()

#Deteriming the count of Null and Not Null values
def plant_bees_info():
    temp = pd.DataFrame(index=df.columns)
    temp["Datatype"] = df.dtypes
    temp["Not null values"] = df.count()
    temp["Null values"] = df.isnull().sum()
    temp["Percentage of Null values"] = (df.isnull().mean())*100
    temp["Unique count"] = df.nunique()
    return temp
print(plant_bees_info())

#Dropping the columns 'specialized_on' and 'status'
df = df.drop(['specialized_on','status'], axis = 1)
print(df.isnull().sum())

#Dropping all rows having null value in 'nonnative_bee' and 'parasitic' column
df_cleaned = df.dropna(subset=['nonnative_bee'])
df_cleaned = df.dropna(subset=['parasitic'])
df = df_cleaned
print(df.isnull().sum())

#Replacing the null values in 'nesting' by global constant 'No nesting'
df["nesting"].fillna("No nesting", inplace = True) 
df.nesting.unique()
array(['ground', 'hive', 'wood', 'parasite [ground]', 'wood/shell',
       'No nesting', 'wood/cavities'], dtype=object)
print(df.isnull().sum())

#Dropping all the duplicate rows
x = df[df.duplicated()].shape[0]
print(f"Number of duplicate rows: {x}")
Number of duplicate rows: 578

df = df.drop_duplicates()
print(f"Shape: {df.shape}")
Shape: (609, 14)

#Changing the data type of 'date' to datetime
df["date"] = pd.to_datetime(df["date"])
df.info()

#Changing the data type of 'nonnative_bee' and 'parasitic' to bool
def convert_bool(val):
    return bool ( val )

df['nonnative_bee'] = df['nonnative_bee'].apply(lambda x: convert_bool(x))
df['parasitic'] = df['parasitic'].apply(lambda x: convert_bool(x))
df.info()
plant_bees_info()

# Selecting the specific sample_id
selected_sample_id = 17425
selected_sample_data = df[df['sample_id'] == selected_sample_id]

# Counting bee and plant species occurrences in the selected sample
bee_species_counts = selected_sample_data['bee_species'].value_counts()
plant_species_counts = selected_sample_data['plant_species'].value_counts()

# Plotting the distribution of bee and plant species
plt.figure(figsize=(10, 6))
plt.bar(bee_species_counts.index, bee_species_counts.values, color='b', alpha=0.7, label='Bee Species')
plt.bar(plant_species_counts.index, plant_species_counts.values, color='g', alpha=0.7, label='Plant Species')
plt.xticks(rotation=90)
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.title(f'Distribution of Bee and Plant Species for Sample ID {selected_sample_id}')
plt.legend()
plt.tight_layout()
plt.show()

#To determine which plants are preferred by native and non-native bees, Random Forest Model was opted due to its robustness and accuracy.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming you already have the dataset loaded in a DataFrame called "df"

# Data Preprocessing
label_encoder = LabelEncoder()

df['season'] = label_encoder.fit_transform(df['season'])
df['site'] = label_encoder.fit_transform(df['site'])
df['native_or_non'] = label_encoder.fit_transform(df['native_or_non'])
df['sampling'] = label_encoder.fit_transform(df['sampling'])
df['plant_species'] = label_encoder.fit_transform(df['plant_species'])
df['bee_species'] = label_encoder.fit_transform(df['bee_species'])
df['sex'] = label_encoder.fit_transform(df['sex'])
df['nesting'] = label_encoder.fit_transform(df['nesting'])

# Split the data into training and testing sets
features = ['species_num', 'season', 'site', 'sampling', 'plant_species', 'time', 'bee_species', 'parasitic', 'nonnative_bee']
X = df[features]
y = df['native_or_non']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model Training
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting native or non-native for plants
df['predicted_native_or_non'] = rf_classifier.predict(X)

# Grouping the data by 'predicted_native_or_non' and 'native_or_non' to count occurrences
grouped_data = df.groupby(['predicted_native_or_non', 'native_or_non']).size().reset_index(name='count')

# Pivot the data for visualization
pivot_data = grouped_data.pivot('predicted_native_or_non', 'native_or_non', 'count').fillna(0)

print("Feature Importance Scores:")
print(rf_classifier.feature_importances_)

accuracy = rf_classifier.score(X_test, y_test)
print("Random Forest Model Accuracy:", accuracy)

# Plotting the grouped bar chart
pivot_data.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
plt.xlabel('Predicted Native or Non-Native')
plt.ylabel('Count')
plt.title('Native or Non-Native Plants')
plt.legend(title='Native or Non-Native', loc='upper left')
plt.show()

#The accuracy of the model is shown above.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Feature Selection
# Select the relevant features for the analysis
features = ['species_num', 'season', 'site', 'sampling', 'plant_species', 'time', 'bee_species', 'parasitic', 'nonnative_bee']

# Extract the selected features and the target variable
X = df[features]
y = df['native_or_non']

# Step 3: Data Preprocessing
# Perform any necessary data preprocessing steps, such as one-hot encoding categorical variables

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 4: Model Training
# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Predict the native/non-native bee species for the test set
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Step 6: Feature Importance
# Extract the feature importances from the trained model
importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Step 8: Predictions
# Using the trained model to make predictions on the test set
y_pred = rf_model.predict(X_test)

# Create a DataFrame with the test set and the corresponding predictions
predictions_df = X_test.copy()
predictions_df['predicted_nonnative_bee'] = y_pred

# Step 9: Plant species preferred by native bees
# Select the rows where the model predicts the bee species as native
preferred_by_native_bees = predictions_df[predictions_df['predicted_nonnative_bee'] == False]['plant_species'].unique()

# Step 10: Plant species preferred by non-native bees
# Select the rows where the model predicts the bee species as non-native
preferred_by_nonnative_bees = predictions_df[predictions_df['predicted_nonnative_bee'] == True]['plant_species'].unique()

from sklearn.preprocessing import LabelEncoder
df2 = pd.read_csv("datacamp_workspace_export_2023-07-18 01_23_51.csv")

# Sample encoded data
encoded_data = df['plant_species']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the label encoder on the original data to establish the mapping
label_encoder.fit(df2['plant_species'])

# Use inverse_transform to decode the encoded data
decoded_data = label_encoder.inverse_transform(encoded_data)

# Fit the label encoder on the data to establish the mapping
label_encoder.fit(decoded_data)

# Access the original labels that were fitted to the encoded values
original_labels = label_encoder.classes_

mapping_dict = {encoded_val: original_label for encoded_val, original_label in zip(label_encoder.transform(original_labels), original_labels)}

# Display the results
print("\nPlant species preferred by native bees:")
plants_list_native = [mapping_dict[i] for i in preferred_by_native_bees]
print(plants_list_native)

print("\nPlant species preferred by non-native bees:")
plants_list_nonnative = [mapping_dict[i] for i in preferred_by_nonnative_bees]
print(plants_list_nonnative)

#selecting the top three plant speciea
native_bees_df = df2[(df2['nonnative_bee'] == False) & (df2['plant_species'] != 'None')]
plant_species_counts = native_bees_df['plant_species'].value_counts()
top_three_species = plant_species_counts.head(3)
colors = sns.color_palette('pastel')

# Bar plot of the top three plant species
plt.figure(figsize=(8,4))
top_three_species.plot(kind='bar', color=colors)
plt.xlabel('Plant Species')
plt.ylabel('Frequency')
plt.title('Top Three Plant Species for Supporting Native Bees')
plt.xticks(rotation=90)
plt.show()

#other analytical data
#Scatter plot of sample_id and species_num
plt.figure(figsize=(8, 6))
plt.scatter(df['sample_id'], df['species_num'])
plt.xlabel('Sample ID')
plt.ylabel('Species Number')
plt.title('Sample ID vs. Species Number')
plt.show()

