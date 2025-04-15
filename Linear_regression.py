import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset (data.csv)
df = pd.read_csv('data.csv')

# Step 2: Check the first few rows of the dataset to verify the structure
print(df.head())

# Step 3: Feature Engineering
# Convert the 'date' column to datetime format and extract year, month, day if needed
df['date'] = pd.to_datetime(df['date'])
df['year_sold'] = df['date'].dt.year
df['month_sold'] = df['date'].dt.month
df['day_sold'] = df['date'].dt.day

# Feature: Age of the house (since it was built)
df['Age'] = 2023 - df['yr_built']

# Feature: Time since last renovation (if it has been renovated)
df['Years_Since_Renovation'] = df['yr_renovated'].apply(lambda x: 2023 - x if x != 0 else 0)

# Step 4: Handle missing values (if any)
df = df.dropna()  # Drop rows with missing values

# Step 5: Check correlation between features
# Select numeric columns only to avoid issues with categorical data (e.g., addresses)
df_numeric = df.select_dtypes(include=[np.number])

# Visualize correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.show()

# Step 6: Define the features (X) and target (y)
# Drop the non-predictive columns such as 'date', 'street', 'city', 'statezip', 'country'
X = df.drop(['price', 'date', 'street', 'city', 'statezip', 'country'], axis=1)
y = df['price']  # Target variable (Price)

# Step 7: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 11: Display predicted prices alongside actual prices for the test set
predicted_prices = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

# Display the first few predicted and actual prices
print(predicted_prices.head())

# Step 12: Visualize the predictions vs actual prices
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Step 13: User Input to predict price
print("\nPlease enter the following information about the house:")

# Get user input for various features (you can adjust these based on your dataset's columns)
bedrooms = float(input("Number of bedrooms: "))
bathrooms = float(input("Number of bathrooms: "))
sqft_living = float(input("Square footage of the house (sqft_living): "))
sqft_lot = float(input("Square footage of the lot (sqft_lot): "))
floors = float(input("Number of floors: "))
waterfront = float(input("Waterfront (1 if yes, 0 if no): "))
view = float(input("View (0-4, with 4 being best): "))
condition = float(input("Condition (1-5, with 5 being excellent): "))
grade = float(input("Grade (1-13, with 13 being the best): "))
sqft_above = float(input("Square footage above ground (sqft_above): "))
sqft_basement = float(input("Square footage of the basement (sqft_basement): "))
yr_built = int(input("Year the house was built (yr_built): "))
yr_renovated = int(input("Year the house was renovated (yr_renovated, 0 if not renovated): "))
zipcode = float(input("Zipcode: "))
lat = float(input("Latitude: "))
long = float(input("Longitude: "))

# Prepare the user's input in a format similar to the training data
user_input = np.array([bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                       condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
                       zipcode, lat, long])

# Reshape the input to match the expected input shape for prediction
user_input = user_input.reshape(1, -1)

# Step 14: Predict the house price using the trained model
predicted_price = model.predict(user_input)

# Step 15: Display the predicted price
print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")

