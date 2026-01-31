# Recommendation System using Collaborative Filtering and Matrix Factorization
# Structured like a Jupyter Notebook for easy submission

# =====================================
# Cell 1: Import Required Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# =====================================
# Cell 2: Load Sample Dataset (User-Item Ratings)
# =====================================
# Sample dataset (You can replace with MovieLens dataset)

data = {
    'user_id': [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5],
    'item_id': [101,102,103,101,102,104,101,103,105,102,104,105,101,104,105],
    'rating':  [5,4,3,4,5,2,2,5,4,3,4,5,4,5,3]
}

df = pd.DataFrame(data)

df.head()

# =====================================
# Cell 3: Exploratory Data Analysis
# =====================================
print("Dataset Shape:", df.shape)
print("Number of Users:", df['user_id'].nunique())
print("Number of Items:", df['item_id'].nunique())
print("\nRating Distribution:\n", df['rating'].value_counts())

# =====================================
# Cell 4: Create User-Item Matrix
# =====================================

user_item_matrix = df.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)

print("User-Item Matrix:\n")
print(user_item_matrix)

# =====================================
# Cell 5: Train-Test Split
# =====================================

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create train matrix
train_matrix = train_data.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)

# Align columns
train_matrix = train_matrix.reindex(columns=user_item_matrix.columns, fill_value=0)

# =====================================
# Cell 6: Convert to Sparse Matrix
# =====================================

sparse_matrix = csr_matrix(train_matrix.values)

# =====================================
# Cell 7: Matrix Factorization using SVD
# =====================================

# Number of latent factors
k = 2

# Perform Singular Value Decomposition
U, sigma, Vt = svds(sparse_matrix, k=k)

sigma = np.diag(sigma)

# Reconstruct matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

predicted_df = pd.DataFrame(
    predicted_ratings,
    index=train_matrix.index,
    columns=train_matrix.columns
)

# =====================================
# Cell 8: Evaluation Metrics (RMSE & MAE)
# =====================================

actual = []
predicted = []

for row in test_data.itertuples():
    user = row.user_id
    item = row.item_id
    rating = row.rating

    if user in predicted_df.index and item in predicted_df.columns:
        actual.append(rating)
        predicted.append(predicted_df.loc[user, item])

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)

# =====================================
# Cell 9: Recommendation Function
# =====================================

def recommend_items(user_id, num_recommendations=3):
    if user_id not in predicted_df.index:
        return "User not found"

    user_ratings = predicted_df.loc[user_id]
    rated_items = df[df['user_id'] == user_id]['item_id'].values

    recommendations = user_ratings.drop(rated_items)
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(num_recommendations)

# Example Recommendation
print("Recommendations for User 1:")
print(recommend_items(1))

# =====================================
# Cell 10: Visualize Predicted Ratings
# =====================================

plt.figure()
plt.hist(predicted_df.values.flatten(), bins=20)
plt.title("Distribution of Predicted Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# =====================================
# Cell 11: Analysis and Observations
# =====================================
"""
Analysis:

1. Collaborative filtering was implemented using matrix factorization (SVD).
2. User-item interaction matrix was created from ratings data.
3. Latent features were extracted using Singular Value Decomposition.
4. Missing ratings were predicted using reconstructed matrix.
5. Performance was evaluated using RMSE and MAE metrics.
6. Recommendation function suggests top items for each user.

Limitations:
- Small dataset reduces accuracy.
- Cold-start problem for new users/items.

Improvements:
- Use larger datasets (MovieLens)
- Increase latent factors (k)
- Apply deep learning models
- Use hybrid recommendation systems

Conclusion:
Matrix factorization provides an effective way to build collaborative filtering recommendation systems.
"""
