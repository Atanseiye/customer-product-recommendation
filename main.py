import joblib
import numpy as np
from models import customer_product_matrix
from models import customer_product_matrix_filled
from models import train_data
from models import product_data
from models import predict_ratings
from models import predict_content_based
from models import hybrid_predict
from models import recommend_products
from models import recommend



# Example: Recommend top 5 products for a specific customer
customer_id = 'C189076'
top_n = 3
recommended_products = recommend(customer_id, top_n)

print(f"Top {top_n} products recommended for customer {customer_id}: {recommended_products}")
