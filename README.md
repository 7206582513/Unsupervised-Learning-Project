# 💳 Bank Customer Segmentation using Unsupervised Learning

This project applies unsupervised machine learning techniques to segment bank customers based on their credit card usage behavior. The aim is to help financial institutions understand customer types for better personalization, risk mitigation, and targeted marketing.

---

## 📌 Project Objective

To identify distinct customer groups using their transaction and credit behavior, enabling:
- 🎯 Personalized promotional offers
- ⚠️ Early identification of high-risk customers
- 📈 Increased customer retention and engagement

---

## 🧾 Dataset Description

The anonymized dataset includes transaction-level behavior of bank customers.  
**Key features:**
- `spending`: Total money spent
- `advance_payments`: Prepaid amounts
- `probability_of_full_payment`: Likelihood of full bill clearance
- `current_balance`: Present outstanding credit
- `credit_limit`: Maximum allowed credit
- `min_payment_amt`, `max_spent_in_single_shopping`
- ➕ **Derived**: `balance_utilization = current_balance / credit_limit`

---

## 🛠️ Pipeline Summary

```text
Raw Data
   ↓
Feature Engineering (e.g., Balance Utilization)
   ↓
Outlier Removal (IQR Method)
   ↓
Standard Scaling
   ↓
PCA → Autoencoder (Dimensionality Reduction)
   ↓
UMAP (2D Projection)
   ↓
HDBSCAN (Density-Based Clustering)
   ↓
Evaluation (Silhouette Score = 0.7272)

![ChatGPT Image May 21, 2025, 02_15_18 PM](https://github.com/user-attachments/assets/e5cf2368-9c2e-4373-9500-3a71d7edae01)


📊 Techniques Used
Stage	Tool / Algorithm
Scaling	StandardScaler (Sklearn)
Dimensionality Reduction	PCA + Autoencoder (TensorFlow)
2D Projection	UMAP
Clustering	HDBSCAN
Evaluation	Silhouette Score (Sklearn)


📈 Clustering Results
Cluster	Profile	Traits
0	High Spenders with Capacity	High max spend, low utilization, high credit
1	Over-utilized Customers	High utilization, medium payments, risky
2	Low Engagement Customers	Low spending, low payments, safe but inactive
3	Irregular High Spenders	High max spend, low payment likelihood
-1	Noise / Outliers	Inconsistent or rare patterns


💡 Business Recommendations
🎁 Cluster 0: Offer reward programs and premium credit cards.

🧾 Cluster 1: Provide financial counseling and EMI plans.

📩 Cluster 2: Cross-sell savings and debit products.

⚠️ Cluster 3: Send alerts and create fraud/risk watchlists.

🔍 Noise: Analyze or flag manually for review.

🧪 How to Run
Clone the repository:
git clone https://github.com/yourusername/bank-customer-segmentation.git
cd bank-customer-segmentation

Install dependencies:
pip install -r requirements.txt
Launch the notebook in Google Colab or Jupyter:

bash
Copy
Edit
jupyter notebook Bank_Segmentation_Model.ipynb
🧠 Learning Outcomes
Hands-on with real-world customer data

Apply deep learning-based non-linear feature reduction

Understand density-based clustering and noise filtering

Evaluate clustering models using Silhouette Score

📂 Project Structure
csharp
Copy
Edit
📁 project-root/
├── 📊 Bank_Segmentation_Model.ipynb
├── 📄 utils.py
├── 📁 static/              # Cluster plots
├── 📁 data/                # CSV file(s)
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 report.pdf           # Business Report
🔖 License
This project is licensed under the MIT License.

👤 Author
Made with ❤️ by Rohit Makani
Student, B.S in Data Science & AI
IIT Guwahati
