import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create outputs folder if it does not exist (relative path)
os.makedirs('outputs', exist_ok=True)

# Load data (update the paths as necessary)
train = pd.read_csv(r'C:\Users\teble\rossmann-sales-prediction\rossmann-sales-prediction\data\train.csv')
test = pd.read_csv(r'C:\Users\teble\rossmann-sales-prediction\rossmann-sales-prediction\data\test.csv')
store = pd.read_csv(r'C:\Users\teble\rossmann-sales-prediction\rossmann-sales-prediction\data\store.csv')

# Merge data on 'Store' column
train = train.merge(store, how='left', on='Store')
test = test.merge(store, how='left', on='Store')

# 1. Distribution of Promotions
plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Promo')
plt.title('Distribution of Promotions')
plt.xlabel('Promo (1: Promotion Active, 0: No Promotion)')
plt.ylabel('Count')
plt.savefig(os.path.join('outputs', 'distribution_of_promotions.png'))
plt.show()

# 2. Sales Behavior Around Holidays
holiday_sales = train.groupby('StateHoliday')['Sales'].mean()
plt.figure(figsize=(8, 6))
holiday_sales.plot(kind='bar', color='skyblue')
plt.title('Sales During Holidays')
plt.xlabel('Holiday')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.savefig(os.path.join('outputs', 'sales_during_holidays.png'))
plt.show()

# 3. Seasonal Behavior (Monthly Sales Trends)
train['Date'] = pd.to_datetime(train['Date'])
train['Month'] = train['Date'].dt.month
monthly_sales = train.groupby('Month')['Sales'].mean()
plt.figure(figsize=(8, 6))
monthly_sales.plot(kind='line', marker='o', color='green')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.savefig(os.path.join('outputs', 'monthly_sales_trends.png'))
plt.show()

# 4. Correlation Between Sales and Customers
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Customers', y='Sales', data=train, color='orange')
plt.title('Sales vs Customers')
plt.xlabel('Number of Customers')
plt.ylabel('Sales')
plt.savefig(os.path.join('outputs', 'sales_vs_customers.png'))
plt.show()

# 5. Promo Effectiveness (Average Sales with and without Promotion)
promo_sales = train.groupby('Promo')['Sales'].mean()
plt.figure(figsize=(8, 6))
promo_sales.plot(kind='bar', color=['lightgreen', 'lightcoral'])
plt.title('Average Sales with Promotions')
plt.xlabel('Promo (0: No, 1: Yes)')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.savefig(os.path.join('outputs', 'average_sales_with_promotions.png'))
plt.show()

# 6. Competitor Distance Impact on Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x='CompetitionDistance', y='Sales', data=train, color='blue')
plt.title('Sales vs Competition Distance')
plt.xlabel('Competition Distance')
plt.ylabel('Sales')
plt.savefig(os.path.join('outputs', 'sales_vs_competition_distance.png'))
plt.show()
