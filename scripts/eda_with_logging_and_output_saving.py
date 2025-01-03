import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Set up logging - ensure logs directory exists
log_dir = r'C:\Users\teble\rossmann-sales-prediction\logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'eda.log'), level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Log start of the process
logger.info("Started data cleaning")

# Create outputs folder if it does not exist (absolute path)
output_dir = r'C:\Users\teble\rossmann-sales-prediction\outputs'
os.makedirs(output_dir, exist_ok=True)

# Load data with explicit dtype for 'StateHoliday' to avoid DtypeWarning
train_path = r'C:\Users\teble\rossmann-sales-prediction\data\train.csv'
test_path = r'C:\Users\teble\rossmann-sales-prediction\data\test.csv'
store_path = r'C:\Users\teble\rossmann-sales-prediction\data\store.csv'
sample_submission_path = r'C:\Users\teble\rossmann-sales-prediction\data\sample_submission.csv'

train = pd.read_csv(train_path, dtype={'StateHoliday': str})
test = pd.read_csv(test_path)
store = pd.read_csv(store_path)
sample_submission = pd.read_csv(sample_submission_path)

# Merge data on 'Store' column
train = train.merge(store, how='left', on='Store')
test = test.merge(store, how='left', on='Store')

# Log the merging step
logger.info("Merged train and test datasets with store data")

# Handle missing values and log the action
logger.info("Handled missing values")
train['CompetitionDistance'] = train['CompetitionDistance'].fillna(train['CompetitionDistance'].median())
train['Sales'] = train['Sales'].fillna(train['Sales'].median())
train['Open'] = train['Open'].fillna(train['Open'].mode()[0])

# Save cleaned data (train, test, store, and sample_submission)
train.to_csv(os.path.join(output_dir, 'cleaned_train.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'cleaned_test.csv'), index=False)
store.to_csv(os.path.join(output_dir, 'cleaned_store.csv'), index=False)
sample_submission.to_csv(os.path.join(output_dir, 'cleaned_sample_submission.csv'), index=False)

# Log the saving of cleaned data
logger.info("Saved cleaned training data to 'cleaned_train.csv'")
logger.info("Saved cleaned test data to 'cleaned_test.csv'")
logger.info("Saved cleaned store data to 'cleaned_store.csv'")
logger.info("Saved cleaned sample_submission data to 'cleaned_sample_submission.csv'")

# Perform Exploratory Data Analysis (EDA) and save visualizations
# Example: Distribution of Promotions
plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Promo')
plt.title('Distribution of Promotions')
plt.xlabel('Promo (1: Promotion Active, 0: No Promotion)')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'distribution_of_promotions.png'))
plt.show()  # Display the plot

# Log that the distribution plot has been saved
logger.info("Saved distribution of promotions plot")

# Sales Behavior Around Holidays (example)
holiday_sales = train.groupby('StateHoliday')['Sales'].mean()
plt.figure(figsize=(8, 6))
holiday_sales.plot(kind='bar', color='skyblue')
plt.title('Sales During Holidays')
plt.xlabel('Holiday')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.savefig(os.path.join(output_dir, 'sales_during_holidays.png'))
plt.show()  # Display the plot

# Log that the holiday sales plot has been saved
logger.info("Saved sales during holidays plot")

# Seasonal Behavior (Monthly Sales Trends)
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
plt.savefig(os.path.join(output_dir, 'monthly_sales_trends.png'))
plt.show()  # Display the plot

# Log the monthly sales trends plot
logger.info("Saved monthly sales trends plot")

# Save further visualizations, such as correlation, promo effectiveness, etc.
# Example: Promo Effectiveness (Average Sales with and without Promotion)
promo_sales = train.groupby('Promo')['Sales'].mean()
plt.figure(figsize=(8, 6))
promo_sales.plot(kind='bar', color=['lightgreen', 'lightcoral'])
plt.title('Average Sales with Promotions')
plt.xlabel('Promo (0: No, 1: Yes)')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.savefig(os.path.join(output_dir, 'average_sales_with_promotions.png'))
plt.show()

# Log the promo effectiveness plot
logger.info("Saved average sales with promotions plot")

# Log completion of EDA
logger.info("Performed EDA and visualizations")

# End of script
logger.info("Completed the EDA process")
