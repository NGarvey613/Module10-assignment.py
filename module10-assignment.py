# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

from io import StringIO

csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""

# Load data
customer_data_csv = StringIO(csv_content)
raw_df = pd.read_csv(customer_data_csv)

# ---- 1. Explore ----
initial_missing_counts = raw_df.isna().sum()
initial_duplicate_count = raw_df.duplicated().sum()

# ---- 2. Missing Values ----
missing_value_report = raw_df.isna().sum()

# Fill satisfaction
satisfaction_median = raw_df["satisfaction_rating"].median()
raw_df["satisfaction_rating"] = raw_df["satisfaction_rating"].fillna(satisfaction_median)

# Fill last_purchase
date_fill_strategy = "forward_fill"
raw_df["last_purchase"] = raw_df["last_purchase"].ffill()

# Fill others
raw_df["phone"] = raw_df["phone"].fillna("Unknown")
raw_df["last_name"] = raw_df["last_name"].fillna("Unknown")
raw_df["loyalty_status"] = raw_df["loyalty_status"].fillna("None")
raw_df["age"] = raw_df["age"].fillna(raw_df["age"].median())

df_no_missing = raw_df.copy()
df_no_missing = df_no_missing.copy()

# ---- 3. Data Types ----
df_typed = df_no_missing.copy()

df_typed["join_date"] = pd.to_datetime(df_typed["join_date"], errors="coerce")
df_typed["last_purchase"] = pd.to_datetime(df_typed["last_purchase"], errors="coerce")

# Clean money column
df_typed["total_spent"] = df_typed["total_spent"].replace('[\$,]', '', regex=True)
df_typed["total_spent"] = pd.to_numeric(df_typed["total_spent"], errors="coerce")

df_typed["total_purchases"] = pd.to_numeric(df_typed["total_purchases"], errors="coerce")
df_typed["age"] = pd.to_numeric(df_typed["age"], errors="coerce")

# ---- 4. Text Cleaning ----
df_text_cleaned = df_typed.copy()

df_text_cleaned["first_name"] = df_text_cleaned["first_name"].str.title()
df_text_cleaned["last_name"] = df_text_cleaned["last_name"].str.title()

df_text_cleaned["preferred_category"] = df_text_cleaned["preferred_category"].str.title()

# Phone formatting
phone_format = "(XXX) XXX-XXXX"
df_text_cleaned["phone"] = df_text_cleaned["phone"].str.replace(r'\D', '', regex=True)
df_text_cleaned["phone"] = df_text_cleaned["phone"].apply(
    lambda x: f"({x[:3]}) {x[3:6]}-{x[6:]}" if len(x) == 10 else "Invalid"
)

# ---- 5. Duplicates ----
duplicate_count = df_text_cleaned.duplicated().sum()
df_no_duplicates = df_text_cleaned.drop_duplicates()

# ---- 6. Derived Features ----
df_no_duplicates = df_no_duplicates.copy()
df_no_duplicates["days_since_last_purchase"] = (datetime.now() - df_no_duplicates["last_purchase"]).dt.days

df_no_duplicates["average_purchase_value"] = (
    df_no_duplicates["total_spent"] / df_no_duplicates["total_purchases"]
)

def purchase_category(x):
    if x >= 10:
        return "High"
    elif x >= 5:
        return "Medium"
    else:
        return "Low"

df_no_duplicates["purchase_frequency_category"] = df_no_duplicates["total_purchases"].apply(purchase_category)

# ---- 7. Cleanup ----
df_renamed = df_no_duplicates.rename(columns={
    "customer_id": "Customer ID",
    "first_name": "First Name",
    "last_name": "Last Name",
    "total_spent": "Total Spent"
})

df_final = df_renamed.copy()
df_final = df_final.sort_values(by="Total Spent", ascending=False)

# ---- 8. Insights ----
avg_spent_by_loyalty = df_final.groupby("loyalty_status")["Total Spent"].mean()
category_revenue = df_final.groupby("preferred_category")["Total Spent"].sum().sort_values(ascending=False)
satisfaction_spend_corr = df_final["satisfaction_rating"].corr(df_final["Total Spent"])

# ---- 9. REPORT ----
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

print("\nData Quality Issues:")
print(f"- Missing Values: {initial_missing_counts.sum()}")
print(f"- Duplicates: {initial_duplicate_count}")
print("- Data Type Issues: currency formatting, mixed date formats")

print("\nStandardization Changes:")
print("- Names: Converted to proper case")
print("- Categories: Standardized capitalization")
print(f"- Phone Numbers: {phone_format}")

print("\nKey Business Insights:")
print(f"- Customer Base: {len(df_final)} customers")
print("- Revenue by Loyalty:")
print(avg_spent_by_loyalty.to_string())
print(f"- Top Category: {category_revenue.idxmax()} with ${category_revenue.max():.2f}")

print("\nFinal Cleaned Data:")
print(df_final.head())