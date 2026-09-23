import pandas as pd
import matplotlib.pyplot as plt
import os

# Task 4 cluster summary from the Spark output
data = {
    "cluster": [0, 1, 2],
    "number_of_customers": [399537, 193231, 399177],
    "avg_sales": [264.0381, 999.1700, 262.3454],
    "avg_profit": [100.9045, 417.1418, 100.2738],
    "avg_loyalty": [75.3760, 49.8514, 24.6104]
}

df = pd.DataFrame(data)

# Make sure output directory exists
os.makedirs("task4_visualisations", exist_ok=True)

# ==========================================
# BAR CHART: Customers by Cluster
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["cluster"].astype(str),
    df["number_of_customers"]
)

plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.title("Number of Customers by Customer Segment")
plt.tight_layout()

plt.savefig(
    "task4_visualisations/customer_segments_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# PIE CHART: Customer Distribution
# ==========================================

plt.figure(figsize=(7, 7))

plt.pie(
    df["number_of_customers"],
    labels=[
        "Cluster 0 - High Loyalty",
        "Cluster 1 - High Value",
        "Cluster 2 - Low Loyalty"
    ],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Distribution by Segment")
plt.tight_layout()

plt.savefig(
    "task4_visualisations/customer_segments_pie_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("========================================")
print("TASK 4 VISUALISATIONS CREATED")
print("========================================")
print("Bar chart:")
print("task4_visualisations/customer_segments_bar_chart.png")
print()
print("Pie chart:")
print("task4_visualisations/customer_segments_pie_chart.png")
