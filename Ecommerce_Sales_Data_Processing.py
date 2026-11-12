# Databricks notebook source
# MAGIC %md
# MAGIC # PEI Project — Sales Data Transformation Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Imports

# COMMAND ----------

# MAGIC %pip install openpyxl
# MAGIC import pandas as pd
# MAGIC from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC Reading Source Files

# COMMAND ----------

base_path = "/Volumes/workspace/default/pei_assessment"

orders_df = spark.read.json(f"{base_path}/Orders.json", multiLine=True)
customers_df = spark.createDataFrame(pd.read_excel(f"{base_path}/Customer.xlsx", dtype={"phone": str}))
#products_df = spark.read.csv(f"{base_path}/Products.csv", header=True, inferSchema=True)
products_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("multiLine", True) \
    .csv(f"{base_path}/Products.csv")

print("orders_df:", orders_df.count(), "rows")
print("customers_df:", customers_df.count(), "rows")
print("products_df:", products_df.count(), "rows")

# COMMAND ----------

# MAGIC %md
# MAGIC Writing Raw tables as Parquet as the columns name have " " and Delta doesnt support it.

# COMMAND ----------

orders_df.write.mode("overwrite").parquet(f"{base_path}/raw_tables/raw_orders")
customers_df.write.mode("overwrite").parquet(f"{base_path}/raw_tables/raw_customers")
products_df.write.mode("overwrite").parquet(f"{base_path}/raw_tables/raw_products")


# COMMAND ----------

# MAGIC %md
# MAGIC Enriched Customer Table after cleaning the Customer's Name

# COMMAND ----------

customer_enriched_df = customers_df.withColumn(
    "Customer Name",
    F.trim(
        F.regexp_replace(
            F.regexp_replace(
                F.col("Customer Name"),
                r"[^a-zA-Z\s]",
                ""
            ),
            r"\s+",
            " "
        )
    )
)
customer_enriched_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Enriched Product Table (Didnt drop the duplicates as the product name and other columns were unique)

# COMMAND ----------

product_enriched_df = products_df
product_enriched_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Created a lookup for product table post dropping the duplicates for creating enriched order.
# MAGIC  

# COMMAND ----------

product_lookup_df = product_enriched_df.select(
    "Product ID",
    "Category",
    "Sub-Category"
).dropDuplicates(["Product ID"])

# COMMAND ----------

# MAGIC %md
# MAGIC Created Enriched Order Table as per requirment.

# COMMAND ----------

enriched_orders_df = orders_df \
    .join(customer_enriched_df.select(
            "Customer ID",
            "Customer Name",
            "Country"
        ),
        on="Customer ID",
        how="left"
    ) .join(
        product_lookup_df,
        on="Product ID",
        how="left"
    ).withColumn(
        "Profit",
        F.round(F.col("Profit"), 2)
    )

# COMMAND ----------

Writing all Enriched table as Parquet Tables.

# COMMAND ----------

customer_enriched_df.write.mode("overwrite").parquet(f"{base_path}/enriched_tables/enriched_customers")
product_enriched_df.write.mode("overwrite").parquet(f"{base_path}/enriched_tables/enriched_products")
enriched_orders_df.write.mode("overwrite").parquet(f"{base_path}/enriched_tables/enriched_orders")

# COMMAND ----------

# MAGIC %md
# MAGIC Creation of Aggregated Table

# COMMAND ----------

orders_with_year_df = enriched_orders_df.withColumn("Order Date",F.to_date(F.col("Order Date"), "d/M/yyyy")
    ).withColumn("Year",F.year(F.col("Order Date")))

# COMMAND ----------

aggregate_df = orders_with_year_df \
    .groupBy(
        "Year","Category","Sub-Category","Customer ID","Customer Name"
    ).agg(F.round(F.sum("Profit"), 2).alias("Total Profit"))

aggregate_df.show(5)

# COMMAND ----------

aggregate_df.write.mode("overwrite").parquet(f"{base_path}/aggregated_table/aggregated_orders")

# COMMAND ----------

orders_with_year_df.createOrReplaceTempView("enriched_orders")

# COMMAND ----------

# MAGIC %md
# MAGIC SQL Output for Profit By Year

# COMMAND ----------

profit_by_year_df = spark.sql("""
    SELECT
        Year,
        ROUND(SUM(Profit), 2) AS Total_Profit
    FROM enriched_orders
    GROUP BY Year
    ORDER BY Year
""")

profit_by_year_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC SQL Output for Profit by Year and Product Category

# COMMAND ----------

profit_by_year_category_df = spark.sql("""
    SELECT
        Year,
        Category,
        ROUND(SUM(Profit), 2) AS Total_Profit
    FROM enriched_orders
    GROUP BY Year, Category
    ORDER BY Year, Category
""")

profit_by_year_category_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Profit By Customer

# COMMAND ----------

profit_by_customer_df = spark.sql("""
    SELECT
        `Customer ID`,
        `Customer Name`,
        ROUND(SUM(Profit), 2) AS Total_Profit
    FROM enriched_orders
    GROUP BY `Customer ID`, `Customer Name`
    ORDER BY `Customer ID`
""")

profit_by_customer_df.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Profit By Customer and Year

# COMMAND ----------

profit_by_customer_year_df = spark.sql("""
    SELECT
        `Customer ID`,
        `Customer Name`,
        Year,
        ROUND(SUM(Profit), 2) AS Total_Profit
    FROM enriched_orders
    GROUP BY
        `Customer ID`,
        `Customer Name`,
        Year
    ORDER BY
        `Customer ID`,
        Year
""")

profit_by_customer_year_df.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Test cases for all the Tables

# COMMAND ----------

import unittest

class TestCustomerEnrichment(unittest.TestCase):

    def test_customer_row_count(self):
        self.assertEqual(customer_enriched_df.count(), 793)

    def test_customer_id_uniqueness(self):
        duplicate_count = (
            customer_enriched_df.count()
            - customer_enriched_df.select("Customer ID").distinct().count()
        )
        self.assertEqual(duplicate_count, 0)

    def test_customer_columns(self):
        self.assertEqual(len(customer_enriched_df.columns), 11)

    def test_null_customer_ids(self):
        null_count = customer_enriched_df.filter(
            F.col("Customer ID").isNull()
        ).count()
        self.assertEqual(null_count, 0)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCustomerEnrichment)
unittest.TextTestRunner(verbosity=2).run(suite)

# COMMAND ----------

class TestProductEnrichment(unittest.TestCase):

    def test_product_row_count(self):
        self.assertEqual(product_enriched_df.count(), 1851)

    def test_product_columns(self):
        self.assertEqual(len(product_enriched_df.columns), 6)

    def test_product_id_nulls(self):
        null_count = product_enriched_df.filter(
            F.col("Product ID").isNull()
        ).count()
        self.assertEqual(null_count, 0)

    def test_product_price_data_type(self):
        price_type = dict(product_enriched_df.dtypes)["Price per product"]
        self.assertEqual(price_type, "double")

    def test_product_lookup_uniqueness(self):
        duplicate_count = (
            product_lookup_df.count()
            - product_lookup_df.select("Product ID").distinct().count()
        )
        self.assertEqual(duplicate_count, 0)

    def test_product_lookup_count(self):
        self.assertEqual(product_lookup_df.count(), 1818)


suite = unittest.TestLoader().loadTestsFromTestCase(TestProductEnrichment)
unittest.TextTestRunner(verbosity=2).run(suite)

# COMMAND ----------

class TestEnrichedOrders(unittest.TestCase):

    def test_order_row_count(self):
        self.assertEqual(enriched_orders_df.count(), 9994)

    def test_order_column_count(self):
        self.assertEqual(len(enriched_orders_df.columns), 15)

    def test_customer_columns_exist(self):
        self.assertIn("Customer Name", enriched_orders_df.columns)
        self.assertIn("Country", enriched_orders_df.columns)

    def test_product_columns_exist(self):
        self.assertIn("Category", enriched_orders_df.columns)
        self.assertIn("Sub-Category", enriched_orders_df.columns)

    def test_profit_rounding(self):
        invalid_count = enriched_orders_df.filter(
            F.col("Profit") != F.round(F.col("Profit"), 2)
        ).count()
        self.assertEqual(invalid_count, 0)

    def test_customer_unmatched_records(self):
        null_count = enriched_orders_df.filter(
            F.col("Customer Name").isNull()
        ).count()
        self.assertEqual(null_count, 137)

    def test_product_unmatched_records(self):
        null_count = enriched_orders_df.filter(
            F.col("Category").isNull()
        ).count()
        self.assertEqual(null_count, 204)


suite = unittest.TestLoader().loadTestsFromTestCase(TestEnrichedOrders)
unittest.TextTestRunner(verbosity=2).run(suite)

# COMMAND ----------

class TestAggregateTable(unittest.TestCase):

    def test_aggregate_columns(self):
        expected_columns = [
            "Year",
            "Category",
            "Sub-Category",
            "Customer ID",
            "Customer Name",
            "Total Profit"
        ]

        self.assertEqual(
            aggregate_df.columns,
            expected_columns
        )

    def test_aggregate_row_count(self):
        self.assertEqual(aggregate_df.count(), 8129)

    def test_no_duplicate_group_combinations(self):
        duplicate_count = (
            aggregate_df.count()
            - aggregate_df.dropDuplicates([
                "Year",
                "Category",
                "Sub-Category",
                "Customer ID",
                "Customer Name"
            ]).count()
        )

        self.assertEqual(duplicate_count, 0)

    def test_total_profit_reconciliation(self):
        source_profit = orders_with_year_df.agg(
            F.round(F.sum("Profit"), 2)
        ).first()[0]

        aggregate_profit = aggregate_df.agg(
            F.round(F.sum("Total Profit"), 2)
        ).first()[0]

        self.assertEqual(source_profit, aggregate_profit)


suite = unittest.TestLoader().loadTestsFromTestCase(TestAggregateTable)
unittest.TextTestRunner(verbosity=2).run(suite)

# COMMAND ----------

