import os
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Define file path for dataset and model
model_path = "./credit_risk_model"
data_path = "german.data"
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Check if file exists, otherwise download it
if not os.path.exists(data_path):
    response = requests.get(data_url)
    with open(data_path, "wb") as file:
        file.write(response.content)

# Initialize Spark Session
spark = SparkSession.builder.appName("CreditScoringTraining").getOrCreate()

# Load data into Spark DataFrame
df = spark.read.csv(data_path, sep=" ", inferSchema=True)

# Rename columns
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRate", 
           "PersonalStatus", "Debtors", "ResidenceDuration", "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", 
           "Job", "PeopleLiable", "Telephone", "ForeignWorker", "CreditRisk"]
df = df.toDF(*columns)

# Convert target variable (1 = Good, 2 = Bad credit risk)
df = df.withColumn("CreditRisk", when(col("CreditRisk") == 1, 0).otherwise(1))

# Split dataset first (Avoid Data Leakage)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Handle class imbalance: Oversample only in training data
majority_train = train_df.filter(col("CreditRisk") == 0)
minority_train = train_df.filter(col("CreditRisk") == 1)

ratio = majority_train.count() / minority_train.count()
minority_train_oversampled = minority_train.sample(withReplacement=True, fraction=ratio, seed=42)
train_df_balanced = majority_train.union(minority_train_oversampled)

# Encode categorical features (Fit StringIndexer on Training Data only)
categorical_cols = ["Status", "CreditHistory", "Purpose", "Savings", "Employment", "PersonalStatus", "Debtors", 
                    "Property", "OtherInstallment", "Housing", "Job", "Telephone", "ForeignWorker"]

indexers = {}
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_Index")
    indexers[col_name] = indexer.fit(train_df_balanced)  # Fit on training data only

# Apply transformations
for col_name in categorical_cols:
    train_df_balanced = indexers[col_name].transform(train_df_balanced)
    test_df = indexers[col_name].transform(test_df)  # Transform test set using the same mapping

# Feature Engineering
feature_cols = ["Duration", "CreditAmount", "InstallmentRate", "ResidenceDuration", "Age", "ExistingCredits", "PeopleLiable"]
feature_cols += [col + "_Index" for col in categorical_cols]

#Selecting top 10 important features
#feature_cols = ["Status_Index","CreditAmount","Duration","Purpose_Index","Age",
#               "Employment_Index","Savings_Index","CreditHistory_Index","Property_Index","InstallmentRate"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df_balanced = assembler.transform(train_df_balanced)
test_df = assembler.transform(test_df)

# Train Random Forest Model with optimized parameters
rf = RandomForestClassifier(
    labelCol="CreditRisk", 
    featuresCol="features", 
    numTrees=300, 
    maxDepth=20, 
    minInstancesPerNode=5, 
    maxBins=40, 
    featureSubsetStrategy="sqrt", 
    impurity="entropy", 
    subsamplingRate=0.8, 
    minInfoGain=0.01, 
    seed=42
)
model = rf.fit(train_df_balanced)

# Save model
model.write().overwrite().save(model_path)

# Save fitted StringIndexer models
indexer_dir = "./indexers"
os.makedirs(indexer_dir, exist_ok=True)

for col_name, indexer_model in indexers.items():
    indexer_model.save(os.path.join(indexer_dir, f"{col_name}_indexer"))

# Evaluate Model
evaluator = BinaryClassificationEvaluator(labelCol="CreditRisk", metricName="areaUnderROC")
predictions = model.transform(test_df)
roc_auc = evaluator.evaluate(predictions)
print(f"Random Forest ROC-AUC Score: {roc_auc}")

# Feature Importance
feature_importances = model.featureImportances
sorted_features = sorted(zip(feature_cols, feature_importances), key=lambda x: x[1], reverse=True)

print("Feature Importance:")
for feature, importance in sorted_features:  # Show top 10 features
    print(f"{feature}: {importance}")





# Stop Spark Session
spark.stop()
