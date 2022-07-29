import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
from pyspark.sql import SparkSession
import sys, os, re
from pyspark.sql.functions import isnan, when, count
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, MultilayerPerceptronClassifier, LinearSVC
from pyspark.mllib.evaluation import MulticlassMetrics

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("nuclear_plants_small_dataset.csv", header=True, inferSchema=True)
for col in df.columns:
    #removes all the whitespaces in the dataframe headers
    df = df.withColumnRenamed(col,re.sub(r'\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*','',col.replace(' ', '')))

# Task 1 - identify any missing data in the dataset ########################################################################################################################################
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

#Task 2 - collect summary statistics ##########################################################################################################################################################
def summary_stat_table(df):
    def mode_table(df, group):
        #gets the most common value from the dataframe and places the results in a pandas dataframe which is transposed
        mode = pd.DataFrame([df.groupby(i).count().orderBy("count", ascending=False).first()[0] for i in df.columns]).transpose()
        #new pandas dataframe to converted into a spark dataframe
        mode_dataframe = spark.createDataFrame(mode)
        #group column name is replaced with 'mode'
        mode_dataframe = mode_dataframe.replace((group), 'mode')
        return mode_dataframe

    def variance_table(df, group):
        #gets the variance values from the dataframe and places the results in a pandas dataframe which is transposed
        variance = pd.DataFrame([df.agg({i:"variance"}).first() for i in df.columns]).transpose()
        #new pandas dataframe to converted into a spark dataframe
        variance_dataframe = spark.createDataFrame(variance)
        #group column name is replaced with 'variance'
        variance_dataframe = variance_dataframe.replace((group), 'variance')
        return variance_dataframe

    def table_union(df, group):
        #groups the values together based on the 'group' variable (Normal or Abnormal)
        mode = mode_table(df.where(df.Status == group).select(df.columns),group)
        variance = variance_table(df.where(df.Status == group).select(df.columns),group)
        #gets the statistical summary of the other required statistics based on the 'group' variable
        table = df.where(df.Status == group).select(df.columns[1:]).summary("min","max","mean","50%")
        #unifies all the tables together
        table = table.union(mode).union(variance)
        #replaces values for better readability
        table = table.replace("50%", 'median')
        table = table.replace("NaN", 'variance')
        return table

    print("Normal Nuclear Reactor Statistics Table: ","\n")
    #displays the statistical values based on the normal status values
    table_union(df,"Normal").show()
    print("Abnormal Nuclear Reactor Statistics Table: ","\n")
    #displays the statistical values based on the abnormal status values
    table_union(df,"Abnormal").show()
summary_stat_table(df)

columns = df.columns
columns.remove('Status')

#Task 2 - boxplot for each group's features ##########################################################################################################################################################
def box_plot(df):
    listids = [list(x.asDict().values())[0] 
            #collects each distinct value from the status feature
           for x in df.select("Status").distinct().collect()]
    #splits the data based on the status features values (normal or abnormal)
    Status_Split = [df.where(df.Status == x) for x in listids]
    #abnormal values are placed in abnormal_group variable
    abnormal_group = Status_Split[0]
    normal_group = Status_Split[1]
    #converts each status group variable to pandas for plotting
    abnormalDF = abnormal_group.toPandas()
    normalDF = normal_group.toPandas() 
    #sets the plotly figure to have 1 row and columns based on the length of the dataset
    fig = make_subplots(rows=1, cols=len(columns))
    #loops through each feature and each column
    for i, column in enumerate(columns):
        fig.add_trace(
            #assigns the boxplot column to the figure
            go.Box(y=abnormalDF[column],name=column),
            #sets the row to 1 and moves the column index across by 1
            row=1, col=i+1
        )
    #adds a title to the figure
    fig.update_layout(title_text='Box Plot for Abnormal Reactor Features')
    fig.update_traces(boxpoints='all', jitter=.3)
    fig.show()
    #does the same as the above code but for the normal dataframe values
    fig = make_subplots(rows=1, cols=len(columns))
    for i, column in enumerate(columns):
        fig.add_trace(
            go.Box(y=normalDF[column],name=column),
            row=1, col=i+1
        )
    fig.update_layout(title_text='Box Plot for Normal Reactor Features')
    fig.update_traces(boxpoints='all', jitter=.3)
    fig.show()

box_plot(df)

#Task 3 - correlation matrix ###############################################################################################################################################################################
def correlation_matrix(df):
    #transforms the dataframe to be used as a correlation matrix
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    #combines all the feature columns into one column and renames to feature
    df_vector = assembler.transform(df).select("features")
    #collects the values from the features column
    matrix = Correlation.corr(df_vector, "features").collect()[0][0]
    #converts the matrix to an array then to a list
    corrmatrix = matrix.toArray().tolist()
    #puts the correlation matrix into a heatmap for analysis
    sns.heatmap(corrmatrix, xticklabels=columns,yticklabels=columns, cmap="Blues", annot=True)
    plt.show()

correlation_matrix(df)

#Task 4 - Shuffle and split dataset ###############################################################################################################################################################################
def split_data(df):
    #splits the data into training and testing data with a 70-30 ratio
    train_data,test_data = df.randomSplit([0.7,0.3])
    #counts the amount of values are in each training set based on group values
    print("Total Dataset Total Count: " + str(df.count()),"\n")
    print("Training Dataset Total Count: " + str(train_data.count()))
    print("Training Dataset Normal Count: " + str(train_data.where(train_data.Status == 'Normal').count()))
    print("Training Dataset Abnormal Count: " + str(train_data.where(train_data.Status == 'Abnormal').count()),"\n")
    print("Testing Dataset Total Count: " + str(test_data.count()))
    print("Testing Dataset Normal Count: " + str(test_data.where(test_data.Status == 'Normal').count()))
    print("Testing Dataset Abnormal Count: " + str(test_data.where(test_data.Status == 'Abnormal').count()))

split_data(df)

#Task 5 - machine learning models ###############################################################################################################################################################################
def dataset_models():
    def model_scores(model):
        #sets the matrix to the status index and the prediction columns
        matrix = model["StatusIndex","prediction"]
        #groups the status index column and gets the count 
        matrix.groupby('StatusIndex').agg({'StatusIndex': 'count'})
        #groups the prediction column and gets the count 
        matrix.groupby('prediction').agg({'prediction': 'count'})
        #calculates the error rate by getting the incorrectly classified samples divided by the classified samples
        error_rate = matrix.filter(matrix.StatusIndex != matrix.prediction).count() / matrix.count()
        #creates a matric using the matrix variable and maps the values
        matrix = MulticlassMetrics(matrix.rdd.map(tuple))
        #converts the matrix to a confusion matrix then to an array
        matrix = matrix.confusionMatrix().toArray()
        #gets the sensitivity score by selecting different parts of the matrix (either TP,FP,TN,FN)
        sensitivity_score = matrix[0,0]/(matrix[0,0]+matrix[0,1])
        #gets the specificity score by selecting different parts of the matrix (either TP,FP,TN,FN)
        specificity_score = matrix[1,1]/(matrix[1,0]+matrix[1,1])
        print("Model error rate: %.2f%%" % (error_rate * 100)) 
        print("Model sensitivity: %.2f%%" % (sensitivity_score * 100))     
        print("Model specificity: %.2f%%" % (specificity_score * 100))
        print("\n")

    #transforms the dataframe to be used as a correlation matrix
    assembler = VectorAssembler(inputCols=columns,outputCol="features")
    #transforms the new assembler variable based on the dataframe values
    output = assembler.transform(df)
    #indexes the status column to replace the string data types with integer data types
    indexer = StringIndexer(inputCol="Status", outputCol="StatusIndex")
    #fits and transforms the output data features
    output_fixed = indexer.fit(output).transform(output)
    #selects the feature column and the statusindex column
    final_data = output_fixed.select("features",'StatusIndex')
    #splits the data into training and testing data with a 70-30 ratio
    train_data,test_data = final_data.randomSplit([0.7,0.3])

    #Task 5 - decision tree model ###################################################################################################################################################################
    #assigns the dtc model with the statusindex values and the features column values
    dtc = DecisionTreeClassifier(labelCol='StatusIndex',featuresCol='features')
    #fits the dtc model with the training data
    dtc_model = dtc.fit(train_data)
    #predicts the output of the dtc model with the testing data
    dtc_prediction = dtc_model.transform(test_data)
    model_scores(dtc_prediction)

    #Task 5 - support vector machine model ###############################################################################################################################################################################
    #assigns the SVC model with the statusindex values and the features column values
    svm = LinearSVC(labelCol='StatusIndex',featuresCol='features')
    #fits the svm model with the training data
    svm_model = svm.fit(train_data)
    #predicts the output of the svm model with the testing data
    svm_prediction = svm_model.transform(test_data)
    model_scores(svm_prediction)

    #Task 5 - artificial neural network ###############################################################################################################################################################################
    #sets the amount of layers for the neural network
    layers = [len(assembler.getInputCols()), 4, 4, 2]
    #assigns the ann model with the statusindex values the features column values and the layers list
    ann = MultilayerPerceptronClassifier(labelCol='StatusIndex',featuresCol='features',layers=layers)
    #fits the ann model with the training data
    ann_model = ann.fit(train_data)
    #predicts the output of the ann model with the testing data
    ann_prediction = ann_model.transform(test_data)
    model_scores(ann_prediction)

dataset_models()

#Task 8 - MapReduce summary statistics ###############################################################################################################################################################################
df2 = spark.read.csv("nuclear_plants_big_dataset.csv",header=True, inferSchema=True)
for col in df2.columns:
    #removes all the whitespaces in the dataframe headers
    df2 = df2.withColumnRenamed(col,re.sub(r'\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*','',col.replace(' ', '')))

#gets minimum values of each feature and turns them into a dataframe
def minimum_value(rdd):
    #uses MapReduce to get the minimum value for each feature - 'i' is the column name iteration. Dataframe is transposed for easier readability
    min = pd.DataFrame([rdd.map(lambda x:(x[i])).reduce(lambda x,y: (x if x < y else y))]for i in df2.columns).transpose()
    #dataframe is converted to a spark dataframe to unify each table together
    min_dataframe = spark.createDataFrame(min)
    return min_dataframe

#gets maximum values of each feature and turns them into a dataframe
def maximum_value(rdd):
    #uses MapReduce to get the maximum value for each feature - 'i' is the column name iteration. Dataframe is transposed for easier readability
    max = pd.DataFrame([rdd.map(lambda x:(x[i])).reduce(lambda x,y: (x if x > y else y))]for i in df2.columns).transpose()
    #dataframe is converted to a spark dataframe to unify each table together
    max_dataframe = spark.createDataFrame(max)
    return max_dataframe

#gets average values of each feature and turns them into a dataframe
def average_value(rdd):
    #uses MapReduce to get the average value for each feature - all values are added together and divided by the total amount of values (rows) - 'i' is the column name iteration. Dataframe is transposed for easier readability
    mean = pd.DataFrame([rdd.map(lambda x:(x[i])).reduce(lambda x,y: (x+y)) / df2.count()] for i in df2.columns).transpose()
    #dataframe is converted to a spark dataframe to unify each table together
    mean_dataframe = spark.createDataFrame(mean)
    return mean_dataframe

#for an rdd
df2 = df2.drop("Status")
#convert to rdd to use map and reduce functions
rdd = df2.rdd
min_dataframe = minimum_value(rdd)
max_dataframe = maximum_value(rdd)
mean_dataframe = average_value(rdd)
#unifies each statistic table together
final_table = min_dataframe.union(max_dataframe).union(mean_dataframe)
final_table.show()