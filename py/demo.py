from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("pyspark_benchmark").getOrCreate()
sc = spark.sparkContext
rdd = sc.parallelize([1, 2, 3, 4])
res = rdd.map(lambda x: x**2).collect()
print(res)
