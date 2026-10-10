"""Chronon Integration Tasks for Uniflow Pipeline.

End-to-end Chronon data preparation with local Spark.

Migrated from core ``michelangelo``'s
``python/examples/amazon_books_qwen/chronon_tasks.py``.

Adaptations from core:
- Import paths repointed from ``examples.amazon_books_qwen.*`` to
  ``michelangelo_examples.amazon_books_qwen.pipelines.train.*``.
- Dropped the ``try/except ModuleNotFoundError`` dual-import fallback
  (not needed for an always-installed package).
- ``data_dir`` resolution uses ``__file__`` instead of branching on
  ``Path(os.getcwd()).name == "python"`` (a core-monorepo CWD assumption).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import michelangelo.uniflow.core as uniflow
from ai.chronon.api.ttypes import GroupBy, StagingQuery
from ai.chronon.repo.compile import thrift_simple_json_protected
from michelangelo.uniflow.plugins.spark import SparkTask
from michelangelo.workflow.variables import DatasetVariable
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    concat_ws,
    count,
    lit,
    rand,
    when,
)
from pyspark.sql.functions import (
    max as spark_max,
)
from pyspark.sql.functions import (
    min as spark_min,
)

from michelangelo_examples.amazon_books_qwen.pipelines.train.data.group_bys.amazon_books.book_features import (
    book_popularity,
    book_velocity,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.data.staging_queries.amazon_books.books_reviews import (
    base_table,
)


def _setup_chronon_environment():
    """Set up Chronon environment with JAR and directories."""
    print("Setting up Chronon environment...")

    # Try local JAR first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_jar_path = os.path.join(script_dir, "data", "chronon-spark.jar")

    if os.path.exists(local_jar_path):
        print(f"Found local Chronon JAR at {local_jar_path}")
        jar_path = local_jar_path
    else:
        # Download from Maven Central
        jar_path = "/tmp/chronon/chronon-spark.jar"
        if not os.path.exists(jar_path):
            print(f"Chronon JAR not found at {jar_path}")
            os.makedirs("/tmp/chronon", exist_ok=True)
            jar_url = (
                "https://repo1.maven.org/maven2/ai/chronon/"
                "spark_uber_2.12/0.0.23/spark_uber_2.12-0.0.23-assembly.jar"
            )
            print(f"Downloading Chronon JAR from {jar_url}")
            urllib.request.urlretrieve(jar_url, jar_path)
            print(f"Downloaded to {jar_path}")

    # Set up S3 dependencies
    print("Setting up S3 dependencies...")
    s3_deps_dir = "/tmp/spark-s3-deps"
    os.makedirs(s3_deps_dir, exist_ok=True)

    hadoop_aws_jar = f"{s3_deps_dir}/hadoop-aws-3.3.4.jar"
    aws_sdk_jar = f"{s3_deps_dir}/aws-java-sdk-bundle-1.12.367.jar"

    if not os.path.exists(hadoop_aws_jar):
        print("Downloading Hadoop AWS JAR...")
        urllib.request.urlretrieve(
            "https://repo1.maven.org/maven2/org/apache/hadoop/"
            "hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar",
            hadoop_aws_jar,
        )

    if not os.path.exists(aws_sdk_jar):
        print("Downloading AWS SDK JAR...")
        urllib.request.urlretrieve(
            "https://repo1.maven.org/maven2/com/amazonaws/"
            "aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar",
            aws_sdk_jar,
        )

    os.makedirs("/tmp/spark", exist_ok=True)
    os.makedirs("/tmp/chronon_data", exist_ok=True)
    os.makedirs("/tmp/chronon_features", exist_ok=True)

    print("Chronon environment ready")
    print("S3 dependencies ready")
    return jar_path


@uniflow.task(
    config=SparkTask(
        driver_memory="4g",
        driver_cpu=2,
        executor_memory="4g",
        executor_cpu=2,
        executor_instances=1,
    )
)
def compute_chronon_features_with_spark(
    dataset_config: dict[str, Any],
    books_dv: DatasetVariable,
    reviews_dv: DatasetVariable,
) -> tuple:
    """Chronon feature computation with Spark.

    Compiles Chronon definitions, executes staging queries and GroupBy
    aggregations, then creates train/val/test splits as DatasetVariables.
    """
    jar_path = _setup_chronon_environment()

    # Resolve data directory relative to this file, not CWD
    data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
    sys.path.insert(0, str(data_dir))

    print("Compiling Chronon definitions on-demand...")

    compiled_staging = thrift_simple_json_protected(base_table, StagingQuery)
    compiled_gb1 = thrift_simple_json_protected(book_popularity, GroupBy)
    compiled_gb2 = thrift_simple_json_protected(book_velocity, GroupBy)

    print("Chronon definitions compiled successfully")

    # Stop any existing session to ensure our configurations take effect
    existing_spark = SparkSession.getActiveSession()
    if existing_spark:
        existing_spark.stop()

    # Create Spark session with Chronon JAR + S3 support
    hadoop_aws_jar = "/tmp/spark-s3-deps/hadoop-aws-3.3.4.jar"
    aws_sdk_jar = "/tmp/spark-s3-deps/aws-java-sdk-bundle-1.12.367.jar"
    all_jars = f"{jar_path},{hadoop_aws_jar},{aws_sdk_jar}"

    spark = (
        SparkSession.builder.appName("ChronoAmazonBooks")
        .config("spark.jars", all_jars)
        .config(
            "spark.sql.sources.commitProtocolClass",
            "org.apache.spark.sql.execution.datasources."
            "SQLHadoopMapReduceCommitProtocol",
        )
        .config(
            "spark.sql.parquet.output.committer.class",
            "org.apache.parquet.hadoop.ParquetOutputCommitter",
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config(
            "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
        )
        .config("spark.hadoop.dfs.client.write.checksum", "false")
        .config("spark.hadoop.dfs.namenode.fs-limits.min-block-size", "1048576")
        .config("spark.hadoop.dfs.checksum", "false")
        .config("spark.hadoop.dfs.client.read.checksum", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.file.impl",
            "org.apache.hadoop.fs.local.RawLocalFs",
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.access.key",
            os.getenv("AWS_ACCESS_KEY_ID", "michelangeloadmin"),
        )
        .config(
            "spark.hadoop.fs.s3a.secret.key",
            os.getenv("AWS_SECRET_ACCESS_KEY", "michelangeloadmin"),
        )
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            os.getenv("AWS_ENDPOINT_URL", "http://localhost:9091"),
        )
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )

    print(f"Using Spark session: {spark.version} with Chronon JAR: {jar_path}")
    print("Setting up Chronon execution environment...")

    runner_args = {
        "mode": "backfill",
        "conf_path": str(data_dir),
        "start_date": "2013-01-01",
        "end_date": "2015-01-01",
        "parallelism": 1,
        "sample_percent": dataset_config.get("sample_size", 100) / 1000.0,
    }

    print(f"Chronon runner configuration: {runner_args}")

    books_dv.load_spark_dataframe()
    books_df = books_dv.value

    reviews_dv.load_spark_dataframe()
    reviews_df = reviews_dv.value

    print(
        f"Using provided DataFrames: {books_df.count()} books, "
        f"{reviews_df.count()} reviews"
    )

    # Register tables for Chronon
    books_df.createOrReplaceTempView("amazon_books_books")
    reviews_df.createOrReplaceTempView("amazon_books_reviews")

    print("Real Amazon Books data loaded and registered for Chronon execution")
    print("Executing Chronon staging query using Chronon Runtime Engine...")

    try:
        from ai.chronon.repo.run import Runner

        print("Setting up Chronon Runtime Engine...")

        class ChronosArgs:
            def __init__(self):
                self.repo = str(data_dir)
                self.conf = None
                self.sub_help = False
                self.mode = "group-by-backfill"
                self.online_jar = jar_path
                self.online_jar_fetch = f"echo {jar_path}"
                self.conf_type = "group-by"
                self.ds = "2015-01-01"
                self.end_ds = "2015-01-01"
                self.start_ds = "2013-01-01"
                self.parallelism = 1
                self.args = ""
                self.online_class = "ai.chronon.online.Api"
                self.app_name = "ChronoAmazonBooks"
                self.spark_submit_path = "spark-submit"
                self.spark_streaming_submit_path = "spark-submit"
                self.render_info = None
                self.list_apps = None

        chronon_args = ChronosArgs()
        Runner(chronon_args, jar_path)

        print("Chronon Runtime Engine initialized")

        print("Extracting feature specifications from compiled Chronon objects...")

        compiled_gb1_dict = json.loads(compiled_gb1)
        compiled_gb2_dict = json.loads(compiled_gb2)
        compiled_staging_dict = json.loads(compiled_staging)

        # Operation mapping (from Chronon thrift definitions)
        operation_names = {
            0: "MIN",
            1: "MAX",
            2: "SUM",
            3: "COUNT",
            4: "MEAN",
            5: "VARIANCE",
            6: "COUNT",
            7: "SUM",
            8: "AVERAGE",
            9: "MAX",
            10: "MIN",
        }

        time_unit_names = {0: "MILLIS", 1: "DAYS", 2: "HOURS"}

        book_popularity_windows = []
        book_velocity_windows = []

        for agg in compiled_gb1_dict["aggregations"]:
            operation_name = operation_names.get(
                agg["operation"], f"OP_{agg['operation']}"
            )
            for window in agg["windows"]:
                time_unit_name = time_unit_names.get(
                    window["timeUnit"], f"UNIT_{window['timeUnit']}"
                )
                window_name = (
                    f"{operation_name.lower()}_{agg['inputColumn']}_"
                    f"{window['length']}{time_unit_name.lower()}"
                )
                book_popularity_windows.append((window_name, agg, window))

        for agg in compiled_gb2_dict["aggregations"]:
            operation_name = operation_names.get(
                agg["operation"], f"OP_{agg['operation']}"
            )
            for window in agg["windows"]:
                time_unit_name = time_unit_names.get(
                    window["timeUnit"], f"UNIT_{window['timeUnit']}"
                )
                window_name = (
                    f"{operation_name.lower()}_{agg['inputColumn']}_"
                    f"{window['length']}{time_unit_name.lower()}"
                )
                book_velocity_windows.append((window_name, agg, window))

        print(
            f"Extracted {len(book_popularity_windows)} temporal windows "
            f"from book_popularity"
        )
        print(
            f"Extracted {len(book_velocity_windows)} temporal windows "
            f"from book_velocity"
        )

        staging_query_sql = compiled_staging_dict["query"]
        staging_df = spark.sql(staging_query_sql)
        print(f"Chronon staging query executed: {staging_df.count()} records")

        print("Computing GroupBy features using Chronon temporal windows...")

        agg_exprs = []

        for _window_name, agg, window in book_popularity_windows[:6]:
            operation_name = operation_names.get(
                agg["operation"], f"OP_{agg['operation']}"
            )
            time_unit_name = time_unit_names.get(
                window["timeUnit"], f"UNIT_{window['timeUnit']}"
            )

            if operation_name == "COUNT":
                agg_exprs.append(
                    count("review_score").alias(
                        f"review_count_{window['length']}{time_unit_name.lower()}"
                    )
                )
            elif operation_name == "AVERAGE":
                agg_exprs.append(
                    avg("review_score").alias(
                        f"avg_rating_{window['length']}{time_unit_name.lower()}"
                    )
                )
            elif operation_name == "MAX":
                agg_exprs.append(
                    spark_max("review_score").alias(
                        f"max_rating_{window['length']}{time_unit_name.lower()}"
                    )
                )
            elif operation_name == "MIN":
                agg_exprs.append(
                    spark_min("review_score").alias(
                        f"min_rating_{window['length']}{time_unit_name.lower()}"
                    )
                )

        book_features = staging_df.groupBy(
            "book_id", "book_title", "book_description"
        ).agg(*agg_exprs)

        print(
            f"Computed features using Chronon Runtime Engine: "
            f"{book_features.count()} books"
        )

    except Exception as e:
        # The Chronon Runtime Engine is required -- there is no fallback.
        # Re-raising as RuntimeError for a clearer stack trace surface.
        print(f"Chronon Runtime Engine execution failed: {e}")
        raise RuntimeError(f"Chronon Runtime Engine failed: {e}") from e

    # Create enhanced training data with Chronon-computed features
    print("Creating training pairs with Chronon features...")

    enhanced_books = book_features.select(
        col("book_id"),
        col("book_title").alias("title"),
        col("book_description").alias("description"),
        col("review_count_30days").alias("recent_review_count"),
        col("avg_rating_30days").alias("recent_avg_rating"),
        when(col("review_count_30days") >= 4, "popular")
        .when(col("review_count_30days") >= 2, "moderate")
        .otherwise("niche")
        .alias("popularity_tier"),
    )

    # Create positive pairs
    positive_pairs = enhanced_books.select(
        col("book_id"),
        col("title").alias("query"),
        concat_ws(" ", col("description"), col("title")).alias("document"),
        lit(1).alias("label"),
        col("popularity_tier"),
        col("recent_avg_rating"),
        col("recent_review_count"),
    )

    # Create negative pairs
    negative_ratio = dataset_config.get("negative_ratio", 1.0)
    negative_count = int(positive_pairs.count() * negative_ratio)

    books_for_negatives = enhanced_books.select(
        "book_id", "title", "description"
    ).alias("b1")
    docs_for_negatives = enhanced_books.select(
        "book_id",
        "description",
        "title",
        "popularity_tier",
        "recent_avg_rating",
        "recent_review_count",
    ).alias("b2")

    negative_pairs = (
        books_for_negatives.crossJoin(docs_for_negatives)
        .filter(col("b1.book_id") != col("b2.book_id"))
        .select(
            col("b1.book_id"),
            col("b1.title").alias("query"),
            concat_ws(" ", col("b2.description"), col("b2.title")).alias("document"),
            lit(0).alias("label"),
            col("b2.popularity_tier"),
            col("b2.recent_avg_rating"),
            col("b2.recent_review_count"),
        )
        .limit(negative_count)
    )

    # Combine and prepare final dataset
    training_pairs = positive_pairs.union(negative_pairs).orderBy(rand())
    print(f"Created {training_pairs.count()} training pairs with Chronon features")

    # Create train/val/test splits
    train_split = dataset_config.get("train_split", 0.7)
    val_split = dataset_config.get("val_split", 0.15)

    train_df, val_test_df = training_pairs.randomSplit(
        [train_split, 1 - train_split], seed=42
    )
    val_df, test_df = val_test_df.randomSplit(
        [val_split / (1 - train_split), 1 - (val_split / (1 - train_split))], seed=42
    )

    print(
        f"Chronon execution completed: {train_df.count()} train, "
        f"{val_df.count()} val, {test_df.count()} test"
    )

    # Convert to DatasetVariable
    train_dv = DatasetVariable.create(train_df)
    train_dv.save_spark_dataframe()

    val_dv = DatasetVariable.create(val_df)
    val_dv.save_spark_dataframe()

    test_dv = DatasetVariable.create(test_df)
    test_dv.save_spark_dataframe()

    print("DatasetVariables created and saved as Spark DataFrames")

    return train_dv, val_dv, test_dv
