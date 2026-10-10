"""Amazon Books Dataset Download Task.

Downloads the Amazon Books Reviews dataset from Kaggle and returns
Spark DataFrames for downstream Chronon feature computation.

Migrated from core ``michelangelo``'s
``python/examples/amazon_books_qwen/download.py``.

Unlike the core version, this module does NOT ship the CSV files
checked-in alongside the code (the ``Books_rating.csv`` file is ~20 MB,
too large to commit to a lightweight examples repo). Instead, the Kaggle
download is the primary path. Users must set up Kaggle API credentials
before running (see the project README for instructions).
"""

import os
import shutil
import zipfile
from typing import Any

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.spark import SparkTask
from michelangelo.workflow.variables import DatasetVariable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


@uniflow.task(
    config=SparkTask(
        driver_memory="2g",
        driver_cpu=1,
        executor_memory="2g",
        executor_cpu=1,
        executor_instances=1,
    )
)
def download_kaggle_dataset(
    dataset_config: dict[str, Any],
) -> tuple[DatasetVariable | None, DatasetVariable | None]:
    """Download Amazon Books dataset from Kaggle using SparkTask.

    Returns Spark DataFrames (books_df, reviews_df) for downstream tasks.
    Requires valid Kaggle API credentials (~/.kaggle/kaggle.json).
    """
    print("Starting Kaggle dataset download with SparkTask...")

    dataset_name = "mohamedbakhet/amazon-books-reviews"
    download_path = "/tmp/amazon_books_dataset"

    # Disable checksum creation before Spark session starts
    os.environ[
        "SPARK_CONF_SPARK.HADOOP.MAPREDUCE.FILEOUTPUTCOMMITTER.MARKSUCCESSFULJOBS"
    ] = "false"
    os.environ["SPARK_CONF_SPARK.HADOOP.DFS.CLIENT.WRITE.CHECKSUM"] = "false"
    os.environ["SPARK_CONF_SPARK.HADOOP.DFS.CHECKSUM"] = "false"
    os.environ["SPARK_CONF_SPARK.HADOOP.DFS.CLIENT.READ.CHECKSUM"] = "false"
    os.environ["SPARK_CONF_SPARK.HADOOP.FS.FILE.IMPL"] = (
        "org.apache.hadoop.fs.RawLocalFileSystem"
    )
    os.environ["SPARK_CONF_SPARK.HADOOP.FS.ABSTRACTFILESYSTEM.FILE.IMPL"] = (
        "org.apache.hadoop.fs.local.RawLocalFs"
    )

    # Get or create Spark session
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = (
            SparkSession.builder.appName("AmazonBooksDownload")
            .config(
                "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
            )
            .config("spark.hadoop.dfs.client.write.checksum", "false")
            .config("spark.hadoop.dfs.checksum", "false")
            .config("spark.hadoop.dfs.client.read.checksum", "false")
            .config(
                "spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem"
            )
            .config(
                "spark.hadoop.fs.AbstractFileSystem.file.impl",
                "org.apache.hadoop.fs.local.RawLocalFs",
            )
            .getOrCreate()
        )

    # Apply configuration to ensure it takes effect
    spark.conf.set(
        "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
    )
    spark.conf.set("spark.hadoop.dfs.client.write.checksum", "false")
    spark.conf.set("spark.hadoop.dfs.checksum", "false")
    spark.conf.set("spark.hadoop.dfs.client.read.checksum", "false")
    spark.conf.set(
        "spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem"
    )
    spark.conf.set(
        "spark.hadoop.fs.AbstractFileSystem.file.impl",
        "org.apache.hadoop.fs.local.RawLocalFs",
    )

    # Define expected file paths
    os.makedirs(download_path, exist_ok=True)
    books_file = f"{download_path}/books_data.csv"
    reviews_file = f"{download_path}/Books_rating.csv"

    # Download if files don't already exist (cached from a previous run)
    if not (os.path.exists(books_file) and os.path.exists(reviews_file)):
        print("Authenticating with Kaggle API...")
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("Kaggle authentication successful")

        # Clean download directory to avoid corrupted files
        if os.path.exists(download_path):
            shutil.rmtree(download_path)
        os.makedirs(download_path, exist_ok=True)

        print(f"Downloading {dataset_name} to {download_path}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                api.dataset_download_files(
                    dataset_name, path=download_path, unzip=False
                )

                zip_files = [f for f in os.listdir(download_path) if f.endswith(".zip")]
                if not zip_files:
                    raise RuntimeError("No zip file found after download")

                zip_path = os.path.join(download_path, zip_files[0])
                print(
                    f"Downloaded zip file: {zip_path} "
                    f"({os.path.getsize(zip_path)} bytes)"
                )

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(download_path)

                os.remove(zip_path)
                print("Download and extraction successful")
                break

            except Exception as e:  # noqa: BLE001
                # Retry logic: Kaggle downloads can be flaky due to network
                # issues or rate limiting; retrying with a clean slate is the
                # most robust recovery strategy.
                print(f"Download attempt {attempt + 1} failed: {e!s}")
                if attempt < max_retries - 1:
                    print("Retrying download...")
                    if os.path.exists(download_path):
                        shutil.rmtree(download_path)
                    os.makedirs(download_path, exist_ok=True)
                else:
                    print("All download attempts failed")
                    return None, None

        # Verify files were downloaded
        if not (os.path.exists(books_file) and os.path.exists(reviews_file)):
            print(f"Download failed: Expected files not found in {download_path}")
            available = (
                os.listdir(download_path) if os.path.exists(download_path) else "None"
            )
            print(f"Available files: {available}")
            return None, None
    else:
        print("Dataset already exists in cache, skipping download")

    # Load data into Spark DataFrames
    print("Loading books dataset into Spark...")
    books_df_full = spark.read.csv(books_file, header=True, inferSchema=True)

    print("Loading reviews dataset into Spark...")
    reviews_df_full = spark.read.csv(reviews_file, header=True, inferSchema=True)

    # Sample a subset for testing
    sample_size = dataset_config.get("sample_size", 100)
    books_df = books_df_full.sample(False, sample_size / 10000.0, seed=42).limit(50)

    # Get reviews for sampled books
    book_titles = books_df.select("Title").collect()
    book_title_list = [row.Title for row in book_titles]
    reviews_df = reviews_df_full.filter(col("Title").isin(book_title_list)).limit(500)

    print(
        f"Successfully loaded {books_df.count()} books and {reviews_df.count()} reviews"
    )

    # Convert to DatasetVariable
    books_dv = DatasetVariable.create(books_df)
    books_dv.save_spark_dataframe()

    reviews_dv = DatasetVariable.create(reviews_df)
    reviews_dv.save_spark_dataframe()

    print("DatasetVariables created and saved as Spark DataFrames")

    return books_dv, reviews_dv
