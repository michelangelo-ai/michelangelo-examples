# amazon-books-qwen / train

Uniflow pipeline for training a Qwen-based dual-encoder recommendation
model on the Amazon Books dataset with Chronon feature engineering.

## What it exercises

- **SparkTask**: `download_kaggle_dataset` (Kaggle API + Spark CSV loading)
  and `compute_chronon_features_with_spark` (Chronon temporal feature
  computation via Spark SQL)
- **RayTask**: `train_dual_encoder` (Qwen dual-encoder training with
  InfoNCE contrastive loss)
- **Chronon SDK**: `chronon-ai` GroupBy/StagingQuery compilation and
  Runner initialization
- **DatasetVariable**: Spark-to-Ray dataset handoff via
  `save_spark_dataframe()` / `load_ray_dataset()`

## Quick start (remote run)

```bash
pip install "michelangelo-examples[amazon-books-qwen]"

# Apply project + pipeline
ma project apply -f src/michelangelo_examples/amazon_books_qwen/config/project.yaml
ma pipeline apply -f src/michelangelo_examples/amazon_books_qwen/pipelines/train/pipeline.yaml

# Run (requires Kaggle credentials, Maven Central access, and a Spark-capable image)
ma pipeline run --namespace amazon-books-qwen --name amazon-books-qwen-train
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | `@uniflow.workflow()` entry point wiring download -> chronon -> train |
| `pipeline.yaml` | Michelangelo pipeline manifest |
| `download.py` | SparkTask: download Amazon Books dataset from Kaggle |
| `chronon_tasks.py` | SparkTask: Chronon feature engineering (JAR download, staging query, GroupBy aggregation) |
| `train.py` | RayTask: Qwen dual-encoder training (local or distributed) |
| `data/` | Chronon configuration tree (group_bys, staging_queries, joins, sources) |

## Deviations from core

- **No checked-in CSV datasets**: core ships `books_data.csv` (~900 KB)
  and `Books_rating.csv` (~20 MB) alongside the code. This migration does
  not commit the large files; the Kaggle download is the primary (and only)
  data path. Users must have valid Kaggle API credentials.
- **No `__main__.py` local runner**: every task requires Spark or Ray;
  see the project-level README for rationale.
- **Import paths**: `examples.amazon_books_qwen.*` rewritten to
  `michelangelo_examples.amazon_books_qwen.pipelines.train.*`.
- **`chronon_tasks.py` path resolution**: uses `__file__`-relative
  resolution instead of branching on `Path(os.getcwd()).name == "python"`.
- **Dropped dual-import fallback**: core's `try/except ModuleNotFoundError`
  in `chronon_tasks.py` is not needed for an always-installed package.

## Spark task image notes

This is the first `michelangelo-examples` project using `SparkTask` for
actual training tasks (not just `push.py` storage). The shared Dockerfile
already handles:
- `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` env vars
- `/app/michelangelo` symlink for `spark-app.yaml`

No additional Dockerfile changes were needed.

## Chronon JAR / network egress

`chronon_tasks.py` downloads JARs from Maven Central at pipeline *run*
time (not just install time):
- `spark_uber_2.12-0.0.23-assembly.jar` (Chronon)
- `hadoop-aws-3.3.4.jar`
- `aws-java-sdk-bundle-1.12.367.jar`

If running in a network-restricted sandbox, ensure Maven Central is
reachable or pre-stage these JARs at the expected paths under `/tmp/`.
