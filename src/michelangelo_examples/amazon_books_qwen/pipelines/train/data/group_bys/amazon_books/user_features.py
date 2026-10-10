"""User behavior features for Amazon Books recommendation system.

Captures user reading patterns and preferences for personalization.

Note: This file is a conceptual definition -- it is NOT imported by the
live ``amazon_books_qwen_workflow``. Its ``get_staging_query_output_table_name``
calls rely on Chronon's garbage-collector-based module introspection, which
requires a flat ``staging_queries/`` top-level layout, not a nested package
path. Importing this file standalone raises ``AssertionError``. This is a
pre-existing limitation inherited from core, not a port regression.
"""

from ai.chronon.api.ttypes import EventSource, Source
from ai.chronon.group_by import (
    Accuracy,
    Aggregation,
    GroupBy,
    Operation,
    TimeUnit,
    Window,
)
from ai.chronon.query import Query, select
from ai.chronon.utils import get_staging_query_output_table_name

from michelangelo_examples.amazon_books_qwen.pipelines.train.data.staging_queries.amazon_books.books_reviews import (
    base_table,
)

# Note: In the current Amazon Books dataset, we don't have explicit user IDs.
# This is a conceptual implementation for when user data becomes available.

# Source for user rating patterns (using review patterns as proxy)
user_rating_source = Source(
    events=EventSource(
        table=get_staging_query_output_table_name(base_table),
        query=Query(
            selects=select("review_id", "review_score", "book_categories"),
            time_column="ts",
        ),
    )
)

# User reading behavior patterns (conceptual - requires user tracking)
user_reading_patterns = GroupBy(
    sources=[user_rating_source],
    keys=["review_id"],
    aggregations=[
        Aggregation(
            input_column="review_score",
            operation=Operation.COUNT,
            windows=[
                Window(length=7, timeUnit=TimeUnit.DAYS),
                Window(length=30, timeUnit=TimeUnit.DAYS),
                Window(length=90, timeUnit=TimeUnit.DAYS),
            ],
        ),
        Aggregation(
            input_column="review_score",
            operation=Operation.AVERAGE,
            windows=[
                Window(length=30, timeUnit=TimeUnit.DAYS),
                Window(length=90, timeUnit=TimeUnit.DAYS),
            ],
        ),
        Aggregation(
            input_column="review_score",
            operation=Operation.VARIANCE,
            windows=[
                Window(length=30, timeUnit=TimeUnit.DAYS),
                Window(length=90, timeUnit=TimeUnit.DAYS),
            ],
        ),
    ],
    accuracy=Accuracy.TEMPORAL,
)

# Genre preference tracking (conceptual)
genre_preference_source = Source(
    events=EventSource(
        table=get_staging_query_output_table_name(base_table),
        query=Query(
            selects=select("review_id", "book_categories", "review_score"),
            time_column="ts",
        ),
    )
)

# User genre preferences (would require user segmentation)
user_genre_preferences = GroupBy(
    sources=[genre_preference_source],
    keys=["review_id", "book_categories"],
    aggregations=[
        Aggregation(
            input_column="review_score",
            operation=Operation.COUNT,
            windows=[
                Window(length=30, timeUnit=TimeUnit.DAYS),
                Window(length=90, timeUnit=TimeUnit.DAYS),
            ],
        ),
        Aggregation(
            input_column="review_score",
            operation=Operation.AVERAGE,
            windows=[
                Window(length=30, timeUnit=TimeUnit.DAYS),
                Window(length=90, timeUnit=TimeUnit.DAYS),
            ],
        ),
    ],
    accuracy=Accuracy.TEMPORAL,
)
