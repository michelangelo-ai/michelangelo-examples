"""Source definitions for Amazon Books Chronon features.

Provides event sources for different aspects of the recommendation system.

Note: This file uses ``get_staging_query_output_table_name``, which relies
on Chronon's garbage-collector-based module introspection. Importing this
file standalone may raise ``AssertionError`` depending on the Chronon
module layout. This is a pre-existing limitation inherited from core.
"""

from ai.chronon.api.ttypes import EventSource, Source
from ai.chronon.query import Query, select
from ai.chronon.utils import get_staging_query_output_table_name

from michelangelo_examples.amazon_books_qwen.pipelines.train.data.staging_queries.amazon_books.books_reviews import (
    base_table,
)


def books_source(*columns):
    """Source for book-centric features based on reviews and metadata."""
    return Source(
        events=EventSource(
            table=get_staging_query_output_table_name(base_table),
            query=Query(
                selects=select(*columns),
                time_column="ts",
            ),
        )
    )


def user_interaction_source(*columns):
    """Source for user interaction events (reviews, ratings)."""
    return Source(
        events=EventSource(
            table=get_staging_query_output_table_name(base_table),
            query=Query(
                selects=select(*columns),
                time_column="ts",
            ),
        )
    )


def content_source(*columns):
    """Source for content-based features (book descriptions, categories)."""
    return Source(
        events=EventSource(
            table=get_staging_query_output_table_name(base_table),
            query=Query(
                selects=select(*columns),
                time_column="ts",
            ),
        )
    )
