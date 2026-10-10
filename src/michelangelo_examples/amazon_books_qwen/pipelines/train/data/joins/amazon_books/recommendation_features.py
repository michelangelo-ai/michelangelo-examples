"""Join definition for Amazon Books recommendation features.

Combines all feature GroupBys into a comprehensive feature set for ML training.

Note: This file is a conceptual definition -- it is NOT imported by the
live ``amazon_books_qwen_workflow``. The ``content_features`` and
``user_features`` GroupBys it references use
``get_staging_query_output_table_name``, which fails on import due to
Chronon's module-introspection assumptions. This is a pre-existing
limitation inherited from core, not a port regression.
"""

from ai.chronon.join import Join, JoinPart

from michelangelo_examples.amazon_books_qwen.pipelines.train.data.group_bys.amazon_books.book_features import (
    book_popularity,
    book_velocity,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.data.group_bys.amazon_books.content_features import (
    author_features,
    category_features,
    publisher_features,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.data.group_bys.amazon_books.user_features import (
    user_genre_preferences,
    user_reading_patterns,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.data.sources.amazon_books.books import (
    books_source,
)

# Main training set join for recommendation features
recommendation_features = Join(
    left=books_source(
        "book_id",
        "book_title",
        "book_authors",
        "book_categories",
        "book_publisher",
        "book_published_date",
        "book_description",
        "review_id",
        "review_score",
        "review_summary",
        "review_text",
    ),
    right_parts=[
        JoinPart(group_by=book_popularity),
        JoinPart(group_by=book_velocity),
        JoinPart(group_by=author_features),
        JoinPart(group_by=category_features),
        JoinPart(group_by=publisher_features),
        JoinPart(group_by=user_reading_patterns),
        JoinPart(group_by=user_genre_preferences),
    ],
)

# Specialized join for book-centric features only (for inference)
book_features_only = Join(
    left=books_source(
        "book_id",
        "book_title",
        "book_authors",
        "book_categories",
        "book_publisher",
        "book_description",
    ),
    right_parts=[
        JoinPart(group_by=book_popularity),
        JoinPart(group_by=book_velocity),
        JoinPart(group_by=author_features),
        JoinPart(group_by=category_features),
        JoinPart(group_by=publisher_features),
    ],
)

# Content similarity features for cold-start recommendations
content_similarity_features = Join(
    left=books_source(
        "book_id", "book_title", "book_authors", "book_categories", "book_description"
    ),
    right_parts=[
        JoinPart(group_by=author_features),
        JoinPart(group_by=category_features),
        JoinPart(group_by=publisher_features),
    ],
)
