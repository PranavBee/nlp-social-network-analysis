"""Tests for the reproducible synthetic dataset generator."""

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_social_network.data_generator import (  # noqa: E402
    DEFAULT_POST_COUNT,
    FIELDS,
    TOPICS,
    USERNAMES,
    QUIET_USER,
    generate_posts,
    write_dataset,
)


class DataGeneratorTests(unittest.TestCase):
    def test_schema_and_post_count(self):
        posts = generate_posts()
        self.assertEqual(len(posts), DEFAULT_POST_COUNT)
        self.assertEqual(tuple(posts[0].keys()), FIELDS)
        self.assertGreaterEqual(len({post["username"] for post in posts}), 30)
        for post in posts:
            self.assertTrue(all(post[field] != "" for field in FIELDS if field != "reply_to_user"))

    def test_post_ids_are_unique(self):
        posts = generate_posts()
        ids = [post["post_id"] for post in posts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_same_seed_produces_same_posts(self):
        self.assertEqual(generate_posts(seed=123), generate_posts(seed=123))
        self.assertNotEqual(generate_posts(seed=123), generate_posts(seed=124))

    def test_mentions_and_replies_reference_existing_users(self):
        posts = generate_posts()
        known_users = {post["username"] for post in posts}
        for post in posts:
            mentions = json.loads(post["mentions"])
            self.assertIsInstance(mentions, list)
            self.assertTrue(set(mentions).issubset(known_users))
            self.assertNotIn(post["username"], mentions)
            if post["reply_to_user"]:
                self.assertIn(post["reply_to_user"], known_users)
                self.assertIn(post["reply_to_user"], mentions)

    def test_timestamps_and_topics_are_valid_and_diverse(self):
        posts = generate_posts()
        for post in posts:
            parsed = datetime.fromisoformat(post["timestamp"].replace("Z", "+00:00"))
            self.assertIsNotNone(parsed.tzinfo)
            self.assertIn(post["topic"], TOPICS)
        self.assertEqual({post["topic"] for post in posts}, set(TOPICS))

    def test_interactions_include_hubs_initiators_and_a_quiet_user(self):
        posts = generate_posts()
        incoming = Counter(
            mentioned
            for post in posts
            for mentioned in json.loads(post["mentions"])
        )
        outgoing = Counter(
            post["username"]
            for post in posts
            if json.loads(post["mentions"])
        )
        self.assertGreater(sum(incoming.values()), 50)
        self.assertGreater(max(incoming.values()), 10)
        self.assertGreaterEqual(outgoing["pixelmoss"], 5)
        self.assertEqual(incoming[QUIET_USER], 0)
        self.assertEqual(outgoing[QUIET_USER], 0)

    def test_csv_writer_writes_all_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample_posts.csv"
            write_dataset(path=output, post_count=DEFAULT_POST_COUNT, seed=7)
            with output.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertEqual(tuple(reader.fieldnames), FIELDS)
                self.assertEqual(sum(1 for _ in reader), DEFAULT_POST_COUNT)


if __name__ == "__main__":
    unittest.main()
