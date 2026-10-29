"""Create a reproducible, fully fictional social media posts dataset.

This module uses only Python's standard library, so it can generate the sample
CSV before the project's analysis dependencies are needed.
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEED = 42
DEFAULT_POST_COUNT = 180
DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_posts.csv"

TOPICS = ("technology", "sports", "education", "environment", "entertainment")
USERNAMES = (
    "maplebyte", "pixelmoss", "orbitowl", "coppercomet", "mintyvector",
    "cloudpebble", "sunnycircuit", "paperrocket", "velvetfern", "lunarloaf",
    "bluejaystudio", "quietquartz", "marblefox", "teacupgalaxy", "echoacorn",
    "willowwidget", "tinyharbor", "plumplanet", "amberkettle", "fableframe",
    "cedarcloud", "buttonmeteor", "softsignal", "pocketpanda", "rainyrobot",
    "doodlecomet", "saffronstar", "honeyvector", "mossyminute", "sleepykiwi",
)

TOPIC_TEXT = {
    "technology": [
        "I tried a tiny coding project today and it finally worked",
        "The new open source tool made my workflow much easier",
        "Debugging this little app is taking longer than I expected",
        "Reading about safer data design gave me a lot to think about",
    ],
    "sports": [
        "Our neighborhood team played with so much heart today",
        "That last minute goal made the whole match exciting",
        "A tough loss, but the team can learn from this one",
        "I watched the game with friends and enjoyed the atmosphere",
    ],
    "education": [
        "A patient teacher helped a difficult idea finally make sense",
        "I finished a short course and feel proud of the progress",
        "This assignment was confusing until I found a clearer example",
        "Libraries are such welcoming places to learn something new",
    ],
    "environment": [
        "The community garden looked beautiful after the rain",
        "We picked up litter along the trail and made a visible difference",
        "It is frustrating to see so much waste after a local event",
        "Small everyday choices can support a healthier neighborhood",
    ],
    "entertainment": [
        "The little concert tonight was joyful from beginning to end",
        "That animated film surprised me with its warmth and humor",
        "The show finale felt rushed and left me disappointed",
        "I spent a quiet evening listening to a new album",
    ],
}

HASHTAGS = {
    "technology": ("#CodeGarden", "#SmallBuilds", "#OpenTools"),
    "sports": ("#LocalLeague", "#GameDay", "#TeamSpirit"),
    "education": ("#LearningTogether", "#StudyNotes", "#CuriousMinds"),
    "environment": ("#GreenNeighbors", "#GardenDay", "#CareForPlaces"),
    "entertainment": ("#MovieNight", "#SoundCheck", "#CreativeTime"),
}

FIELDS = (
    "post_id",
    "user_id",
    "username",
    "text",
    "timestamp",
    "mentions",
    "reply_to_user",
    "topic",
)

# Each group mostly interacts internally. A few bridge users introduce links
# between groups, while one fictional user has no interactions at all.
GROUPS = (range(0, 10), range(10, 20), range(20, 30))
POPULAR_USERS = ("maplebyte", "bluejaystudio", "cedarcloud")
BRIDGE_USERS = {"lunarloaf", "fableframe", "rainyrobot"}
QUIET_USER = "sleepykiwi"


def _user_id(index: int) -> str:
    """Return the stable ID for a username's position in USERNAMES."""
    return f"user_{index + 1:02d}"


def _interaction_targets(author: str, rng: random.Random) -> list[str]:
    """Pick zero or more fictional users to mention in one post."""
    if author == QUIET_USER:
        return []

    author_index = USERNAMES.index(author)
    group = next(group for group in GROUPS if author_index in group)
    group_users = [
        USERNAMES[index]
        for index in group
        if USERNAMES[index] not in {author, QUIET_USER}
    ]

    # These accounts initiate a conversation in most of their posts.
    initiators = {"pixelmoss", "quietquartz", "buttonmeteor"}
    if author in initiators:
        target = rng.choice([name for name in POPULAR_USERS if name != author])
        return [target]

    # Bridges occasionally reach into each of the other groups.
    if author in BRIDGE_USERS and rng.random() < 0.8:
        other_users = [
            name for name in USERNAMES
            if name not in {author, QUIET_USER} and name not in group_users
        ]
        return [rng.choice(other_users)]

    # Popular users receive many incoming interactions from all communities.
    if rng.random() < 0.55:
        return [rng.choice([name for name in POPULAR_USERS if name != author])]

    # Most ordinary interactions stay in the author's community.
    if rng.random() < 0.58:
        target = rng.choice(group_users)
        return [target]

    # Some posts are intentionally conversation-free.
    return []


def generate_posts(post_count: int = DEFAULT_POST_COUNT, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build synthetic posts with stable output for the same seed and count.

    Posts are distributed evenly among 30 fictional users. Every generated
    row has a deterministic ID and a valid UTC timestamp.
    """
    if post_count < len(USERNAMES):
        raise ValueError(f"post_count must be at least {len(USERNAMES)} so every user can post")
    if post_count % len(USERNAMES) != 0:
        raise ValueError(f"post_count must be divisible by {len(USERNAMES)} to distribute posts evenly")

    rng = random.Random(seed)
    posts: list[dict[str, Any]] = []
    start = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

    for post_number in range(post_count):
        user_index = post_number % len(USERNAMES)
        username = USERNAMES[user_index]
        topic = TOPICS[(post_number + rng.randrange(len(TOPICS))) % len(TOPICS)]
        base_text = rng.choice(TOPIC_TEXT[topic])
        hashtag = rng.choice(HASHTAGS[topic])
        targets = _interaction_targets(username, rng)

        if targets:
            base_text += " " + " ".join(f"@{target}" for target in targets)
        text = f"{base_text} {hashtag}"

        # A reply is a subset of mentions, so every reply target is also a
        # referenced fictional account. Other mentions begin new conversations.
        reply_to_user = targets[0] if targets and rng.random() < 0.7 else ""
        timestamp = start + timedelta(minutes=post_number * 23 + rng.randrange(18))

        posts.append(
            {
                "post_id": f"post_{post_number + 1:04d}",
                "user_id": _user_id(user_index),
                "username": username,
                "text": text,
                "timestamp": timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "mentions": json.dumps(targets, separators=(",", ":")),
                "reply_to_user": reply_to_user,
                "topic": topic,
            }
        )

    return posts


def write_dataset(path: Path = DATASET_PATH, post_count: int = DEFAULT_POST_COUNT, seed: int = DEFAULT_SEED) -> Path:
    """Generate posts and write them as a UTF-8 CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    posts = generate_posts(post_count=post_count, seed=seed)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(posts)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictional social media posts as a CSV dataset.")
    parser.add_argument("--output", type=Path, default=DATASET_PATH, help="CSV output path (default: data/sample_posts.csv)")
    parser.add_argument("--posts", type=int, default=DEFAULT_POST_COUNT, help="Number of posts; must be a multiple of 30")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible data")
    args = parser.parse_args()

    output_path = write_dataset(path=args.output, post_count=args.posts, seed=args.seed)
    print(f"Generated {args.posts} fictional posts at {output_path}")


if __name__ == "__main__":
    main()
