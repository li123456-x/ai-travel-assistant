"""Preview by default; --apply updates metadata without deleting plan versions."""

import argparse
import json

from database.memory_migration import migrate_plan_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate_plan_memory(apply=args.apply), ensure_ascii=False))


if __name__ == "__main__":
    main()
