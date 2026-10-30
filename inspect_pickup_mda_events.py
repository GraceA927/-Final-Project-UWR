import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PICKUP = Path(
    "/shared/home/grace/Final-Project-UWR/experiments/"
    "e_grace_XTLH_Pickup13yrs_f86a8aad-9ec8-4288-8d62-e957a390d5ed"
)

EVENT_PATTERN = re.compile(r"^MDA_Now_\d+$")
totals = defaultdict(int)
files_checked = 0


def walk(obj, path="$"):
    """Find MDA event-name strings and report their JSON paths."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"

            if isinstance(value, str) and EVENT_PATTERN.fullmatch(value):
                yield value, child_path, obj

            yield from walk(value, child_path)

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk(value, f"{path}[{i}]")


def short_value(value, limit=180):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


campaigns = sorted(PICKUP.glob("*/campaign.json"))

print(f"Pickup directory: {PICKUP}")
print(f"Campaign files found: {len(campaigns)}")

if not campaigns:
    sys.exit("ERROR: No campaign.json files found.")

for campaign_path in campaigns:
    files_checked += 1

    try:
        data = json.loads(campaign_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\nERROR reading {campaign_path}: {exc}")
        continue

    occurrences = list(walk(data))
    per_file = defaultdict(list)

    for event, json_path, parent in occurrences:
        totals[event] += 1
        per_file[event].append((json_path, parent))

    print("\n" + "=" * 90)
    print(f"SIMULATION: {campaign_path.parent.name}")
    print(f"Total MDA event-name occurrences: {len(occurrences)}")

    if not per_file:
        print("No MDA_Now_<number> strings found.")
        continue

    for event in sorted(per_file, key=lambda s: int(s.rsplit("_", 1)[1])):
        matches = per_file[event]
        label = "POSSIBLE COLLISION" if len(matches) > 2 else "INSPECT"

        print(f"\n{event}: {len(matches)} occurrence(s) [{label}]")

        for json_path, parent in matches:
            print(f"  JSON path: {json_path}")
            print(f"  Object keys: {list(parent.keys())}")

            for key in (
                "class",
                "Event_Trigger",
                "Broadcast_Event",
                "Broadcast_Event_Trigger",
                "Intervention_Config",
                "Actual_IndividualIntervention_Config",
                "Intervention_Name",
            ):
                if key in parent:
                    print(f"  {key}: {short_value(parent[key])}")

print("\n" + "=" * 90)
print(f"Campaign files inspected: {files_checked}")
print("AGGREGATE MDA EVENT-NAME OCCURRENCES:")

for event in sorted(totals, key=lambda s: int(s.rsplit("_", 1)[1])):
    print(f"{event}: {totals[event]} occurrence(s) across pickup campaigns")

print("\nNOTE: Occurrence counts alone do not establish duplicate dosing.")
print("Inspect the JSON paths and intervention relationships before concluding.")
