from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Union


# def build_temporal_fact_chains(
#         relations_selected_file: Union[str, Path],
#         news_full_file: Union[str, Path],
#         output_json_file: Union[str, Path]
# ) -> None:
#     """
#     Extract and construct high-quality temporal fact chains from the full news data
#     based on the selected relations.
#     Includes strict conflict removal, length filtering, adjacent deduplication,
#     and a 'one-strike' veto mechanism for complex entities containing slashes ('/').
#     """
#     # 1. Read the selected golden relations set
#     with open(relations_selected_file, 'r', encoding='utf-8') as f:
#         selected_relations = set(line.strip() for line in f if line.strip())
#
#     # 2. Initially cluster data by (Subject, Relation)
#     raw_data = defaultdict(list)
#     with open(news_full_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 4:
#                 sub, rel, obj, time_str = parts[0], parts[1], parts[2], parts[3]
#                 if rel in selected_relations:
#                     raw_data[(sub, rel)].append((time_str, obj))
#
#     valid_chains = []
#     discarded_due_to_slash = 0
#
#     # 3. Process each data group corresponding to (Subject, Relation)
#     for (sub, rel), time_obj_list in raw_data.items():
#         # --- A. One-strike veto for bad smell entities ---
#         # If the subject contains "/", discard the entire chain immediately
#         if '/' in sub:
#             discarded_due_to_slash += 1
#             continue
#
#         # If any object in the timeline contains "/", discard the entire chain immediately
#         if any('/' in o for t, o in time_obj_list):
#             discarded_due_to_slash += 1
#             continue
#
#         # Sort events chronologically
#         time_obj_list.sort(key=lambda x: x[0])
#
#         # --- B. Conflict removal ---
#         has_conflict = False
#         time_to_objs = defaultdict(set)
#         for t, o in time_obj_list:
#             time_to_objs[t].add(o)
#             if len(time_to_objs[t]) > 1:
#                 has_conflict = True
#                 break
#
#         if has_conflict:
#             continue
#
#         # --- C. Merge adjacent times ---
#         merged_timeline = []
#         for t, o in time_obj_list:
#             if not merged_timeline:
#                 merged_timeline.append({"object": o, "time": t})
#             else:
#                 if merged_timeline[-1]["object"] != o:
#                     merged_timeline.append({"object": o, "time": t})
#
#         # --- D. Filter chains without state transitions ---
#         unique_objs = set(item["object"] for item in merged_timeline)
#         if len(unique_objs) < 2:
#             continue
#
#         # 4. Construct and save the final high-quality chains
#         valid_chains.append({
#             "subject": sub,
#             "relation": rel,
#             "timeline": merged_timeline
#         })
#
#     # 5. Write to JSON file
#     out_path = Path(output_json_file)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     with out_path.open('w', encoding='utf-8') as f:
#         json.dump(valid_chains, f, ensure_ascii=False, indent=4)
#
#     print(f"Extraction and cleaning complete!")
#     print(f"Discarded {discarded_due_to_slash} ambiguous fact chains due to containing the '/' symbol.")
#     print(f"Generated {len(valid_chains)} ultra-high purity temporal fact chains.")
#     print(f"Data saved to: {out_path.resolve()}")


def build_temporal_chains(input_file, output_file, min_length=2):
    print(f"Loading probed facts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        probed_facts = json.load(f)

    # Dictionary to store chains: Key = (clean_subject, relation)
    # Value = List of all events that occurred for this subject under this relation
    chains_dict = defaultdict(list)

    # 1. Grouping
    for fact in probed_facts:
        key = (fact['clean_subject'], fact['relation'])
        chains_dict[key].append(fact)

    print(f"Grouped into {len(chains_dict)} unique (Subject, Relation) combinations.")

    # 2. Sorting and Filtering
    valid_chains = []

    for key, events in chains_dict.items():
        # If the number of events on this chain doesn't meet the minimum threshold, discard it
        if len(events) < min_length:
            continue

        # Sort chronologically (ICEWS time format is usually YYYY-MM-DD)
        try:
            sorted_events = sorted(events, key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d"))
        except ValueError:
            # If the time format in your txt is not YYYY-MM-DD, this might throw an error. Try string sorting directly.
            sorted_events = sorted(events, key=lambda x: x['time'])

        # Construct a complete temporal chain object
        chain_obj = {
            "subject": key[0],
            "relation": key[1],
            "chain_length": len(sorted_events),
            "events": sorted_events
        }
        valid_chains.append(chain_obj)

    # 3. Sort chains from longest to shortest, making it easier to observe the best data
    valid_chains = sorted(valid_chains, key=lambda x: x['chain_length'], reverse=True)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_chains, f, indent=4, ensure_ascii=False)

    # Statistics
    total_events_in_chains = sum(chain['chain_length'] for chain in valid_chains)
    print("\n" + "=" * 40)
    print("🎯 Temporal Chains Building Complete!")
    print(f"Total Valid Chains Found: {len(valid_chains)}")
    print(f"Total Events utilized in chains: {total_events_in_chains} / {len(probed_facts)}")
    print("=" * 40)

    # Print the top 3 longest chains to see the effect
    print("\nTop 3 Longest Chains:")
    for i in range(min(3, len(valid_chains))):
        c = valid_chains[i]
        print(f"[{c['chain_length']} events] {c['subject']} -> {c['relation']}")