import logging
import json
from pathlib import Path
from typing import Union

# Configure basic log output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_relations(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        relation_index: int = 1,
        min_columns: int = 4
) -> None:
    """
    Extract unique relations from the dataset and save them alphabetically to a new file.

    Args:
        input_path: Relative or absolute path to the input data file.
        output_path: Relative or absolute path to the output file.
        relation_index: The index position of the relation after splitting each line. Defaults to 1.
        min_columns: Minimum number of columns to be considered a valid data line. Defaults to 4.
    """
    # Use pathlib to handle paths uniformly
    in_file = Path(input_path)
    out_file = Path(output_path)

    unique_relations = set()

    # 1. Safely read and extract
    try:
        with in_file.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                # Check data column count to avoid out of bounds
                if len(parts) >= min_columns:
                    unique_relations.add(parts[relation_index])
                elif parts:  # Ignore pure empty lines, optionally log malformed lines
                    logging.debug(f"Skipped malformed line (Line {line_num}): {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {in_file.resolve()}")
        return
    except Exception as e:
        logging.error(f"Unknown error occurred while reading file: {e}")
        return

    # 2. Safely write
    try:
        # Ensure the parent directory of the output file exists
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open('w', encoding='utf-8') as f:
            for r in sorted(unique_relations):
                f.write(f"{r}\n")

        logging.info(f"Extraction complete! Found {len(unique_relations)} unique relations.")
        logging.info(f"Saved to: {out_file.resolve()}")
    except Exception as e:
        logging.error(f"Error occurred while writing output file: {e}")

# Call example
# extract_relations("news/xxx.txt", "news/yyy.txt")


def extract_entities(
        chains_json_path: Union[str, Path],
        entity_output_path: Union[str, Path]
) -> None:
    """
    Extract all unique entities (subjects and objects) from the constructed fact chain JSON file,
    and save them alphabetically to a txt file.

    Args:
        chains_json_path: Path to the fact chain JSON file.
        entity_output_path: Path to the txt file where extracted entities will be saved.
    """
    input_path = Path(chains_json_path)
    output_path = Path(entity_output_path)

    # Use set to automatically deduplicate
    unique_entities = set()

    try:
        # 1. Read JSON data
        with input_path.open('r', encoding='utf-8') as f:
            chains_data = json.load(f)

        # 2. Iterate over each fact chain and extract entities
        for chain in chains_data:
            # Extract Subject
            if "subject" in chain:
                unique_entities.add(chain["subject"])

            # Extract all objects on the timeline (Object)
            if "timeline" in chain:
                for event in chain["timeline"]:
                    if "object" in event:
                        unique_entities.add(event["object"])

        # 3. Sort entities alphabetically and write to txt file
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', encoding='utf-8') as f:
            for entity in sorted(unique_entities):
                f.write(f"{entity}\n")

        print(f"✅ Entity extraction complete!")
        print(f"Found {len(unique_entities)} unique entities in total.")
        print(f"Saved to: {output_path.resolve()}")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found {input_path.resolve()}")
    except json.JSONDecodeError:
        print(f"❌ Error: {input_path.resolve()} is not a valid JSON file")
    except Exception as e:
        print(f"❌ Unknown error occurred: {e}")

# Test Call Example
# extract_entities(
#     chains_json_path="news/temporal_fact_chains.json",
#     entity_output_path="news/entity_selected.txt"
# )