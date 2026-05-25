import re
import json
from pathlib import Path


def clean_and_map_entities(input_txt_path, output_json_path):
    # Preposition rule base: determines whether to use 'of' or 'in' before the word in parentheses
    of_keywords = ['military', 'government', 'ministry', 'lawmaker', 'citizen', 'member', 'undersecretary',
                   'legislature', 'department', 'naval', 'mayor', 'police', 'force']
    in_keywords = ['facility', 'hospital', 'rebel', 'airline', 'bank', 'works']

    mapping_dict = {}
    discarded_count = 0  # Counter for discarded entities

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        entities = [line.strip() for line in f if line.strip()]

    for raw in entities:
        # --- 🛡️ Defensive interception: Veto entities with "/" ---
        if '/' in raw:
            discarded_count += 1
            continue

        clean = raw

        # 1. Handle annoying quotation marks: "Austin_""Jack""_Warner" -> Austin "Jack" Warner
        if clean.startswith('"') and clean.endswith('"'):
            clean = clean[1:-1]  # Remove leading and trailing quotes
            clean = clean.replace('""', '"')  # Replace double quotes in the middle

        # 2. Handle parenthesis attribution structure: match the content in the *last* set of parentheses
        match = re.search(r'^(.*?)_\(([^)]+)\)$', clean)

        if match:
            core = match.group(1).replace('_', ' ').strip()
            context = match.group(2).replace('_', ' ').strip()

            core_lower = core.lower()

            # Determine preposition
            if any(k in core_lower for k in of_keywords):
                mapped_nl = f"the {core} of {context}"
            elif any(k in core_lower for k in in_keywords):
                mapped_nl = f"the {core} in {context}"
            else:
                # Default fallback logic
                mapped_nl = f"the {core} associated with {context}"
        else:
            # 3. If there are no parentheses at the end, it's just a normal name, directly replace underscores
            mapped_nl = clean.replace('_', ' ')

        # Store in dictionary
        mapping_dict[raw] = mapped_nl

    # Output as JSON dictionary
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

    print(f"✅ Successfully generated entity mapping dictionary!")
    print(f"A total of {len(mapping_dict)} entities were successfully mapped.")
    if discarded_count > 0:
        print(f"⚠️ Interception triggered: Discarded {discarded_count} ambiguous entities due to containing the '/' symbol.")
    print(f"Dictionary saved to: {out_path.resolve()}")

# Run code
# clean_and_map_entities("news/entity_selected.txt", "news/entity_mapping.json")