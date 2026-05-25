import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_txt_data(filepath):
    """Static data loading function (independent of the model, extracted out)"""
    print(f"Parsing raw TXT data from {filepath}...")
    parsed_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 4:
                continue

            raw_subject, raw_relation, raw_object, raw_time = parts
            clean_subject = raw_subject.replace('_', ' ')
            clean_object = raw_object.replace('_', ' ')

            parsed_data.append({
                "raw_subject": raw_subject,
                "raw_object": raw_object,
                "relation": raw_relation,
                "time": raw_time,
                "clean_subject": clean_subject,
                "clean_object": clean_object
            })
    return parsed_data


class TemporalProber:
    """Core class for the Temporal Knowledge Prober (Supports GPU Batched Inference)"""

    def __init__(self, model_path, map_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Prober...")
        print(f"Device: {self.device}")

        # 1. Load Dictionary
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Relation mapping file not found: {map_path}")
        with open(map_path, 'r', encoding='utf-8') as f:
            self.relations_map = json.load(f)

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # [Key Modification 1]: Decoder-only models MUST use left padding for batched inference!
        # Otherwise, generated words are appended after padding tokens, causing total failure.
        self.tokenizer.padding_side = "left"

        # 3. Load Model
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully!\n")

    def create_prompt(self, clean_subject, relation_key, time_str):
        rel_info = self.relations_map.get(relation_key)
        if not rel_info or "past_verb" not in rel_info:
            return None
        past_verb = rel_info["past_verb"]
        # Reverted to the template based on pure declarative sentence continuation
        return f"On {time_str}, {clean_subject} {past_verb} "

    # [Key Modification 2]: Brand new batched inference pipeline
    def run_probing_batched(self, data_file, output_file, batch_size=32, save_interval=1000):
        """
        batch_size: Amount of data fed into the GPU simultaneously. Higher VRAM allows a larger batch size.
        """
        raw_data = load_txt_data(data_file)
        known_facts = []

        # First, filter out data capable of generating valid Prompts, discarding useless data
        valid_items = []
        for item in raw_data:
            prompt = self.create_prompt(item['clean_subject'], item['relation'], item['time'])
            if prompt:
                item['prompt'] = prompt
                valid_items.append(item)

        print(f"Filtered to {len(valid_items)} valid facts.")
        print(f"Starting batched inference with batch_size={batch_size}...")

        # Iterate in blocks according to batch_size
        for i in tqdm(range(0, len(valid_items), batch_size)):
            batch_items = valid_items[i: i + batch_size]
            prompts = [item['prompt'] for item in batch_items]
            expected_objects = [item['clean_object'].lower() for item in batch_items]

            # Batched Tokenize, enable padding
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=0.0
                )

            # Get the true length of the Prompt after padding, used to accurately intercept the New Tokens
            prompt_length = inputs.input_ids.shape[1]

            # Iterate over each generated result in the Batch
            for j, output_seq in enumerate(outputs):
                # Intercept only the newly generated part
                gen_tokens = output_seq[prompt_length:]
                generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

                gen_clean = generated_text.strip().lower()
                obj_clean = expected_objects[j]

                # Matching logic
                if obj_clean in gen_clean:
                    hit_item = batch_items[j].copy()  # Copy original data
                    hit_item['model_generation'] = generated_text.strip()
                    hit_item['prompt_used'] = hit_item['prompt']
                    del hit_item['prompt']  # Delete unnecessary temporary field
                    known_facts.append(hit_item)

            # Batched saving logic (reduce disk writing frequency to once every 1000 items, significantly improving I/O speed)
            if (i // batch_size) % (save_interval // batch_size) == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(known_facts, f, indent=4, ensure_ascii=False)

        # Final save after loop ends
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(known_facts, f, indent=4, ensure_ascii=False)

        print(f"\nProbing complete! Found {len(known_facts)} perfectly known facts.")
        if len(valid_items) > 0:
            print(f"Retention Rate: {(len(known_facts) / len(valid_items)) * 100:.2f}%")