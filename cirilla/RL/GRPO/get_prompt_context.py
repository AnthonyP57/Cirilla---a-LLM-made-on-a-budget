import json
import os
import re

# 1. Base path
BASE_DIR = './training_datasets/raw'

# 2. Build Indices
# We store tuples: (path, file_size) so we can compare
exact_index = {} # Key: filename -> Value: (path, size)
fuzzy_index = {} # Key: clean_name -> Value: (path, size)

empty_files_count = 0
replaced_count = 0

print(f"Indexing files recursively in {BASE_DIR}...")

for root, dirs, files in os.walk(BASE_DIR):
    for filename in files:
        if not filename.endswith(".txt"):
            continue
            
        full_path = os.path.join(root, filename)
        
        try:
            # Check size first (faster than reading content)
            file_size = os.path.getsize(full_path)
            
            if file_size == 0:
                empty_files_count += 1
                continue
            
            # Read content to verify it's not just whitespace
            # and to get the TRUE length of text (in case of encoding diffs)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                content_len = len(content.strip())
                
                if content_len == 0:
                    empty_files_count += 1
                    continue
                
                # We use content_len for comparison, not bytes, 
                # to prioritize actual text length.
                
        except OSError:
            continue

        # --- LOGIC: Keep the Longest Version ---

        # A. Exact Index Update
        if filename not in exact_index:
            exact_index[filename] = (full_path, content_len)
        else:
            # If new file is longer than the old one, overwrite it
            old_path, old_len = exact_index[filename]
            if content_len > old_len:
                exact_index[filename] = (full_path, content_len)
                replaced_count += 1
        
        # B. Fuzzy Index Update
        # Clean: "The Witcher 3 quests.txt" -> "the witcher quests.txt"
        clean_name = re.sub(r'\d+', '', filename.lower())
        clean_name = re.sub(r'\s+', ' ', clean_name).strip() 
        
        if clean_name not in fuzzy_index:
            fuzzy_index[clean_name] = (full_path, content_len)
        else:
            # If new file is longer than the old one, overwrite it
            old_path, old_len = fuzzy_index[clean_name]
            if content_len > old_len:
                fuzzy_index[clean_name] = (full_path, content_len)
                replaced_count += 1

print(f"Indexed {len(exact_index)} unique filenames.")
print(f"Upgraded to longer versions {replaced_count} times.")
print(f"Skipped {empty_files_count} empty files.")

# 3. Process Prompts
valid_contexts = []
skipped_count = 0

output_path = './training_datasets/RL/prompts_context.jsonl'

try:
    with open('./training_datasets/RL/prompts.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            subject = data.get('subject', '').strip()
            target_filename = f"{subject}.txt"
            
            found_path = None
            
            # Attempt 1: Exact Match
            if target_filename in exact_index:
                found_path = exact_index[target_filename][0] # Get path from tuple
            
            # Attempt 2: Fuzzy Match
            else:
                clean_subject = re.sub(r'\d+', '', target_filename.lower())
                clean_subject = re.sub(r'\s+', ' ', clean_subject).strip()
                
                if clean_subject in fuzzy_index:
                    found_path = fuzzy_index[clean_subject][0] # Get path from tuple
            
            # DECISION: Include or Skip
            if found_path:
                with open(found_path, 'r', encoding='utf-8', errors='ignore') as cf:
                    context_text = cf.read()
                    
                    if context_text.strip():
                        data['context'] = context_text
                        valid_contexts.append(data)
                    else:
                        skipped_count += 1
            else:
                skipped_count += 1

    # 4. Save Output
    with open(output_path, 'w') as f:
        for line in valid_contexts:
            json.dump(line, f)
            f.write('\n')

    print(f"\nProcessing complete.")
    print(f"Total lines saved: {len(valid_contexts)}")
    print(f"Lines skipped: {skipped_count}")

except FileNotFoundError:
    print("Error: Could not find prompts.jsonl")