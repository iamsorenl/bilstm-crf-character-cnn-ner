

def load_data(filepath, with_labels=True):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()

            # Sentence boundary: save collected words (and tags)
            if not line:
                if words:  
                    data.append((words, tags if with_labels else []))
                    words, tags = [], []
                continue

            parts = line.split('\t')

            if with_labels:
                if len(parts) != 2:
                    raise ValueError(f"Expected 2 columns (word, tag) but got {len(parts)} at line: {line}")
                word, tag = parts
                words.append(word)
                tags.append(tag)
            else:
                # For test data without tags
                words.append(parts[0])

        # Handle the last sentence if the file doesn't end with a blank line
        if words:
            data.append((words, tags if with_labels else []))

    print(f"Loaded {len(data)} sentences from {filepath}")
    return data