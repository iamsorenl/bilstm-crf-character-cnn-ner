# Script to calculate the average sequence length in the training data (CoNLL format)
train_file_path = "A2-data/train"

def calculate_average_sequence_length(file_path):
    sequence_lengths = []
    current_length = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # If line is empty, it indicates the end of a sentence
            if not line:
                if current_length > 0:
                    sequence_lengths.append(current_length)
                    current_length = 0
            else:
                # Ensure the line has two columns (word and tag)
                parts = line.split('\t')
                if len(parts) == 2:
                    current_length += 1

        # If the last sentence doesn't end with a blank line
        if current_length > 0:
            sequence_lengths.append(current_length)

    average_length = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
    print(f"Average Sequence Length: {average_length:.2f}")
    print(f"Max Length: {max(sequence_lengths)}, Min Length: {min(sequence_lengths)}")
    print(f"Total Sentences: {len(sequence_lengths)}")
    return average_length, max(sequence_lengths), min(sequence_lengths), len(sequence_lengths)

# Calculate and display the results
average_length, max_length, min_length, num_sentences = calculate_average_sequence_length(train_file_path)
average_length, max_length, min_length, num_sentences
