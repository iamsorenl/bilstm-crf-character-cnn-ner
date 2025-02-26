import sys

def read_file(filepath):
    """
    Reads a file and returns a list of (token, label) tuples.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                data.append((None, None))  # Sentence boundary
                continue

            parts = line.split('\t')
            if len(parts) == 2:
                token, label = parts
                data.append((token, label))
            else:
                print(f"Warning (Line {line_num}): Expected 2 columns but got {len(parts)} - '{line}'")
    return data

def find_label_mismatches(predictions, answers):
    """
    Finds label mismatches between predictions and answers.
    """
    mismatches = []
    for idx, ((pred_token, pred_label), (ans_token, ans_label)) in enumerate(zip(predictions, answers)):
        if pred_token != ans_token and pred_token is not None:
            print(f"[Token Mismatch] Line {idx + 1}: Prediction token '{pred_token}' != Answer token '{ans_token}'")
            continue

        if pred_label != ans_label and pred_token is not None:
            mismatches.append((idx + 1, pred_token, ans_label, pred_label))

    return mismatches

def report_mismatches(mismatches):
    """
    Prints a summary of label mismatches.
    """
    if not mismatches:
        print("✅ No label mismatches found.")
        return

    print(f"\n🚨 Found {len(mismatches)} label mismatches:")
    for line_num, token, true_label, pred_label in mismatches:
        print(f"Line {line_num}: Token='{token}' | True='{true_label}' | Predicted='{pred_label}'")

def main(predictions_path, answers_path):
    print("📂 Reading files...")
    predictions = read_file(predictions_path)
    answers = read_file(answers_path)

    if len(predictions) != len(answers):
        print(f"⚠️ Length mismatch: Predictions={len(predictions)} vs Answers={len(answers)}")
        return

    print("🔎 Checking for label mismatches...")
    mismatches = find_label_mismatches(predictions, answers)
    report_mismatches(mismatches)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_2.py <dev.predictions> <dev.answers>")
        sys.exit(1)

    predictions_path = sys.argv[1]
    answers_path = sys.argv[2]
    main(predictions_path, answers_path)
