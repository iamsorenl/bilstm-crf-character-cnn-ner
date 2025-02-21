from data_preprocessing import read_conll_file

# Test with labeled data
data_with_labels = read_conll_file("A2-data/train", with_labels=True)
print(f"Sample (with labels): {data_with_labels[0]}")

# Test with unlabeled data
data_without_labels = read_conll_file("A2-data/dev.answers", with_labels=True)
print(f"Sample (without labels): {data_without_labels[0]}")
