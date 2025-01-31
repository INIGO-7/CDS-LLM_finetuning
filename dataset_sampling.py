input_file = 'output/dataset_v3/val_set_lt2exp13K.jsonl'
n_samples = 40
output_file = f'output/dataset_v3/val_set_lt2exp13K_{n_samples}samples.jsonl'

# Open the input file and output file
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate over the first 50 lines and write them to the output file
    for i, line in enumerate(infile):
        if i >= n_samples:
            break
        outfile.write(line)

print(f"First {n_samples} lines have been saved to {output_file}")