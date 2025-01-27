from sklearn.model_selection import train_test_split
from Bio import SeqIO
import pandas as pd
import numpy as np
import os, pysam, random, json, gzip

def load_resources(ccds_path, fasta_seq_path, human_gen_path, verbose=False):

    # Read CCDS data with modified header handling
    try:
        with open(ccds_path, 'r') as f:
            header = f.readline().strip('#').strip()
            columns = header.split('\t')
            
        ccds_df = pd.read_csv(
            ccds_path, 
            sep='\t', 
            comment='#', 
            names=columns
        )

        ccds_df = ccds_df[ccds_df["ccds_status"] != "Withdrawn"]

        if verbose:
            print(f"-- Loaded the CCDS dataset with '{ccds_df.shape[0]}' entries")
        
    except Exception as e:
        print(f"Error reading CCDS file: {e}")
        return (pd.DataFrame(), {}, )
    
    # Read sequences
    cds_sequences = {}
    try:
        with gzip.open(fasta_seq_path, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                ccds_id = record.id.split('|')[0] if '|' in record.id else record.id
                cds_sequences[ccds_id] = str(record.seq)
        if verbose:
            print(f"-- Loaded {len(cds_sequences)} cds_sequences from FASTA file")
            print(f"-- Sample sequence IDs: {list(cds_sequences.keys())[:5]}")
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return 0
    
    # Load the human genome
    genome_fasta = pysam.FastaFile(human_gen_path)
    print(f"-- Loaded human sequences from fasta file")

    return ccds_df, cds_sequences, genome_fasta

def find_sequence_by_id(fasta_file, target_id):
    """
    Find and return a sequence with a specific ID from a FASTA file.
    
    Parameters:
    -----------
    fasta_file : biopython..... ?
        The already parsed FASTA file
    target_id : str
        Identifier of the sequence to find
    
    Returns:
    --------
    Biopython SeqRecord object if found, None otherwise
    """
    
    # Iterate through all records in the FASTA file
    for record in fasta_file:
        # Check if the current record's ID matches the target ID
        if record.id == target_id:
            return record
    
    # If no matching sequence is found
    print(f"No sequence found with ID: {target_id}")
    return None

def get_seq_fragment(fasta_file, accession_number, start, end):

    # Retrieve the sequence
    try:
        sequence = fasta_file.fetch(accession_number, start, end)
        return sequence.upper()     # Return uppercase
    except KeyError:
        print(f"Accession number {accession_number} not found in the FASTA file.")

def create_dataset(ccds_df, cds_sequences, genome_fasta):

    dataset = []

    for idx, row in ccds_df.iterrows():
        # Just testing for now...
        # if idx == 2: break

        if row["ccds_id"] in cds_sequences:

            # Getting training feature samples:
            padding_left, padding_right = random.randint(0, 300), random.randint(0, 300)

            feature_start = int( row['cds_from'] ) - padding_left
            if feature_start < 0: feature_start = 0

            feature_end = int( row['cds_to'] ) + padding_right
            # if feature_end < ??: feature_end = len(??)

            train_sample = get_seq_fragment(genome_fasta, row[ "nc_accession" ], feature_start, feature_end)

            # Getting target feature samples
            locs = row[ "cds_locations" ].replace( '[', '' ).replace( ']', '' )

            locs_clean = [ l.strip() for l in locs.split(",") ]
            ccds_start = 0
            sequence_start = padding_left
            output_positions = ""
            output_sequences = ""

            for idx, loc in enumerate(locs_clean):

                split = loc.split( '-' )
                start = int( split[0] )
                
                # Start is the distance between previous stop and new start, added to the previous sequence_stop
                if idx != 0: sequence_start = sequence_stop + (start - stop)

                stop = int( split[1] )

                distance = stop - start
                ccds_stop, sequence_stop = ccds_start + distance, sequence_start + distance

                if idx < len(locs_clean) - 1:
                    output_positions += f"{sequence_start}-{sequence_stop+1}" + ","
                    output_sequences += cds_sequences[row['ccds_id']][ccds_start:ccds_stop+1] + ';'

                else: 
                    output_positions += f"{ccds_start}-{ccds_stop+1}"
                    output_sequences += cds_sequences[row['ccds_id']][ccds_start:ccds_stop+1]

                ccds_start = ccds_stop + 1

            if not train_sample:
                print(f"The training feature for cds'id '{row['ccds_id']}' and " +
                        f"nc_accession '{row[ 'nc_accession' ]}' can't be found")
                
            description = f"CCDS_ID: {row[ 'ccds_id' ]}, NC_ACCESSION: {row[ 'nc_accession' ]}"
            is_consistent = 1 if np.all([cds in train_sample for cds in output_sequences.split(';')]) else 0
            # dataset.append( [ description, train_sample, target, is_consistent ] )
            if is_consistent == 1:
                dataset.append( {
                    "input": train_sample,
                    "output_positions": output_positions,
                    "output_sequences": output_sequences,
                    "input_len": len(train_sample)
                    } 
                )
        else:
            print( f"The ccds_id '{row['ccds_id']}' isn't found in the nucleotide file")
            continue

    return dataset

def main():

    # Define the file path
    genome_file = "GCF_000001405.40_GRCh38.p14_genomic.fna"
    # Different project paths and settings
    res_dir = "res"
    output_dir = "output"
    ccds_file = "CCDS.current.txt"
    fasta_file = "CCDS_nucleotide.current.fna.gz"
    genome_file = "GCF_000001405.40_GRCh38.p14_genomic.fna"
    output_file = "training_examples.json"
    max_seq_length = 1000
    
    ccds_df, cds_sequences, genome_fasta = load_resources(
        os.path.join(res_dir, ccds_file),   # DataFrame
        os.path.join(res_dir, fasta_file),  # Dictionary
        os.path.join(res_dir, genome_file), # Pysam fasta
        verbose=True
    )

    dataset = create_dataset(
        ccds_df, 
        cds_sequences, 
        genome_fasta
    )

    train_val, test = train_test_split(dataset, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 = 0.1

    print("-- Dataset extracted successfully.")
    print(f"   - Training set size: {len(train)}")
    print(f"   - Validation set size: {len(val)}")
    print(f"   - Test set size: {len(test)}")

    # acc = 0
    # for tuple in dataset:
    #     if tuple["input_len"] > 2000:
    #         acc += 1

    # print(f"The number of tuples with more than 2000 chars is: {acc}")

    with open(os.path.join(output_dir, 'train_set.json'), 'w') as json_file:
        json.dump(train, json_file, indent=3)

    with open(os.path.join(output_dir, 'val_set.json'), 'w') as json_file:
        json.dump(val, json_file, indent=3)

    with open(os.path.join(output_dir, 'test_set.json'), 'w') as json_file:
        json.dump(test, json_file, indent=3)

    print(f"Train, Val and Test sets saved in '/{output_dir}' directory")

if __name__ == "__main__":
    main()