from Bio import SeqIO
import pandas as pd
import numpy as np
import gzip
import os

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
    genome = SeqIO.parse(human_gen_path, "fasta-pearson")

    if verbose:
        for i, record in enumerate(genome):
            if i >= 5:
                break
            if i == 0: print(f"-- [ID #{i+1}: {record.id}]", end="")
            else: print(
                f"  [ID #{i+1}: {record.id}]", 
                end="" if i!=4 else "\n"
                )

    return ccds_df, cds_sequences, genome

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
    
    ccds_df, cds_sequences, genome = load_resources(
        os.path.join(res_dir, ccds_file),   # DataFrame
        os.path.join(res_dir, fasta_file),  # Dictionary
        os.path.join(res_dir, genome_file), # Fasta BioSeq
        verbose=True
    )

    consistency_results = []

    for idx, row in ccds_df.iterrows():
        # Just testing for now...
        # if idx == 2: break

        if row["ccds_id"] in cds_sequences:
            locs = row[ "cds_locations" ].replace( '[', '' ).replace( ']', '' )

            locs_clean = [ l.strip() for l in locs.split(",") ]
            ccds_start = 0
            test_counter = 0

            for loc in locs_clean:
    
                split = loc.split( '-' )
                start, stop = int( split[0] ), int( split[1] )

                # Add one because both start and stop elements are included
                distance = stop-start + 1
                ccds_stop = ccds_start + distance

                print( f"Genomic idx (start:stop) -> ({start}:{stop})" )
                print( f"ID -{row['ccds_id']}- start: {ccds_start}, stop: {ccds_stop}" )

                # Add one because string printing in python does not include stop idx
                print( f"CDS SEQ: {cds_sequences[row['ccds_id']][ccds_start:ccds_stop+1]}" )

                ccds_start = ccds_stop
                test_counter += distance
        else:
            print( f"The ccds_id '{row['ccds_id']}' isn't found in the nucleotide file")
            continue
        seqlength = len(cds_sequences[row['ccds_id']])
        print(f"sequence's length = {seqlength}")
        print(f"Thought cds length = {test_counter}")
        consistency_results.append(seqlength == test_counter)

    if np.all(consistency_results):
        print('Consistency test passed')
    else:
        print(f'Inconsistencies found: {np.sum(consistency_results)}')
    
if __name__ == "__main__":
    main()