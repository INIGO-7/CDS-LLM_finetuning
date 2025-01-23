from Bio import SeqIO
import pandas as pd
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
            else: print(f"  [ID #{i+1}: {record.id}]", end="")

    return ccds_df, cds_sequences, genome

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
        os.path.join(res_dir, ccds_file),
        os.path.join(res_dir, fasta_file),
        os.path.join(res_dir, genome_file),
        verbose=True
    )

    # Specify the chromosome and region
    # chromosome = 1      # Replace with the correct identifier
    # start = 925941 - 1  # Convert to 0-based indexing
    # end = 926012        # 1-based indexing end
    # nc_accession = "NC_000001.11"

if __name__ == "__main__":
    main()