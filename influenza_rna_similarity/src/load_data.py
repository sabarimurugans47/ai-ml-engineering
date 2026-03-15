# src/load_data.py

import pandas as pd
from Bio import SeqIO

def load_fasta_files(segment_files):
    
    records = []

    for segment, file_path in segment_files.items():
        for record in SeqIO.parse(file_path, "fasta"):
            records.append({
                "id": record.id,
                "sequence": str(record.seq),
                "length": len(record.seq),
                "segment": segment
            })

    return pd.DataFrame(records)