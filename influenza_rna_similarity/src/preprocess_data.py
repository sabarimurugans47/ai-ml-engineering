from nltk.util import ngrams

def dna_to_rna(seq):
    return seq.replace("T", "U")

def extract_kmers(seq, k=3):
    return [''.join(g) for g in ngrams(seq, k)]

def preprocess_dataframe(df, k=3):
    
    df = df.copy()
    df['sequence'] = df['sequence'].apply(dna_to_rna)
    df['kmers'] = df['sequence'].apply(lambda x: extract_kmers(x, k))

    return df
