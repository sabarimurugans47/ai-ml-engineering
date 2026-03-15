import gradio as gr
from load_data import load_fasta_files
from preprocess_data import preprocess_dataframe
from train_model import build_kmer_vocab, compute_user_similarity
from nltk.util import ngrams
import pandas as pd

# Load reference dataset once
SEGMENT_FILES = {
    "PB2": r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PB2.fa",
    "PB1": r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PB1.fa",
    "PA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PA.fa",
    "HA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\HA.fa",
    "NP":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NP.fa",
    "NA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NA.fa",
    "M":   r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\M.fa",
    "NS":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NS.fa"
}

print("Loading reference data...")
df = load_fasta_files(SEGMENT_FILES)
df = preprocess_dataframe(df, k=3)
vocab = build_kmer_vocab(k=3)
print(f"✓ Loaded {len(df)} sequences successfully!")

def analyze_sequence(sequence_file, sequence_text, segment):
    """Process RNA sequence and compute cosine similarity"""
    
    seq = ""
    if sequence_file is not None:
        with open(sequence_file, "r") as f:
            seq = f.read()
    elif sequence_text.strip():
        seq = sequence_text.strip()
    else:
        return pd.DataFrame([{"Error": "Please provide a sequence"}]), ""

    if segment not in SEGMENT_FILES:
        return pd.DataFrame([{"Error": f"Invalid segment: {segment}"}]), ""

    try:
        lines = seq.splitlines()
        seq = "".join([line for line in lines if not line.startswith(">")])
        seq = seq.replace("T", "U").replace(" ", "").upper()

        if len(seq) < 3:
            return pd.DataFrame([{"Error": "Sequence too short (minimum 3 bases)"}]), ""

        user_kmers = [''.join(g) for g in ngrams(seq, 3)]
        results = compute_user_similarity(user_kmers, df, vocab, segment)

        if not results:
            return pd.DataFrame([{"Error": "No matches found"}]), ""

        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:15]
        
        # Check if best match has low similarity (below 50%)
        best_match_score = results_sorted[0][1]
        alert_message = ""
        
        if best_match_score < 0.50:
            alert_message = f"""
            <style>
                @keyframes blink-alert {{
                    0% {{ 
                        background-color: #FF0000;
                        box-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
                        transform: scale(1);
                    }}
                    25% {{ 
                        background-color: #CC0000;
                        box-shadow: 0 0 30px rgba(255, 0, 0, 1);
                        transform: scale(1.02);
                    }}
                    50% {{ 
                        background-color: #FF0000;
                        box-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
                        transform: scale(1);
                    }}
                    75% {{ 
                        background-color: #CC0000;
                        box-shadow: 0 0 30px rgba(255, 0, 0, 1);
                        transform: scale(1.02);
                    }}
                    100% {{ 
                        background-color: #FF0000;
                        box-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
                        transform: scale(1);
                    }}
                }}
                
                .alert-box {{
                    animation: blink-alert 1s ease-in-out infinite;
                    padding: 1.2rem 1.8rem;
                    border-radius: 10px;
                    border: 3px solid #8B0000;
                    max-width: 480px;
                    margin: 0 auto 1.5rem auto;
                    text-align: center;
                }}
                
                .alert-title {{
                    color: white;
                    font-family: 'Poppins', sans-serif;
                    font-size: 1.15rem;
                    font-weight: 800;
                    margin: 0 0 0.5rem 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                
                .alert-text {{
                    color: white;
                    font-size: 0.95rem;
                    margin: 0;
                    line-height: 1.5;
                    font-weight: 600;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                }}
            </style>
            
            <div class="alert-box">
                <div class="alert-title">⚠️ LOW SIMILARITY ALERT ⚠️</div>
                <div class="alert-text">Best Match: {best_match_score*100:.2f}%<br>Please verify sequence quality</div>
            </div>
            """
        
        output_data = []
        for rank, (strain_id, cos_score) in enumerate(results_sorted, 1):
            output_data.append({
                "Rank": f"#{rank}",
                "Reference Strain": strain_id,
                "Cosine Similarity": f"{cos_score:.6f}",
                "Percentage": f"{cos_score*100:.2f}%"
            })
        
        return pd.DataFrame(output_data), alert_message
        
    except Exception as e:
        return pd.DataFrame([{"Error": f"Analysis failed: {str(e)}"}]), ""

SEGMENT_CHOICES = list(SEGMENT_FILES.keys())

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background-color: #E3F2FD;
    background-image: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    background-attachment: fixed;
    min-height: 100vh;
    margin: 0;
    padding: 0;
}

.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

.hero-section {
    background: white;
    padding: 3rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    text-align: center;
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3F51B5 0%, #283593 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
}

.hero-subtitle {
    color: #000000;
    font-size: 1.2rem;
    font-weight: 500;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}

.stat-box {
    text-align: center;
    padding: 1.25rem 2.5rem;
    background: linear-gradient(135deg, #3F51B5 0%, #283593 100%);
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 12px rgba(63,81,181,0.3);
}

.stat-number {
    font-size: 2.25rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    color: white;
}

.stat-label {
    font-size: 0.95rem;
    color: white;
    opacity: 0.95;
    margin-top: 0.25rem;
}

.content-box {
    background: white;
    padding: 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #283593;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #3F51B5;
    padding-bottom: 0.75rem;
}

.section-text {
    color: #000000;
    font-size: 1.05rem;
    line-height: 1.8;
    margin-bottom: 1rem;
    font-weight: 400;
}

.section-text strong {
    color: #000000;
    font-weight: 700;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.info-card {
    background: linear-gradient(135deg, #E8EAF6 0%, #C5CAE9 100%);
    padding: 1.75rem;
    border-radius: 12px;
    border-left: 5px solid #3F51B5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 20px rgba(63,81,181,0.2);
}

.info-card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #283593;
    margin-bottom: 0.75rem;
}

.info-card-text {
    color: #000000;
    font-size: 0.95rem;
    line-height: 1.6;
    font-weight: 400;
}

label {
    font-weight: 600 !important;
    color: #283593 !important;
    font-size: 0.95rem !important;
}

input, textarea, select {
    border: 2px solid #C5CAE9 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    background: white !important;
    color: #000000 !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #3F51B5 !important;
    box-shadow: 0 0 0 3px rgba(63,81,181,0.15) !important;
}

.dataframe {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

.dataframe th {
    background: linear-gradient(135deg, #3F51B5 0%, #283593 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    font-size: 0.9rem !important;
}

.dataframe td {
    padding: 0.875rem !important;
    border-bottom: 1px solid #f0f0f0 !important;
    color: #000000 !important;
    font-weight: 400 !important;
}

.dataframe tbody tr:hover {
    background: #E8EAF6 !important;
}

button[variant="primary"] {
    background: linear-gradient(135deg, #3F51B5 0%, #283593 100%) !important;
    border: none !important;
    color: white !important;
    padding: 1rem 3rem !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(63,81,181,0.4) !important;
    transition: all 0.3s ease !important;
}

button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(63,81,181,0.5) !important;
}

@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    .info-grid {
        grid-template-columns: 1fr;
    }
    .stats-row {
        gap: 1rem;
    }
}
"""

with gr.Blocks(css=custom_css, title="Influenza RNA Analyzer") as app:
    
    with gr.Column(elem_classes="main-container"):
        
        # Hero Header
        gr.HTML("""
        <div class="hero-section">
            <h1 class="hero-title">🧬 Influenza RNA Similarity Model</h1>
            <p class="hero-subtitle">Advanced K-mer Ba sed Machine Learning System for Rapid Strain Identification</p>
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-number">8</div>
                    <div class="stat-label">Gene Segments</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">3-mer</div>
                    <div class="stat-label">K-mer Analysis</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">&lt;3s</div>
                    <div class="stat-label">Processing Time</div>
                </div>
            </div>
        </div>
        """)
        
        # Project Overview
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>Project Overview</h2>")
            gr.HTML("""
            <p class='section-text'>
                The <strong>Influenza RNA Sequence Analyzer</strong> is a cutting-edge computational biology platform designed 
                to rapidly identify influenza virus strains from RNA sequences. Using advanced k-mer decomposition 
                and machine learning algorithms, this tool enables researchers, clinicians, and public health 
                professionals to perform accurate strain identification in seconds.
            </p>
            <p class='section-text'>
                This project leverages natural language processing techniques applied to biological sequences, 
                treating RNA sequences as a "language" where k-mers serve as the vocabulary. By computing 
                similarity scores using cosine similarity metrics, we match unknown sequences against a 
                comprehensive reference database.
            </p>
            """)
        
        # Key Features
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>Key Features</h2>")
            gr.HTML("""
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">⚡ Lightning Fast</div>
                    <div class="info-card-text">Analysis completed in under 3 seconds with optimized algorithms</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🎯 High Accuracy</div>
                    <div class="info-card-text">Research-grade precision using cosine similarity metrics</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🔬 8 Segments</div>
                    <div class="info-card-text">Complete analysis for all influenza genome segments</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">📊 Similarity Scoring</div>
                    <div class="info-card-text">Detailed cosine similarity scores with percentage match</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">💻 User Friendly</div>
                    <div class="info-card-text">Simple interface for file upload or direct sequence input</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">📁 FASTA Support</div>
                    <div class="info-card-text">Standard bioinformatics file format compatibility</div>
                </div>
            </div>
            """)
        
        # Technical Methodology
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>Technical Methodology</h2>")
            gr.HTML("""
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">1. Sequence Processing</div>
                    <div class="info-card-text">BioPython parses FASTA files and extracts RNA sequences with normalization</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">2. K-mer Decomposition</div>
                    <div class="info-card-text">NLTK generates 3-mer subsequences from RNA using n-gram algorithms</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">3. Vectorization</div>
                    <div class="info-card-text">Sequences transformed into numerical feature vectors for comparison</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">4. Cosine Similarity</div>
                    <div class="info-card-text">Mathematical similarity scores identify closest matching strains</div>
                </div>
            </div>
            """)
        
        # Research Applications
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>Research Applications</h2>")
            gr.HTML("""
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">🏥 Clinical Diagnostics</div>
                    <div class="info-card-text">Rapid strain identification for treatment decisions</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">📊 Surveillance</div>
                    <div class="info-card-text">Monitor circulating strains and emerging variants</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">💉 Vaccine Development</div>
                    <div class="info-card-text">Evaluate coverage and guide formulation strategies</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🔍 Evolution Research</div>
                    <div class="info-card-text">Study genetic drift patterns and reassortment events</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🌍 Outbreak Investigation</div>
                    <div class="info-card-text">Trace transmission chains and geographic spread</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🎓 Education</div>
                    <div class="info-card-text">Bioinformatics training and computational virology</div>
                </div>
            </div>
            """)
        
        # Analysis Section
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>RNA Sequence Analysis</h2>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sequence_file = gr.File(
                        label="Upload FASTA File",
                        file_types=[".fa", ".fasta", ".txt"],
                        type="filepath"
                    )
                
                with gr.Column(scale=2):
                    sequence_text = gr.Textbox(
                        lines=7,
                        label="Or Paste RNA Sequence",
                        placeholder=">Sample_H3N2_HA\nAUGCUAGCUAGCUAGCU...",
                        info="FASTA format or raw sequence (T→U conversion automatic)"
                    )
            
            segment = gr.Dropdown(
                choices=SEGMENT_CHOICES,
                label="Select Influenza Segment",
                info="Choose the viral segment type",
                value="HA"
            )
            
            analyze_btn = gr.Button(
                "🚀 Analyze Sequence",
                variant="primary",
                size="lg"
            )
        
        # Alert Box (smooth continuous blinking)
        alert_box = gr.HTML(visible=True)
        
        # Results
        with gr.Column(elem_classes="content-box"):
            gr.HTML("<h2 class='section-title'>Cosine Similarity Results (Top 15 Matches)</h2>")
            
            output = gr.Dataframe(
                headers=["Rank", "Reference Strain", "Cosine Similarity", "Percentage"],
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                wrap=True
            )
    
    analyze_btn.click(
        analyze_sequence,
        inputs=[sequence_file, sequence_text, segment],
        outputs=[output, alert_box]
    )

if __name__ == "__main__":
    app.launch(server_port=7860, inbrowser=True, share=False)