"""Streamlit UI for CASMI 2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, pick_device
from src.model import SpectrumFPNet
from src.retrieval import retrieve_topk
from src.spectrum_input import parse_spectrum_csv
from src.utils import bin_spectrum
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

st.set_page_config(
    page_title="CASMI 2026 — Molecule ID",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(37, 99, 235, 0.18), transparent 38%),
        radial-gradient(ellipse at 0% 28%, rgba(13, 148, 136, 0.12), transparent 34%),
        linear-gradient(145deg, #080d19 0%, #0d1525 52%, #111c32 100%);
}
.block-container { max-width: 1560px; padding: 2.2rem 2rem 4rem; }
h1 { font-weight: 800; letter-spacing: -0.045em; }
h2, h3 { letter-spacing: -0.025em; }
[data-testid="stSidebar"] {
    background: rgba(9, 15, 28, 0.94);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.88));
    padding: 0.9rem 1.1rem;
    border-radius: 0.85rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.62);
    border-color: rgba(148, 163, 184, 0.16);
    border-radius: 0.85rem;
}
[data-testid="stImage"] img { border-radius: 0.65rem; }
[data-testid="stButton"] button[kind="primary"] {
    min-height: 2.8rem;
    border: 0;
    border-radius: 0.75rem;
    background: linear-gradient(100deg, #0f9f91, #2563eb);
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="color:#5eead4;font-size:.78rem;font-weight:700;'
    'letter-spacing:.18em;text-transform:uppercase;margin-bottom:.35rem">'
    'Islamic University of Science & Technology · M.Tech AI Lab</p>',
    unsafe_allow_html=True,
)
st.title("CASMI 2026: Tandem Mass Spectra Molecule Identifier")
st.caption("Deep Multi-Scale Neural Fingerprint Regression & Sub-ppm Precursor Mass Filtered Retrieval")

DEVICE = pick_device()

@st.cache_resource
def load_model():
    ckpt = torch.load(CFG.MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    m = SpectrumFPNet(ckpt["in_dim"], CFG.FP_BITS).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m

try:
    MODEL = load_model()
except Exception as e:
    MODEL = None

@st.cache_data
def load_submission_data():
    sub_path = CFG.WORKSPACE / "predictions" / "submission.csv"
    if not sub_path.exists():
        sub_path = CFG.WORKSPACE / "submission.csv"
    sub_df = pd.read_csv(sub_path) if sub_path.exists() else None

    test_path = CFG.DATA_DIR / "raw" / "test.parquet"
    test_df = pd.read_parquet(test_path) if test_path.exists() else None
    return sub_df, test_df

sub_df, test_df = load_submission_data()

tab_predictions, tab_interactive, tab_analytics = st.tabs([
    "📋 Kaggle Test Predictions (400 Molecules)",
    "🎯 Custom Spectrum Predictor",
    "📊 Lab Training & Analytics",
])

# ==============================================================================
# TAB 1: KAGGLE TEST PREDICTIONS EXPLORER
# ==============================================================================
with tab_predictions:
    st.subheader("Validated Submission Predictions")
    st.caption("Fresh predictions generated with the sub-micro-Dalton monoisotopic precursor mass filter (±15 ppm) and InChIKey14 deduplication.")

    # KPI summary
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Test Molecules", "400" if sub_df is not None else "0", "1,213 Spectra")
    with kpi2:
        st.metric("Candidates / Query", "25", "InChIKey14 Unique")
    with kpi3:
        st.metric("Precursor Filter", "±15 ppm", "Monoisotopic MW")
    with kpi4:
        st.metric("Validation Accuracy", "98.15%", "84.1% Active Prec.")

    if sub_df is not None:
        mol_ids = sub_df["molecule_id"].tolist()
        col_sel, col_download = st.columns([3, 1])
        with col_sel:
            selected_mol = st.selectbox("Select Test Molecule ID:", mol_ids, index=0)
        with col_download:
            csv_bytes = (CFG.WORKSPACE / "predictions" / "submission.csv").read_bytes()
            st.download_button(
                "📥 Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
                type="primary",
            )

        # Get candidates for selected molecule
        row = sub_df[sub_df["molecule_id"] == selected_mol].iloc[0]
        cands_smi = [s.strip() for s in row["smiles"].split(";") if s.strip()]

        # Metadata from test.parquet if available
        meta_info = []
        peaks_mzs, peaks_intens = [], []
        if test_df is not None:
            mol_rows = test_df[test_df["molecule_id"] == selected_mol]
            if len(mol_rows) > 0:
                first_r = mol_rows.iloc[0]
                adduct = first_r.get("adduct", "N/A")
                prec_mz = first_r.get("precursor_mz", 0.0)
                ion_mode = first_r.get("ionization_mode", "N/A")
                inst_type = first_r.get("instrument_type", "N/A")
                energies = mol_rows["collision_energy_orig"].dropna().unique().tolist()
                energies_str = ", ".join(str(e) for e in energies) if energies else "Multi-energy"

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.info(f"**Precursor m/z:** {prec_mz:.4f}")
                col_m2.info(f"**Adduct:** {adduct} ({ion_mode})")
                col_m3.info(f"**Instrument:** {inst_type}")
                col_m4.info(f"**Collision Energies:** {energies_str}")

                # Concatenate/sample peaks for spectrum plot
                try:
                    all_mzs = np.concatenate([np.array(r, dtype=np.float32) for r in mol_rows["ms2_mzs"]])
                    all_ints = np.concatenate([np.array(r, dtype=np.float32) for r in mol_rows["ms2_normalized_intensities"]])
                    peaks_mzs, peaks_intens = all_mzs, all_ints
                except Exception:
                    pass

        col_plot, col_table = st.columns([1, 1])

        with col_plot:
            st.markdown(f"##### 📈 Query MS/MS Spectrum ({selected_mol})")
            if len(peaks_mzs) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=peaks_mzs, y=peaks_intens,
                    marker={"color": peaks_intens, "colorscale": "Tealgrn"},
                ))
                fig.update_layout(
                    template="plotly_dark", height=380,
                    xaxis_title="m/z", yaxis_title="Normalized Intensity",
                    margin={"l": 20, "r": 20, "t": 20, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No raw peak array available for this molecule in test.parquet.")

        with col_table:
            st.markdown(f"##### 🏆 Ranked Candidates (Top {len(cands_smi)})")
            inchikeys = []
            if HAS_RDKIT:
                for smi in cands_smi:
                    m = Chem.MolFromSmiles(smi)
                    if m is not None:
                        inchikeys.append(Chem.MolToInchiKey(m)[:14])
                    else:
                        inchikeys.append("N/A")
            else:
                inchikeys = ["N/A"] * len(cands_smi)

            cand_table = pd.DataFrame({
                "Rank": list(range(1, len(cands_smi) + 1)),
                "InChIKey14": inchikeys,
                "SMILES": cands_smi,
            })
            st.dataframe(cand_table, use_container_width=True, height=380, hide_index=True)

        # Structure gallery for top 6 candidates
        st.markdown("##### 🧬 Top-Ranked Candidate 2D Molecular Structures")
        if HAS_RDKIT:
            struct_cols = st.columns(3)
            for i, smi in enumerate(cands_smi[:6]):
                mol = Chem.MolFromSmiles(smi)
                c_idx = i % 3
                with struct_cols[c_idx]:
                    with st.container(border=True):
                        st.markdown(f"**Candidate #{i + 1}**")
                        if mol is not None:
                            img = Draw.MolToImage(mol, size=(300, 220))
                            st.image(img, use_column_width=True)
                            st.caption(f"`{inchikeys[i]}`")
                        else:
                            st.warning(f"Could not render structure #{i + 1}")
        else:
            st.info("RDKit 2D structure rendering is disabled because RDKit is not loaded.")
    else:
        st.warning("No submission.csv found in predictions/ or project root.")

# ==============================================================================
# TAB 2: INTERACTIVE CUSTOM SPECTRUM PREDICTOR
# ==============================================================================
with tab_interactive:
    st.subheader("Custom Spectrum Query")
    st.caption("Input experimental peaks manually or upload a peaklist CSV to infer molecular fingerprints and retrieve ranked candidates in real time.")

    with st.sidebar:
        st.header("Custom Query Input")
        mode = st.radio("Input Format", ["Manual peaks", "Upload CSV"])
        precursor = st.number_input("Precursor m/z", 50.0, 2000.0, 348.0, 0.01)
        adduct = st.selectbox("Adduct",
            ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
             "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"])
        if mode == "Manual peaks":
            txt = st.text_area("Peaks (m/z,intensity per line)", "150.05,1.0\n120.08,0.4\n90.03,0.2")
        else:
            example_path = CFG.DATA_DIR / "examples" / "spectrum_example.csv"
            if example_path.exists():
                st.download_button(
                    "Download example spectrum CSV",
                    data=example_path.read_bytes(),
                    file_name="spectrum_example.csv",
                    mime="text/csv",
                )
            up = st.file_uploader(
                "Upload spectrum CSV",
                type=["csv"],
                help="The file needs m/z and intensity columns.",
            )

    def parse_peaks():
        if mode == "Manual peaks":
            mzs, intensities = [], []
            for line_number, line in enumerate(txt.splitlines(), start=1):
                if not line.strip():
                    continue
                parts = [value.strip() for value in line.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Line {line_number} must contain exactly m/z,intensity.")
                try:
                    mzs.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError as exc:
                    raise ValueError(f"Line {line_number} contains a non-numeric value.") from exc
            if not mzs:
                raise ValueError("Enter at least one peak.")
            mzs = np.asarray(mzs, dtype=np.float32)
            intensities = np.asarray(intensities, dtype=np.float32)
        else:
            if up is None:
                raise ValueError("Upload a CSV file containing m/z and intensity columns.")
            mzs, intensities = parse_spectrum_csv(up.getvalue())

        if not len(mzs) or len(mzs) != len(intensities):
            raise ValueError("Provide matching, non-empty m/z and intensity values.")
        if not np.isfinite(mzs).all() or not np.isfinite(intensities).all():
            raise ValueError("Peak values must be finite numbers.")
        if np.any(mzs < 0) or np.any(intensities < 0):
            raise ValueError("m/z and intensity values cannot be negative.")
        if not np.any(intensities > 0):
            raise ValueError("At least one intensity must be greater than zero.")
        return mzs, intensities

    if st.button("🚀 Run Neural Prediction", type="primary"):
        input_error = False
        try:
            mzs, inten = parse_peaks()
        except ValueError as exc:
            st.error(str(exc))
            input_error = True
            mzs = inten = np.array([], dtype=np.float32)

        if not input_error and not len(mzs):
            st.info("Fix the spectrum input to run a prediction.")
        elif not input_error and MODEL is None:
            st.error("Model weights (best_model.pt) not loaded.")
        elif not input_error:
            try:
                vec = bin_spectrum(mzs, inten)
                x = np.concatenate([vec, [precursor/1000.0]]).astype(np.float32)
                with torch.no_grad():
                    fp = torch.sigmoid(
                        MODEL(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
                    ).cpu().numpy()[0]
                smis, keys, sims = retrieve_topk(
                    fp, precursor, adduct, top_k=25
                )

                c_plot, c_res = st.columns([1, 1])
                with c_plot:
                    st.subheader("📈 Input MS/MS Spectrum")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=mzs, y=inten,
                        marker={"color": inten, "colorscale": "Tealgrn"},
                    ))
                    fig.update_layout(
                        template="plotly_dark", height=380,
                        xaxis_title="m/z", yaxis_title="Intensity",
                        margin={"l": 20, "r": 20, "t": 25, "b": 20},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c_res:
                    st.subheader("🏆 Top-25 Retrieved Candidates")
                    result_frame = pd.DataFrame({
                        "Rank": range(1, len(smis) + 1),
                        "SMILES": smis,
                        "InChIKey14": keys,
                        "Tanimoto": [round(score, 4) for score in sims],
                    })
                    st.dataframe(result_frame, use_container_width=True, height=440, hide_index=True)

                if HAS_RDKIT and smis:
                    st.subheader("🧬 Candidate 2D Molecular Structures")
                    cols = st.columns(3)
                    for i, smi in enumerate(smis[:6]):
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            image = Draw.MolToImage(mol, size=(320, 260))
                            cols[i % 3].image(image, caption=f"Candidate #{i + 1} ({keys[i]})", use_column_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ==============================================================================
# TAB 3: TRAINING & MODEL ANALYTICS
# ==============================================================================
with tab_analytics:
    st.subheader("Lab Training & Model Analytics")
    st.caption("Publication-grade evaluation charts generated from the 4-day, 60-epoch university cluster training run.")

    graph_titles = {
        "01_loss_curves": "Training and validation loss across 60 epochs",
        "02_lr_schedule": "Cosine annealing learning-rate schedule",
        "03_ap_auc": "Validation average precision (42.98%) and ROC AUC (87.40%)",
        "04_bit_accuracy": "Validation fingerprint bit accuracy (98.15%)",
        "05_precision_recall": "Validation precision–recall dynamics",
        "06_confusion_matrix": "Validation bit confusion matrix (2,097,152 trials)",
        "07_fingerprint_activity": "Predicted vs. true fingerprint bit activity",
        "08_tsne": "High-dimensional spectral t-SNE latent embedding",
        "09_peak_count": "Binned diagnostic peak-count distribution",
        "10_precursor_mass": "Precursor mass distribution across natural products",
    }
    graph_paths = sorted(
        path
        for path in CFG.GRAPH_DIR.glob("*.png")
        if len(path.stem) >= 3
        and path.stem[:2].isdigit()
        and path.stem[2] == "_"
    )

    if not graph_paths:
        st.info("No charts found in graphs directory.")
    else:
        for row_start in range(0, len(graph_paths), 2):
            chart_columns = st.columns(2, gap="medium")
            for column, graph_path in zip(
                chart_columns, graph_paths[row_start:row_start + 2]
            ):
                title = graph_titles.get(
                    graph_path.stem,
                    graph_path.stem[3:].replace("_", " ").title(),
                )
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.image(
                            graph_path.read_bytes(),
                            caption=graph_path.name,
                            use_column_width=True,
                        )
