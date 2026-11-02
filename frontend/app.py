"""Streamlit UI for CASMI 2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG
from src.model import SpectrumFPNet
from src.retrieval import retrieve_topk
from src.spectrum_input import parse_spectrum_csv
from src.utils import bin_spectrum
import torch
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="CASMI 2026 — Molecule ID",
                   page_icon="🧪", layout="wide")

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(37, 99, 235, 0.18), transparent 38%),
        radial-gradient(ellipse at 0% 28%, rgba(13, 148, 136, 0.12), transparent 34%),
        linear-gradient(145deg, #080d19 0%, #0d1525 52%, #111c32 100%);
}
.block-container { max-width: 1560px; padding: 2.5rem 2rem 4rem; }
h1 { font-weight: 800; letter-spacing: -0.045em; }
h2, h3 { letter-spacing: -0.025em; }
[data-testid="stSidebar"] {
    background: rgba(9, 15, 28, 0.94);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.88));
    padding: 1rem 1.15rem;
    border-radius: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.62);
    border-color: rgba(148, 163, 184, 0.16);
    border-radius: 1rem;
}
[data-testid="stImage"] img { border-radius: 0.65rem; }
[data-testid="stButton"] button[kind="primary"] {
    min-height: 3rem;
    border: 0;
    border-radius: 0.8rem;
    background: linear-gradient(100deg, #0f9f91, #2563eb);
    font-weight: 700;
}
@media (max-width: 768px) {
    .block-container { padding: 1.25rem 1rem 2.5rem; }
    h1 { font-size: 1.8rem !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="color:#5eead4;font-size:.78rem;font-weight:700;'
    'letter-spacing:.18em;text-transform:uppercase;margin-bottom:.35rem">'
    'CASMI 2026 · MS/MS intelligence</p>',
    unsafe_allow_html=True,
)
st.title("Molecule Identifier")
st.caption("Turn tandem mass spectra into ranked molecular candidates.")

from src.config import pick_device
DEVICE = pick_device()

@st.cache_resource
def load_model():
    ckpt = torch.load(CFG.MODEL_DIR / "best_model.pt", map_location=DEVICE)
    m = SpectrumFPNet(ckpt["in_dim"], CFG.FP_BITS).to(DEVICE)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m

try:
    MODEL = load_model()
    st.success(f"Model loaded on {DEVICE}")
except Exception as e:
    st.error(f"Model load failed: {e}")
    MODEL = None

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Manual peaks", "Upload CSV"])
    precursor = st.number_input("Precursor m/z", 50.0, 2000.0, 348.0, 0.01)
    adduct = st.selectbox("Adduct",
        ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
         "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"])
    if mode == "Manual peaks":
        txt = st.text_area("Peaks (m/z,intensity per line)", "150.05,1.0\n120.08,0.4\n90.03,0.2")
    else:
        example_path = CFG.DATA_DIR / "examples" / "spectrum_example.csv"
        st.download_button(
            "Download example spectrum CSV",
            data=example_path.read_bytes(),
            file_name="spectrum_example.csv",
            mime="text/csv",
        )
        up = st.file_uploader(
            "Upload spectrum CSV",
            type=["csv"],
            help=(
                "Upload measured spectrum peaks, not sample_submission.csv. "
                "The file needs m/z and intensity columns."
            ),
        )
        st.caption("Required columns: m/z and intensity (one peak per row).")

def parse_peaks():
    if mode == "Manual peaks":
        mzs, intensities = [], []
        for line_number, line in enumerate(txt.splitlines(), start=1):
            if not line.strip():
                continue
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_number} must contain exactly m/z,intensity."
                )
            try:
                mzs.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} contains a value that is not numeric."
                ) from exc
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

col1, col2 = st.columns([1,1])

if st.button("🚀 Predict", type="primary"):
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
        st.error("Model not loaded.")
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

            with col1:
                st.subheader("📈 Spectrum")
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

            with col2:
                st.subheader("🏆 Top-25 Candidates")
                result_frame = pd.DataFrame({
                    "Rank": range(1, len(smis) + 1),
                    "SMILES": smis,
                    "InChIKey14": keys,
                    "Tanimoto": [round(score, 4) for score in sims],
                })
                st.dataframe(
                    result_frame, use_container_width=True, height=440,
                    hide_index=True,
                )

            st.subheader("🧬 Candidate structures")
            cols = st.columns(3)
            for i, smi in enumerate(smis[:6]):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    cols[i % 3].warning(f"Could not render candidate #{i + 1}.")
                    continue
                try:
                    image = Draw.MolToImage(mol, size=(320, 260))
                    cols[i % 3].image(
                        image, caption=f"Candidate #{i + 1}",
                        use_column_width=True,
                    )
                except (OSError, ValueError) as exc:
                    cols[i % 3].warning(
                        f"Could not render candidate #{i + 1}: {exc}"
                    )
        except (FileNotFoundError, OSError, KeyError, RuntimeError, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
gallery_header, refresh_column = st.columns([5, 1])
with gallery_header:
    st.subheader("Model analytics")
    st.caption(
        "Training and data-distribution charts saved in the project graphs folder. "
        "Training charts update after each completed epoch."
    )
with refresh_column:
    st.button("Refresh gallery", help="Reload the latest chart files.")

graph_titles = {
    "01_loss_curves": "Training and validation loss",
    "02_lr_schedule": "Learning-rate schedule",
    "03_ap_auc": "Validation average precision and ROC AUC",
    "04_bit_accuracy": "Validation fingerprint bit accuracy",
    "05_precision_recall": "Validation precision–recall",
    "06_confusion_matrix": "Validation bit confusion matrix",
    "07_fingerprint_activity": "Predicted vs. true fingerprint activity",
    "08_tsne": "Training spectra t-SNE",
    "09_peak_count": "Binned peak-count distribution",
    "10_precursor_mass": "Precursor mass distribution",
}
graph_paths = sorted(
    path
    for path in CFG.GRAPH_DIR.glob("*.png")
    if len(path.stem) >= 3
    and path.stem[:2].isdigit()
    and path.stem[2] == "_"
)

if not graph_paths:
    st.info(
        "No charts are available yet. Training charts appear after the first "
        "completed epoch; data-distribution charts are generated by "
        "`python -m src.visualize`."
    )
else:
    st.caption(f"{len(graph_paths)} chart{'s' if len(graph_paths) != 1 else ''}")
    for row_start in range(0, len(graph_paths), 2):
        chart_columns = st.columns(2, gap="large")
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