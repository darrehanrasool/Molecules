import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import CFG
from src.utils import get_logger, bin_spectrum, clean_peaks, smiles_to_morgan

log = get_logger("preprocess")


CHUNK_SIZE = 2000


def get_id_column(df):

    if "molecule_id" in df.columns:
        return "molecule_id"

    if "inchikey14" in df.columns:
        return "inchikey14"

    if "inchikey" in df.columns:
        return "inchikey"

    return None


def process_chunk(
    df,
    has_labels
):

    spectra = []
    precursor_mz = []
    adducts = []
    molecule_ids = []
    group_ids = []

    fingerprints = []
    smiles = []
    inchikey14 = []

    id_column = get_id_column(df)

    for _, row in df.iterrows():

        mzs, intensities = clean_peaks(
            row["ms2_mzs"],
            row["ms2_normalized_intensities"],
            row["precursor_mz"]
        )

        spectrum = bin_spectrum(
            mzs,
            intensities
        )

        spectra.append(
            spectrum.astype(np.float32)
        )

        precursor_mz.append(
            row["precursor_mz"]
        )

        adducts.append(
            row["adduct"]
        )

        if id_column is not None:

            molecule_ids.append(
                row[id_column]
            )

        else:

            molecule_ids.append(
                len(molecule_ids)
            )

        if has_labels:

            smi = row.get(
                "normalized_smiles"
            )

            fp = smiles_to_morgan(
                smi
            )

            if fp is None:

                fp = np.zeros(
                    CFG.FP_BITS,
                    dtype=np.uint8
                )

            fingerprints.append(
                fp
            )

            smiles.append(
                smi
            )

            inchikey14.append(
                row.get("inchikey14")
            )
            group_value = row.get("inchikey14")
            if pd.isna(group_value):
                group_value = molecule_ids[-1]
            group_ids.append(group_value)
        else:
            group_ids.append(molecule_ids[-1])

    result = {
        "spectra": np.stack(spectra),
        "precursor_mz": np.asarray(
            precursor_mz,
            dtype=np.float32
        ),
        "adduct": np.asarray(
            adducts,
            dtype=object
        ),
        "molecule_id": np.asarray(
            molecule_ids,
            dtype=object
        ),
        "group_id": np.asarray(
            group_ids,
            dtype=object
        ),
    }

    if has_labels:

        result["fp"] = np.stack(
            fingerprints
        ).astype(np.uint8)

        result["smiles"] = np.asarray(
            smiles,
            dtype=object
        )

        result["inchikey14"] = np.asarray(
            inchikey14,
            dtype=object
        )

    return result


def process_dataset(
    parquet_path,
    output_dir,
    has_labels,
    name
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    total_rows = parquet_file.metadata.num_rows

    log.info(
        f"{name}: {total_rows} rows"
    )

    chunk_number = 0

    for batch in tqdm(
        parquet_file.iter_batches(
            batch_size=CHUNK_SIZE
        ),
        total=(total_rows + CHUNK_SIZE - 1)
        // CHUNK_SIZE,
        desc=name
    ):

        chunk_number += 1

        df = batch.to_pandas()

        result = process_chunk(
            df,
            has_labels
        )

        filename = (
            output_dir
            / f"{name}_{chunk_number:05d}.npz"
        )

        np.savez(
            filename,
            **result
        )

        del result
        del df
        del batch

        gc.collect()

    log.info(
        f"{name}: preprocessing finished"
    )

    log.info(
        f"{name}: saved {chunk_number} chunks"
    )


def build_fp_database(
    parquet_path,
    output_file
):

    log.info(
        "Building fingerprint database..."
    )

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    smiles_seen = set()
    smiles_list = []
    inchikey_list = []

    for batch in tqdm(
        parquet_file.iter_batches(
            batch_size=CHUNK_SIZE,
            columns=[
                "normalized_smiles",
                "inchikey14"
            ]
        ),
        desc="fp-db"
    ):

        df = batch.to_pandas()

        for smi, ik in zip(
            df["normalized_smiles"],
            df["inchikey14"]
        ):

            if pd.isna(smi):
                continue

            if pd.isna(ik):
                continue

            key = str(ik)

            if key in smiles_seen:
                continue

            smiles_seen.add(key)

            smiles_list.append(
                smi
            )

            inchikey_list.append(
                ik
            )

        del df
        del batch

        gc.collect()

    fingerprints = []

    for smi in tqdm(
        smiles_list,
        desc="creating fingerprints"
    ):

        fp = smiles_to_morgan(
            smi
        )

        if fp is None:

            fp = np.zeros(
                CFG.FP_BITS,
                dtype=np.uint8
            )

        fingerprints.append(
            fp
        )

    if len(fingerprints) == 0:

        log.warning(
            "No fingerprints found."
        )

        return

    fingerprints = np.stack(
        fingerprints
    ).astype(np.uint8)

    np.savez_compressed(
        output_file,
        fps=fingerprints,
        smiles=np.asarray(
            smiles_list,
            dtype=object
        ),
        inchikey14=np.asarray(
            inchikey_list,
            dtype=object
        )
    )

    log.info(
        f"Fingerprint DB size: {len(smiles_list)}"
    )

    del fingerprints
    gc.collect()


def main():

    log.info(
        "Reading train + test parquet metadata..."
    )

    log.info(
        f"Train file: {CFG.TRAIN_PARQUET}"
    )

    log.info(
        f"Test file: {CFG.TEST_PARQUET}"
    )

    log.info(
        f"Chunk size: {CHUNK_SIZE}"
    )

    log.info(
        "Processing train..."
    )

    process_dataset(
        CFG.TRAIN_PARQUET,
        CFG.PROC_DIR / "train_chunks",
        True,
        "train"
    )

    gc.collect()

    log.info(
        "Processing test..."
    )

    process_dataset(
        CFG.TEST_PARQUET,
        CFG.PROC_DIR / "test_chunks",
        False,
        "test"
    )

    gc.collect()

    build_fp_database(
        CFG.TRAIN_PARQUET,
        CFG.FP_DB
    )

    log.info(
        "======================================"
    )

    log.info(
        "Preprocessing completed successfully."
    )

    log.info(
        "======================================"
    )


if __name__ == "__main__":
    main()