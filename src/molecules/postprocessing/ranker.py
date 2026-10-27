"""
Candidate Ranker and InChIKey14 Deduplication Engine.
Collapses stereoisomers and tautomers to conserve top-25 ranking budget for MRR@25.
"""

from typing import Dict, List, Tuple
from molecules.evaluation.canonicalization import smiles_to_inchikey14


class CandidateRanker:
    """
    Reranks and filters candidate molecules to maximize MRR@25 performance.
    """

    def __init__(self, max_candidates: int = 25, deduplicate_by_inchikey14: bool = True):
        self.max_candidates = max_candidates
        self.deduplicate_by_inchikey14 = deduplicate_by_inchikey14

    def rank_and_deduplicate(
        self, scored_candidates: List[Tuple[str, float]]
    ) -> List[str]:
        """
        Sorts candidates by score descending, collapses duplicates by InChIKey14,
        and returns at most max_candidates SMILES strings.
        """
        if not scored_candidates:
            return []

        # Sort descending by confidence score
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

        if not self.deduplicate_by_inchikey14:
            return [smiles for smiles, _ in sorted_candidates[: self.max_candidates]]

        seen_skeletons: Dict[str, str] = {}
        unique_smiles_list: List[str] = []

        for smiles, _ in sorted_candidates:
            key14 = smiles_to_inchikey14(smiles)
            if key14 is not None:
                if key14 not in seen_skeletons:
                    seen_skeletons[key14] = smiles
                    unique_smiles_list.append(smiles)
            else:
                # If key cannot be computed, retain smiles if not already seen
                if smiles not in unique_smiles_list:
                    unique_smiles_list.append(smiles)

            if len(unique_smiles_list) >= self.max_candidates:
                break

        return unique_smiles_list
