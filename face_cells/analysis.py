"""Face-cell statistics, following the single-unit literature.

* Responsive (Quiroga et al. 2005, Nature 435:1102): mean response to an identity
  exceeds the blank-trial baseline mean + k*SD (k = 5 in the paper).
* Face/identity cell: responsive to exactly one identity, across views of it.
* Invariance: the cell is *chosen* on probe views of the training photos and
  *evaluated* on views of a held-out photo, so selection and testing never share
  data (no double dipping).
* Selectivity index SI = (R_pref - mean R_other) / (R_pref + mean R_other).
* Sparseness (Rolls & Tovee 1995 / Vinje & Gallant 2000), 0 = flat, 1 = one identity.
* AUC: ROC area for "preferred identity vs everyone else", trial by trial.
* Permutation test: the best-cell margin for an identity, against the same max
  statistic under shuffled identity labels. This controls for picking the
  luckiest of the 30 output cells.
"""
from __future__ import annotations

import numpy as np

SD_FLOOR = 0.3  # spikes/window; stops a silent cell's SD=0 making 1 spike "infinitely" significant


def tuning(R: np.ndarray, labels: np.ndarray, n_ids: int) -> np.ndarray:
    return np.stack([R[labels == i].mean(0) for i in range(n_ids)])  # (ids, cells)


def threshold(B: np.ndarray, k: float) -> np.ndarray:
    return B.mean(0) + k * np.maximum(B.std(0), SD_FLOOR)


def margin(T: np.ndarray) -> np.ndarray:
    """(ids, cells): response to identity i minus the strongest response to any other."""
    out = np.empty_like(T)
    for i in range(T.shape[0]):
        out[i] = T[i] - np.delete(T, i, axis=0).max(0)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):  # average ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sparseness(r: np.ndarray) -> float:
    r = np.maximum(r, 0)
    n = len(r)
    if (r ** 2).mean() == 0:
        return 0.0
    a = r.mean() ** 2 / (r ** 2).mean()
    return float((1 - a) / (1 - 1 / n))


def perm_test(R: np.ndarray, labels: np.ndarray, n_ids: int, n_perm: int = 1000,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    obs = margin(tuning(R, labels, n_ids)).max(1)
    null = np.empty((n_perm, n_ids))
    for p in range(n_perm):
        null[p] = margin(tuning(R, rng.permutation(labels), n_ids)).max(1)
    pvals = (1 + (null >= obs[None]).sum(0)) / (1 + n_perm)
    return obs, pvals


def evaluate(R_sel, lab_sel, R_test, lab_test, B, n_ids, cells, k=5.0, n_perm=1000):
    """Choose a candidate face cell per identity on the selection set; score it on
    the held-out test set. Returns a dict per identity plus population stats."""
    T_sel, T_test = tuning(R_sel, lab_sel, n_ids), tuning(R_test, lab_test, n_ids)
    thr = threshold(B, k)
    resp_test = T_test > thr[None]
    M_sel = margin(T_sel)
    _, p_test = perm_test(R_test, lab_test, n_ids, n_perm)
    per_id = []
    for i in range(n_ids):
        c = int(M_sel[i].argmax())
        sel_ids = np.nonzero(resp_test[:, c])[0].tolist()
        other = T_test[np.arange(n_ids) != i, c]
        si_den = T_test[i, c] + other.mean()
        per_id.append(dict(
            identity=i, cell=int(cells[c]), cell_index=c,
            resp_pref=float(T_test[i, c]), resp_other_max=float(other.max()),
            threshold=float(thr[c]), responds_to=sel_ids,
            z=float((T_test[i, c] - B[:, c].mean()) / max(B[:, c].std(), SD_FLOOR)),
            is_face_cell=bool(sel_ids == [i]),
            SI=float((T_test[i, c] - other.mean()) / si_den) if si_den > 0 else 0.0,
            AUC=auc(R_test[lab_test == i, c], R_test[lab_test != i, c]),
            sparseness=sparseness(T_test[:, c]),
            perm_p_population=float(p_test[i]),
        ))
    # population summary: how many cells are single-identity responsive on test views
    n_resp = resp_test.sum(0)
    return dict(
        per_identity=per_id,
        n_cells_single_identity=int((n_resp == 1).sum()),
        n_cells_multi_identity=int((n_resp > 1).sum()),
        n_cells_responsive=int((n_resp > 0).sum()),
        mean_sparseness=float(np.mean([sparseness(T_test[:, c]) for c in range(T_test.shape[1])])),
        tuning_test=T_test.tolist(), threshold=thr.tolist(),
    )
