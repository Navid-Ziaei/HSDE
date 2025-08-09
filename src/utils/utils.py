from typing import Dict


def extract_curves(train_history: Dict) -> Dict[str, list]:
    """Turn the returned history into plottable lists."""
    H = train_history["history"]
    it = [h["iter"] for h in H]
    log_marg = [h.get("metrics", {}).get("log_marginal", float("nan")) for h in H]
    mse_obs = [h.get("metrics", {}).get("mse_observed", float("nan")) for h in H]
    sigma_x = [h["params"]["sigma_x"] for h in H]
    k_tau  = [h["params"]["k_tau"] for h in H]
    lmb_tau= [h["params"]["lmb_tau"] for h in H]
    return {
        "iter": it,
        "log_marginal": log_marg,
        "mse_observed": mse_obs,
        "sigma_x": sigma_x,
        "k_tau": k_tau,
        "lmb_tau": lmb_tau,
    }