from typing import Dict, Any
import numpy as np


def explain_model(model, X, feature_names=None) -> Dict[str, Any]:
    """Return a lightweight explanation dict for a single sample X.

    If SHAP is available and the model supports it, it will be used. Otherwise
    feature importances or coefficients are used as a fallback.
    """
    try:
        import shap
    except Exception:
        shap = None

    xp = X
    if hasattr(X, "values"):
        xp = X.values

    explanation = {"top_features": [], "raw": None}

    if shap is not None:
        try:
            explainer = shap.Explainer(model)
            sv = explainer(xp)
            # shap returns list-like; take first sample
            vals = sv.values[0]
            names = feature_names if feature_names is not None else [f"f{i}" for i in range(len(vals))]
            feat_imp = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)
            explanation["top_features"] = feat_imp[:10]
            explanation["raw"] = vals.tolist()
            return explanation
        except Exception:
            # fall through to fallback
            pass

    # Fallback: use model feature_importances_ or coef_
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            names = feature_names if feature_names is not None else [f"f{i}" for i in range(len(importances))]
            feat_imp = sorted(zip(names, importances), key=lambda x: abs(x[1]), reverse=True)
            explanation["top_features"] = feat_imp[:10]
            explanation["raw"] = importances.tolist()
            return explanation
        elif hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            names = feature_names if feature_names is not None else [f"f{i}" for i in range(len(coefs))]
            feat_imp = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)
            explanation["top_features"] = feat_imp[:10]
            explanation["raw"] = coefs.tolist()
            return explanation
    except Exception:
        pass

    return explanation
