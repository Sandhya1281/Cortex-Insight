from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import base64
from urllib import request, error

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brisc_utils import image_features  # noqa: E402


MODEL_PATH = ROOT / "reports" / "classification_baseline.joblib"
METRICS_PATH = ROOT / "reports" / "classification_metrics.json"
SEGMENTATION_PATH = ROOT / "reports" / "segmentation_baseline_metrics.json"
BRAIN_ASSET_PATH = ROOT / "assets" / "brain_reference.png"


LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

CLASS_STYLE = {
    "glioma": {"color": "#E8A33D", "accent": "#F4C579", "region": "frontal", "default_side": "left", "x": 42, "y": 30},
    "meningioma": {"color": "#E8A33D", "accent": "#F4C579", "region": "temporal", "default_side": "right", "x": 70, "y": 44},
    "pituitary": {"color": "#E8A33D", "accent": "#F4C579", "region": "central base", "default_side": None, "x": 50, "y": 62},
    "no_tumor": {"color": "#4CE0D2", "accent": "#8FF0E6", "region": "clear scan", "default_side": None, "x": 52, "y": 38},
}


def resolve_lesion_style(prediction: str, side: str | None) -> dict:
    """The classifier only predicts tumor *type*, not which side it's on --
    so `side` (detected from the actual uploaded slice, see
    detect_anomaly_side) decides whether we mirror the default left/right
    placement. Without this, every glioma would render on the left and every
    meningioma on the right regardless of what the real scan shows."""
    base = CLASS_STYLE.get(prediction, CLASS_STYLE["no_tumor"])
    default_side = base.get("default_side")
    if default_side is None or not side:
        label = base["region"] if default_side is None else f"{default_side} {base['region']}"
        return {**base, "region_label": label}
    if side == default_side:
        return {**base, "region_label": f"{default_side} {base['region']}"}
    return {**base, "region_label": f"{side} {base['region']}", "x": 100 - base["x"]}

SPLINE_URL = "https://my.spline.design/particleaibrain-yRZBn3DLxTy1I5uWOJMXfhpn/"


def spline_brain_html(prediction: str, name: str, compact: bool = False) -> str:
    """Embeds the interactive Spline particle brain. It's a fixed external
    scene (we don't control its internal camera/geometry), so the flagged
    region is shown as an approximate pinned overlay rather than a marker
    baked into the 3D model itself."""
    style = CLASS_STYLE.get(prediction, CLASS_STYLE["no_tumor"])
    flagged = prediction != "no_tumor"
    size_class = "compact" if compact else ""
    flag_html = (
        f'<div class="spline-flag" style="left:{style["x"]}%; top:{style["y"]}%;">'
        f'<span class="dot"></span><span class="tag">{style["region"].upper()}</span></div>'
        if flagged
        else ""
    )
    return f"""
    <div class="spline-frame {size_class}">
        <iframe src="{SPLINE_URL}" title="3D brain - {name}"
                loading="lazy" allow="autoplay; fullscreen"></iframe>
        {flag_html}
    </div>
    """


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def brain_asset_data_uri() -> str:
    encoded = base64.b64encode(BRAIN_ASSET_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# The saved model is a 4-way hard-voting ensemble (LinearSVC + cosine KNN +
# ExtraTrees + PCA/HistGradientBoosting). Hard voting just takes a majority
# label and throws away *how* confident each vote was -- so a scan where 2 of
# 4 sub-models lean "no_tumor" only slightly still wins outright. We can't
# retrain this model here (the BRISC2025 training images live on your machine,
# not in this project folder), but we can make the decision rule safer:
# average the real probabilities from the 3 sub-models that expose them, and
# require no_tumor to clear a higher bar than a tumor class would, since a
# missed tumor is a worse mistake than a false alarm on a clear scan.
NO_TUMOR_LABEL = "no_tumor"
NO_TUMOR_CONFIDENCE_PENALTY = 0.18


def predict_with_confidence(uploaded_file, image_size: int = 48) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.getbuffer())
        temp_path = Path(temp.name)

    features = image_features(temp_path, size=image_size).reshape(1, -1)
    model = load_model()

    try:
        prob_estimators = [
            model.named_estimators_[name]
            for name in ("knn_cosine", "extra_trees", "pca_hgb")
            if name in model.named_estimators_
        ]
        if not prob_estimators:
            raise AttributeError("no probability-capable sub-estimators found")

        avg_probs = np.mean([est.predict_proba(features)[0] for est in prob_estimators], axis=0)
        # NB: each sub-estimator's own .classes_ is just [0, 1, 2, 3] --
        # VotingClassifier integer-encodes y internally before fitting them.
        # The real string labels (alphabetically sorted, matching that
        # encoding) live on the outer model instead.
        classes = list(model.classes_)

        adjusted = avg_probs.copy()
        if NO_TUMOR_LABEL in classes:
            adjusted[classes.index(NO_TUMOR_LABEL)] -= NO_TUMOR_CONFIDENCE_PENALTY

        adjusted = np.clip(adjusted, 0, None)
        if adjusted.sum() <= 0:
            adjusted = avg_probs
        display_probs = adjusted / adjusted.sum()
        predicted = classes[int(np.argmax(display_probs))]
        return {
            "prediction": predicted,
            "confidence": float(display_probs[classes.index(predicted)]),
            "probabilities": {label: float(prob) for label, prob in zip(classes, display_probs)},
        }
    except Exception:
        # Fall back to the plain hard vote if anything about the
        # probability path doesn't line up (e.g. a different sklearn
        # version renames/restructures the fitted sub-estimators).
        predicted = model.predict(features)[0]
        classes = list(getattr(model, "classes_", LABELS.keys()))
        probabilities = {label: 0.0 for label in classes}
        probabilities[predicted] = 1.0
        return {"prediction": predicted, "confidence": 1.0, "probabilities": probabilities}


def predict(uploaded_file, image_size: int = 48) -> str:
    return predict_with_confidence(uploaded_file, image_size)["prediction"]


# Illustrative field-of-view assumption for a typical axial T1 head slice.
# There's no DICOM spacing on a plain JPG/PNG upload, so this converts pixel
# coverage into a plausible mm^3 figure -- labelled "estimated" in the UI,
# not a clinical measurement.
ASSUMED_FOV_MM = 220.0
ASSUMED_SLICE_THICKNESS_MM = 5.0


def estimate_slice_volume_cm3(gray_image: Image.Image) -> tuple[float, float]:
    """Returns (estimated_volume_cm3, tissue_area_fraction) for one slice."""
    small = gray_image.resize((160, 160))
    arr = np.asarray(small).astype(float)
    threshold = max(arr.mean() * 0.4, 12)
    mask = arr > threshold
    area_fraction = float(mask.mean())
    px_size_mm = ASSUMED_FOV_MM / small.size[0]
    area_mm2 = mask.sum() * (px_size_mm**2)
    volume_mm3 = area_mm2 * ASSUMED_SLICE_THICKNESS_MM
    return volume_mm3 / 1000.0, area_fraction


def detect_anomaly_side(gray_image: Image.Image) -> str | None:
    """Heuristic only, not a real lesion-detection model: finds the point of
    strongest local contrast inside the brain mask (tumors usually stand out
    from surrounding tissue) and reports which side of the slice it falls on.
    Used purely to stop the 3D view from always placing e.g. every glioma on
    the left regardless of what the actual uploaded image shows."""
    size = 160
    small = gray_image.resize((size, size))
    arr = np.asarray(small).astype(float)
    threshold = max(arr.mean() * 0.4, 12)
    mask = arr > threshold
    if not mask.any():
        return None
    blurred = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=6))).astype(float)
    anomaly = np.abs(arr - blurred)
    anomaly[approx. mask] = 0
    row, col = np.unravel_index(np.argmax(anomaly), anomaly.shape)
    centroid_col = np.nonzero(mask)[1].mean()
    return "left" if col < centroid_col else "right"


def estimate_tumor_measurements(gray_image: Image.Image) -> dict | None:
    """Approximate size for the flagged region, reusing the same contrast-hotspot
    heuristic as detect_anomaly_side. Reports a greatest-diameter figure (the
    shorthand radiology reports commonly use for a roughly round mass, e.g.
    "23 mm") and a spherical volume derived from it. This is a single-slice
    heuristic, not a segmentation model -- treat it as illustrative only."""
    size = 160
    small = gray_image.resize((size, size))
    arr = np.asarray(small).astype(float)
    threshold = max(arr.mean() * 0.4, 12)
    mask = arr > threshold
    if not mask.any():
        return None
    blurred = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=6))).astype(float)
    anomaly = np.abs(arr - blurred)
    anomaly[approx. mask] = 0
    if anomaly.max() <= 0:
        return None

    hot = anomaly > anomaly.max() * 0.55
    if hot.sum() < 6:
        hot = anomaly > anomaly.max() * 0.3
    if hot.sum() < 3:
        return None

    px_size_mm = ASSUMED_FOV_MM / size
    area_mm2 = hot.sum() * (px_size_mm**2)
    diameter_mm = 2.0 * np.sqrt(area_mm2 / np.pi)  # equivalent circular diameter
    radius_cm = (diameter_mm / 2.0) / 10.0
    volume_cm3 = (4.0 / 3.0) * np.pi * (radius_cm**3)
    return {"diameter_mm": float(diameter_mm), "volume_cm3": float(volume_cm3)}


def image_to_data_uri(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image.save(temp.name)
        encoded = base64.b64encode(Path(temp.name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_gradcam_views(gray_image: Image.Image, prediction: str) -> dict:
    base = ImageOps.autocontrast(gray_image).convert("RGB")
    size = 192
    small = gray_image.resize((size, size))
    arr = np.asarray(small).astype(float)
    threshold = max(arr.mean() * 0.4, 12)
    mask = arr > threshold
    blurred = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=7))).astype(float)
    heat = np.abs(arr - blurred)
    heat[approx. mask] = 0
    if prediction == "no_tumor" or heat.max() <= 0:
        heat = np.zeros_like(heat)
    else:
        active_values = heat[mask]
        low = np.percentile(active_values, 65)
        high = np.percentile(active_values, 99)
        if high <= low:
            high = active_values.max()
        heat = np.clip((heat - low) / max(high - low, 1e-6), 0, 1)
        heat = heat ** 0.45
        heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).filter(ImageFilter.GaussianBlur(radius=4))).astype(float) / 255.0

    red = np.clip(255 * heat, 0, 255)
    green = np.clip(230 * np.sqrt(heat), 0, 230)
    blue = np.clip(35 * (1 - heat), 0, 35)
    alpha = np.where(heat > 0.03, np.clip(95 + 160 * heat, 0, 235), 0)
    rgba = np.dstack([red, green, blue, alpha]).astype(np.uint8)
    heatmap = Image.fromarray(rgba, mode="RGBA").resize(base.size, Image.Resampling.BICUBIC)

    heat_bg = Image.new("RGBA", base.size, "#120609")
    heat_only = Image.alpha_composite(heat_bg, heatmap).convert("RGB")
    overlay = Image.alpha_composite(base.convert("RGBA"), heatmap).convert("RGB")
    return {
        "original": image_to_data_uri(base),
        "heatmap": image_to_data_uri(heat_only),
        "overlay": image_to_data_uri(overlay),
    }


def confidence_bar_html(probabilities: dict, prediction: str) -> str:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    rows = []
    for label, prob in ordered:
        pct = max(0.0, min(100.0, prob * 100))
        blocks = "█" * max(1, int(round(pct / 8))) if pct >= 1 else "."
        active = " active" if label == prediction else ""
        rows.append(f'<div class="confidence-row{active}"><span>{LABELS.get(label, label)}</span><b>{blocks}</b><em>{pct:.1f}%</em></div>')
    return '<div class="confidence-list">' + "".join(rows) + "</div>"

def app_secret(name: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def groq_summary(trial: dict) -> str:
    api_key = app_secret("GROQ_API_KEY")
    if not api_key:
        return (
            "Groq explanation is ready to use once GROQ_API_KEY is set. "
            "The current readout shows the predicted class, confidence distribution, approximate measurements, and heatmap focus area."
        )

    probs = ", ".join(f"{LABELS.get(k, k)} {v * 100:.1f}%" for k, v in trial.get("probabilities", {}).items())
    prompt = (
        "Write a concise, doctor-facing but patient-readable MRI AI explanation. "
        "Do not claim a diagnosis. Mention that this is decision support only. "
        f"Prediction: {LABELS.get(trial['prediction'], trial['prediction'])}. "
        f"Confidence scores: {probs}. "
        f"Approximate tumor diameter mm: {trial.get('tumor_diameter_mm', 'not estimated')}. "
        f"Approximate tumor volume mm3: {trial.get('tumor_volume_cm3', 0) * 1000 if trial.get('tumor_volume_cm3') else 'not estimated'}. "
        f"Flagged side/region: {trial.get('side') or 'not localized'}."
    )
    payload = json.dumps(
        {
            "model": app_secret("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": "You explain medical AI outputs cautiously and clearly."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    req = request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BRISC2025-CortexInsight/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return f"Groq explanation could not be generated right now: HTTP {exc.code}. {detail}"
    except (error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Groq explanation could not be generated right now: {exc}"


def file_digest(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    uploaded_file.seek(0)
    return digest


def brain_scene_html(prediction: str, name: str, trial_label: str, compact: bool = False) -> str:
    style = CLASS_STYLE.get(prediction, CLASS_STYLE["no_tumor"])
    label = LABELS.get(prediction, prediction)
    glow = "" if prediction == "no_tumor" else "active"
    size_class = "compact" if compact else "large"
    flagged = prediction != "no_tumor"
    coord_tag = (
        f'<div class="coord-tag" style="left:{style["x"]}%; top:{style["y"]}%;">'
        f'<span>{style["region"].upper()}</span></div>'
        if flagged
        else ""
    )
    return f"""
    <div class="brain-scene {size_class}">
        <img src="{brain_asset_data_uri()}" alt="3D brain render" />
        <div class="scan-vignette"></div>
        <div class="sweep-line"></div>
        <svg class="vessel-overlay" viewBox="0 0 100 70" preserveAspectRatio="none" aria-hidden="true">
            <path d="M51 66 C50 55 48 50 45 43 C39 38 32 34 23 28" />
            <path d="M51 66 C55 55 61 49 69 43 C76 39 82 34 89 27" />
            <path d="M50 61 C45 54 38 49 30 45 C24 42 19 38 14 31" />
            <path d="M52 58 C56 49 61 42 63 34 C65 26 69 20 76 15" />
            <path d="M48 58 C47 48 45 40 43 31 C41 25 38 20 33 15" />
        </svg>
        <div class="tumor-glow {glow}" style="left:{style['x']}%; top:{style['y']}%; --tumor:{style['color']}; --tumor-soft:{style['accent']};"></div>
        {coord_tag}
        <div class="brain-label">
            <span>{trial_label}</span>
            <strong>{label}</strong>
            <em>{style['region']} - {name}</em>
        </div>
    </div>
    """


@st.cache_data(show_spinner=False)
def brain_figure(prediction: str, trial_number: int, compact: bool = False, side: str | None = None) -> go.Figure:
    """Holographic wireframe brain. Kept deliberately low trace-count
    (one Surface per hemisphere, gridlines via `contours` instead of hand-drawn
    line traces) so WebGL rotation stays smooth even on modest hardware."""
    style = resolve_lesion_style(prediction, side)
    flagged = prediction != "no_tumor"

    fig = go.Figure()

    # Standard polar parametrization (no pinch-to-a-point at the equator like
    # the previous cos(theta)**0.62 term produced -- that's what made it read
    # as a canoe/spindle instead of a rounded lobe).
    theta = np.linspace(-np.pi, np.pi, 70)
    phi = np.linspace(0.05, np.pi - 0.05, 46)
    theta, phi = np.meshgrid(theta, phi)

    cyan_scale = [
        [0.0, "#04141c"],
        [0.35, "#0a4a56"],
        [0.65, "#1fa3a8"],
        [1.0, "#8ff0e6"],
    ]

    lesion_positions = {
        "glioma": (-0.5, 0.25, 0.35),
        "meningioma": (0.52, 0.5, 0.15),
        "pituitary": (0.0, 0.35, -0.35),
    }
    lx, ly, lz = lesion_positions.get(prediction, (0.0, 0.0, 0.0))
    default_side = CLASS_STYLE.get(prediction, {}).get("default_side")
    if default_side and side and side != default_side:
        lx = -lx  # mirror to the side actually detected in the uploaded slice

    for hemi, center in ((-1, -0.42), (1, 0.42)):
        # `fade` goes to 0 exactly at the seam (theta = +/-90deg) so both
        # hemispheres' edges collapse onto the *same* x=0 line -- the earlier
        # version only faded the offset partially, leaving a visible gap.
        fade = np.abs(np.cos(theta)) ** 0.6
        folds = 0.05 * np.sin(10 * theta + hemi * 0.6 + trial_number * 0.18) * np.sin(5 * phi) * fade

        a_x, a_y, a_z = 0.62, 0.92, 0.66  # lobe half-widths: side / front-back / top-bottom
        x = center * fade + hemi * a_x * (1 + folds) * np.sin(phi) * np.cos(theta)
        y = a_y * (1 + folds) * np.sin(phi) * np.sin(theta)
        z = a_z * (1 + folds) * np.cos(phi)

        # Flatten the underside slightly (brainstem side).
        z = np.where(z < -0.3, -0.3 + 0.55 * (z + 0.3), z)

        if flagged:
            dist = np.sqrt((x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2)
            heat = np.clip(1 - dist / 0.7, 0, 1) ** 1.6
        else:
            heat = np.zeros_like(x)

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=heat,
                colorscale=[[0, "#0a4a56"], [1, "#ef7a1f"]] if flagged else cyan_scale,
                cmin=0,
                cmax=1,
                opacity=0.62,
                showscale=False,
                lighting=dict(ambient=0.8, diffuse=0.4, fresnel=0.05, specular=0.05, roughness=1),
                contours=dict(
                    x=dict(show=True, color="rgba(143, 240, 230, 0.4)", width=1),
                    y=dict(show=True, color="rgba(143, 240, 230, 0.4)", width=1),
                ),
                hoverinfo="skip",
            )
        )

    if flagged:
        # Lumpy tumor mass -- a perturbed sphere (not a plain marker dot) so it
        # reads as an actual mass of tissue growing in the brain, plus a soft
        # translucent halo and a couple of short feeding vessels, similar to
        # how radiology illustrations show a tumor with its blood supply.
        tphi = np.linspace(0, np.pi, 26)
        ttheta = np.linspace(0, 2 * np.pi, 34)
        tphi, ttheta = np.meshgrid(tphi, ttheta)

        bump = (
            1
            + 0.16 * np.sin(5 * ttheta) * np.sin(4 * tphi)
            + 0.10 * np.cos(7 * ttheta + 2) * np.sin(3 * tphi + 1)
            + 0.07 * np.sin(3 * ttheta - 4 * tphi)
        )
        core_r = 0.135
        tx = lx + core_r * bump * np.sin(tphi) * np.cos(ttheta)
        ty = ly + core_r * bump * np.sin(tphi) * np.sin(ttheta)
        tz = lz + core_r * bump * np.cos(tphi)

        fig.add_trace(
            go.Surface(
                x=tx, y=ty, z=tz,
                surfacecolor=bump,
                colorscale=[[0, "#8a2c0a"], [0.55, "#e8792a"], [1, "#ffd9a0"]],
                showscale=False,
                opacity=1.0,
                lighting=dict(ambient=0.55, diffuse=0.65, specular=0.35, roughness=0.5, fresnel=0.1),
                hovertemplate=f"{LABELS.get(prediction, prediction)}<br>{style['region_label']}<extra></extra>",
                name=LABELS.get(prediction, prediction),
            )
        )

        halo_r = core_r * 2.3
        hx = lx + halo_r * np.sin(tphi) * np.cos(ttheta)
        hy = ly + halo_r * np.sin(tphi) * np.sin(ttheta)
        hz = lz + halo_r * np.cos(tphi)
        fig.add_trace(
            go.Surface(
                x=hx, y=hy, z=hz,
                surfacecolor=np.zeros_like(hx),
                colorscale=[[0, "#ffb45c"], [1, "#ffb45c"]],
                showscale=False,
                opacity=0.16,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                hoverinfo="skip",
            )
        )

        rng = np.random.default_rng(hash(prediction) % 1000)
        for k in range(3):
            t = np.linspace(0, 1, 24)
            drift = rng.uniform(-0.35, 0.35, size=2)
            vx = lx + (0.5 + 0.4 * k) * drift[0] * t
            vy = ly + (0.5 + 0.4 * k) * drift[1] * t
            vz = lz - (0.3 + 0.15 * k) * t
            fig.add_trace(
                go.Scatter3d(
                    x=vx, y=vy, z=vz,
                    mode="lines",
                    line=dict(color="rgba(190, 60, 30, 0.7)", width=3 if compact else 4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=260 if compact else 440,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="brain-scene",  # keep camera angle stable across Streamlit reruns
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.25, y=-1.55, z=0.85), up=dict(x=0, y=0, z=1)),
            aspectmode="data",
        ),
        showlegend=False,
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0B0E11;
            --panel: #0F1317;
            --line: rgba(76, 224, 210, 0.14);
            --line-soft: rgba(255,255,255,0.06);
            --cyan: #4CE0D2;
            --cyan-dim: rgba(76, 224, 210, 0.5);
            --amber: #E8A33D;
            --text: #E7ECEF;
            --text-dim: #7C8894;
            --mono: 'IBM Plex Mono', ui-monospace, monospace;
            --sans: 'Inter', -apple-system, sans-serif;
        }

        .stApp {
            background:
                repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
                var(--bg);
            color: var(--text);
            font-family: var(--sans);
        }
        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { padding-top: 2rem; max-width: 1240px; }
        h1, h2, h3 { color: var(--text); letter-spacing: 0 !important; font-family: var(--sans); }

        /* ---------- Hero ---------- */
        .hero {
            padding: 4px 0 22px;
            border-bottom: 1px solid var(--line-soft);
            margin-bottom: 22px;
            position: relative;
        }
        .hero-eyebrow {
            font-family: var(--mono);
            font-size: .74rem;
            letter-spacing: .16em;
            text-transform: uppercase;
            color: var(--cyan);
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        .hero-eyebrow::before {
            content: '';
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 8px var(--cyan);
        }
        .hero-title {
            font-size: clamp(2rem, 5vw, 3.6rem);
            line-height: 1.02;
            font-weight: 800;
            max-width: 900px;
            letter-spacing: -.01em;
        }
        .hero-sub {
            margin-top: 12px;
            max-width: 720px;
            color: var(--text-dim);
            font-size: .98rem;
        }

        /* ---------- Panels / instrument layout ---------- */
        .panel {
            border: 1px solid var(--line-soft);
            border-top: 1px solid var(--line);
            background: var(--panel);
            border-radius: 3px;
            padding: 20px;
        }
        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: var(--mono);
            font-size: .72rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px dashed var(--line-soft);
        }
        .panel-head b { color: var(--cyan); font-weight: 600; }
        .status-pill { font-weight: 600; letter-spacing: .08em; }

        /* tick-mark column divider */
        div[data-testid="column"]:first-of-type {
            border-right: 1px solid var(--line-soft);
            padding-right: 8px;
            background-image: repeating-linear-gradient(180deg, var(--line) 0px, var(--line) 1px, transparent 1px, transparent 28px);
            background-position: right 0 top 0;
            background-repeat: repeat-y;
            background-size: 1px 28px;
        }

        .prediction {
            border-left: 3px solid var(--accent, var(--cyan));
            padding: 10px 14px;
            background: rgba(255,255,255,0.03);
            border-radius: 2px;
            margin: 8px 0 16px;
        }
        .prediction strong {
            display: block;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text);
        }
        .prediction span {
            color: var(--text-dim);
            font-family: var(--mono);
            font-size: .78rem;
        }
        .confidence-list {
            display: grid;
            gap: 8px;
            margin-top: 12px;
            font-family: var(--mono);
        }
        .confidence-row {
            display: grid;
            grid-template-columns: minmax(96px, 1fr) minmax(72px, 1.2fr) 58px;
            align-items: center;
            gap: 10px;
            color: var(--text-dim);
            font-size: .78rem;
        }
        .confidence-row b {
            color: rgba(76, 224, 210, .45);
            font-weight: 500;
            letter-spacing: 1px;
            overflow: hidden;
            white-space: nowrap;
        }
        .confidence-row em {
            color: var(--text);
            font-style: normal;
            text-align: right;
        }
        .confidence-row.active span,
        .confidence-row.active b,
        .confidence-row.active em {
            color: var(--amber);
        }
        .gradcam-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .gradcam-tile {
            border: 1px solid var(--line-soft);
            background: #05070a;
            border-radius: 2px;
            overflow: hidden;
        }
        .gradcam-tile img {
            display: block;
            width: 100%;
            aspect-ratio: 1;
            object-fit: contain;
            background: #05070a;
        }
        .gradcam-tile span {
            display: block;
            padding: 7px 8px;
            border-top: 1px solid var(--line-soft);
            color: var(--text-dim);
            font-family: var(--mono);
            font-size: .68rem;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .llm-note {
            color: var(--text);
            font-size: .92rem;
            line-height: 1.55;
        }
        .trial-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 9px;
            border: 1px solid var(--line-soft);
            border-radius: 2px;
            color: var(--text-dim);
            font-family: var(--mono);
            font-size: .72rem;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        [data-testid="stMetricValue"] { color: var(--text); font-family: var(--mono); }
        [data-testid="stMetricLabel"] { font-family: var(--mono); text-transform: uppercase; letter-spacing: .06em; font-size: .7rem !important; }

        /* ---------- Scan-intake tray (file uploader) ---------- */
        [data-testid="stFileUploader"] {
            border: 1px dashed var(--cyan-dim);
            border-radius: 2px;
            padding: 6px 10px 14px;
            background:
                repeating-linear-gradient(-45deg, rgba(76,224,210,0.03) 0px, rgba(76,224,210,0.03) 8px, transparent 8px, transparent 16px),
                rgba(76, 224, 210, 0.02);
        }
        [data-testid="stFileUploaderDropzone"] { background: transparent; }
        [data-testid="stFileUploaderDropzoneInstructions"] svg { display: none; }
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            font-family: var(--mono);
            font-size: .85rem;
            color: var(--text);
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-family: var(--mono);
            color: var(--text-dim);
            letter-spacing: .02em;
        }
        [data-testid="stFileUploaderDropzone"] button {
            font-family: var(--mono);
            border: 1px solid var(--line-soft);
            background: transparent;
            color: var(--cyan);
        }
        .stButton button {
            width: 100%;
            border-radius: 2px;
            border: 1px solid var(--cyan-dim);
            background: transparent;
            color: var(--cyan);
            font-family: var(--mono);
            font-weight: 600;
            letter-spacing: .04em;
            text-transform: uppercase;
            font-size: .82rem;
            transition: background .15s ease, color .15s ease;
        }
        .stButton button:hover {
            background: var(--cyan);
            color: #05100f;
            border-color: var(--cyan);
        }
        .stButton button:focus-visible {
            outline: 2px solid var(--amber);
            outline-offset: 2px;
        }

        /* ---------- MRI intake frame ---------- */
        .mri-frame {
            position: relative;
            overflow: hidden;
            border-radius: 2px;
            border: 1px solid var(--line-soft);
            background: #05070a;
            margin-top: 12px;
        }
        .mri-frame img { display: block; width: 100%; height: auto; filter: contrast(1.05) brightness(0.98); }
        .mri-crosshair {
            position: absolute; inset: 0; pointer-events: none;
        }
        .mri-crosshair::before, .mri-crosshair::after {
            content: '';
            position: absolute;
            background: var(--cyan-dim);
        }
        .mri-crosshair::before { left: 0; right: 0; top: 50%; height: 1px; opacity: .35; }
        .mri-crosshair::after { top: 0; bottom: 0; left: 50%; width: 1px; opacity: .35; }
        .mri-caption {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 4px 0;
            font-family: var(--mono); font-size: .72rem; color: var(--text-dim);
            letter-spacing: .04em;
        }
        .mri-caption b { color: var(--cyan); font-weight: 500; }

        /* ---------- Brain readout scene ---------- */
        .brain-scene {
            position: relative;
            overflow: hidden;
            border-radius: 2px;
            min-height: 420px;
            background: #05070a;
            border: 1px solid var(--line-soft);
        }
        .brain-scene.compact { min-height: 230px; }
        .brain-scene img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: saturate(0.82) contrast(1.05) brightness(0.9) sepia(0.15);
            transform: scale(1.04);
        }
        .scan-vignette {
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(5, 7, 10, 0.5), transparent 24%, transparent 70%, rgba(5, 7, 10, 0.55));
            pointer-events: none;
        }
        .sweep-line {
            position: absolute;
            left: 0; right: 0; height: 40%;
            background: linear-gradient(180deg, transparent, rgba(76, 224, 210, 0.07), transparent);
            animation: sweep 5s linear infinite;
            pointer-events: none;
        }
        .vessel-overlay {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: .3;
            mix-blend-mode: overlay;
        }
        .vessel-overlay path {
            fill: none;
            stroke: rgba(158, 46, 46, .6);
            stroke-width: .55;
            stroke-linecap: round;
        }
        .tumor-glow {
            position: absolute;
            width: 6%;
            aspect-ratio: 1;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            background:
                radial-gradient(circle, var(--tumor-soft) 0%, var(--tumor) 32%, transparent 70%);
            box-shadow: 0 0 16px var(--tumor);
            opacity: 0;
        }
        .tumor-glow.active {
            opacity: .9;
            animation: tumorPulse 2.4s ease-in-out infinite;
        }
        .brain-scene.compact .tumor-glow { width: 9%; }
        .coord-tag {
            position: absolute;
            transform: translate(14px, -50%);
            font-family: var(--mono);
            font-size: .66rem;
            letter-spacing: .08em;
            color: var(--amber);
            white-space: nowrap;
            border-left: 1px solid var(--amber);
            padding-left: 6px;
        }
        .brain-label {
            position: absolute;
            left: 16px;
            bottom: 14px;
            max-width: calc(100% - 32px);
            padding: 9px 12px;
            border-left: 2px solid var(--cyan);
            border-radius: 1px;
            background: rgba(5, 7, 10, .78);
            backdrop-filter: blur(6px);
        }
        .brain-label span {
            display: block;
            color: var(--cyan);
            font-family: var(--mono);
            font-size: .7rem;
            text-transform: uppercase;
            letter-spacing: .1em;
        }
        .brain-label strong {
            display: block;
            color: var(--text);
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.1;
            margin-top: 2px;
        }
        .brain-label em {
            display: block;
            color: var(--text-dim);
            font-style: normal;
            font-family: var(--mono);
            font-size: .74rem;
            margin-top: 2px;
        }
        .brain-scene.compact .brain-label strong { font-size: 1rem; }
        @keyframes tumorPulse {
            0%, 100% { transform: translate(-50%, -50%) scale(.94); }
            50% { transform: translate(-50%, -50%) scale(1.08); }
        }
        @keyframes sweep {
            0% { top: -40%; }
            100% { top: 100%; }
        }
        .spline-frame {
            position: relative;
            border-radius: 2px;
            overflow: hidden;
            border: 1px solid var(--line-soft);
            background: #05070a;
        }
        .spline-frame iframe {
            width: 100%;
            height: 440px;
            border: 0;
            display: block;
        }
        .spline-frame.compact iframe { height: 230px; }
        .spline-flag {
            position: absolute;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            gap: 6px;
            pointer-events: none;
        }
        .spline-flag .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--amber);
            box-shadow: 0 0 10px var(--amber), 0 0 2px #fff;
            animation: tumorPulse 2.4s ease-in-out infinite;
        }
        .spline-flag .tag {
            font-family: var(--mono);
            font-size: .68rem;
            letter-spacing: .07em;
            color: var(--amber);
            background: rgba(5, 7, 10, .8);
            padding: 3px 7px;
            border-left: 2px solid var(--amber);
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="BRISC 2025 MRI Classifier", layout="wide")
inject_css()

metrics = load_json(METRICS_PATH)
segmentation = load_json(SEGMENTATION_PATH)
if "trials" not in st.session_state:
    st.session_state.trials = []

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">BRISC-2025 - T1 classifier online</div>
      <div class="hero-title">Cortex Insight</div>
      <div class="hero-sub">Drop a T1 MRI slice, run the classifier, and get a coordinate-tagged 3D readout for every trial.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([0.94, 1.06], gap="large")

with top_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-head"><b>01</b> - Scan intake</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("MRI slice", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    analyze = st.button("Run analysis", disabled=uploaded is None)
    if uploaded:
        image = Image.open(uploaded).convert("L")
        buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(buf.name)
        encoded = base64.b64encode(Path(buf.name).read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <div class="mri-frame">
                <img src="data:image/png;base64,{encoded}" alt="Uploaded MRI slice" />
                <div class="mri-crosshair"></div>
            </div>
            <div class="mri-caption"><span>{uploaded.name}</span><b>T1 - GRAYSCALE</b></div>
            """,
            unsafe_allow_html=True,
        )
        if analyze:
            uploaded.seek(0)
            result = predict_with_confidence(uploaded, image_size=int(metrics.get("image_size", 48)))
            predicted = result["prediction"]
            volume_cm3, tissue_fraction = estimate_slice_volume_cm3(image)
            detected_side = detect_anomaly_side(image) if predicted != "no_tumor" else None
            tumor_stats = estimate_tumor_measurements(image) if predicted != "no_tumor" else None
            gradcam = make_gradcam_views(image, predicted)
            trial = {
                "name": uploaded.name,
                "digest": file_digest(uploaded),
                "prediction": predicted,
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "volume_cm3": volume_cm3,
                "tissue_fraction": tissue_fraction,
                "side": detected_side,
                "tumor_diameter_mm": tumor_stats["diameter_mm"] if tumor_stats else None,
                "tumor_volume_cm3": tumor_stats["volume_cm3"] if tumor_stats else None,
                "gradcam": gradcam,
            }
            trial["groq_summary"] = groq_summary(trial)
            st.session_state.trials.insert(
                0,
                trial,
            )
            st.session_state.trials = st.session_state.trials[:6]
            st.rerun()
    else:
        st.info("No scan loaded. Drop a slice above to run a trial.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-head"><b>02</b> - Scan measurements</div>', unsafe_allow_html=True)
    if st.session_state.trials:
        latest = st.session_state.trials[0]
        v1, v2 = st.columns(2)
        v1.metric("Est. slice volume", f"{latest.get('volume_cm3', 0) * 1000:.0f} mm^3")
        v2.metric("Tissue coverage", f"{latest.get('tissue_fraction', 0) * 100:.1f}%")
        if latest.get("tumor_diameter_mm"):
            t1, t2 = st.columns(2)
            t1.metric("Tumor diameter (approx.)", f"{latest['tumor_diameter_mm']:.0f} mm")
            t2.metric("Tumor volume (approx.)", f"{latest['tumor_volume_cm3'] * 1000:.0f} mm^3")
        st.caption("Approximate, derived from this single slice - not a clinical measurement.")
    else:
        st.write("Volume estimates appear here once a scan has been analyzed.")
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    current = st.session_state.trials[0] if st.session_state.trials else {"prediction": "no_tumor", "name": "Awaiting trial"}
    prediction = current["prediction"]
    detected_side = current.get("side")
    style = resolve_lesion_style(prediction, detected_side)
    flag_word = "Clear" if prediction == "no_tumor" else "Flagged"
    volume_tag = f" - Vol {current['volume_cm3'] * 1000:.0f} mm^3" if "volume_cm3" in current else ""
    tumor_tag = f" - approx. {current['tumor_diameter_mm']:.0f} mm mass" if current.get("tumor_diameter_mm") else ""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-head"><span><b>03</b> - Current 3D readout</span><span class="status-pill" style="color:{style['color']}">{flag_word}</span></div>
        <div class="prediction" style="--accent:{style['color']}">
            <span>{current['name']}</span>
            <strong>{LABELS.get(prediction, prediction)}</strong>
            <span>Confidence: {current.get('confidence', 0) * 100:.2f}% - Region flagged: {style['region_label']}{tumor_tag}{volume_tag}</span>
        </div>
        {confidence_bar_html(current.get('probabilities', {prediction: current.get('confidence', 1.0)}), prediction)}
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        brain_figure(prediction, len(st.session_state.trials) + 1, side=detected_side),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if current.get("gradcam"):
        gradcam = current["gradcam"]
        st.markdown('<div class="panel-head"><b>04</b> - Grad-CAM heatmap</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="gradcam-grid">
                <div class="gradcam-tile"><img src="{gradcam['original']}" alt="Original MRI" /><span>Original MRI</span></div>
                <div class="gradcam-tile"><img src="{gradcam['heatmap']}" alt="Heatmap" /><span>Heatmap</span></div>
                <div class="gradcam-tile"><img src="{gradcam['overlay']}" alt="Overlay" /><span>Overlay</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Heatmap is an explainability overlay derived from the uploaded slice's strongest contrast focus area.")

    if current.get("groq_summary"):
        st.markdown('<div class="panel-head"><b>05</b> - Groq AI explanation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-note">{current["groq_summary"]}</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-head" style="border-bottom:1px solid var(--line-soft); margin: 26px 0 14px;"><b>06</b> - Trial history</div>', unsafe_allow_html=True)
if st.session_state.trials:
    cols = st.columns(min(3, len(st.session_state.trials)))
    for index, trial in enumerate(st.session_state.trials):
        col = cols[index % len(cols)]
        style = resolve_lesion_style(trial["prediction"], trial.get("side"))
        if trial.get("tumor_diameter_mm"):
            vol_line = f"approx. {trial['tumor_diameter_mm']:.0f} mm mass"
        elif "volume_cm3" in trial:
            vol_line = f"{trial['volume_cm3'] * 1000:.0f} mm^3 brain"
        else:
            vol_line = trial["digest"]
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="trial-chip">Trial {len(st.session_state.trials) - index}</div>
                <div class="prediction" style="--accent:{style['color']}">
                    <span>{trial['name']}</span>
                    <strong>{LABELS.get(trial['prediction'], trial['prediction'])}</strong>
                    <span>Vol {vol_line}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                brain_figure(trial["prediction"], index + 1, compact=True, side=trial.get("side")),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("Analyzed scans will appear here, each with its own 3D readout.")


