"""Hinglish sentiment demo (Gradio Blocks UI on top of src/service.py).

Run from the project root:
    pip install -r requirements.txt
    python src/save_tfidf.py     # once, to create models/tfidf_lr.joblib
    python app/app.py            # http://127.0.0.1:7860
"""
import csv
import html
import io
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from service import MAX_BATCH, NAMES, ORDER, Service  # noqa: E402

svc = Service()
esc = html.escape

COLORS = {"negative": "#f87171", "neutral": "#fbbf24", "positive": "#34d399"}
MODEL_CHOICES = [(NAMES[k], k) for k in ORDER]

EXAMPLES = [
    "yaar ye gana bahut mast hai, sunke maza aa gaya",
    "bakwas movie thi, paisa barbaad",
    "kal shaam 5 baje match hai kya",
    "ye to bilkul acha nahi tha",
    "Love u sir, best performance ever",
    "musalman log kal yahan aaye the",
]

CSS = """
:root {
  --bg:#07111f;
  --bg-2:#0d1729;
  --panel:#111d32;
  --panel-strong:#14253d;
  --card:#0f1b2d;
  --card-alt:#121f32;
  --line:rgba(148,163,184,0.18);
  --text:#edf6ff;
  --muted:#9bb0ca;
  --accent:#7dd3fc;
  --accent-2:#a78bfa;
  --positive:#34d399;
  --negative:#f87171;
  --neutral:#fbbf24;
}
html, body, gradio-app, .gradio-container, .app {
  background:radial-gradient(circle at top, #11213d 0%, #07111f 45%, #040b14 100%) !important;
  color:var(--text) !important;
  font-family:'Inter','Segoe UI',system-ui,sans-serif !important;
}
.gradio-container {
  max-width:1200px !important;
  margin:0 auto !important;
  padding-top:18px !important;
}
#hero {
  margin-bottom:18px;
  padding:28px 26px 14px;
  border:1px solid var(--line);
  border-radius:20px;
  background:linear-gradient(135deg, rgba(19,32,58,0.95), rgba(11,17,31,0.85));
  box-shadow:0 18px 42px rgba(2,6,23,0.35);
}
#hero .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
#hero .logo {
  width:42px; height:42px; border-radius:12px;
  display:inline-flex; align-items:center; justify-content:center;
  font-weight:700; background:linear-gradient(135deg, var(--accent), var(--accent-2));
  color:#06111e; box-shadow:0 10px 24px rgba(125,211,252,0.4);
}
#hero h1 {
  margin:0; font-size:2.1rem; letter-spacing:-.04em; font-weight:800; color:var(--text);
}
#hero p {
  margin:8px 0 0; color:var(--muted); max-width:70ch; line-height:1.6; font-size:1rem;
}
.kpi-grid {
  display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin:14px 0 8px;
}
.kpi {
  background:linear-gradient(180deg, rgba(17,29,50,0.96), rgba(13,22,37,0.95));
  border:1px solid var(--line); border-radius:16px; padding:18px 16px; min-height:104px;
}
.kpi .label {
  display:block; color:var(--muted); font-size:.75rem; letter-spacing:.09em; text-transform:uppercase;
}
.kpi strong {
  display:block; margin-top:12px; font-size:1.8rem; letter-spacing:-.05em; font-weight:800;
}
.kpi .note {
  display:block; margin-top:8px; color:var(--muted); font-size:.82rem;
}
.card {
  background:linear-gradient(180deg, rgba(17,29,50,0.97), rgba(12,20,33,0.96));
  border:1px solid var(--line); border-radius:18px; padding:18px 18px; margin-top:12px;
  min-width:0; max-width:100%; box-sizing:border-box; box-shadow:0 10px 22px rgba(2,6,23,0.15);
}
.card h3 {
  margin:0 0 8px; font-size:1.04rem; color:var(--text); font-weight:700;
}
.verdict { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.verdict .lab { font-size:1.8rem; font-weight:800; letter-spacing:-.04em; }
.verdict .conf { color:var(--muted); }
.pill {
  display:inline-block; padding:3px 10px; background:rgba(251,191,36,0.14);
  border:1px solid rgba(251,191,36,0.35); color:#f4d784; border-radius:999px; font-size:.82rem;
}
.bar { display:grid; grid-template-columns:86px 1fr 52px; align-items:center; gap:10px; margin:8px 0; }
.bar span { color:var(--muted); }
.bar .track { background:rgba(148,163,184,0.12); border-radius:999px; height:12px; overflow:hidden; }
.bar .fill { height:100%; border-radius:999px; }
.warn {
  background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.18); color:#f6e39b;
  border-radius:12px; padding:10px 12px; margin-top:12px; font-size:.92rem;
}
.small { color:#cbd5e1 !important; font-size:.86rem; }
.tok { display:inline-block; padding:2px 6px; margin:3px 2px; border-radius:6px; color:var(--text); }
.chip {
  display:inline-block; padding:5px 8px; margin:4px 6px 4px 0; border-radius:999px; font-size:.83rem;
  border:1px solid rgba(148,163,184,0.18);
}
.cmpwrap { overflow-x:auto; max-width:100%; width:100%; }
table.cmp { width:100%; min-width:540px; border-collapse:collapse; font-size:.92rem; }
.gradio-container .card table.cmp th,
.gradio-container .card table.cmp thead th {
  color:#f8fafc !important; font-weight:700; text-align:left; padding:10px 12px;
  border-bottom:1px solid rgba(148,163,184,0.3) !important; background:rgba(15,27,45,0.95) !important;
}
.gradio-container .card table.cmp td,
.gradio-container .card table.cmp tbody td {
  color:#e2e8f0 !important; text-align:left; padding:10px 12px;
  border-bottom:1px solid rgba(148,163,184,0.18) !important; background:transparent !important;
}
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }
.sentiment-positive { color:#86efac !important; font-weight:700; }
.sentiment-negative { color:#fca5a5 !important; font-weight:700; }
.sentiment-neutral { color:#fde68a !important; font-weight:700; }
.form-label { color:var(--text) !important; }
.gradio-container button.primary, .primary {
  background:linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border:none !important; color:#07111f !important; font-weight:700 !important;
  box-shadow:0 12px 20px rgba(125,211,252,0.25);
}
button.primary:hover { filter:brightness(1.08); }
.tab-nav button {
  color:var(--muted) !important; border-color:var(--line) !important; background:rgba(15,27,45,0.6) !important;
}
.tab-nav button.selected, button[role=tab][aria-selected=true] {
  color:var(--text) !important; border-color:rgba(125,211,252,0.7) !important; background:rgba(125,211,252,0.12) !important;
}
textarea, input, .gr-text-input, .gr-textarea, .gr-file, .gr-dataframe {
  background:rgba(13,22,37,0.8) !important; color:var(--text) !important; border:1px solid var(--line) !important;
}
.gradio-container,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .block,
.gradio-container .block p,
.gradio-container .block li,
.gradio-container .block label,
.gradio-container .block legend,
.gradio-container .block h1,
.gradio-container .block h2,
.gradio-container .block h3,
.gradio-container .block h4,
.gradio-container .block summary {
  color:#e2e8f0 !important;
}
.gradio-container .card,
.gradio-container .card p,
.gradio-container .card div,
.gradio-container .card span,
.gradio-container .card small,
.gradio-container .card b { color:#e2e8f0; }
.gradio-container .card .small { color:#cbd5e1 !important; }
.gradio-container .block .small { color:#cbd5e1 !important; }
.gradio-container .card .verdict .lab { color:inherit; }
.gradio-container .card .sentiment-positive,
.gradio-container .card table.cmp td .sentiment-positive { color:#86efac !important; }
.gradio-container .card .sentiment-negative,
.gradio-container .card table.cmp td .sentiment-negative { color:#fca5a5 !important; }
.gradio-container .card .sentiment-neutral,
.gradio-container .card table.cmp td .sentiment-neutral { color:#fde68a !important; }
.gradio-container input,
.gradio-container textarea,
.gradio-container [role="combobox"],
.gradio-container [role="textbox"] {
  background-color:#0d1625 !important;
  color:#f8fafc !important;
  border-color:rgba(148,163,184,0.35) !important;
  opacity:1 !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color:#94a3b8 !important; opacity:1 !important; }
.gradio-container select,
.gradio-container select option,
.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container [role="option"] *,
.gradio-container [role="listbox"] * {
  background-color:#111d32 !important; color:#e2e8f0 !important;
}
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] { background-color:#24385a !important; }
.gradio-container button:not(.primary),
.gradio-container button:not(.primary) span { color:#e2e8f0 !important; }
.gradio-container button:disabled,
.gradio-container [aria-disabled="true"] { opacity:.78 !important; color:#cbd5e1 !important; }
.gradio-container .error,
.gradio-container .warning,
.gradio-container .info,
.gradio-container .success { color:#e2e8f0 !important; }
.gradio-container .dataframe,
.gradio-container .dataframe th,
.gradio-container .dataframe td,
.gradio-container .table-wrap th,
.gradio-container .table-wrap td {
  color:#e2e8f0 !important; background-color:#0d1625 !important;
  border-color:rgba(148,163,184,0.24) !important;
}
.gradio-container .gr-form, .gradio-container .panel, .gradio-container .block {
  border-color:var(--line) !important;
}
@media (max-width: 860px) {
  .kpi-grid { grid-template-columns:repeat(2, minmax(0,1fr)); }
}
@media (max-width: 600px) {
  .kpi-grid { grid-template-columns:1fr; }
  .bar { grid-template-columns:72px 1fr 40px; }
  .verdict .lab { font-size:1.45rem; }
}
"""


def _bars(probs: dict) -> str:
    rows = []
    for lab in ("negative", "neutral", "positive"):
        p = probs[lab]
        rows.append(
            f'<div class="bar"><span>{lab}</span><div class="track"><div class="fill" '
            f'style="width:{p * 100:.1f}%;background:{COLORS[lab]}"></div></div><span>{p * 100:.0f}%</span></div>'
        )
    return "".join(rows)


def _explanation(e: dict, label: str) -> str:
    if e["kind"] == "evidence":
        def chips(items, color, sign):
            return "".join(
                f'<span class="chip" style="background:{color}22;color:{color};border-color:{color}55">{esc(w)} '
                f'<small>{sign}{abs(c):.2f}</small></span>' for w, c in items
            ) or '<span class="small">none</span>'
        if not e["toward"] and not e["against"]:
            return (
                "<h3>Why " + esc(label) + "?</h3><div>None of the words in this comment appeared in the "
                "training data, so there is no word-level evidence. The result comes from character "
                "patterns and the class priors, which makes it close to a default guess.</div>"
            )
        return (
            f"<h3>Why {esc(label)}?</h3><div><b>Pushed towards {esc(label)}</b><br>{chips(e['toward'], COLORS[label], '+')}</div>"
            f"<div style='margin-top:8px'><b>Pushed away</b><br>{chips(e['against'], '#a2b1c7', '−')}</div>"
            f'<div class="small">{esc(e["note"])} Character n-gram features also contribute, so words alone do not '
            f"add up to the full score.</div>"
        )
    mx = max(e["weights"]) if e["weights"] else 1
    toks = "".join(
        f'<span class="tok" style="background:rgba(125,211,252,{0.12 + 0.72 * w / mx:.2f})">{esc(t)}</span>'
        for t, w in zip(e["tokens"], e["weights"])
    )
    return f'<h3>Where the model looked</h3><div>{toks}</div><div class="small">{esc(e["note"])}</div>'


def render(o: dict) -> str:
    if o.get("empty"):
        return '<div class="warn">Type a comment first.</div>'
    lab, col = o["label"], COLORS[o["label"]]
    unsure = '<span class="pill">low confidence, treat as a guess</span>' if o["unsure"] else ""
    warn = "".join(f'<div class="warn">{esc(w)}</div>' for w in o["warnings"])
    ag = o["agreement"]
    rows = []
    for m in o["models"]:
        score = "" if m["test_f1"] is None else f"{m['test_f1']:.2f}"
        sentiment_class = f"sentiment-{m['label']}"
        rows.append(
            f'<tr><td>{esc(m["name"])}</td><td><span class="sw" style="background:{COLORS[m["label"]]}"></span>'
            f'<span class="{sentiment_class}">{esc(m["label"])}</span></td>'
            f'<td>{max(m["probs"].values()) * 100:.0f}%</td><td>{score}</td></tr>'
        )
    rows_html = "".join(rows)
    return f"""
<div class="card">
  <div class="verdict"><span class="lab" style="color:{col}">{lab}</span>
  <span class="conf">{o['confidence'] * 100:.0f}% confidence · {esc(NAMES[o['primary']])}</span>{unsure}</div>
  <div style="margin-top:10px">{_bars(o['probs'])}</div>{warn}
</div>
<div class="card">{_explanation(o['explanation'], lab)}</div>
<div class="card"><h3>All models on this comment: {ag['count']} of {ag['total']} say {ag['label']}</h3>
<div class="cmpwrap"><table class="cmp"><tr><th>Model</th><th>Prediction</th><th>Top prob.</th><th>Test macro-F1</th></tr>{rows_html}</table></div>
<div class="small">Probabilities are not comparable across models (each is calibrated differently); agreement is a rough signal, not proof. Test macro-F1 is on 388 held-out comments, so differences under about 0.03 are noise.</div></div>"""


def _dashboard_html(
    text: str = "",
    model: str = "tfidf_lr",
    result: dict | None = None,
    processed: int = 0,
    last_label: str = "No prediction yet",
    confidence: str | None = None,
) -> str:
    selected_model = NAMES.get(model, model)
    latest_label = last_label if last_label else (result.get("label", "No prediction yet") if result else "No prediction yet")
    visual_conf = confidence if confidence is not None else (
        f"{result['confidence'] * 100:.0f}%" if result and not result.get("empty") else "—"
    )
    return f"""
<div class="card" style="margin-top:0;padding:0;border:none;background:transparent;box-shadow:none;">
  <div class="kpi-grid">
    <div class="kpi"><span class="label">Supported models</span><strong>{len(ORDER)}</strong><span class="note">Actual models loaded</span></div>
    <div class="kpi"><span class="label">Comments processed</span><strong>{processed}</strong><span class="note">Current session total</span></div>
    <div class="kpi"><span class="label">Selected model</span><strong>{esc(selected_model)}</strong><span class="note">Current analysis mode</span></div>
    <div class="kpi"><span class="label">Latest result</span><strong>{esc(latest_label)}</strong><span class="note">Confidence {visual_conf}</span></div>
  </div>
</div>
"""


def _session_dashboard(text, model, state):
    state = state or {"processed": 0, "last_label": "No prediction yet", "confidence": "—"}
    result = svc.analyse(text, model) if (text or "").strip() else None
    if result and not result.get("empty"):
        state = dict(state)
        state["processed"] = int(state.get("processed", 0)) + 1
        state["last_label"] = result["label"]
        state["confidence"] = f"{result['confidence'] * 100:.0f}%"
    else:
        state = dict(state)
        state.setdefault("last_label", "No prediction yet")
        state.setdefault("confidence", "—")
    return _dashboard_html(text, model, result, processed=state.get("processed", 0), last_label=state.get("last_label", "No prediction yet"), confidence=state.get("confidence", "—")), state


def analyse(text, primary):
    result = svc.analyse(text, primary)
    return render(result)


def _read_lines(file, text):
    lines = []
    if file:
        raw = Path(file).read_text(encoding="utf-8", errors="replace")
        if Path(file).suffix.lower() == ".csv":
            rd = list(csv.reader(io.StringIO(raw)))
            if rd and rd[0] and rd[0][0].strip().lower() in {"text", "comment", "comments", "sentence"}:
                rd = rd[1:]
            lines += [r[0] for r in rd if r]
        else:
            lines += raw.splitlines()
    lines += (text or "").splitlines()
    return [l.strip() for l in lines if l.strip()]


def batch(file, text, primary):
    lines = _read_lines(file, text)
    if not lines:
        return "Paste comments (one per line) or upload a .txt/.csv file.", None, None
    rows = svc.analyse_many(lines, primary)
    msg = f"Classified {len(rows)} comments with {NAMES[primary]}."
    if len(lines) > MAX_BATCH:
        msg += f" Only the first {MAX_BATCH} of {len(lines)} lines were used."
    cols = ["comment", "prediction", "confidence", "unsure", "models_agreeing"]
    path = Path(tempfile.mkdtemp()) / "hinglish_sentiment_results.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return msg, [[r[c] for c in cols] for r in rows], str(path)


def compare_models(text, primary):
    if not (text or "").strip():
        return '<div class="card"><h3>Model comparison</h3><div class="small">Type a Hinglish comment to compare the active models.</div></div>'
    result = svc.analyse(text, primary)
    if result.get("empty"):
        return '<div class="card"><h3>Model comparison</h3><div class="small">Write a comment to compare model predictions.</div></div>'
    rows = []
    for m in result["models"]:
        score = "—" if m["test_f1"] is None else f"{m['test_f1']:.3f}"
        sentiment_class = f"sentiment-{m['label']}"
        rows.append(
            f'<tr><td>{esc(m["name"])}</td><td><span class="sw" style="background:{COLORS[m["label"]]}"></span>'
            f'<span class="{sentiment_class}">{esc(m["label"])}</span></td>'
            f'<td>{max(m["probs"].values()) * 100:.0f}%</td><td>{score}</td></tr>'
        )
    rows_html = "".join(rows)
    return f"""
<div class="card"><h3>Model comparison for: {esc(text[:120])}</h3>
<div class="cmpwrap"><table class="cmp"><tr><th>Model</th><th>Prediction</th><th>Top prob.</th><th>Test macro-F1</th></tr>{rows_html}</table></div>
<div class="small">Comparison is per-input and not a direct measure of model quality. Evaluation metrics come from saved held-out results, not a single prediction.</div></div>"""


def analytics_html() -> str:
    rows = "".join(
        f'<tr><td>{esc(NAMES[k])}</td><td>{svc.scores.get(k, 0.0):.3f}</td></tr>' for k in ORDER if k in svc.scores
    )
    return f"""
<div class="card"><h3>Model quality snapshot</h3>
<table class="cmp"><tr><th>Model</th><th>Test macro-F1</th></tr>{rows}</table>
<div class="small">Macro-F1 is the primary metric reported on the held-out set. Differences smaller than about 0.03 are not reliable enough to treat as a material win.</div></div>
<div class="card"><h3>Project notes</h3>
<p class="small">This demo uses the same models and evaluation outputs described in the project report. The interface makes uncertainty visible rather than hiding it, especially for short, noisy, or out-of-vocabulary Hinglish comments.</p></div>
"""


def _distribution_html(rows: list[dict]) -> str:
    if not rows:
        return '<div class="card"><h3>Sentiment distribution</h3><div class="small">No rows yet.</div></div>'
    counts = Counter(r["prediction"] for r in rows)
    total = sum(counts.values()) or 1
    parts = []
    for label in ("negative", "neutral", "positive"):
        count = counts.get(label, 0)
        pct = count / total * 100
        parts.append(
            f'<div class="bar"><span>{label}</span><div class="track"><div class="fill" '
            f'style="width:{pct:.1f}%;background:{COLORS[label]}"></div></div><span>{count}</span></div>'
        )
    return f'<div class="card"><h3>Sentiment distribution</h3>{"".join(parts)}</div>'


def about_html() -> str:
    rows = "".join(f"<tr><td>{esc(NAMES[k])}</td><td>{svc.scores[k]:.3f}</td></tr>" for k in ORDER if k in svc.scores)
    return f"""
<div class="card"><h3>What this is</h3>
<p>Sentiment (negative / neutral / positive) for Hindi-English code-mixed text in Roman script. Models were trained on 3,103 comments from the IIITH corpus (Prabhu et al., COLING 2016, arXiv:1611.00472) and compared on 388 held-out comments. The default model is TF-IDF + logistic regression because it scored best.</p>
<table class="cmp"><tr><th>Model</th><th>Test macro-F1</th></tr>{rows}</table></div>
<div class="card"><h3>Limitations</h3>
<p>Small data and a single source: social-media comments about Indian public figures, collected around 2016. Expect mistakes on negation, sarcasm, slang and anything outside that domain. Devanagari script is not supported.</p></div>
<div class="card" style="border-color:rgba(248,113,113,0.6)"><h3>Bias warning</h3>
<p>The training data carries community bias: comments mentioning "musalman" are 81% negative in the corpus. In a probe with neutral sentences, CharCNN and the scratch Transformer rate such sentences as strongly negative. Try the last example to see it. Do not use this for moderation or any decision about people.</p></div>"""


with gr.Blocks(title="Hinglish Sentiment Analyzer") as demo:
    gr.HTML(
        '<div id="hero"><div class="brand-row"><span class="logo">H</span><span style="font-size:.8rem;letter-spacing:.14em;color:#9bb0ca;text-transform:uppercase;">AI-powered sentiment analysis</span></div>'
        '<h1>Hinglish Sentiment Analyzer</h1><p>Modern sentiment analysis for Hinglish comments, with real model comparison, explanation, and robust batch inference.</p></div>'
    )
    session_state = gr.State({"processed": 0, "last_label": "No prediction yet", "confidence": "—"})
    overview = gr.HTML(_dashboard_html(processed=0))
    with gr.Tabs():
        with gr.Tab("Analyse"):
            txt = gr.Textbox(lines=4, max_lines=10, label="Comment", placeholder="Yaar ye gana bahut mast hai, sunke maza aa gaya.")
            with gr.Row():
                model = gr.Dropdown(MODEL_CHOICES, value="tfidf_lr", label="Main model", scale=3)
                go = gr.Button("Analyse", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=txt, label="Try an example")
            out = gr.HTML(label="Result")
            go.click(analyse, [txt, model], out, api_name="analyse")
            go.click(lambda t, m, s: _session_dashboard(t, m, s), [txt, model, session_state], [overview, session_state], api_name=False)
            txt.submit(analyse, [txt, model], out, api_name=False)
            txt.submit(lambda t, m, s: _session_dashboard(t, m, s), [txt, model, session_state], [overview, session_state], api_name=False)
        with gr.Tab("Batch Analysis"):
            gr.Markdown(f"One comment per line, or a CSV with the comment in the first column. Up to {MAX_BATCH} comments.")
            bf = gr.File(label="Upload .txt or .csv", file_types=[".txt", ".csv"], type="filepath")
            bt = gr.Textbox(lines=6, label="Paste comments here")
            bm = gr.Dropdown(MODEL_CHOICES, value="tfidf_lr", label="Model")
            bgo = gr.Button("Classify all", variant="primary")
            bmsg = gr.Markdown()
            btab = gr.Dataframe(headers=["comment", "prediction", "confidence", "unsure", "models_agreeing"], interactive=False, wrap=True)
            bdl = gr.File(label="Download results (CSV)")
            bgo.click(batch, [bf, bt, bm], [bmsg, btab, bdl], api_name="batch")
            bgo.click(lambda file, text, model: _distribution_html(svc.analyse_many(_read_lines(file, text), model)), [bf, bt, bm], gr.HTML(), api_name=False)
        with gr.Tab("Model Comparison"):
            comp_txt = gr.Textbox(lines=4, max_lines=10, label="Comment", placeholder="Compare all models on one Hinglish sentence.")
            comp_model = gr.Dropdown(MODEL_CHOICES, value="tfidf_lr", label="Primary model")
            comp_go = gr.Button("Compare models", variant="primary")
            comp_out = gr.HTML()
            comp_go.click(compare_models, [comp_txt, comp_model], comp_out, api_name="compare")
        with gr.Tab("Results & Analytics"):
            stats = gr.HTML(analytics_html())
            batch_preview = gr.HTML()
            bgo.click(lambda file, text, model: _distribution_html(svc.analyse_many(_read_lines(file, text), model)), [bf, bt, bm], batch_preview, api_name=False)
        with gr.Tab("About"):
            gr.HTML(about_html())

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Base(font=[gr.themes.GoogleFont("Hind"), "system-ui", "sans-serif"]),
        server_name="0.0.0.0",
        share=False,
    )
