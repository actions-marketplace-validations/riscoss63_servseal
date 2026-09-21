"""The attestation report: one page a reviewer can read without the tool.

Everything a decision needs on one screen -- verdict, severity, signature, the two
numbers that justify them, and the certified bound -- followed by everything an audit
needs below the fold: per-position distribution of movement, snapshot identities, probe
hash, versions. The report is a static, self-contained HTML file with no external
scripts, so it can be attached to a ticket or archived with a release.
"""
from __future__ import annotations

import html
import json

__all__ = ["render_report", "render_body", "render_api_body"]

_CSS = """
:root {
  --paper:#fbfaf7; --card:#ffffff; --ink:#22262d; --muted:#5c6472;
  --line:#e5e2da; --accent:#3d566f; --mono-bg:#f2f0ea;
  --sealed:#1b6f5f; --changed:#b03230; --incomparable:#9a6b1a;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#191b1f; --card:#212429; --ink:#e8e6e1; --muted:#9aa1ac;
    --line:#33373e; --accent:#8fb0cf; --mono-bg:#26292f;
    --sealed:#4dbfa4; --changed:#e0706c; --incomparable:#d8a94e;
  }
}
:root[data-theme="dark"] {
  --paper:#191b1f; --card:#212429; --ink:#e8e6e1; --muted:#9aa1ac;
  --line:#33373e; --accent:#8fb0cf; --mono-bg:#26292f;
  --sealed:#4dbfa4; --changed:#e0706c; --incomparable:#d8a94e;
}
body { background:var(--paper); color:var(--ink);
  font:16px/1.55 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; margin:0; }
.wrap { max-width:860px; margin:0 auto; padding:2.2rem 1.4rem 3rem; }
.tool { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); }
h1 { font-size:1.65rem; margin:.25rem 0 1.2rem; text-wrap:balance; font-weight:600; }
.banner { border-radius:8px; padding:1rem 1.25rem; color:#fff; margin:0 0 1.5rem;
  display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; }
.banner .status { font-size:1.25rem; font-weight:700; letter-spacing:.02em; }
.banner .sub { opacity:.92; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:10px; margin:0 0 1.5rem; }
.cell { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:.7rem .9rem; }
.cell .k { font-size:.74rem; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); }
.cell .v { font-size:1.3rem; font-variant-numeric:tabular-nums; margin-top:.15rem; }
.cell .n { font-size:.8rem; color:var(--muted); }
h2 { font-size:1.02rem; margin:1.8rem 0 .6rem; }
p { margin:.5rem 0; max-width:68ch; }
.sig { border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
td, th { text-align:left; padding:.35rem .6rem .35rem 0; vertical-align:top;
  border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:500; }
code, .mono { font-family:"IBM Plex Mono", ui-monospace, Consolas, monospace;
  font-size:.85em; background:var(--mono-bg); border-radius:4px; padding:.08em .35em; }
.hist { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding: .9rem; }
.hist svg { display:block; width:100%; height:auto; }
.foot { margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:.82rem; }
.scroll { overflow-x:auto; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior:smooth; } }
"""

_FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
          'family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono&display=swap">')

_COLOR = {"sealed": "var(--sealed)", "changed": "var(--changed)",
          "incomparable": "var(--incomparable)"}
_TITLE = {"sealed": "SEALED — behaviour matches the reference",
          "changed": "CHANGED — the served behaviour is not the reference",
          "incomparable": "INCOMPARABLE — not the same measurement"}


def _esc(x):
    return html.escape(str(x))


def _histogram_svg(hist, floor):
    edges, counts = hist["edges"], hist["counts"]
    top = max(max(counts), 1)
    W, H, pad = 720, 150, 24
    bw = (W - 2 * pad) / len(counts)
    bars = []
    for i, c in enumerate(counts):
        h = 0 if c == 0 else max(2.0, (H - 2 * pad) * c / top)
        x, y = pad + i * bw, H - pad - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1.5:.1f}" '
                    f'height="{h:.1f}" fill="var(--accent)" opacity="0.85">'
                    f'<title>{edges[i]:.2f}-{edges[i + 1]:.2f}: {c} positions</title></rect>')
    fx = pad + (W - 2 * pad) * min(floor, 1.0)
    bars.append(f'<line x1="{fx:.1f}" y1="{pad / 2}" x2="{fx:.1f}" y2="{H - pad}" '
                f'stroke="var(--changed)" stroke-dasharray="4 3"/>'
                f'<text x="{fx + 5:.1f}" y="{pad}" fill="var(--muted)" '
                f'font-size="11">noise floor {floor:.3f}</text>')
    axis = (f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" '
            f'stroke="var(--line)"/>'
            f'<text x="{pad}" y="{H - 6}" fill="var(--muted)" font-size="11">0.0</text>'
            f'<text x="{W - pad - 18}" y="{H - 6}" fill="var(--muted)" '
            f'font-size="11">1.0</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distribution of '
            f'per-position Hellinger distances">{"".join(bars)}{axis}</svg>')


EMDASH = "—"

_API_TITLE = {
    "sealed": "SEALED \u2014 the sampled behaviour is inside the reference's band",
    "changed": "CHANGED \u2014 the sampled behaviour is outside the reference's band",
    "incomparable": "INCOMPARABLE \u2014 not the same measurement",
}


def _redact(url):
    """Endpoint URLs reach tickets and archives; credentials must not ride along."""
    try:
        from urllib.parse import urlsplit, urlunsplit
        u = urlsplit(str(url))
        host = u.hostname or ""
        if u.port:
            host = f"{host}:{u.port}"
        if u.username:
            host = f"***@{host}"
        return urlunsplit((u.scheme, host, u.path, "", ""))
    except Exception:
        return str(url)


def _band_svg(rows):
    """Where each statistic landed against the band the unchanged endpoint produces.

    The verdict here is "inside or outside", so the figure is the interval and the
    point -- not a distance against a threshold, which is the weights-mode picture and
    would misdescribe what was measured.
    """
    W, rowh, pad = 720, 64, 8
    H = rowh * len(rows) + 24
    out = []
    for i, (name, val, lo, hi, note) in enumerate(rows):
        span = max(hi - lo, 1e-9)
        xmin = min(lo - 3 * span, val - 0.8 * span)
        xmax = max(hi + 3 * span, val + 0.8 * span)
        rng = xmax - xmin

        def px(x, _a=xmin, _r=rng):
            return pad + (W - 2 * pad) * (x - _a) / _r

        y = 30 + i * rowh
        inside = lo <= val <= hi
        colour = "var(--sealed)" if inside else "var(--changed)"
        out.append(
            f'<text x="{pad}" y="{y - 16}" fill="var(--ink)" font-size="13" '
            f'font-weight="600">{_esc(name)}</text>'
            f'<text x="{pad}" y="{y - 3}" fill="var(--muted)" font-size="10">'
            f'{_esc(note)}</text>'
            f'<line x1="{px(xmin):.1f}" y1="{y + 13}" x2="{px(xmax):.1f}" '
            f'y2="{y + 13}" stroke="var(--line)" stroke-width="2"/>'
            f'<rect x="{px(lo):.1f}" y="{y + 4}" width="{max(px(hi) - px(lo), 1):.1f}" '
            f'height="18" fill="var(--accent)" opacity="0.22" rx="3"/>'
            f'<line x1="{px(lo):.1f}" y1="{y + 2}" x2="{px(lo):.1f}" y2="{y + 24}" '
            f'stroke="var(--accent)"/>'
            f'<line x1="{px(hi):.1f}" y1="{y + 2}" x2="{px(hi):.1f}" y2="{y + 24}" '
            f'stroke="var(--accent)"/>'
            f'<circle cx="{px(val):.1f}" cy="{y + 13}" r="5.5" fill="{colour}">'
            f'<title>{_esc(name)}: {val:.4f}, band {lo:.4f}-{hi:.4f}</title></circle>'
            f'<text x="{px(val):.1f}" y="{y + 37}" fill="{colour}" font-size="11" '
            f'text-anchor="middle" font-weight="600">{val:.4f}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Observed statistics '
            f'against their acceptance bands">{"".join(out)}</svg>')


def render_api_body(result: dict) -> str:
    """The attestation for a sampled endpoint.

    A different measurement from weights mode, and therefore a different page: there
    is no per-position distance to draw a histogram of, and SEALED is the weaker claim
    that the sample fell inside what the unchanged reference produces at this budget.
    Printing the weights-mode page here would be a lie of format.
    """
    v = result["verdict"]
    ref = result.get("ref_meta", {})
    b = result.get("bands", {})
    wire = result.get("wire", {})
    ep = result.get("endpoint", {})
    s1, s2 = result["s1"], result["s2"]
    tc = ref.get("api", {}).get("text_convention", {})
    color, title = _COLOR[v["status"]], _API_TITLE[v["status"]]
    head, _, tail = title.partition(" \u2014 ")

    cells = []

    def cell(k, val, note=""):
        cells.append(f'<div class="cell"><div class="k">{_esc(k)}</div>'
                     f'<div class="v">{_esc(val)}</div>'
                     + (f'<div class="n">{_esc(note)}</div>' if note else "")
                     + "</div>")

    n_pos = ref.get("n_positions")
    total = b.get("total", wire.get("resolved", 0))
    per = (f", {total / n_pos:.1f} each"
           if isinstance(n_pos, int) and n_pos and isinstance(total, int) else "")
    cell("sampled tokens", f"{total:,}" if isinstance(total, int) else total,
         f"over {n_pos if n_pos else '?'} positions{per}")
    if b.get("s1"):
        cell("S1 distributional", f"{s1:.4f}",
             f"band {b['s1'][0]:.4f}-{b['s1'][1]:.4f}")
    if b.get("s2"):
        cell("S2 head frequency", f"{s2:.4f}",
             f"band {b['s2'][0]:.4f}-{b['s2'][1]:.4f}")
    conv = wire.get("convention", "?")
    cell("wire convention", conv,
         "exact: nothing to invert" if conv != "text"
         else "text: ids recovered by re-encoding")
    cell("unresolved tokens", f"{wire.get('unresolved_fraction', 0):.3%}",
         f"{wire.get('resolved', 0):,} of {wire.get('returned', 0):,} used")
    if conv == "text" and tc:
        cell("probe-set exposure", f"{100 * tc.get('wrong_mass', 0):.4f} %",
             "reference mass this convention moves")

    fig = ""
    if b.get("s1") and b.get("s2"):
        fig = ('<div class="hist">' + _band_svg([
            ("S1 " + EMDASH + " distributional, sees the tail",
             s1, b["s1"][0], b["s1"][1],
             "Bhattacharyya coefficient of the sample against the stored sketches"),
            ("S2 " + EMDASH + " how often the reference argmax is emitted",
             s2, b["s2"][0], b["s2"][1],
             "above the band = head concentrated; below = head moved"),
        ]) + "</div>")

    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(ref.get(k, EMDASH))}</td></tr>"
        for k in ("model", "model_id", "dtype", "template", "n_positions",
                  "created_utc", "servseal"))

    return f"""<style>{_CSS}</style>{_FONTS}
<div class="wrap">
  <div class="tool">servseal &middot; behavioural attestation &middot; sampled endpoint</div>
  <h1>{_esc(ep.get('model', '?'))} at {_esc(_redact(ep.get('url', '?')))}</h1>
  <div class="banner" style="background:{color}">
    <span class="status">{_esc(head)}</span>
    <span class="sub">{_esc(tail)}
      {' &middot; severity: ' + _esc(v['severity']) if v['status'] == 'changed' else ''}</span>
  </div>
  <div class="grid">{''.join(cells)}</div>
  <h2>Reading</h2>
  <div class="sig"><p><strong>{_esc(v['signature'])}.</strong> {_esc(v['explanation'])}</p></div>
  <h2>Where the statistics landed</h2>
  {fig}
  <h2>Reference identity</h2>
  <div class="scroll"><table>
    {rows}
    <tr><th>probe set</th><td><code>{_esc(ref.get('probe', '?'))}</code>
        ({_esc(ref.get('probe_file', '?'))}) &mdash; the endpoint was asked for the
        prefixes this hash covers, and nothing else</td></tr>
    <tr><th>bands</th><td>calibrated at snapshot time from
        {_esc(b.get('reps', '?'))} simulated runs of the <em>unchanged</em> reference
        at this budget, family false-positive rate {_esc(b.get('alpha', '?'))}{
        ', through the text round trip' if b.get('through_wire') else ''}</td></tr>
    <tr><th>sampling</th><td>{_esc(wire.get('requests', '?'))} requests,
        {_esc(wire.get('retries', 0))} retries, <code>max_tokens=1</code>,
        <code>temperature=1</code>, <code>top_p</code> not sent</td></tr>
  </table></div>
  <div class="foot">
    <p>Method: the reference stores a {_esc(ref.get('D', '?'))}-coordinate square-root
    sketch of the full next-token distribution at every probe position. A black-box
    endpoint exposes no distribution, so it is sampled one token at a time and two
    statistics are formed against those stored sketches: S1, the estimated
    Bhattacharyya coefficient of the empirical sample, which sees the tail; and S2,
    the rate at which the reference's most likely token comes back, which sees only
    the head. A serving filter pushes S2 <em>up</em> &mdash; cutting the tail
    concentrates the sample on the head &mdash; whereas changed weights push it down.</p>
    <p><strong>What SEALED does and does not say here.</strong> It says the sample fell
    inside the interval the unchanged reference produces at this budget. It is not
    proof of identity: what is guaranteed is the measured detection power at the
    budget spent, not certainty, and more tokens buy more power.
    <code>top_p</code> is deliberately never sent, because sending it would switch off
    the server-side filter this test exists to find.</p>
    <p>Exit code {v['exit_code']} &middot; generated by servseal
    {_esc(ref.get('servseal', ''))} &middot; the bands and the probe-set exposure were
    computed where the reference distributions still existed, and travel inside the
    snapshot.</p>
  </div>
</div>"""


def render_body(result: dict) -> str:
    """The report content, headless (no doctype/html wrapper): metrics + verdict from a
    `servseal verify --json` result. Dispatches on how the candidate was measured."""
    if result.get("mode") == "endpoint":
        return render_api_body(result)
    m, v = result["metrics"], result["verdict"]
    ref = m.get("ref_meta", {})
    cand = m.get("cand_meta", {})
    color, title = _COLOR[v["status"]], _TITLE[v["status"]]

    cells = []

    def cell(k, val, note=""):
        cells.append(f'<div class="cell"><div class="k">{_esc(k)}</div>'
                     f'<div class="v">{_esc(val)}</div>'
                     + (f'<div class="n">{_esc(note)}</div>' if note else "")
                     + "</div>")

    if not m.get("incomparable"):
        npos = m["n_positions"]
        npos = f"{npos[0]} vs {npos[1]}" if isinstance(npos, (list, tuple)) else npos
        cell("mean Hellinger", f"{m['mean_hellinger']:.4f}",
             f"over {npos} probe positions")
        if m.get("top1_agreement") is not None:
            cell("top-1 agreement", f"{m['top1_agreement']:.1%}",
                 "most likely token unchanged")
        if m.get("positions_moved") is not None:
            cell("positions moved", f"{m['positions_moved']}",
                 f"above the noise floor {m['noise_floor']:.3f}")
        klb = m.get('mean_kl_lower_bound',
                    max(0.0, m['aggregate_kl_lower_bound']))
        cell("certified KL", f">= {klb:.3f}",
             "mean per-position lower bound; 0 certifies nothing")
        pa, pb = ref.get("perplexity"), cand.get("perplexity")
        if pa and pb:
            cell("perplexity", f"{pa:.2f} -> {pb:.2f}",
                 "what a log-loss check would see")

    hist_html = ""
    if m.get("hellinger_histogram"):
        hist_html = ('<h2>How far each probe position moved</h2><div class="hist">'
                     + _histogram_svg(m["hellinger_histogram"], m["noise_floor"])
                     + "</div>")

    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(ref.get(k, chr(0x2014)))}</td>"
        f"<td>{_esc(cand.get(k, chr(0x2014)))}</td></tr>"
        for k in ("model", "dtype", "template", "n_positions", "created_utc",
                  "servseal"))
    probe = ref.get("probe", "?")

    return f"""<style>{_CSS}</style>{_FONTS}
<div class="wrap">
  <div class="tool">servseal &middot; behavioural attestation</div>
  <h1>{_esc(ref.get('model', '?'))} vs {_esc(cand.get('model', '?'))}</h1>
  <div class="banner" style="background:{color}">
    <span class="status">{_esc(title.split(' — ')[0])}</span>
    <span class="sub">{_esc(title.split(' — ')[1])}
      {' &middot; severity: ' + _esc(v['severity']) if v['status'] == 'changed' else ''}</span>
  </div>
  <div class="grid">{''.join(cells)}</div>
  <h2>Reading</h2>
  <div class="sig"><p><strong>{_esc(v['signature'])}.</strong> {_esc(v['explanation'])}</p></div>
  {hist_html}
  <h2>Snapshot identities</h2>
  <div class="scroll"><table>
    <tr><th></th><th>reference</th><th>candidate</th></tr>{rows}
    <tr><th>probe set</th><td colspan="2"><code>{_esc(probe)}</code>
        ({_esc(ref.get('probe_file', '?'))}) &mdash; snapshots refuse comparison unless
        this hash matches</td></tr>
  </table></div>
  <div class="foot">
    <p>Method: each snapshot stores a {_esc(ref.get('D', '?'))}-coordinate square-root
    sketch of the full next-token distribution at every probe position
    (&asymp;1&nbsp;KB/position), plus the most likely token. The sketch estimates the
    Bhattacharyya coefficient of the complete distributions &mdash; tail included,
    which is where serving filters and quantisation act and where top-k logprobs are
    blind by construction. The KL line is a one-sided certificate: it proves the
    behaviour moved at least that much, never that it did not.</p>
    <p>Exit code {v['exit_code']} &middot; generated by servseal
    {_esc(ref.get('servseal', ''))} &middot; verdict thresholds are calibrated on
    measured perturbations of real models; see the project's committed calibration run.</p>
  </div>
</div>"""


def render_report(result: dict, title: str | None = None) -> str:
    """Standalone HTML document for `servseal report -o report.html`."""
    body = render_body(result)
    t = title or ("servseal attestation \u2014 sampled endpoint"
                  if result.get("mode") == "endpoint" else "servseal attestation")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(t)}</title></head><body>{body}</body></html>')


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
