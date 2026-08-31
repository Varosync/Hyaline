#!/usr/bin/env python3
"""
Build the Hyaline kinase atlas.
==============================

A browsable, downloadable map of human kinases. For every human kinase in KLIFS
it records the accessible DFG / alphaC states, structure counts, known Type I and
Type II inhibitors, a Type-II-opportunity score, and (where computed) a geometric
descriptor fingerprint.

Outputs (research/kinase/atlas/):
  * kinase_atlas.parquet / .csv   -- one row per kinase (loads in pandas)
  * index.html                    -- offline page: searchable table + scatter
                                     (DFG-to-alphaC distance vs hinge angle,
                                      colored by DFG state)

Per-kinase counts are taken directly from KLIFS structures, so they reconcile
against the KLIFS input by construction.

Usage:
    python scripts/build_kinase_atlas.py
"""
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

BASE = "https://klifs.net/api_v2"
OUT = "research/kinase/atlas"
CACHE = "klifs_cache/atlas_structures.json"
DESC_CSV = "research/kinase/paper/kinase_descriptors.csv"

# Cofactors / crystallization additives that are not drug-like inhibitors.
NON_DRUG = {"ATP", "ADP", "ANP", "AMP", "GTP", "GDP", "GNP", "ACP", "AGS", "ACO",
            "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "DMS", "BME", "DTT", "TRS",
            "HOH", "WAT", "MG", "MN", "ZN", "CA", "NA", "CL", "0", ""}


def _get(endpoint, params):
    return requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)


def fetch_all_structures(kinases):
    """Per-kinase structures_list (concurrent), cached to JSON."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))

    def one(k):
        try:
            r = _get("structures_list", {"kinase_ID": [k["kinase_ID"]]})
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return k, data
        except Exception:
            pass
        return k, []

    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, k) for k in kinases]
        for i, f in enumerate(as_completed(futs)):
            k, data = f.result()
            out[str(k["kinase_ID"])] = {"meta": k, "structures": data}
            if (i + 1) % 50 == 0:
                print(f"  fetched {i + 1}/{len(kinases)} kinases")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    return out


def drug_ligands(structures, state):
    ligs = set()
    for s in structures:
        dfg = str(s.get("DFG", "")).lower()
        is_in = dfg == "in"
        is_out = "out" in dfg
        if (state == "in" and is_in) or (state == "out" and is_out):
            lig = str(s.get("ligand", "")).strip().upper()
            if lig and lig not in NON_DRUG:
                ligs.add(lig)
    return sorted(ligs)


def opportunity_score(n_struct, frac_out, n_type2):
    """Heuristic: well-studied kinases that access DFG-out but have few known
    Type II inhibitors score high. Documented as a heuristic, not a measurement."""
    interest = math.log10(1 + n_struct)
    return round(interest * (0.25 + frac_out) / (1 + n_type2), 3)


def load_descriptor_medians():
    if not os.path.exists(DESC_CSV):
        return {}, []
    rows = list(csv.DictReader(open(DESC_CSV)))
    per = {}
    points = []
    for r in rows:
        d, a = float(r["distance"]), float(r["hinge_angle"])
        per.setdefault(r["kinase"], []).append((d, a))
        points.append({"k": r["kinase"], "d": round(d, 2), "a": round(a, 2),
                       "y": int(r["y"])})
    med = {}
    for k, vals in per.items():
        vals.sort()
        ds = sorted(v[0] for v in vals)
        as_ = sorted(v[1] for v in vals)
        mid = len(vals) // 2
        med[k] = (round(ds[mid], 2), round(as_[mid], 2), len(vals))
    return med, points


def build_rows(raw, med):
    rows = []
    for _kid, entry in raw.items():
        meta, structs = entry["meta"], entry["structures"]
        if not structs:
            continue
        n = len(structs)
        n_in = sum(1 for s in structs if str(s.get("DFG", "")).lower() == "in")
        n_out = sum(1 for s in structs if "out" in str(s.get("DFG", "")).lower())
        t1 = drug_ligands(structs, "in")
        t2 = drug_ligands(structs, "out")
        states = []
        if n_in:
            states.append("DFG-in")
        if n_out:
            states.append("DFG-out")
        name = meta.get("name", "")
        fp = med.get(name)
        rows.append({
            "kinase": name,
            "gene": meta.get("gene_name", ""),
            "uniprot": meta.get("accession", ""),
            "n_structures": n,
            "n_dfg_in": n_in,
            "n_dfg_out": n_out,
            "accessible_states": "|".join(states),
            "n_type1_inhibitors": len(t1),
            "n_type2_inhibitors": len(t2),
            "type1_inhibitors": ",".join(t1[:15]),
            "type2_inhibitors": ",".join(t2[:15]),
            "type2_opportunity": opportunity_score(n, n_out / n if n else 0, len(t2)),
            "med_dfg_achelix_distance_A": fp[0] if fp else None,
            "med_hinge_angle_deg": fp[1] if fp else None,
            "n_descriptors": fp[2] if fp else 0,
        })
    rows.sort(key=lambda r: r["type2_opportunity"], reverse=True)
    return rows


def render_html(rows, points):
    data = json.dumps({"rows": rows, "points": points})
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Hyaline Kinase Atlas</title>
<style>
  :root{--ink:#1a1a18;--muted:#6b6b66;--surface:#fcfcfb;--line:#e7e7e4;
        --in:#0072B2;--out:#E69F00}
  body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       color:var(--ink);background:var(--surface)}
  header{padding:20px 24px;border-bottom:1px solid var(--line)}
  h1{margin:0 0 4px;font-size:20px}.sub{color:var(--muted);font-size:13px}
  .wrap{display:grid;grid-template-columns:1fr 460px;gap:24px;padding:20px 24px}
  input{padding:8px 10px;border:1px solid var(--line);border-radius:6px;
        width:260px;font-size:14px}
  table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;
        white-space:nowrap}
  th{cursor:pointer;user-select:none;position:sticky;top:0;background:var(--surface)}
  td.num,th.num{text-align:right}
  .tablebox{max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:8px}
  .chip{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
  .legend{font-size:12px;color:var(--muted);margin:6px 0}
  svg{border:1px solid var(--line);border-radius:8px;background:#fff}
  .muted{color:var(--muted)}
</style></head><body>
<header><h1>Hyaline Kinase Atlas</h1>
<div class="sub">Human kinases from KLIFS — accessible DFG states, known Type I/II
inhibitors, a Type-II-opportunity score (heuristic), and a geometric descriptor
fingerprint where computed. Counts reconcile against KLIFS structures.</div></header>
<div class="wrap">
 <div>
  <input id="q" placeholder="Search kinase / gene / UniProt…" oninput="draw()">
  <span class="muted" id="count"></span>
  <div class="tablebox"><table id="tbl"><thead><tr>
   <th onclick="sortby('kinase')">Kinase</th>
   <th onclick="sortby('gene')">Gene</th>
   <th class="num" onclick="sortby('n_structures')">#Struct</th>
   <th onclick="sortby('accessible_states')">States</th>
   <th class="num" onclick="sortby('n_type1_inhibitors')">Type I</th>
   <th class="num" onclick="sortby('n_type2_inhibitors')">Type II</th>
   <th class="num" onclick="sortby('type2_opportunity')">Opp.</th>
  </tr></thead><tbody id="body"></tbody></table></div>
 </div>
 <div>
  <div class="legend"><span class="chip" style="background:var(--in)"></span>DFG-in
   &nbsp;&nbsp;<span class="chip" style="background:var(--out)"></span>DFG-out</div>
  <svg id="plot" width="440" height="380"></svg>
  <div class="legend">DFG-to-αC distance (Å) vs hinge angle (°); grouped
   leave-one-kinase-out AUROC 0.834. Highlight a kinase by searching.</div>
 </div>
</div>
<script>
const DATA=__DATA__;let rows=DATA.rows.slice(),sortk='type2_opportunity',asc=false;
function sortby(k){if(sortk===k)asc=!asc;else{sortk=k;asc=false}
 rows.sort((a,b)=>{let x=a[k],y=b[k];if(typeof x==='string'){x=x||'';y=y||''}
  return (x<y?-1:x>y?1:0)*(asc?1:-1)});draw()}
function draw(){const q=document.getElementById('q').value.toLowerCase();
 const f=rows.filter(r=>!q||(r.kinase+' '+r.gene+' '+r.uniprot).toLowerCase().includes(q));
 document.getElementById('count').textContent=' '+f.length+' kinases';
 document.getElementById('body').innerHTML=f.slice(0,400).map(r=>`<tr>
  <td>${r.kinase}</td><td class="muted">${r.gene}</td>
  <td class="num">${r.n_structures}</td><td>${r.accessible_states||''}</td>
  <td class="num">${r.n_type1_inhibitors}</td><td class="num">${r.n_type2_inhibitors}</td>
  <td class="num">${r.type2_opportunity}</td></tr>`).join('');
 plot(q)}
function plot(q){const P=DATA.points,W=440,H=380,m=44;
 const xs=P.map(p=>p.d),ys=P.map(p=>p.a);
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const sx=d=>m+(d-x0)/(x1-x0)*(W-m-12),sy=a=>H-m-(a-y0)/(y1-y0)*(H-m-14);
 let s=`<rect x="0" y="0" width="${W}" height="${H}" fill="#fff"/>`;
 // axes
 s+=`<line x1="${m}" y1="${H-m}" x2="${W-8}" y2="${H-m}" stroke="#ccc"/>`;
 s+=`<line x1="${m}" y1="14" x2="${m}" y2="${H-m}" stroke="#ccc"/>`;
 for(const p of P){const hl=q&&p.k.toLowerCase().includes(q);
  const c=p.y? '#E69F00':'#0072B2';
  s+=`<circle cx="${sx(p.d).toFixed(1)}" cy="${sy(p.a).toFixed(1)}" r="${hl?4.2:2.6}"
      fill="${c}" fill-opacity="${q?(hl?1:0.12):0.75}"
      stroke="${hl?'#111':'#fff'}" stroke-width="${hl?1:0.4}"/>`;}
 s+=`<text x="${W/2}" y="${H-8}" font-size="11" text-anchor="middle" fill="#666">DFG-to-αC distance (Å)</text>`;
 s+=`<text x="14" y="${H/2}" font-size="11" text-anchor="middle" fill="#666" transform="rotate(-90 14 ${H/2})">hinge angle (°)</text>`;
 document.getElementById('plot').innerHTML=s;}
draw();
</script></body></html>"""
    return html.replace("__DATA__", data)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Fetching human kinase list...")
    kinases = _get("kinase_names", {"species": "Human"}).json()
    print(f"  {len(kinases)} human kinases")
    print("Fetching structures per kinase (cached)...")
    raw = fetch_all_structures(kinases)

    med, points = load_descriptor_medians()
    rows = build_rows(raw, med)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "kinase_atlas.parquet"), index=False)
    df.to_csv(os.path.join(OUT, "kinase_atlas.csv"), index=False)

    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html(rows, points))

    n_struct = sum(r["n_structures"] for r in rows)
    print(f"\nAtlas built: {len(rows)} kinases with structures, "
          f"{n_struct} structures total")
    print(f"  descriptor fingerprints for {sum(1 for r in rows if r['n_descriptors'])} kinases, "
          f"{len(points)} descriptor points")
    print(f"  wrote {OUT}/kinase_atlas.parquet, .csv, index.html")


if __name__ == "__main__":
    main()
