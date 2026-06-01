#!/usr/bin/env python3
"""
Generate an interactive HTML timeline visualization from session_results.json
using the vis-timeline (vis.js) library.

Each worker gets a swimlane row. Hovering a request highlights all requests
belonging to the same SID and shows detail in a panel below the timeline.
"""

import argparse
import json
import os


def fmt_ts(seconds_relative):
    """Format relative seconds as HH:MM:SS.mmm"""
    neg = seconds_relative < 0
    s = abs(seconds_relative)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    sign = "-" if neg else ""
    return f"{sign}{h:02d}:{m:02d}:{sec:06.3f}"


def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    workers = {}
    for entry in data:
        if entry["end_time"] is not None:
            wid = entry["workid"]
            if wid not in workers:
                workers[wid] = []
            workers[wid].append(entry)

    for wid in workers:
        workers[wid].sort(key=lambda e: e["start_time"])

    return workers


def generate_html(workers):
    all_start, all_end = [], []
    for entries in workers.values():
        for e in entries:
            all_start.append(e["start_time"])
            all_end.append(e["end_time"])

    t_min = min(all_start)
    t_max = max(all_end)
    span = t_max - t_min
    t_min -= span * 0.02
    t_max += span * 0.02

    MS = 1000.0
    t0 = t_min
    t_min_ms = t_min * MS
    t_max_ms = t_max * MS

    sorted_workers = sorted(workers.keys())

    groups_js = []
    for wid in sorted_workers:
        label = f"Trigger {wid + 1:02d}"
        groups_js.append(f'    {{id: {wid}, content: "{label}"}}')
    groups_str = ",\n".join(groups_js)

    items_js = []
    for wid in sorted_workers:
        for e in workers[wid]:
            start = e["start_time"] * MS
            end = e["end_time"] * MS
            dur = (end - start) / MS

            if dur < 1:
                color = "#58a6ff"
            elif dur < 5:
                color = "#3fb950"
            elif dur < 20:
                color = "#d29922"
            else:
                color = "#f85149"

            sid = (
                e.get("sid", "?")
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("'", "\\'")
            )
            dep_idx = e["dep_idx"]
            workid = e["workid"]

            start_rel = e["start_time"] - t0
            end_rel = e["end_time"] - t0
            start_fmt = fmt_ts(start_rel)
            end_fmt = fmt_ts(end_rel)
            short_sid = os.path.basename(e.get("sid", "?"))
            try:
                detail_json = (
                    '{"dep_idx":%d,"workid":%d,"sid":"%s",'
                    '"start_time":"%s","end_time":"%s",'
                    '"start_raw":%.6f,"end_raw":%.6f,'
                    '"first_token_time":%.6f,"completion_time":%.6f,'
                    '"prompt_tokens":%d,"cached_tokens":%d,'
                    '"reasoning_tokens":%d,"completion_tokens":%d,'
                    '"tok_model_output_len":%d,"is_target":"%s"}'
                    % (
                        dep_idx,
                        workid,
                        short_sid,
                        start_fmt,
                        end_fmt,
                        e["start_time"],
                        e["end_time"],
                        e["first_token_time"] or 0,
                        e["completion_time"] or 0,
                        e["prompt_tokens"],
                        e["cached_tokens"],
                        e["reasoning_tokens"],
                        e["completion_tokens"],
                        e["tok_model_output_len"],
                        str(e["is_target"]).lower(),
                    )
                )
            except:
                print(e)
                raise
            shorts_sid = (
                short_sid[-8:] + "-" + str(e.get("pos")) + "-" + str(e.get("end_at"))
            )
            if e["role"] == "user":
                shorts_sid = "👨‍🔧" + shorts_sid
            orig_style = (
                f"background-color:{color};color:#fff;font-size:11px;"
                f"border:1px solid rgba(255,255,255,0.2);border-radius:3px;"
            )

            items_js.append(
                f"    {{id: {dep_idx}, group: {wid}, "
                f"start: {start}, end: {end}, "
                f'content: "{shorts_sid}", '
                f'sid: "{sid}", '
                f"origStyle: '{orig_style}', "
                f'style: "{orig_style}", '
                f'title: "idx={shorts_sid} | dur={dur:.3f}s", '
                f"detail: '{detail_json}'}}"
            )

    items_str = ",\n".join(items_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session Timeline - Per Worker (vis.js)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-timeline/7.7.3/vis-timeline-graph2d.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis-timeline/7.7.3/vis-timeline-graph2d.min.css" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
}}
h1 {{ text-align: center; margin-bottom: 4px; font-size: 20px; color: #f0f6fc; }}
.subtitle {{ text-align: center; margin-bottom: 12px; font-size: 13px; color: #8b949e; }}
.legend {{ display: flex; gap: 16px; margin: 8px 0 14px; font-size: 12px;
          align-items: center; flex-wrap: wrap; justify-content: center; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px;
                 border: 1px solid rgba(255,255,255,0.25); }}
#timeline {{ border: 1px solid #30363d; border-radius: 6px 6px 0 0; background: #161b22; }}

.vis-timeline {{ border: none !important; }}
.vis-panel {{ border-color: #30363d !important; }}
.vis-labelset .vis-label {{
    background: #1c2129 !important; color: #e6edf3 !important;
    border-color: #30363d !important; font-weight: 600 !important; font-size: 13px !important;
}}
.vis-foreground .vis-group {{ border-color: #21262d !important; }}
.vis-time-axis .vis-grid.vis-minor,
.vis-time-axis .vis-grid.vis-major {{ border-color: #21262d !important; }}
.vis-time-axis .vis-text {{ color: #8b949e !important; font-size: 11px !important; }}
.vis-item {{ cursor: pointer !important; border-radius: 4px !important; }}

.vis-item.sid-highlight {{
    box-shadow: 0 0 0 3px #ff0000, 0 0 16px rgba(255,0,0,0.6) !important;
    z-index: 20 !important;
}}

/* ---- detail panel below timeline ---- */
#detail-panel {{
    border: 1px solid #30363d; border-top: none; border-radius: 0 0 6px 6px;
    background: #161b22; min-height: 60px; padding: 12px 16px;
    display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start;
    font-size: 13px;
}}
#detail-panel .detail-col {{
    display: flex; flex-direction: column; gap: 2px;
}}
#detail-panel .detail-label {{ color: #8b949e; font-size: 11px; }}
#detail-panel .detail-value {{ color: #e6edf3; }}
#detail-panel .detail-empty {{ color: #484f58; padding: 8px 0; font-style: italic; }}
</style>
</head>
<body>
<h1>&#128202; Session Request Timeline</h1>
<p class="subtitle">
    Hover a request box &mdash; same-SID items are highlighted in
    <span style="color:#ffa500;">orange</span>
    &bull; Detail shown below &bull; Ctrl+scroll to zoom &bull; Drag to pan
</p>
<div class="legend">
    <span style="color:#8b949e;">Duration:</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#58a6ff;"></span> &lt;1s</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#3fb950;"></span> 1-5s</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#d29922;"></span> 5-20s</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#f85149;"></span> &gt;20s</span>
</div>
<div id="timeline"></div>
<div id="detail-panel">
    <span class="detail-empty">Hover over a request box to see details</span>
</div>

<script>
var groups = new vis.DataSet([
{groups_str}
]);

var items = new vis.DataSet([
{items_str}
]);

var options = {{
    start: {t_min_ms},
    end: {t_max_ms},
    height: '62vh',
    groupOrder: 'id',
    stack: false,
    showCurrentTime: false,
    zoomable: true,
    moveable: true,
    horizontalScroll: true,
    verticalScroll: true,
    zoomKey: 'ctrlKey',
    orientation: {{ axis: 'top', item: 'top' }},
    timeAxis: {{ scale: 'second', step: 30 }},
    margin: {{ axis: 10, item: {{ vertical: 6 }} }},
    tooltip: {{ followMouse: true, overflowMethod: 'flip' }},
}};

var container = document.getElementById('timeline');
var timeline = new vis.Timeline(container, items, groups, options);

var highlightedIds = [];

function highlightBySid(sid) {{
    clearHighlight();
    var all = items.get();
    for (var i = 0; i < all.length; i++) {{
        if (all[i].sid === sid) {{
            highlightedIds.push(all[i].id);
            items.update({{ id: all[i].id, className: 'sid-highlight' }});
        }}
    }}
}}

function clearHighlight() {{
    for (var i = 0; i < highlightedIds.length; i++) {{
        items.update({{ id: highlightedIds[i], className: '' }});
    }}
    highlightedIds = [];
}}

function updateDetailPanel(item) {{
    var panel = document.getElementById('detail-panel');
    if (!item || !item.detail) {{
        panel.innerHTML = '<span class="detail-empty">Hover over a request box to see details</span>';
        return;
    }}
    if (item.sid) highlightBySid(item.sid);
    var d = JSON.parse(item.detail);
    var dur = (d.end_raw - d.start_raw).toFixed(4);
    panel.innerHTML =
        '<div class="detail-col">' +
        '<span class="detail-label">Dep Index</span><span class="detail-value">' + d.dep_idx + '</span>' +
        '<span class="detail-label">Worker ID</span><span class="detail-value">' + d.workid + '</span>' +
        '<span class="detail-label">Is Target</span><span class="detail-value">' + d.is_target + '</span>' +
        '</div>' +
        '<div class="detail-col">' +
        '<span class="detail-label">Session ID</span><span class="detail-value" style="max-width:420px;overflow:hidden;text-overflow:ellipsis;">' + d.sid + '</span>' +
        '<span class="detail-label">Start &rarr; End</span><span class="detail-value">' + d.start_time + ' &rarr; ' + d.end_time + '</span>' +
        '<span class="detail-label">Duration</span><span class="detail-value">' + dur + ' s</span>' +
        '</div>' +
        '<div class="detail-col">' +
        '<span class="detail-label">First Token</span><span class="detail-value">' + d.first_token_time + ' s</span>' +
        '<span class="detail-label">Completion</span><span class="detail-value">' + d.completion_time + ' s</span>' +
        '<span class="detail-label">Token Output Len</span><span class="detail-value">' + d.tok_model_output_len + '</span>' +
        '</div>' +
        '<div class="detail-col">' +
        '<span class="detail-label">Prompt Tokens</span><span class="detail-value">' + d.prompt_tokens + '</span>' +
        '<span class="detail-label">Cached Tokens</span><span class="detail-value">' + d.cached_tokens + '</span>' +
        '<span class="detail-label">Reasoning Tokens</span><span class="detail-value">' + d.reasoning_tokens + '</span>' +
        '<span class="detail-label">Completion Tokens</span><span class="detail-value">' + d.completion_tokens + '</span>' +
        '</div>';
}}

timeline.on('click', function (props) {{
    if (props.item) {{
        var item = items.get(props.item);
        updateDetailPanel(item);
    }}
}});
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML timeline from trace_results.json"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to trace_results.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: timeline.html in same dir as input)",
    )
    args = parser.parse_args()

    json_path = args.input
    output_path = args.output or os.path.join(
        os.path.dirname(json_path), "timeline.html"
    )

    print(f"Loading data from: {json_path}")
    workers = load_data(json_path)

    total_requests = sum(len(v) for v in workers.values())
    print(f"Loaded {total_requests} requests across {len(workers)} workers.")
    for wid in sorted(workers.keys()):
        print(f"  work{wid + 1:02d} (id={wid}): {len(workers[wid])} requests")

    html = generate_html(workers)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML timeline written to: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")
    print("Open it in a browser to explore the timeline.")


if __name__ == "__main__":
    main()
