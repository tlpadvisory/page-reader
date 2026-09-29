"""
page-reader — reads the Yes/No tick boxes on a Queensland Form 2 (Seller
Disclosure Statement) directly from the rendered PDF pages.

WHY THIS EXISTS
The Form 2's tick states are not part of the text that OpenAI's file search can
see: each item reads only as "<printed statement> Yes No", with nothing showing
which box is ticked, so a language model can only guess. This service does not
guess. It finds every printed "Yes" / "No" label using the PDF's text layer,
looks at the small box drawn immediately to the left of it, and checks whether
that box contains a tick mark. It runs on Railway because rendering pages is
CPU-heavy and Supabase Edge Functions only allow ~200ms of CPU per invocation.

ENDPOINTS
  GET  /health        -> {"ok": true}
  POST /form2-ticks   -> body = the raw PDF bytes (Content-Type: application/pdf)
                         header  x-api-key: <PAGE_READER_KEY>
                         optional query ?debug=1 to include per-box details
"""
import hmac
import os
import re
import time
from collections import deque

import pymupdf
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="page-reader")

API_KEY = os.environ.get("PAGE_READER_KEY", "")

ZOOM = 300 / 72.0          # render tiny clips at 300 dpi so thin box outlines survive
INK = 200                  # grey level (0=black, 255=white) below which a pixel counts as ink
LABELS = ("Yes", "No")
GLYPH_CHECKED = set("\u2611\u2612\u2713\u2714")   # ☑ ☒ ✓ ✔
GLYPH_EMPTY = set("\u2610\u25a1\u25a2")            # ☐ □ ▢
STOP_FIRST_WORDS = {"if", "or", "note", "note\u2014", "note-"}
# a glyph box and its label can come out of the text layer as one fused word, e.g. "\u2610Yes"
FUSED_LABEL = re.compile("^([\u2610\u2611\u2612\u25a1\u25a2\u2713\u2714])\\s*(Yes|No)$")


# --------------------------------------------------------------------------
# Finding the Form 2 pages
# --------------------------------------------------------------------------
def find_form2_pages(doc):
    """Every page whose text carries the Form 2 footer wording."""
    pages = []
    for i in range(len(doc)):
        try:
            text = doc[i].get_text("text")
        except Exception:
            continue
        if re.search(r"seller\s+disclosure\s+statement", text, re.I) and re.search(r"\bform\s*2\b", text, re.I):
            pages.append(i)
    return pages


# --------------------------------------------------------------------------
# Finding the printed Yes / No labels
# --------------------------------------------------------------------------
def find_labels(words):
    """
    A tick-box label is a standalone word 'Yes' or 'No' with a gap on its left
    (where the box sits) and nothing running on after it. A 'Yes'/'No' inside
    running text ('If Yes, ...', 'No further information...') has ordinary word
    gaps and is skipped.

    Ticked form-field checkboxes often leave a stray single-character 'word' in
    the text layer inside the box (e.g. a '3' from a check-mark font). That is
    the box, not running text, so it must not cause the label to be dropped. If
    the box is a proper glyph (☐ / ☑) its state is read straight from the
    character; otherwise the state is read from the rendered pixels.
    """
    labels = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        fused = FUSED_LABEL.match(text)
        if fused:
            fused_glyph, text = fused.group(1), fused.group(2)
        elif text in LABELS:
            fused_glyph = None
        else:
            continue
        yc = (y0 + y1) / 2
        prev = None
        nxt_gap = None
        for o in words:
            if o is w:
                continue
            oyc = (o[1] + o[3]) / 2
            if abs(oyc - yc) > 3:
                continue
            if o[2] <= x0 + 0.5:
                if prev is None or o[2] > prev[2]:
                    prev = o
            elif o[0] >= x1 - 0.5:
                g = o[0] - x1
                if nxt_gap is None or g < nxt_gap:
                    nxt_gap = g
        if nxt_gap is not None and nxt_gap < 8:
            continue  # running text such as "No further information"
        glyph_state = None
        box_word = None
        if fused_glyph is not None:
            glyph_state = fused_glyph in GLYPH_CHECKED
        elif prev is not None:
            ptext = prev[4]
            gap = x0 - prev[2]
            if ptext and len(ptext) <= 2 and ptext[-1] in GLYPH_CHECKED:
                glyph_state, box_word = True, prev
            elif ptext and len(ptext) <= 2 and ptext[-1] in GLYPH_EMPTY:
                glyph_state, box_word = False, prev
            elif len(ptext) == 1 and not ptext.isalpha() and gap <= 14:
                box_word = prev   # stray check-mark character inside the box: read pixels
            elif gap < 8:
                continue  # running text such as "If Yes"
        # where does real text end on the left? the box lives in the space between
        left_text_x1 = None
        for o in words:
            if o is w or (box_word is not None and o is box_word):
                continue
            oyc = (o[1] + o[3]) / 2
            if abs(oyc - yc) <= 3 and o[2] <= x0 + 0.5:
                if left_text_x1 is None or o[2] > left_text_x1:
                    left_text_x1 = o[2]
        labels.append({
            "text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "yc": yc,
            "glyph": glyph_state,
            "left_text_x1": left_text_x1,
            "box_word_pos": (round(box_word[0], 1), round(box_word[1], 1)) if box_word else None,
        })
    return labels


# --------------------------------------------------------------------------
# Reading the box next to a label
# --------------------------------------------------------------------------
def read_box(page, label):
    """
    Renders a small clip just left of the label, finds the square outline, and
    returns how much ink sits inside it (a tick mark => high, empty box => ~0).
    Returns None if no box-shaped outline can be found.
    """
    # The box sits in the gap between the end of the statement text and the label.
    # Use that whole gap (capped at 40pt) so the box is fully inside the window
    # regardless of how far the form places it from the label.
    left = label["x0"] - 40
    if label.get("left_text_x1") is not None:
        left = max(left, label["left_text_x1"] + 0.5)
    clip = pymupdf.Rect(left, label["y0"] - 3, label["x0"] - 0.5, label["y1"] + 3) & page.rect
    if clip.is_empty or clip.width < 4:
        return None
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=clip,
                          colorspace=pymupdf.csGRAY, alpha=False)
    w, h, stride = pix.width, pix.height, pix.stride
    data = pix.samples
    dark = [[data[y * stride + x] < INK for x in range(w)] for y in range(h)]

    seen = [[False] * w for _ in range(h)]
    best = None
    for sy in range(h):
        for sx in range(w):
            if not dark[sy][sx] or seen[sy][sx]:
                continue
            q = deque([(sx, sy)])
            seen[sy][sx] = True
            minx = maxx = sx
            miny = maxy = sy
            n = 0
            while q:
                cx, cy = q.popleft()
                n += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and dark[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            q.append((nx, ny))
            bw, bh = maxx - minx + 1, maxy - miny + 1
            bw_pt, bh_pt = bw / ZOOM, bh / ZOOM
            aspect = bw / bh if bh else 0
            fill = n / (bw * bh)
            # a checkbox: roughly square, 5-15pt, mostly hollow
            if 5 <= bw_pt <= 15 and 5 <= bh_pt <= 15 and 0.75 <= aspect <= 1.33 and fill < 0.7:
                area = bw * bh
                if best is None or area > best[4]:
                    best = (minx, miny, maxx, maxy, area)
    if best is None:
        return None
    minx, miny, maxx, maxy, _ = best
    bw, bh = maxx - minx + 1, maxy - miny + 1
    inset = max(2, round(0.22 * min(bw, bh)))
    ix0, iy0, ix1, iy1 = minx + inset, miny + inset, maxx - inset, maxy - inset
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    total = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    ink = sum(1 for y in range(iy0, iy1 + 1) for x in range(ix0, ix1 + 1) if dark[y][x])
    return ink / total


def choose_threshold(ratios):
    """Empty boxes cluster near 0, ticked boxes well above — split at the biggest gap."""
    r = sorted(ratios)
    if len(r) >= 4:
        best_gap, best_mid = 0.0, None
        for a, b in zip(r, r[1:]):
            if b - a > best_gap:
                best_gap, best_mid = b - a, (a + b) / 2
        if best_gap >= 0.05 and best_mid is not None and best_mid < 0.30:
            return best_mid
    return 0.07


# --------------------------------------------------------------------------
# Recovering the printed statement that belongs to a row of labels
# --------------------------------------------------------------------------
def build_lines(words):
    lines = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for key, ws in lines.items():
        ws.sort(key=lambda t: t[0])
        out.append({
            "key": key, "words": ws,
            "y0": min(t[1] for t in ws), "y1": max(t[3] for t in ws),
        })
    return out


def statement_for(group, words, lines, label_pos):
    lx = min(l["x0"] for l in group)
    yc = sum(l["yc"] for l in group) / len(group)
    cands = [w for w in words
             if abs((w[1] + w[3]) / 2 - yc) <= 4 and w[2] <= lx - 6
             and w[4] not in LABELS
             and (round(w[0], 1), round(w[1], 1)) not in label_pos]
    if not cands:
        return ""
    nearest = max(cands, key=lambda w: w[2])
    key = (nearest[5], nearest[6])
    first_ln = next((ln for ln in lines if ln["key"] == key), None)
    if first_ln is None:
        return ""

    def usable(ln):
        return [w for w in ln["words"] if (round(w[0], 1), round(w[1], 1)) not in label_pos]

    # First line: split at big horizontal gaps (a left-hand heading column shares
    # the baseline but sits far from the statement) and keep the segment that
    # holds the word nearest the boxes.
    segments, cur = [], []
    for w in usable(first_ln):
        if cur and w[0] - cur[-1][2] > 25:
            segments.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segments.append(cur)
    seg = next((sg for sg in segments if any((w[0], w[1]) == (nearest[0], nearest[1]) for w in sg)), None)
    if seg is None:
        return ""
    stmt_x0 = seg[0][0]
    texts = [" ".join(w[4] for w in seg)]
    seg_y0 = min(w[1] for w in seg)
    seg_y1 = max(w[3] for w in seg)
    prev_c = (seg_y0 + seg_y1) / 2
    prev_h = max(seg_y1 - seg_y0, 6)

    # Continuation lines: the wrapped statement continues in the same left-aligned
    # column, closely spaced below, and never on a row that has its own boxes.
    # Compare line CENTRES — the boxes of consecutive lines of one paragraph
    # overlap slightly, so comparing top/bottom edges skips real continuation lines.
    for _ in range(3):
        best, best_c = None, None
        for ln in lines:
            ws = usable(ln)
            if not ws or abs(ws[0][0] - stmt_x0) > 4:
                continue
            c = (ln["y0"] + ln["y1"]) / 2
            if c <= prev_c + 3:
                continue
            if best is None or c < best_c:
                best, best_c = ln, c
        if best is None:
            break
        if best_c - prev_c > 1.5 * prev_h:
            break
        if any((round(w[0], 1), round(w[1], 1)) in label_pos for w in best["words"]):
            break
        ws = usable(best)
        first = re.sub(r"[^\w\u2014-]", "", ws[0][4]).lower()
        if first in STOP_FIRST_WORDS:
            break
        texts.append(" ".join(w[4] for w in ws))
        prev_c = best_c
        prev_h = max(best["y1"] - best["y0"], 6)
    text = re.sub(r"\s+", " ", " ".join(texts)).strip()
    text = re.sub(r"\s+OR$", "", text)
    return text[:400]


# --------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------
def widget_stats(doc, page_indexes):
    """Diagnostics only: are the boxes real form-field checkboxes, and how many are on?"""
    total = checked = 0
    try:
        for pi in page_indexes:
            for w in (doc[pi].widgets() or []):
                if w.field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX:
                    total += 1
                    if w.field_value not in (False, None, "", "Off", "off", 0):
                        checked += 1
    except Exception:
        pass
    return total, checked


def process(data: bytes, debug: bool = False):
    t0 = time.time()
    doc = pymupdf.open(stream=data, filetype="pdf")
    if doc.needs_pass:
        raise ValueError("PDF is password protected")

    form2 = find_form2_pages(doc)
    per_page = []
    for pi in form2:
        page = doc[pi]
        if page.rotation:
            page.set_rotation(0)
        words = page.get_text("words")
        labels = find_labels(words)
        for lb in labels:
            if lb["glyph"] is not None:
                lb["ratio"], lb["source"] = None, "glyph"
            else:
                lb["ratio"] = read_box(page, lb)
                lb["source"] = "pixels" if lb["ratio"] is not None else None
        per_page.append((pi, page, words, labels))

    ratios = [lb["ratio"] for _, _, _, labels in per_page for lb in labels if lb["ratio"] is not None]
    threshold = choose_threshold(ratios)

    items = []
    unreadable_boxes = 0
    debug_labels = []
    for pi, page, words, labels in per_page:
        for lb in labels:
            if lb["glyph"] is not None:
                lb["state"] = lb["glyph"]
            elif lb["ratio"] is not None:
                lb["state"] = lb["ratio"] >= threshold
            else:
                lb["state"] = None
                unreadable_boxes += 1
            if debug:
                debug_labels.append({"page": pi + 1, "text": lb["text"], "yc": round(lb["yc"], 1),
                                     "x0": round(lb["x0"], 1), "ratio": lb["ratio"], "state": lb["state"],
                                     "source": lb["source"]})

        label_pos = {(round(l["x0"], 1), round(l["y0"], 1)) for l in labels}
        label_pos |= {l["box_word_pos"] for l in labels if l["box_word_pos"]}
        lines = build_lines(words)
        rows = []
        for lb in sorted(labels, key=lambda l: (l["yc"], l["x0"])):
            if rows and abs(lb["yc"] - rows[-1][0]["yc"]) <= 4:
                rows[-1].append(lb)
            else:
                rows.append([lb])
        for group in rows:
            yes_l = [l for l in group if l["text"] == "Yes"]
            no_l = [l for l in group if l["text"] == "No"]
            ambiguous = len(yes_l) > 1 or len(no_l) > 1
            yes = yes_l[0]["state"] if len(yes_l) == 1 else None
            no = no_l[0]["state"] if len(no_l) == 1 else None
            if (yes_l and yes is None) or (no_l and no is None):
                ambiguous = True   # a box we could not read
            if yes is True and no is True:
                ambiguous = True   # both ticked: cannot trust either
            items.append({
                "page": pi + 1,
                "y": round(group[0]["yc"], 1),
                "statement": statement_for(group, words, lines, label_pos),
                "yes": yes,
                "no": no,
                "ambiguous": ambiguous,
            })

    w_total, w_checked = widget_stats(doc, form2)
    result = {
        "ok": True,
        "items": items,
        "diagnostics": {
            "pdf_pages": len(doc),
            "form2_pages": [p + 1 for p in form2],
            "labels_found": sum(len(labels) for _, _, _, labels in per_page),
            "boxes_unreadable": unreadable_boxes,
            "threshold": round(threshold, 3),
            "ratios_sorted": [round(r, 3) for r in sorted(ratios)],
            "checkbox_widgets": w_total,
            "checkbox_widgets_checked": w_checked,
            "elapsed_ms": int((time.time() - t0) * 1000),
        },
    }
    if debug:
        result["labels"] = debug_labels
    doc.close()
    return result


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/form2-ticks")
async def form2_ticks(request: Request, x_api_key: str = Header(default=""), debug: int = 0):
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="unauthorized")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        return await run_in_threadpool(process, data, bool(debug))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # unreadable / corrupt PDF etc.
        raise HTTPException(status_code=422, detail=f"could not process PDF: {e}")
