import warnings
warnings.filterwarnings("ignore")
import os

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import stable_whisper
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import urllib.parse
import re
import unicodedata
import tempfile
import time
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

model = None

AUDIO_EXTS   = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff"}
MIN_DURATION = 0.3


def is_audio_only(filepath):
    return os.path.splitext(filepath)[1].lower() in AUDIO_EXTS


def load_model():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading stable-whisper medium on {device}...")
    model = stable_whisper.load_model("medium", device=device)
    print("Model loaded")


# ── GOOGLE TRANSLATE GTX ──────────────────────────────────────────────────────

GTX_URL = "https://translate.googleapis.com/translate_a/single"
GTX_SEP = "\n<<<SEP>>>\n"


def _gtx_parse(data):
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("Response GTX tidak valid.")
    out = []
    for item in data[0]:
        if isinstance(item, list) and item and isinstance(item[0], str):
            out.append(item[0])
    return "".join(out)


def _gtx_batch(texts, source="ko", target="id", timeout=30):
    joined = GTX_SEP.join(texts)
    params = {"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": joined}
    url = GTX_URL + "?" + urlencode(params)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    translated = _gtx_parse(r.json())
    parts = translated.split(GTX_SEP)
    if len(parts) != len(texts):
        raise ValueError(f"Jumlah hasil tidak cocok: expected={len(texts)} got={len(parts)}")
    return parts


def translate_lines(lines, batch_size=30):
    """Translate Korean -> Indonesian via Google Translate GTX (batched)."""
    results = [""] * len(lines)
    for start in range(0, len(lines), batch_size):
        end   = min(start + batch_size, len(lines))
        batch = lines[start:end]
        last_err = None
        for attempt in range(1, 4):
            try:
                out = _gtx_batch(batch)
                results[start:end] = out
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  GTX retry {attempt}/3: {e}")
                time.sleep(attempt)
        if last_err is not None:
            results[start:end] = batch
        time.sleep(0.3)
    return results


def seconds_to_srt_time(s):
    s  = max(0.0, s)
    ms = int((s % 1) * 1000)
    ss = int(s) % 60
    m  = (int(s) // 60) % 60
    h  = int(s) // 3600
    return f"{h:02}:{m:02}:{ss:02},{ms:03}"


def make_path_url(media_file):
    abs_path = os.path.abspath(media_file).replace("\\", "/")
    if len(abs_path) >= 2 and abs_path[1] == ":":
        abs_path = abs_path[0].upper() + abs_path[1:]
    parts   = abs_path.split("/")
    encoded = "/".join(urllib.parse.quote(p, safe=":") for p in parts)
    return "file:///" + encoded


def build_rate(parent, timebase="30", ntsc="FALSE"):
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = timebase
    ET.SubElement(r, "ntsc").text     = ntsc


def clean_line(text):
    text = text.strip()
    text = re.sub(r'[.,\s]+$', '', text)
    return text.strip()


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


# ── AUDIO MERGE ───────────────────────────────────────────────────────────────

def merge_audio_files(file_list):
    import subprocess
    tmp_dir     = tempfile.mkdtemp()
    merged_path = os.path.join(tmp_dir, "merged.wav")

    if len(file_list) == 1:
        cmd = ["ffmpeg", "-y", "-i", file_list[0], "-ar", "16000", "-ac", "1", merged_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return merged_path, tmp_dir

    concat_list_path = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for fp in file_list:
            safe = fp.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-ar", "16000", "-ac", "1",
        merged_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return merged_path, tmp_dir


def get_file_duration(fp):
    import subprocess, json
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", fp]
    out = subprocess.check_output(cmd)
    return float(json.loads(out)["format"]["duration"])


# ── FORCED ALIGNMENT ──────────────────────────────────────────────────────────

def align_subtitles(audio_files, subtitle_lines, progress_cb=None):
    global model
    if model is None:
        load_model()

    if progress_cb:
        progress_cb(10, "Merging audio files..." if len(audio_files) > 1 else "Preparing audio...")

    merged_path, tmp_dir = merge_audio_files(audio_files)

    try:
        full_text = " ".join(subtitle_lines)
        print(f"Aligning {len(subtitle_lines)} lines...")

        if progress_cb:
            progress_cb(25, "Running forced alignment (stable-whisper)...")

        result = model.align(merged_path, full_text, language="ko")

        words = []
        for seg in result.segments:
            if not hasattr(seg, "words") or seg.words is None:
                continue
            for w in seg.words:
                word = w.word.strip()
                if word:
                    words.append({
                        "word":  word,
                        "norm":  normalize(word),
                        "start": w.start,
                        "end":   w.end,
                    })

        if not words:
            raise ValueError("stable_whisper returned no word timestamps.")

        print(f"Got {len(words)} word timestamps")
        return map_lines_to_words(subtitle_lines, words)

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def map_lines_to_words(lines, words):
    results   = []
    n_words   = len(words)
    cursor    = 0
    total_dur = words[-1]["end"] if words else 0

    for i, line in enumerate(lines):
        if cursor >= n_words:
            prev_end  = results[-1]["end"] if results else 0
            remaining = len(lines) - i
            slot      = (total_dur - prev_end) / max(remaining, 1)
            t_start   = prev_end
            t_end     = min(t_start + slot, total_dur)
            results.append({"text": line, "start": t_start, "end": t_end})
            continue

        tokens = [normalize(w) for w in line.split() if normalize(w)]
        if not tokens:
            prev_end = results[-1]["end"] if results else 0
            results.append({"text": line, "start": prev_end,
                            "end": prev_end + MIN_DURATION})
            continue

        start_idx   = cursor
        for ci in range(cursor, min(cursor + len(tokens) * 3 + 10, n_words)):
            wn = normalize(words[ci]["word"])
            tn = tokens[0]
            if wn == tn or tn in wn or wn in tn:
                start_idx = ci
                break

        end_idx = min(start_idx + len(tokens) - 1, n_words - 1)
        t_start = words[start_idx]["start"]
        t_end   = words[end_idx]["end"]

        if t_end - t_start < MIN_DURATION:
            t_end = t_start + MIN_DURATION

        results.append({"text": line, "start": t_start, "end": t_end})
        cursor = end_idx + 1

    for i in range(len(results) - 1):
        results[i]["end"] = results[i + 1]["start"]

    return results


# ── SRT ───────────────────────────────────────────────────────────────────────

def create_srt(output_path, timed_lines):
    out_lines = []
    for i, entry in enumerate(timed_lines, start=1):
        out_lines.append(str(i))
        out_lines.append(
            f"{seconds_to_srt_time(entry['start'])} --> {seconds_to_srt_time(entry['end'])}")
        out_lines.append(entry["text"])
        out_lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"SRT saved -> {output_path}")
    return output_path


def create_srt_id(output_path, timed_lines, id_texts):
    out_lines = []
    for i, (entry, id_text) in enumerate(zip(timed_lines, id_texts), start=1):
        out_lines.append(str(i))
        out_lines.append(
            f"{seconds_to_srt_time(entry['start'])} --> {seconds_to_srt_time(entry['end'])}")
        out_lines.append(id_text)
        out_lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"SRT (ID) saved -> {output_path}")
    return output_path


# ── XML ───────────────────────────────────────────────────────────────────────

def create_xml(output_path, audio_files, timed_lines, total_audio_duration, mode="full"):
    import subprocess, json

    fps          = 30
    audio_only   = all(is_audio_only(f) for f in audio_files)
    total_frames = int(total_audio_duration * fps)

    file_durations = [get_file_duration(f) for f in audio_files]

    xmeml    = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence", id="sequence-1")
    ET.SubElement(sequence, "name").text = "Whisper Timeline"
    build_rate(sequence)
    ET.SubElement(sequence, "duration").text = str(total_frames)

    tc = ET.SubElement(sequence, "timecode")
    build_rate(tc)
    ET.SubElement(tc, "string").text        = "00:00:00:00"
    ET.SubElement(tc, "frame").text         = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    media = ET.SubElement(sequence, "media")

    if not audio_only:
        video    = ET.SubElement(media, "video")
        v_format = ET.SubElement(video, "format")
        v_sc     = ET.SubElement(v_format, "samplecharacteristics")
        build_rate(v_sc)
        ET.SubElement(v_sc, "width").text  = "1920"
        ET.SubElement(v_sc, "height").text = "1080"
        v_track = ET.SubElement(video, "track")

    audio_el = ET.SubElement(media, "audio")
    a_format = ET.SubElement(audio_el, "format")
    a_sc     = ET.SubElement(a_format, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text      = "16"
    ET.SubElement(a_sc, "samplerate").text = "48000"
    a_track = ET.SubElement(audio_el, "track")

    parent_track = a_track if audio_only else v_track

    def make_file_node(parent, file_path, file_id, frames):
        fn = ET.SubElement(parent, "file", id=file_id)
        ET.SubElement(fn, "name").text     = os.path.basename(file_path)
        ET.SubElement(fn, "pathurl").text  = make_path_url(file_path)
        ET.SubElement(fn, "duration").text = str(frames)
        build_rate(fn)
        fm = ET.SubElement(fn, "media")
        if not audio_only:
            fv  = ET.SubElement(fm, "video")
            fvs = ET.SubElement(fv, "samplecharacteristics")
            build_rate(fvs)
            ET.SubElement(fvs, "width").text  = "1920"
            ET.SubElement(fvs, "height").text = "1080"
        fa  = ET.SubElement(fm, "audio")
        fas = ET.SubElement(fa, "samplecharacteristics")
        ET.SubElement(fas, "depth").text      = "16"
        ET.SubElement(fas, "samplerate").text = "48000"

    if mode == "full":
        timeline_pos = 0
        for fi, (fp, dur) in enumerate(zip(audio_files, file_durations)):
            file_id     = f"masterclip-{fi + 1}"
            file_frames = int(dur * fps)
            tl_start    = timeline_pos
            tl_end      = timeline_pos + file_frames

            clip = ET.SubElement(parent_track, "clipitem", id=f"clipitem-{fi + 1}")
            ET.SubElement(clip, "name").text     = os.path.basename(fp)
            ET.SubElement(clip, "enabled").text  = "TRUE"
            ET.SubElement(clip, "duration").text = str(file_frames)
            build_rate(clip)
            ET.SubElement(clip, "start").text = str(tl_start)
            ET.SubElement(clip, "end").text   = str(tl_end)
            ET.SubElement(clip, "in").text    = "0"
            ET.SubElement(clip, "out").text   = str(file_frames)
            make_file_node(clip, fp, file_id, file_frames)

            file_start_sec = timeline_pos / fps
            file_end_sec   = tl_end / fps
            for entry in timed_lines:
                if entry["start"] >= file_start_sec and entry["start"] < file_end_sec:
                    text = entry.get("text", "").strip()
                    if text:
                        local_start = int((entry["start"] - file_start_sec) * fps)
                        local_end   = int((entry["end"]   - file_start_sec) * fps)
                        marker = ET.SubElement(clip, "marker")
                        ET.SubElement(marker, "name").text    = text[:60]
                        ET.SubElement(marker, "comment").text = text
                        ET.SubElement(marker, "in").text      = str(local_start)
                        ET.SubElement(marker, "out").text     = str(local_end)

            timeline_pos = tl_end

    else:
        file_offsets = []
        offset = 0.0
        for fi, (fp, dur) in enumerate(zip(audio_files, file_durations)):
            file_offsets.append((offset, offset + dur, fp, f"masterclip-{fi + 1}"))
            offset += dur

        seen_file_ids = set()
        last_end = 0

        for i, entry in enumerate(timed_lines):
            start = int(entry["start"] * fps)
            end   = int(entry["end"]   * fps)
            if start < last_end:
                start = last_end
            if end <= start:
                end = start + 1

            ref_fp     = audio_files[0]
            ref_fid    = "masterclip-1"
            ref_offset = 0.0
            ref_dur    = file_durations[0]
            for (fs, fe, fp, fid) in file_offsets:
                if entry["start"] >= fs and entry["start"] < fe:
                    ref_fp     = fp
                    ref_fid    = fid
                    ref_offset = fs
                    ref_dur    = fe - fs
                    break

            local_start = int((entry["start"] - ref_offset) * fps)
            local_end   = int((entry["end"]   - ref_offset) * fps)
            ref_frames  = int(ref_dur * fps)

            clip = ET.SubElement(parent_track, "clipitem", id=f"clipitem-{i + 1}")
            ET.SubElement(clip, "name").text     = os.path.basename(ref_fp)
            ET.SubElement(clip, "enabled").text  = "TRUE"
            ET.SubElement(clip, "duration").text = str(end - start)
            build_rate(clip)
            ET.SubElement(clip, "start").text = str(start)
            ET.SubElement(clip, "end").text   = str(end)
            ET.SubElement(clip, "in").text    = str(local_start)
            ET.SubElement(clip, "out").text   = str(local_end)

            if ref_fid not in seen_file_ids:
                make_file_node(clip, ref_fp, ref_fid, ref_frames)
                seen_file_ids.add(ref_fid)
            else:
                ET.SubElement(clip, "file", id=ref_fid)

            text = entry.get("text", "").strip()
            if text:
                marker = ET.SubElement(clip, "marker")
                ET.SubElement(marker, "name").text    = text[:60]
                ET.SubElement(marker, "comment").text = text
                ET.SubElement(marker, "in").text      = str(local_start)
                ET.SubElement(marker, "out").text     = str(local_end)

            last_end = end

    raw_str      = ET.tostring(xmeml, encoding="unicode")
    pretty       = minidom.parseString(raw_str).toprettyxml(indent="  ", encoding=None)
    xml_lines    = pretty.split("\n")
    xml_lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    final_xml    = "\n".join(xml_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_xml)
    print(f"XML saved -> {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  —  Light / Claude-inspired
# ══════════════════════════════════════════════════════════════════════════════

C = {
    # Backgrounds
    "bg":       "#f9f8f6",   # warm off-white — main window
    "surface":  "#ffffff",   # card / textarea
    "surface2": "#f2f1ef",   # secondary surface (chips, option blocks)
    "hover":    "#ede9e3",   # hover state

    # Borders
    "border":   "#e3e1dc",   # default border
    "border2":  "#ccc9c3",   # stronger border

    # Typography
    "text":     "#1a1917",   # primary text
    "text2":    "#4a4845",   # secondary text
    "muted":    "#9e9b96",   # placeholder / caption

    # Accent — warm amber/clay (Claude brand feel)
    "accent":   "#c96442",   # primary CTA
    "accent_h": "#b05538",   # hover
    "accent_lt":"#fdf0eb",   # light tint for badges

    # Status
    "success":  "#2d7a4f",
    "error":    "#c0392b",
    "warn":     "#b06a10",
}

FONT_UI   = ("Segoe UI",          9)
FONT_SM   = ("Segoe UI",          8)
FONT_MONO = ("Consolas",         10)
FONT_HEAD = ("Segoe UI Semibold", 12)
FONT_CAP  = ("Segoe UI",          7)
FONT_BTN  = ("Segoe UI Semibold", 10)
FONT_BODY = ("Segoe UI",         10)


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _sep(parent, bg=None, pady=(8, 0)):
    tk.Frame(parent, height=1, bg=bg or C["border"]).pack(fill="x", pady=pady)


def _label(parent, text, font=FONT_SM, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or C["text2"], bg=bg or C["bg"], **kw)


def _radio(parent, text, var, value, bg):
    return tk.Radiobutton(
        parent, text=text, variable=var, value=value,
        font=FONT_UI, bg=bg, fg=C["text2"],
        selectcolor=bg,
        activebackground=bg, activeforeground=C["text"],
        relief="flat", bd=0, cursor="hand2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SUBTITLE EDITOR
# ══════════════════════════════════════════════════════════════════════════════

def open_subtitle_editor(audio_files):
    base        = os.path.splitext(audio_files[0])[0]
    srt_path    = base + ".srt"
    srt_id_path = base + "_id.srt"
    xml_path    = base + "_timeline.xml"

    editor = tk.Toplevel(app_root)
    editor.title("Subtitle Editor")
    editor.geometry("660x580")
    editor.resizable(True, True)
    editor.configure(bg=C["bg"])

    # ── HEADER BAR ────────────────────────────────────────────────────────────
    hdr = tk.Frame(editor, bg=C["bg"])
    hdr.pack(fill="x", padx=24, pady=(22, 0))

    tk.Label(hdr, text="Subtitle Editor", font=FONT_HEAD,
             bg=C["bg"], fg=C["text"]).pack(side="left")

    line_var = tk.StringVar(value="0 baris")
    tk.Label(hdr, textvariable=line_var, font=FONT_SM,
             bg=C["bg"], fg=C["muted"]).pack(side="right", padx=(0, 2))

    _sep(editor, pady=(10, 0))

    # ── HINT ROW ──────────────────────────────────────────────────────────────
    hint_row = tk.Frame(editor, bg=C["bg"])
    hint_row.pack(fill="x", padx=24, pady=(8, 0))
    tk.Label(hint_row, text="1 baris = 1 subtitle  ·  tanpa tanda baca di akhir",
             font=FONT_SM, bg=C["bg"], fg=C["muted"]).pack(side="left")

    # ── FILE CHIPS ────────────────────────────────────────────────────────────
    chip_row = tk.Frame(editor, bg=C["bg"])
    chip_row.pack(fill="x", padx=24, pady=(10, 0))

    lbl = f"File audio:" if len(audio_files) == 1 else f"{len(audio_files)} file:"
    tk.Label(chip_row, text=lbl, font=FONT_SM,
             bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0, 8))

    for f in audio_files:
        chip = tk.Frame(chip_row, bg=C["accent_lt"],
                        highlightbackground=C["border"], highlightthickness=1)
        chip.pack(side="left", padx=(0, 5))
        tk.Label(chip, text=os.path.basename(f), font=FONT_SM,
                 bg=C["accent_lt"], fg=C["accent"], padx=9, pady=3).pack()

    # ── TEXT AREA ─────────────────────────────────────────────────────────────
    ta_frame = tk.Frame(editor, bg=C["border"], bd=0)
    ta_frame.pack(fill="both", expand=True, padx=24, pady=(12, 0))

    text_box = scrolledtext.ScrolledText(
        ta_frame,
        font=("Malgun Gothic", 11),
        wrap="word", undo=True,
        height=14,
        bg=C["surface"], fg=C["text"],
        insertbackground=C["accent"],
        selectbackground="#fad9cf",
        selectforeground=C["text"],
        relief="flat", bd=0,
        padx=16, pady=12,
        spacing1=3, spacing3=3,
    )
    text_box.pack(fill="both", expand=True, padx=1, pady=1)

    def update_count(event=None):
        c = len([l for l in text_box.get("1.0", "end-1c").splitlines() if l.strip()])
        line_var.set(f"{c} baris")
    text_box.bind("<KeyRelease>", update_count)

    # ── OPTIONS ROW ───────────────────────────────────────────────────────────
    opts = tk.Frame(editor, bg=C["bg"])
    opts.pack(fill="x", padx=24, pady=(14, 0))

    def _opt_block(parent, title):
        frame = tk.Frame(parent, bg=C["surface2"],
                         highlightbackground=C["border"], highlightthickness=1)
        frame.pack(side="left", fill="y", padx=(0, 10), ipadx=14, ipady=8)
        tk.Label(frame, text=title, font=("Segoe UI", 7, "bold"),
                 bg=C["surface2"], fg=C["muted"]).pack(anchor="w", pady=(0, 4))
        return frame

    xml_block = _opt_block(opts, "XML MODE")
    xml_mode  = tk.StringVar(value="full")
    _radio(xml_block, "Full clip  (markers)", xml_mode, "full", C["surface2"]).pack(anchor="w")
    _radio(xml_block, "Cut per subtitle",      xml_mode, "cut",  C["surface2"]).pack(anchor="w")

    srt_block = _opt_block(opts, "SRT OUTPUT")
    srt_mode  = tk.StringVar(value="ko")
    _radio(srt_block, "Korean saja",                srt_mode, "ko",    C["surface2"]).pack(anchor="w")
    _radio(srt_block, "Korean + Indonesia (2 file)", srt_mode, "ko+id", C["surface2"]).pack(anchor="w")

    # ── PROGRESS ──────────────────────────────────────────────────────────────
    prog_outer = tk.Frame(editor, bg=C["bg"])
    prog_outer.pack(fill="x", padx=24, pady=(14, 0))

    prog_track = tk.Frame(prog_outer, bg=C["border"], height=3)
    prog_track.pack(fill="x")

    prog_fill  = tk.Frame(prog_track, bg=C["accent"], height=3, width=0)
    prog_fill.place(x=0, y=0, relheight=1)

    def _update_prog(val):
        prog_track.update_idletasks()
        total_w = prog_track.winfo_width()
        fill_w  = int(total_w * val / 100)
        prog_fill.place(x=0, y=0, width=fill_w, relheight=1)

    edit_status = tk.Label(prog_outer, text="Paste subtitle di atas, lalu klik Proses",
                           font=FONT_SM, bg=C["bg"], fg=C["muted"], anchor="w")
    edit_status.pack(fill="x", pady=(5, 0))

    # ── BUTTONS ───────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(editor, bg=C["bg"])
    btn_frame.pack(fill="x", padx=24, pady=(12, 24))

    confirm_btn = tk.Button(
        btn_frame,
        text="Proses  →  SRT + XML",
        bg=C["accent"], fg="white",
        font=FONT_BTN,
        relief="flat", padx=20, pady=9,
        cursor="hand2",
        activebackground=C["accent_h"], activeforeground="white",
        bd=0,
    )
    confirm_btn.pack(side="left", padx=(0, 8))

    def _btn_cancel_enter(e):
        cancel_btn.config(bg=C["hover"])
    def _btn_cancel_leave(e):
        cancel_btn.config(bg=C["surface2"])

    cancel_btn = tk.Button(
        btn_frame, text="Batal",
        command=editor.destroy,
        bg=C["surface2"], fg=C["text2"],
        font=FONT_UI, relief="flat", padx=14, pady=9,
        cursor="hand2", activebackground=C["hover"], activeforeground=C["text"],
        highlightbackground=C["border"], highlightthickness=1, bd=0,
    )
    cancel_btn.pack(side="left")
    cancel_btn.bind("<Enter>", _btn_cancel_enter)
    cancel_btn.bind("<Leave>", _btn_cancel_leave)

    def set_ed(value, msg, color=None):
        _update_prog(value)
        col = color or C["warn"]
        edit_status.config(text=msg, fg=col)

    def on_confirm():
        content = text_box.get("1.0", "end-1c")
        lines   = [clean_line(l) for l in content.splitlines() if clean_line(l)]
        mode    = xml_mode.get()
        s_mode  = srt_mode.get()

        if not lines:
            messagebox.showwarning("Warning", "No subtitle content found.", parent=editor)
            return

        confirm_btn.config(state="disabled")
        set_ed(5, "Starting...")

        def run():
            try:
                def progress_cb(val, msg):
                    editor.after(0, lambda v=val, m=msg: set_ed(v, m))

                timed     = align_subtitles(audio_files, lines, progress_cb=progress_cb)
                total_dur = timed[-1]["end"] if timed else 0

                editor.after(0, lambda: set_ed(65, "Generating Korean SRT..."))
                create_srt(srt_path, timed)

                id_path_out = None
                if s_mode == "ko+id":
                    editor.after(0, lambda: set_ed(72, "Translating to Indonesian..."))
                    id_texts    = translate_lines([e["text"] for e in timed])
                    editor.after(0, lambda: set_ed(82, "Generating Indonesian SRT..."))
                    create_srt_id(srt_id_path, timed, id_texts)
                    id_path_out = srt_id_path

                editor.after(0, lambda: set_ed(90, f"Generating XML ({mode} mode)..."))
                create_xml(xml_path, audio_files, timed, total_dur, mode=mode)

                editor.after(0, lambda: set_ed(100, "Selesai!", C["success"]))
                editor.after(0, lambda: confirm_btn.config(state="normal"))
                editor.after(0, lambda: on_done(mode, s_mode, id_path_out))

            except Exception as e:
                err_msg = str(e)
                editor.after(0, lambda m=err_msg: set_ed(0, f"Error: {m}", C["error"]))
                editor.after(0, lambda: confirm_btn.config(state="normal"))
                editor.after(0, lambda m=err_msg: messagebox.showerror("Error", m, parent=editor))

        threading.Thread(target=run, daemon=True).start()

    confirm_btn.config(command=on_confirm)

    def on_done(mode, s_mode, id_path_out):
        mode_label = "full clip" if mode == "full" else "cut per subtitle"
        update_status(f"Selesai  ·  {os.path.basename(srt_path)}")
        msg = f"File tersimpan [{mode_label}]:\n\nSRT (KO)  →  {srt_path}\n"
        if id_path_out:
            msg += f"SRT (ID)  →  {id_path_out}\n"
        msg += f"\nXML  →  {xml_path}"
        messagebox.showinfo("Selesai", msg, parent=editor)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".m4a", ".flac", ".aac"}


def parse_drop_files(data: str):
    import shlex
    data = data.strip()
    if data.startswith("{"):
        files = []
        i = 0
        while i < len(data):
            if data[i] == "{":
                end = data.index("}", i)
                files.append(data[i + 1:end])
                i = end + 1
            elif data[i] == " ":
                i += 1
            else:
                end = data.find(" ", i)
                if end == -1:
                    files.append(data[i:])
                    break
                files.append(data[i:end])
                i = end + 1
        return files
    else:
        try:
            return shlex.split(data)
        except Exception:
            return data.split()


def open_files(files):
    valid = [f for f in files if os.path.splitext(f)[1].lower() in MEDIA_EXTS]
    if not valid:
        messagebox.showwarning("Format tidak didukung",
                               "Tidak ada file media yang valid.\n"
                               "Format: mp4 mov avi mkv mp3 wav m4a flac aac")
        return
    sorted_files = sorted(valid, key=lambda f: os.path.basename(f))
    open_subtitle_editor(sorted_files)


def select_file():
    files = filedialog.askopenfilenames(
        title="Select Audio / Video File(s)",
        filetypes=[
            ("Media files", "*.mp4 *.mov *.avi *.mkv *.mp3 *.wav *.m4a *.flac *.aac"),
            ("All files",   "*.*"),
        ]
    )
    if not files:
        return
    open_files(list(files))


# ── STARTUP ───────────────────────────────────────────────────────────────────

import torch as _torch
print(f"[STARTUP] CUDA available: {_torch.cuda.is_available()}")
if _torch.cuda.is_available():
    print(f"[STARTUP] GPU: {_torch.cuda.get_device_name(0)}")

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    app_root = TkinterDnD.Tk()
    HAS_DND = True
except Exception:
    app_root = tk.Tk()
    HAS_DND = False
    print("[INFO] tkinterdnd2 not found — drag-and-drop disabled.")

app_root.title("Whisper Auto Editor")
app_root.geometry("460x300")
app_root.resizable(False, False)
app_root.configure(bg=C["bg"])

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr = tk.Frame(app_root, bg=C["bg"])
hdr.pack(fill="x", padx=26, pady=(22, 0))

tk.Label(hdr, text="Whisper", font=("Segoe UI Semibold", 16),
         bg=C["bg"], fg=C["text"]).pack(side="left")
tk.Label(hdr, text="  Auto Editor", font=("Segoe UI Light", 16),
         bg=C["bg"], fg=C["text2"]).pack(side="left")

gpu_txt = (f"CUDA · {_torch.cuda.get_device_name(0)[:22]}"
           if _torch.cuda.is_available() else "CPU only")
tk.Label(hdr, text=gpu_txt, font=FONT_SM,
         bg=C["bg"], fg=C["muted"]).pack(side="right", pady=(4, 0))

_sep(app_root, pady=(10, 0))

# ── DROP ZONE ─────────────────────────────────────────────────────────────────
drop_frame = tk.Frame(
    app_root,
    bg=C["surface"],
    highlightbackground=C["border"],
    highlightthickness=1,
    width=408, height=128,
)
drop_frame.pack(padx=26, pady=16)
drop_frame.pack_propagate(False)

drop_icon = tk.Label(drop_frame, text="↑", font=("Segoe UI Light", 24),
                     bg=C["surface"], fg=C["muted"])
drop_icon.place(relx=0.5, rely=0.30, anchor="center")

_dnd_hint = ("Seret file video / audio ke sini\natau klik untuk pilih"
             if HAS_DND else "Klik untuk pilih file")
drop_label = tk.Label(drop_frame, text=_dnd_hint,
                      font=FONT_SM, bg=C["surface"], fg=C["muted"],
                      justify="center", cursor="hand2")
drop_label.place(relx=0.5, rely=0.72, anchor="center")

def _zone_enter(e=None):
    drop_frame.config(highlightbackground=C["accent"], bg=C["accent_lt"])
    drop_icon.config(fg=C["accent"],  bg=C["accent_lt"])
    drop_label.config(fg=C["accent"], bg=C["accent_lt"])

def _zone_leave(e=None):
    drop_frame.config(highlightbackground=C["border"], bg=C["surface"])
    drop_icon.config(fg=C["muted"], bg=C["surface"])
    drop_label.config(fg=C["muted"], bg=C["surface"])

def _on_drop(e):
    _zone_leave()
    files = parse_drop_files(e.data)
    open_files(files)

def _on_click(e):
    select_file()

for widget in (drop_frame, drop_icon, drop_label):
    widget.bind("<Button-1>", _on_click)
    widget.bind("<Enter>",    lambda e: _zone_enter())
    widget.bind("<Leave>",    lambda e: _zone_leave())

if HAS_DND:
    drop_frame.drop_target_register(DND_FILES)
    drop_frame.dnd_bind("<<DropEnter>>", lambda e: _zone_enter())
    drop_frame.dnd_bind("<<DropLeave>>", lambda e: _zone_leave())
    drop_frame.dnd_bind("<<Drop>>",      _on_drop)

# ── BOTTOM STATUS ─────────────────────────────────────────────────────────────
bot = tk.Frame(app_root, bg=C["bg"])
bot.pack(fill="x", padx=26, pady=(0, 18))

prog_track_m = tk.Frame(bot, bg=C["border"], height=2)
prog_track_m.pack(fill="x")

prog_fill_m = tk.Frame(prog_track_m, bg=C["accent"], height=2, width=0)
prog_fill_m.place(x=0, y=0, relheight=1)

def _update_main_prog(val):
    prog_track_m.update_idletasks()
    total_w = prog_track_m.winfo_width()
    prog_fill_m.place(x=0, y=0, width=int(total_w * val / 100), relheight=1)

def set_progress(value, status_text):
    app_root.after(0, lambda: _update_main_prog(value))
    app_root.after(0, lambda: status.config(text=status_text))

def update_status(msg):
    app_root.after(0, lambda: status.config(text=msg))

status = tk.Label(bot, text="Siap",
                  font=FONT_SM, fg=C["muted"], bg=C["bg"], anchor="w")
status.pack(fill="x", pady=(5, 0))

app_root.mainloop()
