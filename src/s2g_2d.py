from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traceback
import threading
import sys

import torch
from torch import nn
from transformers import AutoFeatureExtractor, HubertModel


# -------------------------
# Copy of main with light/fast tweaks
# -------------------------
@dataclass
class Joint:
    name: str
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    channels: list[str] = field(default_factory=list)
    children: list["Joint"] = field(default_factory=list)
    parent: "Joint | None" = None


def rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def euler_from_channels(ch_names: list[str], ch_vals: dict[str, float]) -> np.ndarray:
    R = np.eye(3, dtype=np.float32)
    for ch in ch_names:
        if ch.endswith("rotation"):
            v = float(ch_vals.get(ch, 0.0))
            if ch.startswith("X"):
                R = R @ rot_x(v)
            elif ch.startswith("Y"):
                R = R @ rot_y(v)
            elif ch.startswith("Z"):
                R = R @ rot_z(v)
    return R


def pos_from_channels(ch_names: list[str], ch_vals: dict[str, float]) -> np.ndarray:
    p = np.zeros(3, dtype=np.float32)
    for ch in ch_names:
        if ch.endswith("position"):
            v = float(ch_vals.get(ch, 0.0))
            if ch.startswith("X"):
                p[0] = v
            elif ch.startswith("Y"):
                p[1] = v
            elif ch.startswith("Z"):
                p[2] = v
    return p


def parse_bvh_hierarchy(template_bvh: Path):
    txt = template_bvh.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(txt) and txt[i].strip() != "HIERARCHY":
        i += 1
    if i >= len(txt):
        raise RuntimeError("HIERARCHY not found in template BVH")
    i += 1

    stack: list[Joint] = []
    root: Joint | None = None
    channel_order: list[tuple[Joint, str]] = []

    def cur() -> Joint:
        return stack[-1]

    while i < len(txt):
        line = txt[i].strip()
        if line == "MOTION":
            break

        if line.startswith("ROOT"):
            name = line.split(maxsplit=1)[1]
            root = Joint(name=name)
            stack.append(root)
            i += 1
            continue

        if line.startswith("JOINT"):
            name = line.split(maxsplit=1)[1]
            j = Joint(name=name, parent=cur())
            cur().children.append(j)
            stack.append(j)
            i += 1
            continue

        if line.startswith("End Site") or line == "End Site":
            j = Joint(name="EndSite", parent=cur())
            cur().children.append(j)
            stack.append(j)
            i += 1
            continue

        if line == "{":
            i += 1
            continue

        if line == "}":
            stack.pop()
            i += 1
            continue

        if line.startswith("OFFSET"):
            parts = line.split()
            cur().offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            i += 1
            continue

        if line.startswith("CHANNELS"):
            parts = line.split()
            n = int(parts[1])
            chs = parts[2 : 2 + n]
            cur().channels = chs
            for ch in chs:
                channel_order.append((cur(), ch))
            i += 1
            continue

        i += 1

    if root is None:
        raise RuntimeError("ROOT not found in template BVH")

    edges = []

    def edges_dfs(j: Joint):
        for c in j.children:
            edges.append((j, c))
            edges_dfs(c)

    edges_dfs(root)

    return root, channel_order, edges


def compute_positions(root: Joint, channel_order, edges, frame_channels_deg: np.ndarray):
    ch_map: dict[int, dict[str, float]] = {}
    for (j, ch), v in zip(channel_order, frame_channels_deg.tolist()):
        k = id(j)
        if k not in ch_map:
            ch_map[k] = {}
        ch_map[k][ch] = float(v)

    pos_g: dict[int, np.ndarray] = {}

    def dfs(j: Joint, parent_pos: np.ndarray, parent_rot: np.ndarray):
        vals = ch_map.get(id(j), {})
        p_local = j.offset + pos_from_channels(j.channels, vals)
        R_local = euler_from_channels(j.channels, vals)

        p_global = parent_pos + parent_rot @ p_local
        R_global = parent_rot @ R_local

        pos_g[id(j)] = p_global

        for c in j.children:
            dfs(c, p_global, R_global)

    dfs(root, np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    return pos_g


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    t = np.arange(len(x), dtype=np.float32) / float(sr)
    t2 = np.arange(int(round(len(x) * 16000 / sr)), dtype=np.float32) / 16000.0
    y = np.interp(t2, t, x).astype(np.float32)
    return y


def hubert_extract(model, fe, audio_16k: np.ndarray, device: str) -> np.ndarray:
    inputs = fe(audio_16k, sampling_rate=16000, return_tensors="pt")
    iv = inputs["input_values"].to(device)
    with torch.no_grad():
        out = model(iv)
    return out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)


def resample_feats_to_T(feats: np.ndarray, T: int) -> np.ndarray:
    Th, F = feats.shape
    if Th == T:
        return feats
    if Th <= 1:
        return np.repeat(feats[:1], T, axis=0)
    src = np.linspace(0.0, 1.0, Th, endpoint=False)
    tgt = np.linspace(0.0, 1.0, T, endpoint=False)
    out = np.zeros((T, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(tgt, src, feats[:, f])
    return out


def normalize_sincos(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    C2 = y.shape[1]
    C = C2 // 2
    sin = y[:, :C]
    cos = y[:, C:]
    r = np.sqrt(sin * sin + cos * cos) + eps
    sin = sin / r
    cos = cos / r
    return np.concatenate([sin, cos], axis=1)


def sincos_to_deg(y: np.ndarray) -> np.ndarray:
    C2 = y.shape[1]
    C = C2 // 2
    sin = y[:, :C]
    cos = y[:, C:]
    rad = np.arctan2(sin, cos)
    return (rad * (180.0 / np.pi)).astype(np.float32)


def main():
    ckpt_path = Path(__file__).resolve().parent.parent / "models" / "lstmp_trim_final.pt"
    template_bvh = Path(__file__).resolve().parent.parent / "data" / "wayne_skeleton.bvh"

    # Tweaked defaults for snappy realtime
    chunk_sec = 0.2
    hubert_win_sec = 1.0  # shortened
    fps_motion = 30.0
    view = "xy"
    fps_out = 30

    # keep ema as parameter but default to 0.0 for responsiveness
    ema_alpha = 0

    mic_sr = 48000
    block_sec = 0.05
    channels = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # BVH
    root, channel_order, edges = parse_bvh_hierarchy(template_bvh)
    n_channels = len(channel_order)

    # HuBERT
    fe = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    # LSTMP
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = ckpt["in_dim"]
    out_dim = ckpt["out_dim"]
    expected_out = 2 * n_channels
    if out_dim != expected_out:
        raise RuntimeError(f"out_dim mismatch: ckpt out_dim={out_dim}, template expects {expected_out}")

    model = LSTMPRegressor = None
    # reuse LSTMPRegressor definition from original file by loading the module
    # to keep this file self-contained we construct minimal wrapper class
    class LSTMPRegressor(nn.Module):
        def __init__(self, in_dim, hidden_dim=512, proj_dim=256, num_layers=2, out_dim=0, dropout=0.1):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.proj_dim = proj_dim
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                proj_size=proj_dim,
            )
            self.head = nn.Linear(proj_dim, out_dim)

        def forward(self, x, state=None):
            y, new_state = self.lstm(x, state)
            y = self.head(y)
            return y, new_state

        def init_state(self, batch_size: int, device: str):
            h = torch.zeros((self.num_layers, batch_size, self.proj_dim), device=device)
            c = torch.zeros((self.num_layers, batch_size, self.hidden_dim), device=device)
            return (h, c)

    model = LSTMPRegressor(in_dim=in_dim, out_dim=out_dim).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    state = model.init_state(batch_size=1, device=device)

    take_frames = int(round(chunk_sec * fps_motion))
    if take_frames < 1:
        take_frames = 1

    T_win = int(round(hubert_win_sec * fps_motion))
    if T_win < take_frames:
        T_win = take_frames

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    chunk_len = int(round(chunk_sec * mic_sr))
    acc = np.zeros((0,), dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal acc
        x = indata[:, 0].astype(np.float32)
        acc = np.concatenate([acc, x], axis=0)
        while len(acc) >= chunk_len:
            one = acc[:chunk_len].copy()
            acc = acc[chunk_len:]
            try:
                audio_q.put_nowait(one)
            except queue.Full:
                pass

    win_len = int(round(hubert_win_sec * mic_sr))
    audio_buf = np.zeros((0,), dtype=np.float32)

    y_queue: deque[np.ndarray] = deque()
    cur_frame = np.zeros((n_channels,), dtype=np.float32)
    ema_frame_sc: np.ndarray | None = None

    bench_stats: list[tuple[float, float, float, float, float, float]] = []
    enable_bench = True

    def infer_and_enqueue_latest_window():
        nonlocal audio_buf, state, ema_frame_sc

        while audio_q.qsize() > 1:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

        t_start_get = time.time()
        try:
            ch = audio_q.get_nowait()
        except queue.Empty:
            return
        t_got = time.time()

        audio_buf = np.concatenate([audio_buf, ch], axis=0)
        if len(audio_buf) > win_len:
            audio_buf = audio_buf[-win_len:]

        if len(audio_buf) < win_len:
            return

        win_audio = audio_buf[-win_len:].astype(np.float32)
        x16 = resample_to_16k(win_audio, mic_sr)
        t_after_resample = time.time()

        feats = hubert_extract(hubert, fe, x16, device=device)
        t_after_hubert = time.time()
        Xwin = resample_feats_to_T(feats, T_win)

        if Xwin.shape[1] != in_dim:
            raise RuntimeError(f"in_dim mismatch: got {Xwin.shape[1]}, expected {in_dim}")

        Xtake = Xwin[-take_frames:]
        Xt = torch.from_numpy(Xtake[None]).to(device)
        with torch.no_grad():
            Yt, state = model(Xt, state)
        t_after_model = time.time()

        Y = Yt[0].detach().cpu().numpy().astype(np.float32)
        Y = normalize_sincos(Y)

        for t in range(Y.shape[0]):
            frame_sc = Y[t]

            if ema_frame_sc is None:
                ema_frame_sc = frame_sc.copy()
            else:
                ema_frame_sc = ema_alpha * ema_frame_sc + (1.0 - ema_alpha) * frame_sc
                ema_frame_sc = normalize_sincos(ema_frame_sc[None])[0]

            frame_deg = sincos_to_deg(ema_frame_sc[None])[0]
            y_queue.append(frame_deg)

        t_end = time.time()
        if enable_bench:
            bench_stats.append((t_got - t_start_get, t_after_resample - t_got, t_after_hubert - t_after_resample, t_after_model - t_after_hubert, t_end - t_after_model, t_end - t_start_get))
            try:
                print(f"[BENCH] get={(t_got-t_start_get):.3f}s resample={(t_after_resample-t_got):.3f}s hubert={(t_after_hubert-t_after_resample):.3f}s model={(t_after_model-t_after_hubert):.3f}s post={(t_end-t_after_model):.3f}s total={(t_end-t_start_get):.3f}s")
            except Exception:
                pass

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    lines = []
    for _ in edges:
        ln, = ax.plot([], [], linewidth=1, color="tab:blue")
        lines.append(ln)

    ax.set_xlabel(view[0].upper())
    ax.set_ylabel(view[1].upper())
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_title("Realtime gesture (fast) — press 'q' to quit and show bench average")

    last_draw = time.time()
    draw_fps = 0.0

    def on_key(event):
        # debug: print key so we can see what matplotlib reports
        try:
            print("key event:", event.key)
        except Exception:
            print("key event: <no key attribute>")

        if event.key and str(event.key).lower() == 'q':
            print("quitting on 'q' press")
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_):
        nonlocal cur_frame, last_draw, draw_fps
        try:
            # inference runs in background worker thread now
            # keep only latest frame to avoid backlog
            while len(y_queue) > 1:
                y_queue.pop()

            if y_queue:
                cur_frame = y_queue.popleft()

            pos = compute_positions(root, channel_order, edges, cur_frame)

            for k, (a, b) in enumerate(edges):
                pa = pos[id(a)]
                pb = pos[id(b)]
                if view == "xz":
                    xa, ya = pa[0], pa[2]
                    xb, yb = pb[0], pb[2]
                elif view == "xy":
                    xa, ya = pa[0], pa[1]
                    xb, yb = pb[0], pb[1]
                else:
                    xa, ya = pa[1], pa[2]
                    xb, yb = pb[1], pb[2]
                lines[k].set_data([xa, xb], [ya, yb])

            now = time.time()
            dt = now - last_draw
            last_draw = now
            if dt > 1e-6:
                draw_fps = 0.9 * draw_fps + 0.1 * (1.0 / dt)

            ax.set_title(f"Realtime gesture (hop={chunk_sec:.1f}s, hubert_win={hubert_win_sec:.1f}s, EMA={ema_alpha:.2f}) | draw_fps≈{draw_fps:.1f}")
            return lines
        except Exception:
            print("Exception in update:")
            traceback.print_exc()
            return lines

    block = int(round(block_sec * mic_sr))
    print("starting mic stream... (press 'q' in figure to stop)")
    # start a background thread to allow quitting from the terminal (type 'q' + Enter)
    stop_event = threading.Event()
    def _stdin_watcher():
        try:
            while not stop_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip().lower() == 'q':
                    print("stdin: quitting")
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                    break
        except Exception:
            pass

    t_stdin = threading.Thread(target=_stdin_watcher, daemon=True)
    t_stdin.start()
    # start background worker thread for inference to avoid blocking the GUI
    def _worker():
        try:
            while not stop_event.is_set():
                try:
                    infer_and_enqueue_latest_window()
                except Exception:
                    print("Exception in worker:")
                    traceback.print_exc()
                # small sleep to yield and avoid tight loop when idle
                time.sleep(0.001)
        except Exception:
            pass

    t_worker = threading.Thread(target=_worker, daemon=True)
    t_worker.start()
    with sd.InputStream(
        samplerate=mic_sr,
        channels=channels,
        blocksize=block,
        dtype="float32",
        callback=audio_callback,
    ):
        ani = FuncAnimation(fig, update, blit=True, interval=1000 / fps_out)
        try:
            plt.show()
        except KeyboardInterrupt:
            # allow Ctrl+C to stop cleanly
            print("KeyboardInterrupt received, closing figure...")
            try:
                plt.close(fig)
            except Exception:
                pass
        finally:
            stop_event.set()
            # wait briefly for worker to finish
            try:
                t_worker.join(timeout=1.0)
            except Exception:
                pass
    # after figure closed — print bench averages
    if bench_stats:
        arr = np.array(bench_stats, dtype=np.float32)
        mean = arr.mean(axis=0)
        print("\nBENCH AVERAGE (s):")
        print(f" get={mean[0]:.3f} resample={mean[1]:.3f} hubert={mean[2]:.3f} model={mean[3]:.3f} post={mean[4]:.3f} total={mean[5]:.3f}")
    else:
        print("no bench samples collected")


if __name__ == "__main__":
    main()
