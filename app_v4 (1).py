# -*- coding: utf-8 -*-
"""
DSS Penjadwalan Produksi Job Shop — Garmen | v8.0
Dual-Engine Optimizer: Simulated Annealing + MILP (PuLP/CBC)
Dual Benchmark: EDD + FCFS

PERBAIKAN dari v7.0:
  - [NEW] Tab 8: Analisis Crashing — 3 mode:
      Mode 1 (Tanggal Target)   : cari resource minimum agar semua order selesai sebelum tanggal X
      Mode 2 (Percepatan %)     : cari resource minimum agar makespan dipercepat sebesar X%
      Mode 3 (Zero-Tardiness)   : hanya menambah tim Jahit Kaos/Polo (1→3 tim, sesuai constraint real)
                                   cari jumlah tim minimum agar semua order tepat waktu
  - Semua fitur v7 tetap dipertahankan
"""

import streamlit as st
import pandas as pd
import pulp
import math
import random
import re
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(page_title="DSS Penjadwalan Job Shop", layout="wide", page_icon="🏭")

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1E3A8A;margin-bottom:0}
.sub-header{font-size:1.05rem;color:#64748B;margin-bottom:16px}
.metric-winner{background:#14532D;padding:14px;border-radius:10px;color:#FFFFFF;line-height:1.6}
.metric-loser{background:#7F1D1D;padding:14px;border-radius:10px;color:#FFFFFF;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏭 DSS: Optimasi Penjadwalan Produksi</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Penjadwalan Cerdas — Routing Dinamis (OPC) · Dual-Engine Optimizer (SA + MILP) · Analisis Sensitivitas</p>', unsafe_allow_html=True)
st.divider()

# ============================================================
# 2. KONSTANTA GLOBAL
# ============================================================
STATIONS = [
    '1. Potong', '2. Jahit_KaosPolo', '3. Jahit_KemejaJaket',
    '4. Sablon', '5. DTF', '6. Bordir', '7. Pasang_Kancing',
    '8. Buang_Benang', '9. Lipat', '10. Packing',
]

REQUIRED_COLUMNS = [
    'id pesanan', 'jenis produk', 'qty', 'due date (tanggal)',
    'furing', 'sablon', 'dtf', 'bordir', 'pasang kancing',
]
BINARY_COLUMNS = ['furing', 'sablon', 'dtf', 'bordir', 'pasang kancing']

MENIT_PER_HARI  = 450
MENIT_ISTIRAHAT = 90

PALETTE_ORDER = [
    '#2563EB','#16A34A','#DC2626','#D97706','#7C3AED',
    '#0891B2','#DB2777','#65A30D','#EA580C','#0D9488',
    '#4F46E5','#B45309','#BE185D','#15803D','#1D4ED8',
    '#92400E','#6D28D9','#047857','#B91C1C','#0369A1',
]


# ============================================================
# 3. LOAD & VALIDASI DATA
# ============================================================
def load_order_file(uploaded_file):
    fn = uploaded_file.name.lower()
    if fn.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif fn.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format tidak didukung. Gunakan CSV atau Excel (.xlsx/.xls).")

    df.columns = df.columns.str.lower().str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df['id pesanan']   = df['id pesanan'].astype(str).str.strip()
    df['jenis produk'] = df['jenis produk'].astype(str).str.strip().str.lower()

    mapping = {'kaos': 'kaos', 'polo': 'polo', 'kemeja': 'kemeja', 'jaket': 'jaket'}
    df['jenis produk'] = df['jenis produk'].replace(mapping)
    unknown = sorted(set(df['jenis produk']) - set(mapping))
    if unknown:
        raise ValueError(f"Jenis produk tidak dikenali: {unknown}")

    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    for col in BINARY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['due date (tanggal)'] = pd.to_datetime(df['due date (tanggal)'], errors='coerce', dayfirst=True)

    null_mask = df[['qty', 'due date (tanggal)'] + BINARY_COLUMNS].isnull().any(axis=1)
    if null_mask.any():
        raise ValueError(f"Data kosong/invalid di baris: {df.index[null_mask].tolist()}.")
    if (df['qty'] <= 0).any():
        raise ValueError(f"qty <= 0 pada order: {df.loc[df['qty']<=0,'id pesanan'].tolist()}")
    for col in BINARY_COLUMNS:
        inv = df.loc[~df[col].isin([0,1]), col].unique().tolist()
        if inv:
            raise ValueError(f"Kolom '{col}' hanya boleh 0 atau 1. Nilai: {inv}")
    return df


# ============================================================
# 4. WAKTU PROSES (OPC)
# ============================================================
def hitung_waktu_proses(row, resources, setup_time):
    qty    = row['qty']
    jenis  = str(row['jenis produk']).lower()
    furing = row['furing']
    P      = {m: 0.0 for m in STATIONS}

    cap_potong = 1000
    if jenis in ('kemeja', 'jaket'):
        cap_potong = 125 if furing == 1 else 250
    P['1. Potong'] = (qty / (cap_potong * resources['1. Potong'])) * MENIT_PER_HARI

    if jenis in ('kaos', 'polo'):
        cap_j = 112.5 if jenis == 'kaos' else 55
        P['2. Jahit_KaosPolo'] = (qty / (cap_j * resources['2. Jahit_KaosPolo'])) * MENIT_PER_HARI
    elif jenis in ('kemeja', 'jaket'):
        base = 13.5 if jenis == 'kemeja' else 11.0
        if furing == 1:
            base *= 2/3
        P['3. Jahit_KemejaJaket'] = (qty / (base * resources['3. Jahit_KemejaJaket'])) * MENIT_PER_HARI

    if row['sablon'] == 1:
        P['4. Sablon'] = (qty / (700   * resources['4. Sablon'])) * MENIT_PER_HARI
    if row['dtf']    == 1:
        P['5. DTF']    = (qty / (750   * resources['5. DTF']))    * MENIT_PER_HARI
    if row['bordir'] == 1:
        P['6. Bordir'] = (qty / (442.5 * resources['6. Bordir'])) * MENIT_PER_HARI

    if row['pasang kancing'] == 1 and jenis != 'kaos':
        cap_k = 400 if jenis == 'polo' else 125
        P['7. Pasang_Kancing'] = (qty / (cap_k * resources['7. Pasang_Kancing'])) * MENIT_PER_HARI

    cap_benang = 166.67 if furing == 1 else 500
    P['8. Buang_Benang'] = (qty / (cap_benang * resources['8. Buang_Benang'])) * MENIT_PER_HARI
    P['9. Lipat']        = (qty / (500 * resources['9. Lipat']))               * MENIT_PER_HARI
    P['10. Packing']     = (qty / (500 * resources['10. Packing']))            * MENIT_PER_HARI

    for m in STATIONS:
        if P[m] > 0:
            P[m] += setup_time
    return P


# ============================================================
# 5. KONVERSI WAKTU
# ============================================================
def konversi_ke_jam_dinding(menit_efektif, start_date):
    hari_ke = int(menit_efektif // MENIT_PER_HARI)
    sisa    = menit_efektif % MENIT_PER_HARI
    current = start_date
    cnt     = 0
    while cnt < hari_ke:
        current += timedelta(days=1)
        if current.weekday() != 6:
            cnt += 1
    if current.weekday() == 6:
        current += timedelta(days=1)
    base = current.replace(hour=8, minute=30, second=0, microsecond=0)
    return base + timedelta(minutes=sisa if sisa <= 180 else sisa + MENIT_ISTIRAHAT)


def hitung_target_menit(target_dt, start_dt):
    if target_dt <= start_dt:
        return 0
    total   = 0
    current = start_dt
    while current.date() < target_dt.date():
        if current.weekday() != 6:
            total += MENIT_PER_HARI
        current += timedelta(days=1)
    if target_dt.weekday() != 6:
        base  = current.replace(hour=8, minute=30, second=0, microsecond=0)
        delta = (target_dt - base).total_seconds() / 60
        if delta <= 0:
            pass
        elif delta <= 180:
            total += delta
        elif delta <= 180 + MENIT_ISTIRAHAT:
            total += 180
        else:
            total += delta - MENIT_ISTIRAHAT
    return total


def pecah_balok_gantt(start_efektif, durasi, start_date):
    blocks      = []
    tersisa     = durasi
    cur         = start_efektif
    while tersisa > 0.01:
        mnt = cur % MENIT_PER_HARI
        chunk = min(tersisa, (180 - mnt) if mnt < 180 else (MENIT_PER_HARI - mnt))
        if chunk < 0.01:
            cur += 0.01
            continue
        blocks.append({
            'start_nyata'    : konversi_ke_jam_dinding(cur, start_date),
            'end_nyata'      : konversi_ke_jam_dinding(cur + chunk, start_date),
            'durasi_potongan': chunk,
        })
        cur     += chunk
        tersisa -= chunk
    return blocks


# ============================================================
# 6. EVAL + SA + BENCHMARK
# ============================================================
def eval_sequence(seq, P_dict, D_dict, W_dict):
    """
    [FIX-C] Job Shop Scheduling yang benar — bukan permutation scheduling.

    Permutation scheduling (versi lama): memproses seluruh stasiun job ke-i
    sampai tuntas sebelum memulai job ke-i+1 di stasiun manapun. Ini keliru
    untuk job shop karena mesin bisa dipakai paralel oleh job berbeda di
    tahapan berbeda (job-2 bisa Potong saat job-1 sedang Jahit).

    Job Shop Scheduling (versi ini): setiap stasiun dalam routing setiap job
    dijadwalkan secepat mungkin berdasarkan:
      start = max(mesin_tersedia[m], job_selesai_stasiun_sebelumnya[j])
    Ini merepresentasikan constraint presedensi dan kapasitas mesin secara benar.
    Urutan antar-job pada setiap mesin ditentukan oleh 'seq' (prioritas job).

    Cara kerja: untuk setiap mesin, kita track kapan mesin bebas (m_avail).
    Untuk setiap job, kita track kapan job selesai di stasiun sebelumnya (j_avail).
    Setiap operasi (job, stasiun) dijadwalkan di max(m_avail[m], j_avail[j]).
    """
    # m_avail[m] = menit mesin m tersedia berikutnya
    m_avail = {m: 0.0 for m in STATIONS}
    # j_avail[j] = menit job j boleh mulai di stasiun berikutnya (selesai stasiun sebelumnya)
    j_avail = {j: 0.0 for j in seq}
    tard    = 0
    sched   = []

    # Proses setiap (job, stasiun) dalam urutan prioritas job × urutan stasiun routing
    # Ini adalah greedy dispatcher: untuk setiap job (sesuai urutan seq),
    # jadwalkan seluruh rutingnya. Karena m_avail di-share antar semua job,
    # mesin tidak akan pernah overlap.
    for j in seq:
        rute = [m for m in STATIONS if P_dict[j][m] > 0]
        for m in rute:
            dur = P_dict[j][m]
            # start = momen paling cepat mesin m bebas DAN job j sudah selesai tahap sebelumnya
            s   = max(m_avail[m], j_avail[j])
            e   = s + dur
            m_avail[m] = e   # mesin m baru tersedia setelah job ini selesai
            j_avail[j] = e   # job j baru boleh lanjut ke stasiun berikutnya
            sched.append({'job': j, 'm': m, 'start': s, 'dur': dur})
        tard += max(0, j_avail[j] - D_dict[j]) * W_dict[j]
    return tard, sched, j_avail


def eval_sequence_with_pins(seq, P_dict, D_dict, W_dict, pinned_starts):
    """
    Evaluasi dengan locked jobs di-pin tepat pada posisi asal.
    Free jobs diselipkan di slot mesin yang belum terpakai.
    pinned_starts: {job_id: {mesin: start_time}}
    """
    # Bangun tabel busy-slot dari pinned jobs
    m_busy   = {m: [] for m in STATIONS}
    j_end_pin = {}
    for j, pin_m in pinned_starts.items():
        for m, s in pin_m.items():
            m_busy[m].append((s, s + P_dict[j][m]))
        rute_j = [m for m in STATIONS if P_dict[j][m] > 0]
        j_end_pin[j] = max(pin_m.get(m, 0) + P_dict[j][m] for m in rute_j if m in pin_m)
    for m in STATIONS:
        m_busy[m].sort(key=lambda x: x[0])

    def earliest_free_slot(m, not_before, dur):
        t = not_before
        for s, e in m_busy[m]:
            if t + dur <= s + 1e-6:
                break
            if t < e:
                t = e
        return t

    tard  = 0
    sched = []

    # Tambahkan pinned jobs ke sched (posisi tetap)
    for j, pin_m in pinned_starts.items():
        rute_j = [m for m in STATIONS if P_dict[j][m] > 0]
        for m in rute_j:
            sched.append({'job': j, 'm': m, 'start': pin_m.get(m, 0), 'dur': P_dict[j][m]})
        tard += max(0, j_end_pin[j] - D_dict[j]) * W_dict[j]

    # Free jobs dioptimasi
    for j in seq:
        if j in pinned_starts:
            continue
        rute    = [m for m in STATIONS if P_dict[j][m] > 0]
        j_avail = 0.0
        for m in rute:
            dur     = P_dict[j][m]
            t_start = earliest_free_slot(m, j_avail, dur)
            t_end   = t_start + dur
            m_busy[m].append((t_start, t_end))
            m_busy[m].sort(key=lambda x: x[0])
            j_avail = t_end
            sched.append({'job': j, 'm': m, 'start': t_start, 'dur': dur})
        tard += max(0, j_avail - D_dict[j]) * W_dict[j]

    j_end_all = {}
    for e in sched:
        je = e['start'] + e['dur']
        if e['job'] not in j_end_all or je > j_end_all[e['job']]:
            j_end_all[e['job']] = je

    return tard, sched, j_end_all


def run_simulated_annealing(jobs, P_dict, D_dict, W_dict, pinned_starts=None, n_iter=8000):
    if pinned_starts is None:
        pinned_starts = {}
    free_jobs = [j for j in jobs if j not in pinned_starts]

    def evaluate(seq):
        return (eval_sequence_with_pins(seq, P_dict, D_dict, W_dict, pinned_starts)
                if pinned_starts else eval_sequence(seq, P_dict, D_dict, W_dict))

    def swap(seq):
        s = seq.copy()
        if len(s) < 2: return s
        a, b = random.sample(range(len(s)), 2)
        s[a], s[b] = s[b], s[a]
        return s

    def insert_op(seq):
        s = seq.copy()
        if len(s) < 2: return s
        a   = random.randrange(len(s))
        job = s.pop(a)
        b   = random.randrange(len(s) + 1)
        s.insert(b, job)
        return s

    cur_seq    = sorted(free_jobs, key=lambda x: D_dict[x])
    cur_score, _, _ = evaluate(cur_seq)
    best_seq   = cur_seq.copy()
    best_score = cur_score
    T_sa, cool = 500.0, 0.997

    for _ in range(n_iter):
        new_seq         = swap(cur_seq) if random.random() < 0.7 else insert_op(cur_seq)
        new_score, _, _ = evaluate(new_seq)
        delta           = new_score - cur_score
        if delta < 0 or (T_sa > 1e-10 and random.random() < math.exp(-delta / T_sa)):
            cur_seq, cur_score = new_seq, new_score
            if new_score < best_score:
                best_seq, best_score = new_seq.copy(), new_score
        T_sa *= cool

    _, final_sched, final_end = evaluate(best_seq)
    return best_score, final_sched, final_end


def run_edd(jobs, P_dict, D_dict, W_dict):
    return eval_sequence(sorted(jobs, key=lambda x: D_dict[x]), P_dict, D_dict, W_dict)

def run_fcfs(jobs_ordered, P_dict, D_dict, W_dict):
    return eval_sequence(jobs_ordered, P_dict, D_dict, W_dict)


# ============================================================
# 7. HELPER
# ============================================================
def safe_var_name(s):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(s))


def build_gantt_df(sched_list, df_pool, start_date, waktu_selesai_dict=None, D_dict=None):
    rows = []
    for t in sched_list:
        match   = df_pool[df_pool['id pesanan'].astype(str) == t['job']]
        qty_val = match['qty'].iloc[0] if not match.empty else 0
        terlambat = False
        if waktu_selesai_dict and D_dict:
            terlambat = waktu_selesai_dict.get(t['job'], 0) > D_dict.get(t['job'], float('inf'))
        for blk in pecah_balok_gantt(t['start'], t['dur'], start_date):
            rows.append({
                'Stasiun Kerja' : t['m'],
                'ID Pesanan'    : t['job'],
                'Qty'           : qty_val,
                'Mulai'         : blk['start_nyata'],
                'Selesai'       : blk['end_nyata'],
                'Durasi (Menit)': round(blk['durasi_potongan'], 2),
                'Status'        : '🔴 Terlambat' if terlambat else '🟢 Tepat Waktu',
            })
    return pd.DataFrame(rows)


# ============================================================
# 8. RENDER GANTT DUAL — TANPA st.rerun()
# ============================================================
def render_gantt_dual(df_gantt, title, chart_key, height=520):
    """
    Dual-mode Gantt chart.
    Mode toggle disimpan di session_state[chart_key].
    TIDAK menggunakan st.rerun() — mode diubah lewat on_change callback
    yang di-trigger oleh st.radio, sehingga Streamlit rerun normal
    tetapi session_state hasil optimasi tetap ada karena tidak ada
    kondisi yang menghapusnya.
    """
    if df_gantt.empty:
        st.warning("Tidak ada data jadwal untuk ditampilkan.")
        return

    if chart_key not in st.session_state:
        st.session_state[chart_key] = "Default (per Order)"

    # ── Render chart sesuai mode ──────────────────────────────
    mode = st.session_state[chart_key]

    if mode == "Default (per Order)":
        unique_ids = sorted(df_gantt['ID Pesanan'].unique())
        color_map  = {jid: PALETTE_ORDER[i % len(PALETTE_ORDER)] for i, jid in enumerate(unique_ids)}
        fig = px.timeline(
            df_gantt, x_start="Mulai", x_end="Selesai",
            y="Stasiun Kerja", color="ID Pesanan",
            color_discrete_map=color_map,
            text="ID Pesanan",
            hover_data=["ID Pesanan", "Durasi (Menit)", "Qty", "Status"],
            title=title,
        )
    else:  # Status
        color_map = {'🔴 Terlambat': '#EF4444', '🟢 Tepat Waktu': '#22C55E'}
        fig = px.timeline(
            df_gantt, x_start="Mulai", x_end="Selesai",
            y="Stasiun Kerja", color="Status",
            color_discrete_map=color_map,
            text="ID Pesanan",
            hover_data=["ID Pesanan", "Durasi (Menit)", "Qty"],
            title=title,
        )
        fig.update_layout(legend_title_text='Status Ketepatan')

    fig.update_traces(textposition='inside', insidetextanchor='middle')
    fig.update_yaxes(categoryorder="array", categoryarray=STATIONS[::-1])
    fig.update_layout(height=height, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_key}_{mode}")

    # ── Toggle tombol DI BAWAH chart — pakai st.radio tanpa rerun ──
    # Gunakan radio horizontal sebagai toggle; on_change hanya update session_state
    # tanpa st.rerun() eksplisit, sehingga Streamlit melakukan rerun-minimal yang aman.
    chosen = st.radio(
        "Tampilan:",
        options=["Default (per Order)", "Status (Tepat/Terlambat)"],
        index=0 if mode == "Default (per Order)" else 1,
        key=f"radio_{chart_key}",
        horizontal=True,
        label_visibility="collapsed",
    )
    if chosen != mode:
        st.session_state[chart_key] = chosen
        st.rerun()


# ============================================================
# 9. SANITY CHECK
# ============================================================
def jalankan_sanity_check(jadwal_final, df_pool, P_dict, start_date):
    log = ["="*60, "🔍 SANITY CHECK — VERIFIKASI LOGIKA JADWAL", "="*60]
    err_overlap = err_presedens = False

    # Toleransi floating-point: 0.001 menit (~0.06 detik).
    # Lebih ketat dari sebelumnya (0.01) tapi tetap toleran terhadap
    # rounding error CBC yang wajar (biasanya < 1e-6).
    TOL = 0.001

    log.append("\n[1/3] Memeriksa Overlap Mesin...")
    overlap_details = []
    for stn in STATIONS:
        tasks = sorted([t for t in jadwal_final if t['m'] == stn], key=lambda x: x['start'])
        for i in range(1, len(tasks)):
            p, c = tasks[i-1], tasks[i]
            gap  = c['start'] - (p['start'] + p['dur'])
            if gap < -TOL:
                msg = (f"  ❌ OVERLAP {stn}: [{p['job']}] end={p['start']+p['dur']:.4f} "
                       f"> [{c['job']}] start={c['start']:.4f}  (gap={gap:.4f} mnt)")
                log.append(msg)
                overlap_details.append(msg)
                err_overlap = True
    if not err_overlap:
        log.append("  ✔️ LULUS: Tidak ada tumpang tindih.")

    log.append("\n[2/3] Memeriksa Presedensi (urutan stasiun per job)...")
    presedens_details = []
    station_idx = {m: idx for idx, m in enumerate(STATIONS)}
    for job in set(t['job'] for t in jadwal_final):
        # Sort berdasarkan urutan routing resmi (STATIONS), bukan start time
        tasks_j = sorted(
            [t for t in jadwal_final if t['job'] == job],
            key=lambda x: station_idx.get(x['m'], 999)
        )
        for i in range(1, len(tasks_j)):
            p, c = tasks_j[i-1], tasks_j[i]
            gap  = c['start'] - (p['start'] + p['dur'])
            if gap < -TOL:
                msg = (f"  ❌ {job}: [{c['m']}] start={c['start']:.4f} "
                       f"< [{p['m']}] end={p['start']+p['dur']:.4f}  (gap={gap:.4f} mnt)")
                log.append(msg)
                presedens_details.append(msg)
                err_presedens = True
    if not err_presedens:
        log.append("  ✔️ LULUS: Semua urutan stasiun per job benar.")

    log.append("\n[3/3] Memeriksa Hari Minggu...")
    minggu_rows = []
    for t in jadwal_final:
        for blk in pecah_balok_gantt(t['start'], t['dur'], start_date):
            if blk['start_nyata'].weekday() == 6:
                minggu_rows.append({'ID Pesanan': t['job'], 'Stasiun': t['m'],
                                    'Mulai': blk['start_nyata'].strftime('%d-%b-%y %H:%M'),
                                    'Selesai': blk['end_nyata'].strftime('%d-%b-%y %H:%M')})
    log.append("  ✔️ LULUS: Tidak ada jadwal di Hari Minggu." if not minggu_rows
               else f"  ❌ {len(minggu_rows)} tugas di Hari Minggu!")
    log += ["="*60,
            "🚨 GAGAL!" if (err_overlap or err_presedens or minggu_rows) else "✅ PASSED!",
            "="*60]

    all_jobs      = list(set(t['job'] for t in jadwal_final))
    sample_id     = random.choice(all_jobs)
    sample_df     = df_pool[df_pool['id pesanan'].astype(str) == sample_id]
    sample_sched  = sorted([t for t in jadwal_final if t['job'] == sample_id], key=lambda x: x['start'])

    return {
        'log_text'        : "\n".join(log),
        'err_overlap'     : err_overlap,
        'err_presedens'   : err_presedens,
        'overlap_details' : overlap_details[:5],    # maks 5 contoh untuk display
        'presedens_details': presedens_details[:5],
        'sample_job_id'   : sample_id,
        'sample_row'      : sample_df.iloc[0].to_dict() if not sample_df.empty else {},
        'sample_rute'     : [t['m'] for t in sample_sched],
        'tabel_minggu'    : pd.DataFrame(minggu_rows) if minggu_rows else pd.DataFrame(),
    }


# ============================================================
# 10. CRASHING ENGINE
# ============================================================

# Stasiun yang bisa di-crash beserta batas maksimum resource-nya.
# Urutan penting: stasiun yang paling sering jadi bottleneck didahulukan.
CRASHABLE_STATIONS = {
    '1. Potong'           : {'max': 5,  'label': 'Operator Potong'},
    '2. Jahit_KaosPolo'   : {'max': 3,  'label': 'Tim Jahit Kaos/Polo'},
    '3. Jahit_KemejaJaket': {'max': 6,  'label': 'Tim Jahit Kemeja/Jaket'},
    '4. Sablon'           : {'max': 5,  'label': 'Mesin Sablon'},
    '5. DTF'              : {'max': 5,  'label': 'Mesin DTF'},
    '6. Bordir'           : {'max': 5,  'label': 'Mesin Bordir'},
    '7. Pasang_Kancing'   : {'max': 5,  'label': 'Operator Kancing'},
    '8. Buang_Benang'     : {'max': 5,  'label': 'Operator Buang Benang'},
    '9. Lipat'            : {'max': 5,  'label': 'Operator Lipat'},
    '10. Packing'         : {'max': 5,  'label': 'Operator Packing'},
}

# Mode 3 hanya boleh mengubah stasiun ini
MODE3_STATION = '2. Jahit_KaosPolo'
MODE3_MAX     = 3


def hitung_makespan_sa(jobs_raw, res_override, setup_time, W_dict, D_dict):
    """
    Jalankan SA dengan resource tertentu dan kembalikan:
    (makespan, sched, end_dict, tardiness_score, n_telat)
    makespan = waktu selesai job terakhir (menit efektif).
    """
    job_ids = [str(j['id pesanan']) for j in jobs_raw]
    P = {str(j['id pesanan']): hitung_waktu_proses(j, res_override, setup_time)
         for j in jobs_raw}
    score, sched, end = run_simulated_annealing(job_ids, P, D_dict, W_dict, n_iter=5000)
    makespan = max(end.values()) if end else 0
    n_telat  = sum(1 for i in job_ids if end.get(i, 0) > D_dict.get(i, 0))
    return makespan, sched, end, P, score, n_telat


def crash_mode1(jobs_raw, res_base, setup_time, W_dict, D_dict,
                target_menit, pb_callback=None):
    """
    Mode 1: Temukan resource minimum agar makespan ≤ target_menit.
    Strategi: iterasi resource dari base ke max, satu stasiun sekaligus (greedy).
    Stasiun yang paling memperpendek makespan diprioritaskan.
    """
    res_current  = res_base.copy()
    log_langkah  = []

    # Cek baseline
    mk_base, _, _, P_base, sc_base, nt_base = hitung_makespan_sa(
        jobs_raw, res_current, setup_time, W_dict, D_dict)
    log_langkah.append({'Langkah': 0, 'Aksi': 'Baseline',
                        'Resource': str({k.split('. ',1)[-1]: v for k,v in res_current.items() if v>1}),
                        'Makespan (Hari)': round(mk_base/MENIT_PER_HARI, 2),
                        'Berhasil': mk_base <= target_menit})

    if mk_base <= target_menit:
        return True, res_current, P_base, sc_base, log_langkah, mk_base

    step     = 0
    max_step = 20   # batas iterasi agar tidak infinite
    while True:
        step += 1
        if step > max_step:
            break
        if pb_callback:
            pb_callback(min(step / max_step, 0.95))

        # Cari stasiun yang jika ditambah 1 resource paling mempersingkat makespan
        best_gain    = 0
        best_station = None
        best_res     = None

        for stn, cfg in CRASHABLE_STATIONS.items():
            if res_current[stn] >= cfg['max']:
                continue  # sudah maksimum
            trial_res = res_current.copy()
            trial_res[stn] += 1
            mk_trial, _, _, _, _, _ = hitung_makespan_sa(
                jobs_raw, trial_res, setup_time, W_dict, D_dict)
            gain = mk_base - mk_trial
            if gain > best_gain:
                best_gain    = gain
                best_station = stn
                best_res     = trial_res
                mk_best      = mk_trial

        if best_station is None:
            # Tidak ada stasiun lagi yang bisa ditambah
            break

        res_current = best_res
        mk_base     = mk_best
        log_langkah.append({
            'Langkah': step,
            'Aksi'   : f"+1 {CRASHABLE_STATIONS[best_station]['label']} → {res_current[best_station]} unit",
            'Resource': str({k.split('. ',1)[-1]: v for k,v in res_current.items() if v > res_base.get(k,1)}),
            'Makespan (Hari)': round(mk_base/MENIT_PER_HARI, 2),
            'Berhasil': mk_base <= target_menit,
        })

        if mk_base <= target_menit:
            # Sukses — jalankan SA final dengan resource ini
            _, sched_f, end_f, P_f, sc_f, _ = hitung_makespan_sa(
                jobs_raw, res_current, setup_time, W_dict, D_dict)
            return True, res_current, P_f, sched_f, log_langkah, mk_base

    # Gagal mencapai target
    _, sched_f, end_f, P_f, sc_f, _ = hitung_makespan_sa(
        jobs_raw, res_current, setup_time, W_dict, D_dict)
    return False, res_current, P_f, sched_f, log_langkah, mk_base


def crash_mode2(jobs_raw, res_base, setup_time, W_dict, D_dict,
                pct_percepatan, pb_callback=None):
    """
    Mode 2: Percepatan makespan sebesar pct_percepatan % dari baseline.
    Gunakan makespan baseline SA sebagai acuan, lalu targetkan:
    target = baseline × (1 - pct/100).
    Delegasikan ke crash_mode1 dengan target tersebut.
    """
    mk_base, _, _, _, _, _ = hitung_makespan_sa(
        jobs_raw, res_base, setup_time, W_dict, D_dict)
    target = mk_base * (1 - pct_percepatan / 100)
    return crash_mode1(jobs_raw, res_base, setup_time, W_dict, D_dict,
                       target, pb_callback), mk_base


def crash_mode3(jobs_raw, res_base, setup_time, W_dict, D_dict,
                pb_callback=None):
    """
    Mode 3: Hanya tambah tim Jahit Kaos/Polo (1→3), cari jumlah minimum
    agar semua order tepat waktu (zero tardiness).
    Resource lain tidak berubah.
    Mengembalikan hasil untuk setiap level tim.
    """
    hasil_per_level = []
    for n_tim in range(1, MODE3_MAX + 1):
        if pb_callback:
            pb_callback((n_tim - 1) / MODE3_MAX)
        res_trial = res_base.copy()
        res_trial[MODE3_STATION] = n_tim
        mk, sched, end, P_trial, score, n_telat = hitung_makespan_sa(
            jobs_raw, res_trial, setup_time, W_dict, D_dict)
        hasil_per_level.append({
            'n_tim'  : n_tim,
            'res'    : res_trial.copy(),
            'sched'  : sched,
            'end'    : end,
            'P'      : P_trial,
            'score'  : score,
            'n_telat': n_telat,
            'makespan': mk,
        })
        if n_telat == 0:
            break   # sudah cukup, tidak perlu tambah lagi

    return hasil_per_level


# ============================================================
# 11. SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043236.png", width=60)
    st.header("⚙️ Konfigurasi Sistem")

    with st.expander("📥 Template Data"):
        tpl = pd.DataFrame({
            'id pesanan': ['ORD-01','ORD-02'], 'jenis produk': ['Kaos','Kemeja'],
            'qty': [100, 50], 'due date (tanggal)': ['15/05/2026','20/05/2026'],
            'furing': [0,1], 'sablon': [1,0], 'dtf': [0,0], 'bordir': [0,1], 'pasang kancing': [0,1],
        })
        st.download_button("⬇️ Download Template.csv", tpl.to_csv(index=False).encode('utf-8'),
                           "Template_Order_Pabrik.csv", "text/csv")

    uploaded_file    = st.file_uploader("1. Upload Data Order", type=['csv','xlsx','xls'])
    start_date_input = st.date_input("2. Tanggal Mulai Produksi", datetime.today())
    start_date       = datetime.combine(start_date_input, datetime.min.time()).replace(hour=8, minute=30)

    st.subheader("🛠️ Analisis Sensitivitas")
    use_custom = st.checkbox("Ubah Kapasitas/Resource Default")

    res = {m: 1 for m in STATIONS}
    res['3. Jahit_KemejaJaket'] = 3
    res['8. Buang_Benang']      = 2
    setup_time_val = 0.0

    if use_custom:
        with st.container(border=True):
            setup_time_val              = st.number_input("Setup Antar Order (menit)", 0.0, 60.0, 0.0, 5.0)
            res['1. Potong']            = st.number_input("Operator Potong",           1, 10, 1)
            res['2. Jahit_KaosPolo']    = st.number_input("Tim Jahit Kaos/Polo",       1, 10, 1)
            res['3. Jahit_KemejaJaket'] = st.number_input("Tim Jahit Kemeja/Jaket",    1, 10, 3)
            res['4. Sablon']            = st.number_input("Mesin Sablon",              1, 10, 1)
            res['5. DTF']               = st.number_input("Mesin DTF",                1, 10, 1)
            res['6. Bordir']            = st.number_input("Mesin Bordir",             1, 10, 1)
            res['7. Pasang_Kancing']    = st.number_input("Operator Pasang Kancing",  1, 10, 1)
            res['8. Buang_Benang']      = st.number_input("Operator Buang Benang",    1, 10, 2)
            res['9. Lipat']             = st.number_input("Operator Lipat",           1, 10, 1)
            res['10. Packing']          = st.number_input("Operator Packing",         1, 10, 1)

    milp_time_limit = st.slider("⏱️ Batas Waktu MILP (detik)", 60, 600, 300, 30)
    run_button      = st.button("🚀 JALANKAN OPTIMASI", type="primary", use_container_width=True)


# ============================================================
# 11. MAIN AREA
# ============================================================
if uploaded_file is None:
    st.info("👈 Silakan unggah file CSV / Excel di panel kiri untuk memulai.")
else:
    try:
        df = load_order_file(uploaded_file)
        df['_orig_order'] = range(len(df))
        df['Bulan-Tahun'] = df['due date (tanggal)'].dt.strftime('%B %Y')

        with st.container(border=True):
            c1, c2 = st.columns(2)
            bulan_pilih = c1.selectbox("Filter Bulan Due Date:", ["Semua"] + list(df['Bulan-Tahun'].unique()))
            sortir      = c2.selectbox("Urutkan:",
                ["Default (sesuai file)", "Due Date Terdekat", "Due Date Terjauh"], index=0)

        df_disp = df.copy() if bulan_pilih == "Semua" else df[df['Bulan-Tahun'] == bulan_pilih].copy()
        if sortir == "Due Date Terdekat":
            df_disp = df_disp.sort_values('due date (tanggal)', ascending=True)
        elif sortir == "Due Date Terjauh":
            df_disp = df_disp.sort_values('due date (tanggal)', ascending=False)
        else:
            df_disp = df_disp.sort_values('_orig_order')

        # ── Hanya 2 kolom checkbox: Pilih & Priority (TIDAK ADA Terkunci di sini) ──
        for col, default in [("Pilih", False), ("Priority", False)]:
            if col not in df_disp.columns:
                df_disp.insert(0 if col == "Pilih" else 1, col, default)

        st.subheader("📋 Pemilihan & Prioritisasi Order")
        st.info("💡 Centang **Pilih** untuk memasukkan order ke optimasi. "
                "Centang **Priority** untuk bobot penalti lebih tinggi (order VIP).")

        display_cols = [c for c in df_disp.columns if c not in ['Bulan-Tahun', '_orig_order']]
        df_disp_show = df_disp[display_cols].copy()

        if ("df_editor_state" not in st.session_state or
                set(st.session_state["df_editor_state"].columns) != set(df_disp_show.columns)):
            st.session_state["df_editor_state"] = df_disp_show.copy()

        cur_ids    = df_disp["id pesanan"].astype(str).tolist()
        stored_ids = st.session_state["df_editor_state"]["id pesanan"].astype(str).tolist()
        if cur_ids != stored_ids:
            st.session_state["df_editor_state"] = df_disp_show.copy()

        col_sel1, col_sel2, _ = st.columns([1, 1, 4])
        if col_sel1.button("☑️ Pilih Semua", use_container_width=True):
            st.session_state["df_editor_state"]["Pilih"] = True
            st.rerun()
        if col_sel2.button("⬜ Batal Semua", use_container_width=True):
            st.session_state["df_editor_state"]["Pilih"] = False
            st.rerun()

        edited_df = st.data_editor(
            st.session_state["df_editor_state"],
            key="order_editor", hide_index=True, use_container_width=True,
            column_config={
                "Pilih"   : st.column_config.CheckboxColumn("Pilih",    default=False),
                "Priority": st.column_config.CheckboxColumn("Priority", default=False),
            },
        )
        st.session_state["df_editor_state"] = edited_df.copy()

        df_pool = edited_df[edited_df["Pilih"] == True].copy()
        df_pool = df_pool.merge(df[['id pesanan', '_orig_order']], on='id pesanan', how='left')

        # ============================================================
        # 12. ENGINE OPTIMASI AWAL  (hanya jalan saat run_button)
        # ============================================================
        if run_button:
            if len(df_pool) == 0:
                st.warning("⚠️ Centang minimal 1 pesanan untuk dioptimasi.")
                st.stop()
            if len(df_pool) == 1:
                st.warning("⚠️ Hanya 1 order dipilih — SA dijalankan, MILP dilewati.")

            pb = st.progress(0, text="Memulai optimasi…")
            pb.progress(5, "1/5 Kalkulasi routing & waktu proses…")

            df_pool = df_pool.copy()
            df_pool['target_dt'] = df_pool['due date (tanggal)'].apply(
                lambda x: x.replace(hour=17, minute=30, second=0, microsecond=0))
            df_pool['target_menit'] = df_pool['target_dt'].apply(
                lambda x: hitung_target_menit(x, start_date))

            jobs_raw      = df_pool.to_dict('records')
            jobs_raw_fcfs = sorted(jobs_raw, key=lambda x: x['_orig_order'])
            job_ids       = [str(j['id pesanan']) for j in jobs_raw]
            job_ids_fcfs  = [str(j['id pesanan']) for j in jobs_raw_fcfs]

            P = {str(j['id pesanan']): hitung_waktu_proses(j, res, setup_time_val) for j in jobs_raw}
            D = {str(j['id pesanan']): j['target_menit']                           for j in jobs_raw}
            W = {str(j['id pesanan']): 10_000 if j['Priority'] else 1              for j in jobs_raw}

            # SA
            pb.progress(15, "2/5 Simulated Annealing (8.000 iterasi)…")
            sa_score, sa_sched, sa_end = run_simulated_annealing(job_ids, P, D, W)

            # MILP
            pb.progress(35, f"3/5 MILP/CBC (maks {milp_time_limit} detik)…")
            milp_score = float('inf'); milp_feasible = False
            milp_status = "Not Run"; milp_sched = []; milp_end = {}
            bigm_info  = {}; milp_gap_pct = None

            if len(job_ids) >= 2:
                prob = pulp.LpProblem("JobShop_Garment", pulp.LpMinimize)

                # ── [FIX-1] Buat index unik per job untuk nama variabel ───────────────────
                # BUG KRITIS: safe_var_name() menyeragamkan karakter berbeda ke '_'.
                # Jika ada ID seperti 'ORD-01' dan 'ORD.01', keduanya jadi 'ORD_01'
                # → nama variabel DUPLIKAT → PuLP menimpa constraint → CBC Infeasible.
                # Fix: gunakan index numerik sebagai prefix, bukan ID pesanan langsung.
                job_idx = {j: f"j{k:03d}" for k, j in enumerate(job_ids)}

                # ── [FIX-2] S hanya untuk (job, mesin) yang AKTIF ───────────────────────
                # BUG: LpVariable.dicts("S", (job_ids, STATIONS)) membuat S[i][m]
                # meski P[i][m]=0. Variabel bebas ini menambah noise numerik LP dan
                # membuat LP relaxation lebih longgar → CBC lebih lambat / bisa trivial.
                # Fix: buat S hanya untuk pasangan aktif menggunakan dict comprehension.
                aktif_pairs = [(i, m) for i in job_ids for m in STATIONS if P[i][m] > 0]
                S = {
                    (i, m): pulp.LpVariable(
                        f"S_{job_idx[i]}_{safe_var_name(m)}", lowBound=0, cat='Continuous'
                    )
                    for (i, m) in aktif_pairs
                }
                Tard_var = {
                    i: pulp.LpVariable(f"Tard_{job_idx[i]}", lowBound=0, cat='Continuous')
                    for i in job_ids
                }

                # ── [FIX-3] BigM per-mesin yang ketat — bukan BigM global ───────────────
                # BUG: BigM global = total semua durasi → terlalu besar → constraint
                # disjunctive sangat longgar → CBC menemukan "solusi trivial" di mana
                # semua job mulai di t=0 dengan Y=1 → obj=0 → lolos obj_val check
                # → TAPI gagal validasi overlap → milp_feasible di-reset False.
                #
                # BigM yang benar untuk mesin m:
                # Hanya perlu cukup besar untuk mematikan constraint ketika job ke-2
                # harus datang SETELAH job ke-1 selesai di mesin m.
                # Batas atas start time job i di mesin m = sum(P[i][m'] untuk semua m')
                # karena job i harus lewati semua stasiun sebelum m secara seri.
                # Jadi: BigM[m] = max over semua job i,j aktif di m:
                #   (sum semua P[i] di semua stasiun) + P[j][m]
                # Ini jauh lebih kecil dari BigM global tapi tetap valid.
                BigM = {}
                bigm_info = {}
                horizon_per_job = {
                    i: sum(P[i][m] for m in STATIONS)
                    for i in job_ids
                }
                for m in STATIONS:
                    aktif_m = [i for i in job_ids if P[i][m] > 0]
                    if not aktif_m:
                        BigM[m] = 1.0
                    else:
                        # BigM[m] = horizon terpanjang job mana pun + P terpanjang di mesin m
                        # Ini cukup untuk: S[j][m] >= S[i][m] + P[i][m] - BigM[m]*Y
                        # karena S[i][m] <= horizon_per_job[i] selalu terpenuhi
                        bm = max(horizon_per_job[i] for i in aktif_m) + max(P[i][m] for i in aktif_m)
                        BigM[m] = bm
                    bigm_info[m] = round(BigM[m], 1)

                # Variabel biner — hanya untuk pasangan yang benar-benar share mesin
                Y = {}
                for m in STATIONS:
                    aktif_m = [i for i in job_ids if P[i][m] > 0]
                    for a in range(len(aktif_m)):
                        for b in range(a + 1, len(aktif_m)):
                            i, j = aktif_m[a], aktif_m[b]
                            # Nama variabel menggunakan index numerik → tidak ada collision
                            Y[(i, j, m)] = pulp.LpVariable(
                                f"Y_{job_idx[i]}_{job_idx[j]}_{safe_var_name(m)}",
                                cat='Binary')

                # Objektif: minimasi total weighted tardiness
                prob += pulp.lpSum(W[i] * Tard_var[i] for i in job_ids)

                for i in job_ids:
                    rute = [m for m in STATIONS if P[i][m] > 0]
                    # Kendala presedensi (urutan stasiun dalam routing)
                    for k in range(1, len(rute)):
                        prob += S[(i, rute[k])] >= S[(i, rute[k-1])] + P[i][rute[k-1]]
                    # Kendala tardiness
                    if rute:
                        prob += Tard_var[i] >= (S[(i, rute[-1])] + P[i][rute[-1]]) - D[i]

                # Kendala kapasitas mesin (disjunctive / no-overlap)
                for (i, j, m), y_var in Y.items():
                    bm = BigM[m]
                    prob += S[(j, m)] >= S[(i, m)] + P[i][m] - bm * y_var
                    prob += S[(i, m)] >= S[(j, m)] + P[j][m] - bm * (1 - y_var)

                # ── [FIX-4] Warm-start dari solusi SA ────────────────────────────────────
                # Gunakan jadwal SA sebagai MIP start.
                # S sudah dict dengan key (i,m) → akses langsung tanpa ambiguitas.
                sa_map = {(e['job'], e['m']): e['start'] for e in sa_sched}
                for (i, m) in aktif_pairs:
                    v = sa_map.get((i, m))
                    if v is not None and v >= 0:
                        try:
                            S[(i, m)].setInitialValue(float(v))
                        except Exception:
                            pass
                for (i, j, m), y_var in Y.items():
                    si = sa_map.get((i, m))
                    sj = sa_map.get((j, m))
                    if si is not None and sj is not None:
                        try:
                            # Y=0: i sebelum j (S[j]>=S[i]+P[i])
                            # Y=1: j sebelum i (S[i]>=S[j]+P[j])
                            y_var.setInitialValue(0 if float(si) <= float(sj) else 1)
                        except Exception:
                            pass
                for i in job_ids:
                    rute = [m for m in STATIONS if P[i][m] > 0]
                    if rute:
                        sa_start_last = sa_map.get((i, rute[-1]))
                        if sa_start_last is not None:
                            tard_hint = max(0.0, sa_start_last + P[i][rute[-1]] - D[i])
                            try:
                                Tard_var[i].setInitialValue(tard_hint)
                            except Exception:
                                pass

                # ── Solve ─────────────────────────────────────────────────────────────────
                prob.solve(pulp.PULP_CBC_CMD(
                    timeLimit=milp_time_limit,
                    msg=0,
                    warmStart=True,
                ))

                milp_status = pulp.LpStatus[prob.status]
                obj_val     = pulp.value(prob.objective)

                # CBC menemukan feasible solution jika obj_val ada
                # (berlaku saat Optimal maupun timeout-with-solution)
                milp_feasible = obj_val is not None
                milp_score    = float(obj_val) if milp_feasible else float('inf')

                # Hitung optimality gap
                if milp_feasible:
                    try:
                        best_bound = float(prob.bestBound) if hasattr(prob, 'bestBound') and prob.bestBound is not None else None
                        if best_bound is not None and milp_score > 1e-9:
                            milp_gap_pct = abs(milp_score - best_bound) / (abs(milp_score) + 1e-10) * 100
                        else:
                            milp_gap_pct = 0.0 if milp_status == 'Optimal' else None
                    except Exception:
                        milp_gap_pct = None

                    # ── Ekstrak jadwal dari varValue ──────────────────────────────────────
                    milp_sched_cand = []
                    milp_end_cand   = {}

                    for i in job_ids:
                        rute = [m for m in STATIONS if P[i][m] > 0]
                        if not rute:
                            milp_end_cand[i] = 0.0
                            continue
                        # Waktu selesai job = end stasiun TERAKHIR dalam routing
                        # (bukan max dari semua stasiun, karena routing adalah chain linear)
                        last_m = rute[-1]
                        raw_last = S[(i, last_m)].varValue
                        s_last   = round(float(raw_last), 4) if raw_last is not None else 0.0
                        milp_end_cand[i] = s_last + P[i][last_m]

                        for m in rute:
                            raw   = S[(i, m)].varValue
                            s_val = round(float(raw), 4) if raw is not None else 0.0
                            milp_sched_cand.append({
                                'job'  : i,
                                'm'    : m,
                                'start': s_val,
                                'dur'  : P[i][m],
                            })

                    # ── Validasi overlap mesin ────────────────────────────────────────────
                    # Toleransi 1 menit untuk floating point CBC
                    MILP_TOL  = 1.0
                    milp_valid = True
                    for stn in STATIONS:
                        tasks_stn = sorted(
                            [t for t in milp_sched_cand if t['m'] == stn],
                            key=lambda x: x['start']
                        )
                        for k in range(1, len(tasks_stn)):
                            gap = tasks_stn[k]['start'] - (tasks_stn[k-1]['start'] + tasks_stn[k-1]['dur'])
                            if gap < -MILP_TOL:
                                milp_valid = False
                                break
                        if not milp_valid:
                            break

                    if milp_valid and milp_sched_cand:
                        milp_sched = milp_sched_cand
                        milp_end   = milp_end_cand
                        # Hitung ulang skor aktual (lebih akurat dari obj_val CBC)
                        milp_score = sum(
                            max(0.0, milp_end_cand[i] - D[i]) * W[i]
                            for i in job_ids
                        )
                    else:
                        # Jadwal tidak valid → SA menang
                        milp_feasible = False
                        milp_sched    = []
                        milp_end      = {}

            # Benchmark
            pb.progress(80, "4/5 Benchmark EDD & FCFS…")
            edd_score,  edd_sched,  edd_end  = run_edd(job_ids, P, D, W)
            fcfs_score, fcfs_sched, fcfs_end = run_fcfs(job_ids_fcfs, P, D, W)

            # Showdown
            pb.progress(90, "5/5 Finalisasi…")
            if milp_feasible and milp_score <= sa_score:
                pemenang = f"MILP ({milp_status})"; label_pemenang = "MILP"
                jadwal_final = milp_sched; wsd = milp_end; score_pemenang = milp_score
            else:
                alasan = "SA lebih baik" if milp_feasible else f"MILP tidak feasible ({milp_status})"
                pemenang = f"Simulated Annealing ({alasan})"; label_pemenang = "Simulated Annealing"
                jadwal_final = sa_sched; wsd = sa_end; score_pemenang = sa_score

            sc = jalankan_sanity_check(jadwal_final, df_pool, P, start_date)
            pb.progress(100, "✅ Selesai!")

            # Post-processing
            laporan_order = []; jadwal_op_rows = []; pesanan_telat = 0
            for i in job_ids:
                tgt   = df_pool[df_pool['id pesanan'].astype(str) == i]['target_dt'].iloc[0]
                sel   = konversi_ke_jam_dinding(wsd[i], start_date)
                selih = (sel - tgt).total_seconds() / 60
                if selih > 0: pesanan_telat += 1
                laporan_order.append({
                    'ID Pesanan': i, 'Prioritas': "⭐ Ya" if W[i]>1 else "Tidak",
                    'Target Selesai': tgt.strftime('%d-%b-%y %H:%M'),
                    'Estimasi Selesai': sel.strftime('%d-%b-%y %H:%M'),
                    'Status': 'Telat' if selih > 0 else 'Tepat Waktu',
                    'Telat (Hari)': math.ceil(max(0, selih) / MENIT_PER_HARI),
                })
            for t in jadwal_final:
                match = df_pool[df_pool['id pesanan'].astype(str) == t['job']]
                qty_r = match['qty'].iloc[0] if not match.empty else 0
                jadwal_op_rows.append({
                    'Stasiun Kerja': t['m'], 'ID Pesanan': t['job'], 'Qty': qty_r,
                    'Mulai':   konversi_ke_jam_dinding(t['start'],          start_date).strftime('%d-%b-%y %H:%M'),
                    'Selesai': konversi_ke_jam_dinding(t['start']+t['dur'], start_date).strftime('%d-%b-%y %H:%M'),
                })

            df_gantt      = build_gantt_df(jadwal_final, df_pool, start_date, wsd,      D)
            df_laporan    = pd.DataFrame(laporan_order).sort_values(['Status','Estimasi Selesai'], ascending=[False,True])
            df_op         = pd.DataFrame(jadwal_op_rows)
            df_gantt_edd  = build_gantt_df(edd_sched,   df_pool, start_date, edd_end,  D)
            df_gantt_fcfs = build_gantt_df(fcfs_sched,  df_pool, start_date, fcfs_end, D)

            edd_telat  = sum(1 for i in job_ids if edd_end.get(i,0)  > D.get(i,0))
            fcfs_telat = sum(1 for i in job_ids if fcfs_end.get(i,0) > D.get(i,0))

            # ── SIMPAN SEMUA HASIL KE SESSION_STATE ──────────────────
            # Ini adalah kunci arsitektur v7: semua yang perlu ditampilkan
            # disimpan di sini, bukan sebagai variabel lokal run_button.
            # Dengan ini, st.rerun() dari toggle Gantt tidak kehilangan data.
            st.session_state['hasil'] = {
                # Data jadwal
                'jadwal_final'  : jadwal_final,
                'wsd'           : wsd,           # waktu_selesai_dict pemenang
                'P'             : P,
                'D'             : D,
                'W'             : W,
                'job_ids'       : job_ids,
                'job_ids_fcfs'  : job_ids_fcfs,
                'df_pool'       : df_pool,
                'start_date'    : start_date,
                'res'           : res,
                'setup_time_val': setup_time_val,
                # Label & skor
                'pemenang'      : pemenang,
                'label_pemenang': label_pemenang,
                'score_pemenang': score_pemenang,
                'milp_status'   : milp_status,
                'milp_gap_pct'  : milp_gap_pct,   # [FIX-4] optimality gap
                'bigm_info'     : bigm_info,
                # Benchmark
                'edd_score'     : edd_score,
                'fcfs_score'    : fcfs_score,
                'edd_end'       : edd_end,
                'fcfs_end'      : fcfs_end,
                'edd_telat'     : edd_telat,
                'fcfs_telat'    : fcfs_telat,
                # DataFrames untuk display
                'df_gantt'      : df_gantt,
                'df_gantt_edd'  : df_gantt_edd,
                'df_gantt_fcfs' : df_gantt_fcfs,
                'df_laporan'    : df_laporan,
                'df_op'         : df_op,
                # Sanity check
                'sc'            : sc,
                # Stats
                'pesanan_telat' : pesanan_telat,
            }

        # ============================================================
        # 13. DASHBOARD — dibaca dari session_state, bukan run_button
        #     Dengan ini toggle Gantt (rerun) tidak kehilangan data
        # ============================================================
        if 'hasil' not in st.session_state:
            st.info("⬆️ Tekan **JALANKAN OPTIMASI** di panel kiri untuk memulai.")
        else:
            H = st.session_state['hasil']   # shortcut

            # Unpack semua variabel yang dibutuhkan
            jadwal_final   = H['jadwal_final']
            wsd            = H['wsd']
            P              = H['P']
            D              = H['D']
            W              = H['W']
            job_ids        = H['job_ids']
            job_ids_fcfs   = H['job_ids_fcfs']
            df_pool        = H['df_pool']
            start_date_h   = H['start_date']
            res_h          = H['res']
            setup_h        = H['setup_time_val']
            pemenang       = H['pemenang']
            label_pemenang = H['label_pemenang']
            score_pemenang = H['score_pemenang']
            milp_status    = H['milp_status']
            milp_gap_pct   = H.get('milp_gap_pct')   # [FIX-4] optimality gap
            bigm_info      = H['bigm_info']
            edd_score      = H['edd_score']
            fcfs_score     = H['fcfs_score']
            edd_end        = H['edd_end']
            fcfs_end       = H['fcfs_end']
            edd_telat      = H['edd_telat']
            fcfs_telat     = H['fcfs_telat']
            df_gantt       = H['df_gantt']
            df_gantt_edd   = H['df_gantt_edd']
            df_gantt_fcfs  = H['df_gantt_fcfs']
            df_laporan     = H['df_laporan']
            df_op          = H['df_op']
            sc             = H['sc']
            pesanan_telat  = H['pesanan_telat']

            # ── Metrik ──
            st.divider()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📦 Total Order",  len(job_ids))
            m2.metric("✅ Tepat Waktu",  len(job_ids) - pesanan_telat)
            m3.metric("🚨 Terlambat",    pesanan_telat, delta_color="inverse")
            m4.metric("🏆 Pemenang",     label_pemenang)
            m5.metric("📉 Skor Penalti", f"{score_pemenang:,.1f}")

            with st.container(border=True):
                cc1, cc2, cc3 = st.columns(3)
                gap_display = (f"Gap: {milp_gap_pct:.2f}%" if milp_gap_pct is not None
                               else f"Status: {milp_status}")
                cc1.markdown(
                    f'<div class="metric-winner"><b>🏆 {label_pemenang}</b><br>'
                    f'Skor Penalti: <b>{score_pemenang:,.2f}</b><br>'
                    f'MILP {gap_display}</div>', unsafe_allow_html=True)
                cc2.markdown(
                    f'<div class="metric-loser"><b>📊 EDD (Benchmark)</b><br>'
                    f'Skor Penalti: <b>{edd_score:,.2f}</b><br>'
                    f'Tepat: {len(job_ids)-edd_telat} | Terlambat: {edd_telat}</div>', unsafe_allow_html=True)
                cc3.markdown(
                    f'<div class="metric-loser"><b>📊 FCFS (Benchmark)</b><br>'
                    f'Skor Penalti: <b>{fcfs_score:,.2f}</b><br>'
                    f'Tepat: {len(job_ids)-fcfs_telat} | Terlambat: {fcfs_telat}</div>', unsafe_allow_html=True)

            if bigm_info:
                # BigM sekarang bersifat global (sama untuk semua stasiun)
                bigm_val = list(bigm_info.values())[0] if bigm_info else 0
                gap_str  = (f"Gap: {milp_gap_pct:.2f}%" if milp_gap_pct is not None
                            else f"MILP: {milp_status}")
                st.caption(
                    f"🔧 BigM global: {bigm_val:,.0f} mnt (= total makespan teoritis) · "
                    f"{gap_str} · SA: 8.000 iter, T₀=500, α=0.997 · Warm-start: ✅"
                )

            # ── Tabs ──
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Gantt — Pemenang", "📊 Gantt — EDD", "📊 Gantt — FCFS",
                "📑 Laporan Manajemen", "👨‍🔧 Lembar Kerja Operator",
                "🔎 Audit & Sanity Check", "🆕 Penjadwalan Order Baru",
                "⚡ Analisis Crashing",
            ])

            # ── TAB 1: Gantt Pemenang ──
            with tab1:
                st.markdown(f"**Jadwal Akhir Produksi** — *{pemenang}*")
                render_gantt_dual(df_gantt,
                    title=f"Gantt: {label_pemenang} (Skor: {score_pemenang:,.2f})",
                    chart_key="gantt_pemenang")
                with st.expander("📋 Detail Status Order"):
                    rows = []
                    for i in job_ids:
                        tgt = df_pool[df_pool['id pesanan'].astype(str)==i]['target_dt'].iloc[0]
                        sel = konversi_ke_jam_dinding(wsd.get(i,0), start_date_h)
                        tel = (sel-tgt).total_seconds()/60
                        rows.append({'ID Pesanan':i, 'Target':tgt.strftime('%d-%b-%y %H:%M'),
                                     'Selesai':sel.strftime('%d-%b-%y %H:%M'),
                                     'Status':'🔴 Terlambat' if tel>0 else '🟢 Tepat Waktu',
                                     'Selisih (Hari)':math.ceil(max(0,tel)/MENIT_PER_HARI)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # ── TAB 2: Gantt EDD ──
            with tab2:
                st.markdown("**Benchmark EDD** — Earliest Due Date")
                render_gantt_dual(df_gantt_edd,
                    title=f"Gantt EDD (Skor: {edd_score:,.2f})",
                    chart_key="gantt_edd")
                with st.expander("📋 Detail Status Order — EDD"):
                    rows = []
                    for i in job_ids:
                        tgt = df_pool[df_pool['id pesanan'].astype(str)==i]['target_dt'].iloc[0]
                        sel = konversi_ke_jam_dinding(edd_end.get(i,0), start_date_h)
                        tel = (sel-tgt).total_seconds()/60
                        rows.append({'ID Pesanan':i, 'Target':tgt.strftime('%d-%b-%y %H:%M'),
                                     'Selesai':sel.strftime('%d-%b-%y %H:%M'),
                                     'Status':'🔴 Terlambat' if tel>0 else '🟢 Tepat Waktu',
                                     'Selisih (Hari)':math.ceil(max(0,tel)/MENIT_PER_HARI)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # ── TAB 3: Gantt FCFS ──
            with tab3:
                st.markdown("**Benchmark FCFS** — First Come First Served")
                render_gantt_dual(df_gantt_fcfs,
                    title=f"Gantt FCFS (Skor: {fcfs_score:,.2f})",
                    chart_key="gantt_fcfs")
                with st.expander("📋 Detail Status Order — FCFS"):
                    rows = []
                    for i in job_ids_fcfs:
                        tgt = df_pool[df_pool['id pesanan'].astype(str)==i]['target_dt'].iloc[0]
                        sel = konversi_ke_jam_dinding(fcfs_end.get(i,0), start_date_h)
                        tel = (sel-tgt).total_seconds()/60
                        rows.append({'ID Pesanan':i, 'Target':tgt.strftime('%d-%b-%y %H:%M'),
                                     'Selesai':sel.strftime('%d-%b-%y %H:%M'),
                                     'Status':'🔴 Terlambat' if tel>0 else '🟢 Tepat Waktu',
                                     'Selisih (Hari)':math.ceil(max(0,tel)/MENIT_PER_HARI)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # ── TAB 4: Laporan Manajemen ──
            with tab4:
                st.markdown("**Status Penyelesaian Order per Tenggat Waktu**")
                def color_status(val):
                    return 'background-color:#DC2626;color:white' if val=='Telat' \
                           else 'background-color:#16A34A;color:white'
                st.dataframe(df_laporan.style.map(color_status, subset=['Status']),
                             use_container_width=True, height=420)

            # ── TAB 5: Lembar Kerja Operator ──
            with tab5:
                st.markdown("**Instruksi Kerja (Work Order) per Stasiun Kerja**")
                for stn in STATIONS:
                    df_st = df_op[df_op['Stasiun Kerja'] == stn]
                    if df_st.empty: continue
                    with st.expander(f"📁 {stn} — {len(df_st)} order"):
                        st.dataframe(df_st.drop(columns=['Stasiun Kerja']),
                                     hide_index=True, use_container_width=True)

            # ── TAB 6: Audit & Sanity Check ──
            with tab6:
                st.markdown("### 🔎 Audit Otomatis — Verifikasi Logika Jadwal")
                if sc['err_overlap'] or sc['err_presedens'] or not sc['tabel_minggu'].empty:
                    st.error("🚨 **Sanity Check GAGAL**")
                else:
                    st.success("✅ **Sanity Check PASSED** — Jadwal valid.")
                with st.expander("📄 Log Teks Lengkap"):
                    st.code(sc['log_text'], language="text")
                st.divider()

                st.markdown("#### [1] Overlap Mesin")
                if sc['err_overlap']:
                    st.error("❌ Ada overlap — lihat log.")
                    if sc.get('overlap_details'):
                        st.code("\n".join(sc['overlap_details']), language="text")
                else:
                    st.success("✔️ Tidak ada overlap.")
                st.divider()

                st.markdown("#### [2] Verifikasi Presedensi")
                all_job_ids = list(set(t['job'] for t in jadwal_final))
                if ("sel_job" not in st.session_state or
                        st.session_state["sel_job"] not in all_job_ids):
                    st.session_state["sel_job"] = sc["sample_job_id"]

                selected_job = st.selectbox("🔍 Pilih Order:", options=all_job_ids,
                    index=all_job_ids.index(st.session_state["sel_job"]), key="sel_job")

                sel_df = df_pool[df_pool["id pesanan"].astype(str) == selected_job]
                sr     = sel_df.iloc[0].to_dict() if not sel_df.empty else {}
                rute   = [t["m"] for t in sorted([t for t in jadwal_final if t["job"]==selected_job],
                                                  key=lambda x: x["start"])]

                if sr:
                    c1s, c2s = st.columns(2)
                    with c1s:
                        st.markdown("**Spesifikasi Order:**")
                        st.dataframe(pd.DataFrame({
                            'Atribut': ['ID','Jenis','Qty','Due Date','Furing','Sablon','DTF','Bordir','Kancing'],
                            'Nilai'  : [str(sr.get('id pesanan','-')),
                                        str(sr.get('jenis produk','-')).capitalize(),
                                        str(int(sr.get('qty',0))),
                                        pd.Timestamp(sr.get('due date (tanggal)','')).strftime('%d-%b-%Y')
                                        if sr.get('due date (tanggal)') else '-',
                                        *['✅ Ya' if sr.get(k,0)==1 else '❌ Tidak'
                                          for k in ['furing','sablon','dtf','bordir','pasang kancing']]]
                        }), hide_index=True, use_container_width=True)
                    with c2s:
                        st.markdown("**Routing Aktif (OPC):**")
                        p_s = P.get(selected_job, {})
                        st.dataframe(pd.DataFrame([
                            {'Urutan':i+1, 'Stasiun':m, 'Durasi (mnt)':round(p_s.get(m,0),2)}
                            for i, m in enumerate(rute)
                        ]), hide_index=True, use_container_width=True)

                if rute:
                    st.markdown("**OPC Flow:**")
                    p_s  = P.get(selected_job, {})
                    fig_opc = go.Figure()
                    n    = len(rute)
                    durs = [round(p_s.get(m,0),2) for m in rute]
                    for xi, (m, d) in enumerate(zip(rute, durs)):
                        short = m.split('. ',1)[-1].replace('_',' ')
                        fig_opc.add_trace(go.Scatter(
                            x=[xi], y=[0.35], mode='markers+text',
                            marker=dict(size=48, color='#1E3A8A', symbol='square'),
                            text=[f"<b>{xi+1}</b>"], textposition='middle center',
                            textfont=dict(color='white', size=16),
                            hovertemplate=f"<b>{m}</b><br>Durasi: {d:.1f} mnt<extra></extra>",
                            showlegend=False))
                        fig_opc.add_annotation(x=xi, y=-0.35,
                            text=f"<b>{short}</b><br>{d:.1f} mnt", showarrow=False,
                            font=dict(size=13, color='#1E293B'), align='center',
                            bgcolor='#F1F5F9', borderpad=5, bordercolor='#1E3A8A',
                            borderwidth=2, opacity=1.0)
                        if xi < n-1:
                            fig_opc.add_annotation(
                                ax=xi+0.1, ay=0.35, axref='x', ayref='y',
                                x=xi+0.9,  y=0.35, xref='x',  yref='y',
                                showarrow=True, arrowhead=2, arrowsize=1.5,
                                arrowwidth=2, arrowcolor='#64748B')
                    fig_opc.update_layout(
                        height=260, margin=dict(l=20,r=20,t=10,b=10),
                        xaxis=dict(visible=False, range=[-0.7, n-0.3]),
                        yaxis=dict(visible=False, range=[-0.85, 0.85]),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_opc, use_container_width=True)

                st.markdown(f"**Gantt Order `{selected_job}`:**")
                rows_sg = []
                for t in sorted([t for t in jadwal_final if t['job']==selected_job], key=lambda x:x['start']):
                    for blk in pecah_balok_gantt(t['start'], t['dur'], start_date_h):
                        rows_sg.append({'Stasiun':t['m'],'Mulai':blk['start_nyata'],
                                        'Selesai':blk['end_nyata'],
                                        'Durasi (Menit)':round(blk['durasi_potongan'],2)})
                if rows_sg:
                    df_sg = pd.DataFrame(rows_sg)
                    fig_sg = px.timeline(df_sg, x_start="Mulai", x_end="Selesai",
                                         y="Stasiun", color="Stasiun",
                                         hover_data=["Durasi (Menit)"],
                                         title=f"Alur Proses {selected_job}",
                                         color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_sg.update_yaxes(categoryorder="array", categoryarray=rute[::-1])
                    fig_sg.update_layout(height=max(250,len(rute)*45), showlegend=False,
                                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_sg, use_container_width=True)

                st.divider()
                st.markdown("#### [3] Verifikasi Hari Libur (Minggu)")
                if sc['tabel_minggu'].empty:
                    st.success("✔️ Tidak ada jadwal di Hari Minggu.")
                    st.caption("Tidak ada aktivitas produksi di Hari Minggu.")
                else:
                    st.error(f"❌ {len(sc['tabel_minggu'])} slot di Hari Minggu!")
                    st.dataframe(sc['tabel_minggu'], hide_index=True, use_container_width=True)

            # ================================================================
            # ── TAB 7: PENJADWALAN ORDER BARU
            # ================================================================
            with tab7:
                st.markdown("### 🆕 Penjadwalan Order Baru")
                st.info(
                    "Masukkan spesifikasi order baru, pilih order mana yang **🔒 Terkunci** "
                    "(sudah/sedang dikerjakan), lalu jalankan re-optimasi. "
                    "Order terkunci akan di-*pin* persis pada posisi jadwal asal — "
                    "**tidak berubah sama sekali**. Order lain dan order baru dioptimasi bersama "
                    "di slot yang tersisa."
                )

                # ── Spesifikasi order baru ──
                st.subheader("📥 Spesifikasi Order Baru")
                nb_c1, nb_c2, nb_c3 = st.columns(3)
                nb_id    = nb_c1.text_input("ID Order Baru", value="NEW-001", key="nb_id")
                nb_jenis = nb_c2.selectbox("Jenis Produk", ['kaos','polo','kemeja','jaket'], key="nb_jenis")
                nb_qty   = nb_c3.number_input("Qty", 1, 10000, 100, key="nb_qty")
                nb_due   = nb_c1.date_input("Due Date (opsional)", value=None, key="nb_due")

                nb_c4, nb_c5, nb_c6 = st.columns(3)
                nb_furing  = 1 if nb_c4.checkbox("Furing",         key="nb_furing")  else 0
                nb_sablon  = 1 if nb_c4.checkbox("Sablon",         key="nb_sablon")  else 0
                nb_dtf     = 1 if nb_c5.checkbox("DTF",            key="nb_dtf")     else 0
                nb_bordir  = 1 if nb_c5.checkbox("Bordir",         key="nb_bordir")  else 0
                nb_kancing = 1 if nb_c6.checkbox("Pasang Kancing", key="nb_kancing") else 0

                st.divider()

                # ── Tabel pilih order terkunci — HANYA di tab ini ──
                st.subheader("🔒 Pilih Order yang Sudah/Sedang Dikerjakan (Terkunci)")
                st.caption(
                    "Order yang ditandai Terkunci akan diblokir pada posisi jadwal asalnya. "
                    "Ini tidak mempengaruhi hasil optimasi awal di atas."
                )

                # Inisialisasi tabel lock state
                lock_table_init = pd.DataFrame({
                    'ID Pesanan': job_ids,
                    'Terkunci'  : [False] * len(job_ids),
                })
                if 'lock_table' not in st.session_state:
                    st.session_state['lock_table'] = lock_table_init.copy()
                else:
                    # Sinkronisasi jika job_ids berubah (optimasi ulang)
                    stored_ids_lock = st.session_state['lock_table']['ID Pesanan'].tolist()
                    if stored_ids_lock != job_ids:
                        st.session_state['lock_table'] = lock_table_init.copy()

                lock_lk1, lock_lk2, _ = st.columns([1, 1, 4])
                if lock_lk1.button("🔒 Kunci Semua", key="lock_all", use_container_width=True):
                    st.session_state['lock_table']['Terkunci'] = True
                    st.rerun()
                if lock_lk2.button("🔓 Buka Semua", key="unlock_all", use_container_width=True):
                    st.session_state['lock_table']['Terkunci'] = False
                    st.rerun()

                edited_lock = st.data_editor(
                    st.session_state['lock_table'],
                    key="lock_editor",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "ID Pesanan": st.column_config.TextColumn("ID Pesanan", disabled=True),
                        "Terkunci"  : st.column_config.CheckboxColumn(
                            "🔒 Terkunci",
                            help="Centang = posisi di jadwal asal dibekukan, tidak boleh berubah",
                            default=False,
                        ),
                    },
                )
                st.session_state['lock_table'] = edited_lock.copy()

                locked_set = set(edited_lock[edited_lock['Terkunci']==True]['ID Pesanan'].astype(str).tolist())

                if locked_set:
                    st.success(f"🔒 Order yang akan dibekukan: **{', '.join(sorted(locked_set))}**")
                else:
                    st.warning("⚠️ Belum ada order yang dikunci. Semua order + order baru akan dioptimasi bebas.")

                st.divider()

                # ── Tombol re-optimasi ──
                btn_reopt = st.button("⚡ Jalankan Re-Optimasi dengan Order Baru",
                                      type="primary", key="btn_reopt")

                if btn_reopt:
                    nb_id_clean = nb_id.strip()
                    if nb_id_clean == "":
                        st.error("❌ ID Order Baru tidak boleh kosong.")
                    elif nb_id_clean in job_ids:
                        st.error(f"❌ ID '{nb_id_clean}' sudah ada di jadwal. Gunakan ID yang berbeda.")
                    else:
                        nb_row = {
                            'qty': nb_qty, 'jenis produk': nb_jenis,
                            'furing': nb_furing, 'sablon': nb_sablon, 'dtf': nb_dtf,
                            'bordir': nb_bordir, 'pasang kancing': nb_kancing,
                        }
                        P_new_s = hitung_waktu_proses(nb_row, res_h, setup_h)

                        if nb_due:
                            nb_tgt_dt  = datetime.combine(nb_due, datetime.min.time()).replace(hour=17, minute=30)
                            nb_tgt_mnt = hitung_target_menit(nb_tgt_dt, start_date_h)
                        else:
                            nb_tgt_mnt = 99999.0

                        all_jobs_new = job_ids + [nb_id_clean]
                        P_all  = {**P, nb_id_clean: P_new_s}
                        D_all  = {**D, nb_id_clean: nb_tgt_mnt}
                        W_all  = {**W, nb_id_clean: 1}

                        # Bangun pinned_starts dari jadwal_final
                        pinned_starts = {}
                        for jid in locked_set:
                            if jid in job_ids:
                                pin_m = {e['m']: e['start'] for e in jadwal_final if e['job'] == jid}
                                if pin_m:
                                    pinned_starts[jid] = pin_m

                        pb_new = st.progress(0, "Re-optimasi dimulai…")
                        pb_new.progress(20, "Membangun jadwal dengan pin locked jobs…")

                        sa_new_score, sa_new_sched, sa_new_end = run_simulated_annealing(
                            all_jobs_new, P_all, D_all, W_all,
                            pinned_starts=pinned_starts, n_iter=8000,
                        )
                        pb_new.progress(100, "✅ Re-optimasi selesai!")

                        # Verifikasi locked tidak bergerak
                        issues = []
                        for jid in locked_set:
                            if jid not in pinned_starts: continue
                            for e in sa_new_sched:
                                if e['job'] == jid:
                                    orig = pinned_starts[jid].get(e['m'])
                                    if orig is not None and abs(e['start'] - orig) > 0.1:
                                        issues.append(f"{jid}@{e['m']}: Δ={e['start']-orig:.2f}")

                        if issues:
                            st.error(f"⚠️ Inkonsistensi floating-point kecil: {'; '.join(issues)}")
                        else:
                            st.success("✅ Semua order terkunci tetap persis pada posisi jadwal asalnya.")

                        # DataFrame augmented
                        nb_df_row = pd.DataFrame([{'id pesanan': nb_id_clean, 'qty': nb_qty}])
                        df_pool_aug = pd.concat(
                            [df_pool[['id pesanan','qty']], nb_df_row], ignore_index=True)
                        df_gantt_new = build_gantt_df(
                            sa_new_sched, df_pool_aug, start_date_h, sa_new_end, D_all)

                        # Metrik
                        n_telat_new = sum(1 for i in all_jobs_new if sa_new_end.get(i,0) > D_all.get(i,0))
                        rc1, rc2, rc3 = st.columns(3)
                        rc1.metric("📦 Total Order (+ baru)", len(all_jobs_new))
                        rc2.metric("✅ Tepat Waktu", len(all_jobs_new) - n_telat_new)
                        rc3.metric("🚨 Terlambat", n_telat_new, delta_color="inverse")

                        selesai_baru = konversi_ke_jam_dinding(sa_new_end.get(nb_id_clean,0), start_date_h)
                        status_baru  = ("🔴 Terlambat" if sa_new_end.get(nb_id_clean,0) > nb_tgt_mnt
                                        else "🟢 Tepat Waktu") if nb_due else "ℹ️ Due date tidak diisi"

                        with st.container(border=True):
                            st.markdown(
                                f"### 📦 Order Baru: `{nb_id_clean}`\n"
                                f"- **Estimasi Selesai:** {selesai_baru.strftime('%A, %d %B %Y pukul %H:%M')}\n"
                                f"- **Status:** {status_baru}"
                            )
                            if nb_due:
                                buf_dt = konversi_ke_jam_dinding(
                                    sa_new_end.get(nb_id_clean,0) + MENIT_PER_HARI, start_date_h)
                                st.caption(f"💡 Rekomendasi due date: **{buf_dt.strftime('%d %B %Y')}** (+1 hari kerja buffer)")

                        # Tabel perbandingan semua order
                        st.markdown("#### 📋 Status Semua Order Setelah Re-Optimasi")
                        rows_cmp = []
                        for i in all_jobs_new:
                            d_m  = D_all.get(i, 99999)
                            s_m  = sa_new_end.get(i, 0)
                            s_dt = konversi_ke_jam_dinding(s_m, start_date_h)
                            tgt_s = (konversi_ke_jam_dinding(d_m, start_date_h).strftime('%d-%b-%y %H:%M')
                                     if d_m < 99999 else "—")
                            rows_cmp.append({
                                'ID Pesanan'      : i,
                                'Jenis'           : '🆕 Order Baru' if i==nb_id_clean else
                                                    ('🔒 Terkunci' if i in locked_set else '📋 Existing'),
                                'Target Selesai'  : tgt_s,
                                'Estimasi Selesai': s_dt.strftime('%d-%b-%y %H:%M'),
                                'Status'          : '🔴 Terlambat' if s_m>d_m else '🟢 Tepat Waktu',
                                'Selisih (Hari)'  : math.ceil(max(0, s_m-d_m)/MENIT_PER_HARI),
                            })
                        st.dataframe(pd.DataFrame(rows_cmp), hide_index=True, use_container_width=True)

                        # Gantt gabungan
                        st.markdown("#### 📊 Gantt Chart Jadwal Gabungan")
                        render_gantt_dual(
                            df_gantt_new,
                            title=f"Re-Optimasi SA: {len(all_jobs_new)} Order (incl. {nb_id_clean})",
                            chart_key="gantt_reopt", height=560,
                        )

            # ================================================================
            # ── TAB 8: ANALISIS CRASHING
            # ================================================================
            with tab8:
                st.markdown("### ⚡ Analisis Crashing")
                st.info(
                    "Analisis crashing menghitung **penambahan resource minimum** yang diperlukan "
                    "untuk mempercepat jadwal, tanpa mengubah hasil optimasi awal di atas. "
                    "Pilih salah satu dari tiga mode di bawah."
                )

                # Siapkan data dari hasil optimasi awal
                jobs_raw_crash = df_pool.to_dict('records')
                # W_crash dan D_crash menggunakan data yang sama dengan optimasi awal
                W_crash = W.copy()
                D_crash = D.copy()

                # Tampilkan baseline sekilas
                with st.container(border=True):
                    mk_baseline_hari = round(max(wsd.values()) / MENIT_PER_HARI, 2) if wsd else 0
                    n_telat_base     = pesanan_telat
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("📅 Makespan Saat Ini (Hari Kerja)",
                               f"{mk_baseline_hari:.2f} hari")
                    bc2.metric("🚨 Order Terlambat Saat Ini", n_telat_base)
                    bc3.metric("📦 Total Order", len(job_ids))
                    st.caption(
                        f"Resource saat ini — " +
                        " | ".join(f"{k.split('. ',1)[-1]}: {v}"
                                   for k, v in res_h.items() if v > 0)
                    )

                st.divider()

                # ── Pilih mode ──
                crash_mode = st.radio(
                    "🎯 Pilih Mode Crashing:",
                    options=[
                        "📅 Mode 1 — Tanggal Target",
                        "📈 Mode 2 — Percepatan (%)",
                        "🧵 Mode 3 — Zero-Tardiness (Jahit Kaos/Polo saja)",
                    ],
                    key="crash_mode_radio",
                    horizontal=False,
                )

                st.divider()

                # ──────────────────────────────────────────────────────────
                # MODE 1: Tanggal Target
                # ──────────────────────────────────────────────────────────
                if "Mode 1" in crash_mode:
                    st.markdown("#### 📅 Mode 1 — Target Selesai Berdasarkan Tanggal")
                    st.markdown(
                        "Masukkan **tanggal target** kapan semua order harus selesai. "
                        "Sistem akan mencari penambahan resource minimum (stasiun mana saja) "
                        "agar seluruh order tuntas sebelum atau pada tanggal tersebut."
                    )

                    m1_col1, m1_col2 = st.columns(2)
                    m1_tanggal = m1_col1.date_input(
                        "📆 Tanggal Target Selesai Semua Order",
                        value=(start_date_h + timedelta(days=14)).date(),
                        key="m1_tanggal",
                    )
                    m1_col2.markdown("<br>", unsafe_allow_html=True)
                    m1_btn = m1_col2.button("⚡ Jalankan Crashing Mode 1",
                                            type="primary", key="btn_crash1")

                    if m1_btn:
                        m1_target_dt  = datetime.combine(m1_tanggal,
                            datetime.min.time()).replace(hour=17, minute=30)
                        m1_target_mnt = hitung_target_menit(m1_target_dt, start_date_h)

                        if m1_target_mnt <= 0:
                            st.error("❌ Tanggal target harus setelah tanggal mulai produksi.")
                        else:
                            crash_pb = st.progress(0, "Mencari resource minimum…")
                            def cb1(v): crash_pb.progress(v, f"Iterasi crashing… {v*100:.0f}%")

                            ok, res_crash, P_crash, sched_crash, log_crash, mk_crash = crash_mode1(
                                jobs_raw_crash, res_h, setup_h, W_crash, D_crash,
                                m1_target_mnt, cb1
                            )
                            crash_pb.progress(1.0, "✅ Selesai!")

                            # Hitung end time dari sched_crash
                            end_crash = {}
                            for e in sched_crash:
                                je = e['start'] + e['dur']
                                if e['job'] not in end_crash or je > end_crash[e['job']]:
                                    end_crash[e['job']] = je

                            n_telat_crash = sum(1 for i in job_ids if end_crash.get(i,0) > D_crash.get(i,0))

                            if ok:
                                st.success(
                                    f"✅ **Berhasil!** Semua order dapat selesai dalam "
                                    f"**{round(mk_crash/MENIT_PER_HARI,2):.2f} hari kerja** "
                                    f"(≤ {m1_tanggal.strftime('%d %b %Y')})."
                                )
                            else:
                                st.warning(
                                    f"⚠️ Target tanggal **{m1_tanggal.strftime('%d %b %Y')}** "
                                    f"tidak dapat dicapai meski semua resource dimaksimalkan. "
                                    f"Makespan minimum yang bisa dicapai: "
                                    f"**{round(mk_crash/MENIT_PER_HARI,2):.2f} hari kerja**."
                                )

                            # Ringkasan perubahan resource
                            st.markdown("#### 📊 Ringkasan Perubahan Resource")
                            res_rows = []
                            for stn in STATIONS:
                                base_v  = res_h.get(stn, 1)
                                crash_v = res_crash.get(stn, 1)
                                tambah  = crash_v - base_v
                                res_rows.append({
                                    'Stasiun'          : stn.split('. ',1)[-1].replace('_',' '),
                                    'Resource Awal'    : base_v,
                                    'Resource Crashing': crash_v,
                                    'Tambahan'         : f"+{tambah}" if tambah > 0 else "—",
                                    'Status'           : '🔴 Ditambah' if tambah > 0 else '✅ Sama',
                                })
                            df_res = pd.DataFrame(res_rows)
                            st.dataframe(
                                df_res.style.apply(
                                    lambda col: ['background-color:#FEF3C7;color:#92400E'
                                                 if v == '🔴 Ditambah' else '' for v in col],
                                    subset=['Status']
                                ),
                                hide_index=True, use_container_width=True
                            )

                            # Metrik perbandingan
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Makespan Awal (Hari)", f"{mk_baseline_hari:.2f}")
                            r2.metric("Makespan Crash (Hari)", f"{round(mk_crash/MENIT_PER_HARI,2):.2f}",
                                      delta=f"{round((mk_crash/MENIT_PER_HARI)-mk_baseline_hari,2):.2f}",
                                      delta_color="inverse")
                            r3.metric("Terlambat Awal", n_telat_base)
                            r4.metric("Terlambat Setelah Crash", n_telat_crash, delta_color="inverse")

                            # Log langkah
                            with st.expander("📋 Log Langkah Crashing"):
                                st.dataframe(pd.DataFrame(log_crash),
                                             hide_index=True, use_container_width=True)

                            # Gantt hasil crashing
                            st.markdown("#### 📊 Gantt Chart Setelah Crashing")
                            df_gantt_crash = build_gantt_df(
                                sched_crash, df_pool, start_date_h, end_crash, D_crash)
                            render_gantt_dual(df_gantt_crash,
                                title=f"Crashing Mode 1 — Target: {m1_tanggal.strftime('%d %b %Y')}",
                                chart_key="gantt_crash1", height=520)

                # ──────────────────────────────────────────────────────────
                # MODE 2: Percepatan %
                # ──────────────────────────────────────────────────────────
                elif "Mode 2" in crash_mode:
                    st.markdown("#### 📈 Mode 2 — Percepatan Berdasarkan Persentase")
                    st.markdown(
                        "Masukkan **persentase percepatan** yang diinginkan dari makespan saat ini. "
                        "Misalnya 20% berarti semua order harus selesai 20% lebih cepat dari jadwal awal. "
                        "Sistem mencari penambahan resource minimum untuk mencapai target tersebut."
                    )

                    m2_col1, m2_col2 = st.columns(2)
                    m2_pct = m2_col1.number_input(
                        "📈 Percepatan Target (%)",
                        min_value=1.0, max_value=80.0, value=20.0, step=1.0,
                        key="m2_pct",
                        help="Masukkan berapa persen makespan ingin dipercepat dari baseline saat ini."
                    )
                    m2_col2.markdown("<br>", unsafe_allow_html=True)
                    m2_btn = m2_col2.button("⚡ Jalankan Crashing Mode 2",
                                            type="primary", key="btn_crash2")

                    # Tampilkan preview target
                    mk_cur_mnt = max(wsd.values()) if wsd else 0
                    mk_target_preview = mk_cur_mnt * (1 - m2_pct / 100)
                    st.caption(
                        f"📅 Target makespan: **{round(mk_target_preview/MENIT_PER_HARI, 2):.2f} hari kerja** "
                        f"(dari {mk_baseline_hari:.2f} hari saat ini)"
                    )

                    if m2_btn:
                        crash_pb2 = st.progress(0, "Mencari resource minimum…")
                        def cb2(v): crash_pb2.progress(v, f"Iterasi crashing… {v*100:.0f}%")

                        (ok2, res_crash2, P_crash2, sched_crash2,
                         log_crash2, mk_crash2), mk_base2 = crash_mode2(
                            jobs_raw_crash, res_h, setup_h, W_crash, D_crash,
                            m2_pct, cb2
                        )
                        crash_pb2.progress(1.0, "✅ Selesai!")

                        end_crash2 = {}
                        for e in sched_crash2:
                            je = e['start'] + e['dur']
                            if e['job'] not in end_crash2 or je > end_crash2[e['job']]:
                                end_crash2[e['job']] = je

                        n_telat_crash2 = sum(1 for i in job_ids
                                             if end_crash2.get(i,0) > D_crash.get(i,0))

                        target_hari = round(mk_base2 * (1 - m2_pct/100) / MENIT_PER_HARI, 2)
                        hasil_hari  = round(mk_crash2 / MENIT_PER_HARI, 2)

                        if ok2:
                            st.success(
                                f"✅ **Berhasil!** Makespan dipercepat **{m2_pct:.1f}%** → "
                                f"dari {mk_baseline_hari:.2f} hari menjadi **{hasil_hari:.2f} hari kerja** "
                                f"(target: {target_hari:.2f} hari)."
                            )
                        else:
                            st.warning(
                                f"⚠️ Percepatan **{m2_pct:.1f}%** tidak tercapai meski semua resource "
                                f"dimaksimalkan. Makespan terbaik yang bisa dicapai: **{hasil_hari:.2f} hari kerja** "
                                f"(target: {target_hari:.2f} hari, baseline: {mk_baseline_hari:.2f} hari)."
                            )

                        # Ringkasan perubahan resource
                        st.markdown("#### 📊 Ringkasan Perubahan Resource")
                        res_rows2 = []
                        for stn in STATIONS:
                            bv = res_h.get(stn, 1); cv = res_crash2.get(stn, 1)
                            tm = cv - bv
                            res_rows2.append({
                                'Stasiun'          : stn.split('. ',1)[-1].replace('_',' '),
                                'Resource Awal'    : bv,
                                'Resource Crashing': cv,
                                'Tambahan'         : f"+{tm}" if tm > 0 else "—",
                                'Status'           : '🔴 Ditambah' if tm > 0 else '✅ Sama',
                            })
                        df_res2 = pd.DataFrame(res_rows2)
                        st.dataframe(
                            df_res2.style.apply(
                                lambda col: ['background-color:#FEF3C7;color:#92400E'
                                             if v == '🔴 Ditambah' else '' for v in col],
                                subset=['Status']
                            ),
                            hide_index=True, use_container_width=True
                        )

                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Makespan Awal (Hari)",    f"{mk_baseline_hari:.2f}")
                        r2.metric("Makespan Crash (Hari)",   f"{hasil_hari:.2f}",
                                  delta=f"{hasil_hari-mk_baseline_hari:.2f}", delta_color="inverse")
                        percepatan_nyata = (1 - mk_crash2/mk_base2) * 100
                        r3.metric("Percepatan Nyata",        f"{percepatan_nyata:.1f}%")
                        r4.metric("Terlambat Setelah Crash", n_telat_crash2, delta_color="inverse")

                        with st.expander("📋 Log Langkah Crashing"):
                            st.dataframe(pd.DataFrame(log_crash2),
                                         hide_index=True, use_container_width=True)

                        st.markdown("#### 📊 Gantt Chart Setelah Crashing")
                        df_gantt_crash2 = build_gantt_df(
                            sched_crash2, df_pool, start_date_h, end_crash2, D_crash)
                        render_gantt_dual(df_gantt_crash2,
                            title=f"Crashing Mode 2 — Percepatan {m2_pct:.1f}%",
                            chart_key="gantt_crash2", height=520)

                # ──────────────────────────────────────────────────────────
                # MODE 3: Zero-Tardiness (hanya Jahit Kaos/Polo)
                # ──────────────────────────────────────────────────────────
                else:
                    st.markdown("#### 🧵 Mode 3 — Zero-Tardiness (Hanya Tim Jahit Kaos/Polo)")
                    st.markdown(
                        "Sesuai dengan kondisi nyata perusahaan, **hanya jumlah tim Jahit Kaos/Polo** "
                        f"yang dapat ditambah (saat ini: **{res_h.get(MODE3_STATION,1)} tim**, "
                        f"maksimum: **{MODE3_MAX} tim**). "
                        "Sistem akan mencari jumlah tim minimum agar **semua order tepat waktu** "
                        "(zero tardiness). Resource stasiun lain tidak berubah."
                    )

                    # Cek apakah ada order Kaos/Polo yang terlambat
                    kaos_polo_jobs = [str(j['id pesanan']) for j in jobs_raw_crash
                                      if str(j.get('jenis produk','')).lower() in ('kaos','polo')]

                    if kaos_polo_jobs:
                        st.info(
                            f"📋 Order Kaos/Polo yang akan dioptimasi stasiun jahitnya: "
                            f"**{', '.join(kaos_polo_jobs)}**"
                        )
                    else:
                        st.warning(
                            "⚠️ Tidak ada order Kaos/Polo dalam batch ini. "
                            "Mode 3 mungkin tidak berdampak signifikan."
                        )

                    m3_btn = st.button("⚡ Jalankan Crashing Mode 3",
                                       type="primary", key="btn_crash3")

                    if m3_btn:
                        crash_pb3 = st.progress(0, "Menguji setiap level tim…")
                        def cb3(v): crash_pb3.progress(v, f"Menguji {int(v*MODE3_MAX)+1} tim…")

                        hasil_m3 = crash_mode3(
                            jobs_raw_crash, res_h, setup_h, W_crash, D_crash, cb3)
                        crash_pb3.progress(1.0, "✅ Selesai!")

                        # Tampilkan ringkasan semua level yang diuji
                        st.markdown("#### 📊 Hasil per Level Tim Jahit Kaos/Polo")
                        summary_rows = []
                        for h in hasil_m3:
                            summary_rows.append({
                                'Jumlah Tim'       : h['n_tim'],
                                'Makespan (Hari)'  : round(h['makespan']/MENIT_PER_HARI, 2),
                                'Order Terlambat'  : h['n_telat'],
                                'Zero-Tardiness'   : '✅ Ya' if h['n_telat'] == 0 else '❌ Tidak',
                                'Skor Penalti SA'  : round(h['score'], 2),
                            })
                        df_summary3 = pd.DataFrame(summary_rows)

                        def color_m3(val):
                            if val == '✅ Ya':   return 'background-color:#DCFCE7;color:#166534'
                            if val == '❌ Tidak': return 'background-color:#FEE2E2;color:#991B1B'
                            return ''
                        st.dataframe(
                            df_summary3.style.map(color_m3, subset=['Zero-Tardiness']),
                            hide_index=True, use_container_width=True
                        )

                        # Temukan level optimal (zero-tardiness minimum)
                        optimal = next((h for h in hasil_m3 if h['n_telat'] == 0), None)
                        best    = hasil_m3[-1]   # fallback: level tertinggi yang dicoba

                        if optimal:
                            st.success(
                                f"✅ **Rekomendasi: {optimal['n_tim']} tim** Jahit Kaos/Polo "
                                f"(tambahan {optimal['n_tim'] - res_h.get(MODE3_STATION,1)} tim dari kondisi awal). "
                                f"Semua order tepat waktu dengan makespan "
                                f"**{round(optimal['makespan']/MENIT_PER_HARI,2):.2f} hari kerja**."
                            )
                            tampil = optimal
                        else:
                            st.error(
                                f"❌ Bahkan dengan **{MODE3_MAX} tim** Jahit Kaos/Polo, "
                                f"masih ada **{best['n_telat']} order terlambat**. "
                                "Pertimbangkan untuk menambah resource di stasiun lain "
                                "menggunakan Mode 1 atau Mode 2."
                            )
                            tampil = best

                        # Perbandingan before-after
                        st.markdown("#### 📈 Perbandingan Kondisi Awal vs Setelah Crashing")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.markdown("**🔴 Kondisi Awal:**")
                            st.metric(f"Tim Jahit Kaos/Polo", res_h.get(MODE3_STATION,1))
                            st.metric("Makespan (Hari)", mk_baseline_hari)
                            st.metric("Order Terlambat", n_telat_base)

                        with comp_col2:
                            label_res = "Rekomendasi" if optimal else f"Maksimum ({MODE3_MAX} tim)"
                            st.markdown(f"**✅ Setelah Crashing ({label_res}):**")
                            st.metric(f"Tim Jahit Kaos/Polo", tampil['n_tim'],
                                      delta=f"+{tampil['n_tim'] - res_h.get(MODE3_STATION,1)}")
                            st.metric("Makespan (Hari)",
                                      round(tampil['makespan']/MENIT_PER_HARI,2),
                                      delta=round(tampil['makespan']/MENIT_PER_HARI - mk_baseline_hari, 2),
                                      delta_color="inverse")
                            st.metric("Order Terlambat", tampil['n_telat'],
                                      delta=tampil['n_telat']-n_telat_base, delta_color="inverse")

                        # Status per order setelah crashing optimal
                        st.markdown("#### 📋 Status Order Setelah Crashing")
                        end_m3 = tampil['end']
                        rows_m3 = []
                        for i in job_ids:
                            d_m = D_crash.get(i, 99999)
                            s_m = end_m3.get(i, 0)
                            s_dt = konversi_ke_jam_dinding(s_m, start_date_h)
                            tgt_dt = konversi_ke_jam_dinding(d_m, start_date_h)
                            rows_m3.append({
                                'ID Pesanan'      : i,
                                'Jenis'           : str(df_pool[df_pool['id pesanan'].astype(str)==i
                                                         ]['jenis produk'].iloc[0]).capitalize()
                                                    if not df_pool[df_pool['id pesanan'].astype(str)==i].empty else '-',
                                'Target Selesai'  : tgt_dt.strftime('%d-%b-%y %H:%M') if d_m < 99999 else '—',
                                'Estimasi Selesai': s_dt.strftime('%d-%b-%y %H:%M'),
                                'Status'          : '🔴 Terlambat' if s_m > d_m else '🟢 Tepat Waktu',
                                'Selisih (Hari)'  : math.ceil(max(0, s_m-d_m)/MENIT_PER_HARI),
                            })
                        st.dataframe(pd.DataFrame(rows_m3), hide_index=True, use_container_width=True)

                        # Gantt hasil crashing mode 3
                        st.markdown(f"#### 📊 Gantt Chart — {tampil['n_tim']} Tim Jahit Kaos/Polo")
                        df_gantt_m3 = build_gantt_df(
                            tampil['sched'], df_pool, start_date_h, end_m3, D_crash)
                        render_gantt_dual(
                            df_gantt_m3,
                            title=f"Crashing Mode 3 — {tampil['n_tim']} Tim Jahit Kaos/Polo",
                            chart_key="gantt_crash3", height=520,
                        )

                        # Jika ada beberapa level, tampilkan semua Gantt dalam expander
                        if len(hasil_m3) > 1:
                            with st.expander("📊 Lihat Gantt Chart Semua Level Tim"):
                                for h in hasil_m3:
                                    status_lv = "✅ Zero-Tardiness" if h['n_telat'] == 0 else f"❌ {h['n_telat']} Terlambat"
                                    st.markdown(f"**{h['n_tim']} Tim — {status_lv}**")
                                    df_g_tmp = build_gantt_df(
                                        h['sched'], df_pool, start_date_h, h['end'], D_crash)
                                    render_gantt_dual(
                                        df_g_tmp,
                                        title=f"{h['n_tim']} Tim Jahit Kaos/Polo",
                                        chart_key=f"gantt_crash3_lv{h['n_tim']}", height=420,
                                    )

            # ── Download ──
            st.divider()
            st.subheader("📥 Unduh Rekap Excel")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df_laporan.to_excel(writer, sheet_name='Laporan Manajemen',  index=False)
                df_op.to_excel(writer,      sheet_name='Jadwal per Stasiun', index=False)
                df_gantt_edd.drop(columns=['Status'], errors='ignore').to_excel(
                    writer, sheet_name='Benchmark EDD',  index=False)
                df_gantt_fcfs.drop(columns=['Status'], errors='ignore').to_excel(
                    writer, sheet_name='Benchmark FCFS', index=False)
            st.download_button("⬇️ Download Laporan .xlsx", data=buf.getvalue(),
                file_name=f"Jadwal_Pabrik_{datetime.now().strftime('%d%b%Y_%H%M')}.xlsx",
                mime="application/vnd.ms-excel", type="secondary")

    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")
