import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
import io
import os
import tempfile

# ==========================================
# 1. KONFIGURASI DAN ARSITEKTUR MODEL
# ==========================================
SAMPLE_RATE   = 16000
N_FFT         = 512
HOP_LENGTH    = 128
PATCH_SIZE    = 64
PATCH_STRIDE  = 32

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class DnCNNPlus(nn.Module):
    def __init__(self, depth=20, n_channels=96, in_channels=1, out_channels=1, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size, padding=pad, bias=True),
            nn.ReLU(inplace=True)
        )
        mid_layers = []
        for i in range(depth - 2):
            if i % 4 == 3:
                mid_layers.append(ResidualBlock(n_channels))
            else:
                mid_layers += [
                    nn.Conv2d(n_channels, n_channels, kernel_size, padding=pad, bias=False),
                    nn.BatchNorm2d(n_channels),
                    nn.ReLU(inplace=True),
                ]
        mid_layers.append(nn.Dropout2d(p=0.1))
        self.middle = nn.Sequential(*mid_layers)
        self.tail = nn.Conv2d(n_channels, out_channels, kernel_size, padding=pad, bias=True)

    def forward(self, x):
        noise = self.tail(self.middle(self.head(x)))
        return x - noise

# ==========================================
# 2. PERSIAPAN MODEL AI (OTAK)
# ==========================================
device = torch.device('cpu')

@st.cache_resource 
def load_model():
    model = DnCNNPlus(depth=20, n_channels=96).to(device)
    model.load_state_dict(torch.load('dncnn_best.pth', map_location=device))
    model.eval()
    return model

model = load_model()

# ==========================================
# 3. FUNGSI UTAMA PEMBERSIH SUARA
# ==========================================
def enhance_audio(uploaded_file, model):
    # Ambil ekstensi asli file (misal: .m4a, .mp3, .wav)
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Buat file sementara di harddisk lokal agar librosa bisa memanggil decoder internal
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Jalankan pembacaan file dari path fisik lokal
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
    finally:
        # Segera hapus file sementara setelah berhasil dimuat ke memori
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Proses normalisasi amplitudo
    y = y / (np.max(np.abs(y)) + 1e-10)

    # Ekstraksi Fitur Spektrogram
    D          = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
    mag, phase = np.abs(D), np.angle(D)
    log_mag    = np.log1p(mag)

    # Normalisasi Z-Score
    mu, std    = log_mag.mean(), log_mag.std() + 1e-8
    log_mag_n  = (log_mag - mu) / std

    freq_bins, T = log_mag_n.shape
    output_mag   = np.zeros_like(log_mag_n)
    weight_map   = np.zeros_like(log_mag_n)
    hann_win     = np.hanning(PATCH_SIZE)[np.newaxis, :]

    # Proses Pembersihan oleh Model AI per-Patch
    with torch.no_grad():
        for start in range(0, T - PATCH_SIZE + 1, PATCH_STRIDE):
            patch = log_mag_n[:, start:start + PATCH_SIZE]
            patch = np.clip(patch, -5, 5)
            p_t   = torch.tensor(patch[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
            enh   = model(p_t).squeeze().numpy()

            output_mag[:, start:start + PATCH_SIZE] += enh * hann_win
            weight_map[:, start:start + PATCH_SIZE] += np.ones_like(enh) * hann_win

    # Rekonstruksi Kembali Menjadi Gelombang Suara (ISTFT)
    output_mag_n    = output_mag / np.maximum(weight_map, 1e-8)
    output_mag_orig = output_mag_n * std + mu
    output_linear   = np.expm1(np.maximum(output_mag_orig, 0))

    y_enhanced = librosa.istft(output_linear * np.exp(1j * phase), hop_length=HOP_LENGTH, n_fft=N_FFT)
    
    # Simpan ke dalam buffer memori WAV untuk diserahkan ke Streamlit
    out_buffer = io.BytesIO()
    sf.write(out_buffer, y_enhanced, SAMPLE_RATE, format='WAV')
    return out_buffer

# ==========================================
# 4. TAMPILAN WEB (UI) - VERSI PREMIUM
# ==========================================
st.set_page_config(page_title="AI Speech Enhancement", page_icon="🎙️", layout="wide")

st.markdown("""
<div style="text-align: center; padding-bottom: 20px;">
    <h1 style="color: #4CAF50;">🎙️ Sistem Cerdas Pembersih Suara (DnCNN)</h1>
    <p style="color: gray; font-size: 18px;">Unggah rekaman suara yang berisik, dan biarkan AI memisahkan suara manusia dari bising latar belakang.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Pintu uploader dibuka untuk format WAV dan format rekaman HP (.mp3, .m4a, dll)
uploaded_file = st.file_uploader("📂 Pilih atau Tarik File Audio ke Sini", 
                                 type=['wav', 'mp3', 'm4a', 'ogg', 'aac', 'flac'])

if uploaded_file is not None:
    st.info("")
    
    # Pembagian layout kolom Bersebelahan (Kiri & Kanan)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔊 Audio Asli (Berisik)")
        st.audio(uploaded_file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        mulai_proses = st.button("Bersihkan Suara Sekarang", use_container_width=True, type="primary")

    with col2:
        st.subheader("Audio Bersih (Output AI)")
        
        if mulai_proses:
            with st.spinner("AI sedang memproses suara... Mohon tunggu..."):
                # Menjalankan fungsi pembersih
                clean_audio_buffer = enhance_audio(uploaded_file, model)
                
                st.success("✅ Selesai! Dengarkan hasilnya di bawah ini:")
                st.audio(clean_audio_buffer, format='audio/wav')
                
                # Menambahkan tombol download untuk memudahkan dosen mengunduh hasilnya
                st.download_button(
                    label="📥 Unduh Audio Bersih",
                    data=clean_audio_buffer.getvalue(),
                    file_name="hasil_audio_bersih.wav",
                    mime="audio/wav",
                    use_container_width=True
                )