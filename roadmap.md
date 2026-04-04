# 🧭 0. Prinsip utama (ini fondasi sistem)

Sebelum ke teknis, ini yang harus “tertanam” di sistem kamu:

1.  **Hook > Content**
2.  **Retention > Duration**
3.  **Story > Raw Clip**
4.  **Quality > Quantity (awal)**

👉 Jadi pipeline kamu harus optimize ke:

- “apakah orang nonton sampai habis?”

---

# 🏗️ 1. High-Level Architecture

```
[Input Video]
      ↓
[Transcription]
      ↓
[Semantic + Emotion Analysis]
      ↓
[Clip Scoring Engine]
      ↓
[Clip Selection]
      ↓
[Story Packaging Engine]
      ↓
[Video Rendering Engine]
      ↓
[Final Shorts Output]

```

---

# ⚙️ 2. Detail tiap komponen

## 🔹 2.1 Input Layer

Support:

- local video
- YouTube download
- podcast file

**Preprocess:**

- convert ke mp4 (FFmpeg)
- normalize audio

---

## 🔹 2.2 Transcription Engine

Gunakan:

- `faster-whisper` (recommended: fast + murah)

Output:

```json
[
  {
    "start": 12.4,
    "end": 15.2,
    "text": "most people fail because..."
  }
]
```

👉 penting: word-level timestamp (buat subtitle nanti)

---

## 🔹 2.3 Semantic + Emotion Analysis

Gunakan LLM untuk tagging tiap segment:

Contoh output:

```json
{
  "text": "...",
  "tags": ["advice", "controversial"],
  "emotion": 0.8,
  "virality_score": 0.7
}
```

### Yang dianalisis:

- 🔥 controversial?
- 💡 insight?
- 😡 emotional?
- 💰 money-related?
- ⚠️ shocking?

👉 Ini bahan utama scoring

---

## 🎯 2.4 Clip Scoring Engine (CORE)

Ini “otak” sistem kamu.

Contoh formula awal:

```id="score1"
score =
  (emotion * 0.3) +
  (virality * 0.3) +
  (keyword_match * 0.2) +
  (speech_intensity * 0.2)

```

### Keyword boost (penting banget)

Kasih bobot tinggi untuk kata seperti:

- “nobody tells you”
- “truth”
- “mistake”
- “rich”
- “fail”

---

## ✂️ 2.5 Clip Selection

Strategi:

- ambil top N (misal 10 clip)
- durasi ideal:
  - 15–45 detik

Tambahkan:

- buffer 1–2 detik sebelum & sesudah (biar natural)

---

# 🧠 3. Story Packaging Engine (INI PEMBEDA)

Jangan langsung output clip.

## 🔹 3.1 Hook Generator

Gunakan LLM:

Input:

- transcript clip

Output:

- 3 hook variasi

Contoh:

- “This is why you’re still broke”
- “Nobody talks about this mistake”
- “If you understand this, you’re ahead of 90% people”

👉 pilih best (atau random untuk testing)

---

## 🔹 3.2 Context Enhancement

Tambahkan:

- intro text (0–2 detik)
- optional:
  - “From a $0 to millionaire story…”

---

## 🔹 3.3 Subtitle Generator

WAJIB:

- word-by-word highlight
- uppercase keyword
- timing cepat

Contoh:

```
MOST people FAIL because...

```

---

# 🎬 4. Video Rendering Engine

Gunakan:

- FFmpeg + MoviePy

## Output format:

- 9:16 (1080x1920)
- max 60 detik

---

## 🔹 4.1 Layout

- center: speaker
- bottom: subtitle
- top: hook

---

## 🔹 4.2 Visual enhancement

Auto:

- zoom in tiap 2–3 detik
- cut silence
- add background music ringan

---

## 🔹 4.3 Subtitle style (krusial)

- warna berubah per kata
- highlight keyword
- font besar

👉 ini langsung impact ke retention

---

# 📊 5. Quality Control Layer (advanced tapi penting)

Tambahkan rule:

## Reject clip kalau:

- terlalu flat (emotion < threshold)
- terlalu panjang tanpa punchline
- terlalu banyak filler words

---

# 🧪 6. Iteration Loop (biar makin pintar)

Karena kamu pakai sendiri:

Simpan:

- clip yang kamu pilih
- clip yang kamu buang

👉 nanti bisa jadi dataset untuk:

- fine-tune scoring

---

# ⚡ 7. Stack rekomendasi (practical)

## Backend:

- Python (karena AI ecosystem kuat)

## Tools:

- Whisper → transcription
- OpenAI / LLM → analysis + hook
- FFmpeg → rendering
- MoviePy → editing

## Optional:

- Redis → queue
- RocksDB → fast local state (kalau batch besar)

---

# ⏱️ 8. Target performa realistis

Untuk 1 video panjang (1 jam):

- transcription: ~3–5 menit
- analysis: ~2–4 menit
- rendering: ~5–10 menit

👉 total: ~10–20 menit

---

# 🔥 9. Definition of “berhasil”

Output kamu harus punya:

✔ Hook kuat di 2 detik pertama  
✔ Subtitle engaging  
✔ Tempo cepat  
✔ Punchline jelas  
✔ Durasi optimal

Kalau nggak → jangan publish

---

# 🧭 10. Roadmap build (biar nggak overengineering)

## Phase 1 (MVP – WAJIB)

- transcription
- simple scoring
- basic clip
- subtitle

👉 target: “udah usable”

---

## Phase 2

- hook generator
- better scoring
- auto zoom

---

## Phase 3

- emotion detection
- retention prediction
- multi-style output

---

# 💡 Insight terakhir (ini penting banget)

Banyak engineer gagal di sini:

> terlalu fokus ke “AI sophistication”  
> tapi lupa “content taste”

👉 Saran gue:

- tetap manual review output
- train “sense of viral”
