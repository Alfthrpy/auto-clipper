"""
Content Profiles — niche-aware presets for semantic scoring, hook generation,
and metadata generation.

Each profile defines:
  - scoring_dimensions:  rubric text + blend weights for the semantic scorer
  - hook_frameworks:     framework definitions for the hook generator
  - hook_signals:        keyword signals to auto-select frameworks
  - audience / tone:     context strings injected into every LLM prompt
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContentProfile:
    """Immutable descriptor for a content niche."""
    niche: str
    label: str                          # human-readable name
    audience: str                       # injected into prompts
    tone: str                           # injected into prompts
    scoring_rubric: str                 # full rubric with anchor points
    scoring_weights: dict[str, float]   # dimension → weight
    scoring_dimensions: list[str]       # ordered list of dimension keys
    hook_frameworks: list[dict[str, Any]]
    hook_signals: dict[str, list[str]]
    few_shot_example: str               # scoring few-shot


# ═══════════════════════════════════════════════════════════════════════════
# PRESET: money  (existing behavior, migrated from hardcoded)
# ═══════════════════════════════════════════════════════════════════════════

_MONEY_RUBRIC = """\
NICHE CONTEXT: Money, Business, and Smart Financial/Economic Insight content.
Score each segment from 0–10 using these anchor points:

[insight_depth] — How valuable and substantive is the financial/business knowledge here?
  0  : Zero substance. Empty motivation, vague platitudes, or filler talk.
  3  : Surface-level. Common knowledge most people already have (e.g., "spend less than you earn").
  5  : Decent point. Useful but not surprising — something you'd find in any finance article.
  7  : Genuinely valuable. Specific mechanism, surprising statistic, or non-obvious insight.
  10 : Mind-opening. Reveals how money/systems actually work in a way most people never knew.
       Example: explaining how inflation is a hidden tax, or how compound interest works against debtors.

[viral_potential] — How likely is this to make a money-conscious viewer stop scrolling and share?
  0  : Completely forgettable. No emotional pull, no tension, nothing to react to.
  3  : Mildly interesting but no reason to share or save.
  5  : Solid content. Worth watching but won't trigger a strong share impulse.
  7  : Strong hook potential. Has a clear "wait, what?" or "I need to show this to someone" moment.
  10 : Explosive stop-scroll energy. Triggers immediate emotional reaction (anger, shock, or revelation).
       Example: a claim that directly contradicts what most people do with their money.

[contrarian_edge] — Does it challenge mainstream financial advice or conventional wisdom?
  0  : Completely safe and agreeable. Nobody would push back.
  3  : Slightly opinionated but not really debate-worthy.
  5  : Takes a clear stance — some people would agree, some would disagree.
  7  : Bold financial take. Challenges popular advice (buying a house, saving in a bank, etc.)
  10 : Polarizing. Will generate passionate comments from both "finally someone said it" and "this is wrong".
       Example: "Mutual funds are designed to make fund managers rich, not you."
"""

_MONEY_FEW_SHOT = """\
INPUT EXAMPLE:
[{"id": 42, "text": "Most people think a savings account protects their money, but with inflation running at 5% and your bank paying 0.5% interest, you're actually losing purchasing power every single year. The bank profits from lending your money at 10% while paying you almost nothing."}]

OUTPUT EXAMPLE:
{"results": [{"id": 42, "insight_depth": 8.5, "viral_potential": 7.5, "contrarian_edge": 7.0}]}
"""

_MONEY_HOOK_FRAMEWORKS = [
    {
        "name": "wealth_revelation",
        "instruction": (
            "Reveal a financial truth, money mechanic, or economic insight that most "
            "people are completely unaware of. Frame it as something hidden, suppressed, "
            "or overlooked — not taught in school, not discussed in mainstream media."
        ),
        "examples_id": [
            "Bank tidak pernah ngajarin ini ke kamu",
            "Cara orang kaya nyimpen uang beda banget",
            "Inflasi diam-diam nyuri kekayaanmu tiap hari",
            "Ini yang bikin gaji besar tapi tetap bokek",
        ],
        "examples_en": [
            "Banks will never tell you this exists",
            "The wealthy use a money rule no one teaches",
            "Your savings account is quietly draining you",
            "This is why high earners still go broke",
        ],
    },
    {
        "name": "contrarian_finance",
        "instruction": (
            "Challenge a widely-held belief about money, success, or business that "
            "most people treat as gospel. The take must be bold and defensible — "
            "not just provocative for its own sake."
        ),
        "examples_id": [
            "Rajin nabung justru bikin kamu miskin",
            "Kerja keras bukan cara jadi kaya",
            "Utang itu bukan musuh — salah paham terbesar soal duit",
            "Beli rumah sekarang bisa jadi keputusan terburukmu",
        ],
        "examples_en": [
            "Saving money is keeping you poor",
            "Hard work alone has never made anyone wealthy",
            "Debt is not your enemy — you've been lied to",
            "Buying a house right now might ruin you financially",
        ],
    },
    {
        "name": "system_expose",
        "instruction": (
            "Expose how an economic system, financial institution, or business structure "
            "is designed in a way that disadvantages ordinary people — or advantages "
            "those who understand it. Make it feel like pulling back a curtain."
        ),
        "examples_id": [
            "Sistem ini dirancang biar kamu tetap butuh gaji",
            "Kenapa pajak orang kaya lebih kecil dari kamu",
            "Cara perusahaan besar bayar nol pajak secara legal",
            "Ini kenapa harga rumah terus naik dan sengaja dibiarkan",
        ],
        "examples_en": [
            "The system is designed to keep you employed forever",
            "Why the wealthy pay less tax than you legally",
            "How billion-dollar companies pay zero tax — legally",
            "Housing prices don't rise by accident — here's why",
        ],
    },
    {
        "name": "insight_urgency",
        "instruction": (
            "Create a sense that the viewer is currently missing out, making a costly "
            "mistake, or losing money right now due to lack of this specific knowledge. "
            "The urgency must be grounded in real economic or financial stakes."
        ),
        "examples_id": [
            "Tiap bulan kamu tunda ini, kerugianmu makin besar",
            "Keputusan finansial ini bisa beda hasilnya 20 tahun lagi",
            "Orang-orang yang skip ini nyesel 5 tahun kemudian",
            "Satu kesalahan ini bisa biaya kamu ratusan juta",
        ],
        "examples_en": [
            "Every month you delay this costs you more",
            "This one decision separates you 20 years from now",
            "People who ignored this regretted it within five years",
            "One financial mistake quietly costing you millions",
        ],
    },
]

_MONEY_SIGNALS = {
    "contrarian_finance": [
        "saving", "save", "work hard", "hustle", "buy house", "property",
        "nabung", "kerja keras", "beli rumah", "investasi saham", "kripto",
    ],
    "system_expose": [
        "tax", "bank", "government", "corporation", "institution", "policy",
        "pajak", "pemerintah", "korporasi", "sistem", "regulasi",
    ],
    "insight_urgency": [
        "delay", "late", "miss", "regret", "opportunity", "compound",
        "terlambat", "rugi", "nyesel", "kesempatan", "bunga",
    ],
}

MONEY_PROFILE = ContentProfile(
    niche="money",
    label="Money / Business / Financial Insight",
    audience="people who want to understand money, build wealth, or navigate the economy smarter",
    tone="sharp, confident, slightly provocative — NOT motivational fluff or generic advice",
    scoring_rubric=_MONEY_RUBRIC,
    scoring_weights={"insight_depth": 0.45, "viral_potential": 0.30, "contrarian_edge": 0.25},
    scoring_dimensions=["insight_depth", "viral_potential", "contrarian_edge"],
    hook_frameworks=_MONEY_HOOK_FRAMEWORKS,
    hook_signals=_MONEY_SIGNALS,
    few_shot_example=_MONEY_FEW_SHOT,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRESET: tech
# ═══════════════════════════════════════════════════════════════════════════

_TECH_RUBRIC = """\
NICHE CONTEXT: Technology, Software, AI, Gadgets, and Digital Innovation content.
Score each segment from 0–10 using these anchor points:

[technical_value] — How useful or informative is the technical knowledge here?
  0  : Zero substance. Vague hype or empty marketing speak.
  3  : Surface-level. Common knowledge (e.g., "AI is changing the world").
  5  : Decent — useful tip or explanation you'd find in a blog post.
  7  : Genuinely valuable. Specific technique, tool comparison, or non-obvious workflow.
  10 : Expert-level insight. Reveals inner workings, architecture, or approach most devs/users never discover.
       Example: explaining how browser rendering pipeline actually prioritizes content paint.

[wow_factor] — How likely is this to make a tech-savvy viewer stop scrolling and share?
  0  : Completely boring. No novelty.
  3  : Mildly interesting but nothing new.
  5  : Solid — worth watching, but won't blow anyone's mind.
  7  : Impressive demo, surprising capability, or a "wait, you can do THAT?" moment.
  10 : Mind-blowing. Makes the viewer immediately want to try it or show someone.
       Example: a 30-second demo of a tool that automates something people thought required hours.

[practical_use] — Can the viewer immediately apply or benefit from this?
  0  : Pure theory with no actionable takeaway.
  3  : Vaguely useful but needs too much extra context.
  5  : Practical for a specific audience segment.
  7  : Widely applicable — most tech enthusiasts could use this right away.
  10 : Universal game-changer. Saves real time or money for anyone who watches.
       Example: a hidden OS shortcut that doubles productivity.
"""

_TECH_FEW_SHOT = """\
INPUT EXAMPLE:
[{"id": 7, "text": "Most people don't know this, but if you hold Ctrl+Shift+T in Chrome it reopens your last closed tab. But here's the real trick — it works with MULTIPLE tabs, and it remembers the full session even after a crash."}]

OUTPUT EXAMPLE:
{"results": [{"id": 7, "technical_value": 5.0, "wow_factor": 7.0, "practical_use": 9.0}]}
"""

_TECH_HOOK_FRAMEWORKS = [
    {
        "name": "mind_blown",
        "instruction": (
            "Reveal a tech capability, tool, or trick that most people had no idea existed. "
            "Make them feel like they've been missing out on something obvious."
        ),
        "examples_id": [
            "Fitur ini udah ada dari dulu tapi gak ada yang tau",
            "AI sekarang bisa lakuin ini dan ini gila",
            "Laptop kamu bisa ini tapi kamu gak pernah coba",
            "Tool gratis ini harusnya udah viral dari dulu",
        ],
        "examples_en": [
            "This feature existed for years and nobody knew",
            "AI can now do this and it's insane",
            "Your laptop can do this but you never tried",
            "This free tool should have gone viral years ago",
        ],
    },
    {
        "name": "future_tech",
        "instruction": (
            "Highlight how a technology is shaping the near future in a way that feels "
            "both exciting and slightly unsettling. Make it feel urgent and real."
        ),
        "examples_id": [
            "Dalam 2 tahun pekerjaan ini bakal hilang",
            "Teknologi ini bakal ubah cara kamu kerja selamanya",
            "Yang terjadi di AI minggu ini beneran serem",
            "Dunia 5 tahun lagi gak bakal sama",
        ],
        "examples_en": [
            "This job will not exist in two years",
            "This tech will change how you work forever",
            "What happened in AI this week is genuinely scary",
            "The world in five years will be unrecognizable",
        ],
    },
    {
        "name": "hidden_feature",
        "instruction": (
            "Surface an underused or buried feature in popular software/hardware that "
            "provides immediate, tangible value. Make the viewer feel smart for learning it."
        ),
        "examples_id": [
            "Setting tersembunyi di HP kamu yang wajib diubah",
            "Trik Windows ini bikin kerja 2x lebih cepet",
            "Fitur rahasia Google yang jarang orang pakai",
            "Cara pakai ChatGPT yang bener beda dari yang kamu lakuin",
        ],
        "examples_en": [
            "The hidden phone setting you need to change now",
            "This Windows trick makes you work twice as fast",
            "A secret Google feature almost nobody uses",
            "You're using ChatGPT wrong — here's the real way",
        ],
    },
    {
        "name": "productivity_hack",
        "instruction": (
            "Share a workflow, tool combination, or automation that dramatically improves "
            "productivity. It must feel achievable, not theoretical."
        ),
        "examples_id": [
            "Workflow ini hemat 3 jam kerja tiap hari",
            "Automasi sederhana yang harusnya semua orang pakai",
            "Setup kerja ini bikin gue gak pernah lembur lagi",
            "3 tool gratis yang gantiin software jutaan",
        ],
        "examples_en": [
            "This workflow saves three hours every single day",
            "A simple automation everyone should be using",
            "This setup means I never work overtime anymore",
            "Three free tools that replace expensive software",
        ],
    },
]

_TECH_SIGNALS = {
    "mind_blown": [
        "ai", "machine learning", "neural", "gpt", "chatgpt", "model",
        "kecerdasan buatan", "algoritma", "deep learning",
    ],
    "future_tech": [
        "future", "replace", "automate", "job", "robot", "autonomous",
        "masa depan", "otomatis", "pekerjaan", "digantikan",
    ],
    "hidden_feature": [
        "setting", "feature", "shortcut", "trick", "hack", "tip",
        "fitur", "trik", "rahasia", "tersembunyi",
    ],
    "productivity_hack": [
        "workflow", "productivity", "automate", "tool", "app", "setup",
        "produktif", "efisien", "hemat waktu", "aplikasi",
    ],
}

TECH_PROFILE = ContentProfile(
    niche="tech",
    label="Technology / Software / AI / Gadgets",
    audience="tech enthusiasts, developers, and digital workers who want to stay ahead",
    tone="excited but grounded, demo-driven — NOT corporate marketing or hype",
    scoring_rubric=_TECH_RUBRIC,
    scoring_weights={"technical_value": 0.40, "wow_factor": 0.35, "practical_use": 0.25},
    scoring_dimensions=["technical_value", "wow_factor", "practical_use"],
    hook_frameworks=_TECH_HOOK_FRAMEWORKS,
    hook_signals=_TECH_SIGNALS,
    few_shot_example=_TECH_FEW_SHOT,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRESET: gaming
# ═══════════════════════════════════════════════════════════════════════════

_GAMING_RUBRIC = """\
NICHE CONTEXT: Gaming content — clips, tips, commentary, and entertainment.
Score each segment from 0–10 using these anchor points:

[entertainment_value] — How entertaining or engaging is this moment?
  0  : Dead air. Nothing interesting happening.
  3  : Filler talk or mundane gameplay with no stakes.
  5  : Decent moment — worth including but forgettable.
  7  : Genuinely funny, hype, or exciting — makes you want to keep watching.
  10 : Peak content. Clip-worthy moment that gets clipped and shared everywhere.
       Example: insane clutch play, perfect comedic timing, or a rage moment.

[skill_showcase] — Does it demonstrate impressive skill or game knowledge?
  0  : No skill or knowledge displayed.
  3  : Basic gameplay anyone could do.
  5  : Above average — shows competence.
  7  : Impressive play or deep game knowledge that makes viewers respect the player.
  10 : Pro-level or galaxy-brain play that makes viewers say "HOW did they do that?"
       Example: pixel-perfect execution, 200 IQ strategy, or frame-perfect trick.

[community_appeal] — Will the gaming community engage with this (comment, debate, share)?
  0  : No community relevance.
  3  : Generic content that doesn't spark discussion.
  5  : Relatable to gamers but not particularly shareable.
  7  : Will get comments — hot take, meta discussion, or community drama.
  10 : Will blow up in community — controversial tier list, broken exploit, or meme-worthy moment.
       Example: discovering a game-breaking glitch or exposing a hidden mechanic.
"""

_GAMING_FEW_SHOT = """\
INPUT EXAMPLE:
[{"id": 15, "text": "Okay so most people don't know this but if you cancel your reload animation with a melee swap you actually save like half a second per reload and in a high-level lobby that's literally the difference between winning and losing the gunfight"}]

OUTPUT EXAMPLE:
{"results": [{"id": 15, "entertainment_value": 6.0, "skill_showcase": 7.5, "community_appeal": 8.0}]}
"""

_GAMING_HOOK_FRAMEWORKS = [
    {
        "name": "epic_moment",
        "instruction": (
            "Hype up an incredible gameplay moment — clutch, fail, or unexpected twist. "
            "Make the viewer feel the stakes and want to see what happens."
        ),
        "examples_id": [
            "Momen paling gila yang pernah terjadi di game ini",
            "Gue gak percaya ini beneran kejadian",
            "1 HP dan musuh tinggal 4 orang",
            "Plot twist yang bikin gue teriak",
        ],
        "examples_en": [
            "The craziest moment to ever happen in this game",
            "I cannot believe this actually happened",
            "1 HP left and four enemies remaining",
            "The plot twist that made me scream",
        ],
    },
    {
        "name": "pro_tip",
        "instruction": (
            "Share a game tip, trick, or mechanic that gives a real competitive advantage. "
            "Make the viewer feel like they unlocked a cheat code."
        ),
        "examples_id": [
            "Trik ini bikin rank kamu naik 2x lebih cepet",
            "99 persen player gak tau mekanik ini",
            "Setting tersembunyi yang bikin aim jauh lebih bagus",
            "Cara main yang pro pakai tapi gak pernah di-share",
        ],
        "examples_en": [
            "This trick makes you rank up twice as fast",
            "99 percent of players don't know this mechanic",
            "A hidden setting that massively improves your aim",
            "How pros play but never share publicly",
        ],
    },
    {
        "name": "rage_bait",
        "instruction": (
            "Take a strong, slightly provocative opinion about a game, meta, or gaming culture "
            "that will make people rush to the comments. Must be defensible, not random."
        ),
        "examples_id": [
            "Game ini overrated dan kalian tau itu",
            "Meta sekarang itu broken dan dev gak peduli",
            "Character ini harusnya di-nerf dari dulu",
            "Unpopular opinion tapi ini game terbaik tahun ini",
        ],
        "examples_en": [
            "This game is overrated and you know it",
            "The current meta is broken and devs don't care",
            "This character should have been nerfed ages ago",
            "Unpopular opinion but this is game of the year",
        ],
    },
    {
        "name": "underrated_find",
        "instruction": (
            "Spotlight an underrated game, weapon, strategy, or feature that deserves more "
            "attention. Make the viewer want to try it immediately."
        ),
        "examples_id": [
            "Senjata ini tidur tapi diam-diam OP banget",
            "Game indie ini lebih bagus dari AAA manapun",
            "Strategi yang gak ada yang pakai padahal broken",
            "Item ini di-skip semua orang padahal best in slot",
        ],
        "examples_en": [
            "This weapon is secretly the most OP thing in the game",
            "This indie game is better than any AAA release",
            "A strategy nobody uses that's actually broken",
            "Everyone skips this item but it's best in slot",
        ],
    },
]

_GAMING_SIGNALS = {
    "epic_moment": [
        "clutch", "win", "kill", "ace", "insane", "gg",
        "menang", "gila", "momen", "epic", "clutch",
    ],
    "pro_tip": [
        "tip", "trick", "guide", "how to", "best", "setting",
        "trik", "cara", "setting", "guide", "tutorial",
    ],
    "rage_bait": [
        "worst", "broken", "nerf", "buff", "overrated", "trash",
        "sampah", "overrated", "broken", "nerf", "buff",
    ],
    "underrated_find": [
        "underrated", "hidden", "secret", "nobody", "sleeper",
        "tidur", "rahasia", "gak ada yang tau", "indie",
    ],
}

GAMING_PROFILE = ContentProfile(
    niche="gaming",
    label="Gaming — Clips, Tips, and Commentary",
    audience="gamers who want entertainment, tips, and community discussion",
    tone="high-energy, authentic gamer voice — NOT scripted or corporate",
    scoring_rubric=_GAMING_RUBRIC,
    scoring_weights={"entertainment_value": 0.45, "skill_showcase": 0.25, "community_appeal": 0.30},
    scoring_dimensions=["entertainment_value", "skill_showcase", "community_appeal"],
    hook_frameworks=_GAMING_HOOK_FRAMEWORKS,
    hook_signals=_GAMING_SIGNALS,
    few_shot_example=_GAMING_FEW_SHOT,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRESET: self_improvement
# ═══════════════════════════════════════════════════════════════════════════

_SELF_IMPROVEMENT_RUBRIC = """\
NICHE CONTEXT: Self-Improvement, Personal Development, Mindset, and Life Advice content.
Score each segment from 0–10 using these anchor points:

[actionability] — Can the viewer immediately apply this to their life?
  0  : Pure fluff. Motivational poster energy with no substance.
  3  : Vague advice like "just be confident" or "believe in yourself".
  5  : Decent tip but requires significant context to execute.
  7  : Specific, actionable advice with clear steps or a framework.
  10 : Life-changing action item. Clear, doable, and powerful.
       Example: a specific morning routine hack with measurable results.

[emotional_resonance] — Does it trigger a genuine emotional response?
  0  : Emotionally flat. No connection.
  3  : Mildly relatable but forgettable.
  5  : Resonates with a particular struggle or aspiration.
  7  : Hits hard — makes the viewer reflect on their own life.
  10 : Deeply moving. Makes people save, screenshot, or share because it felt personal.
       Example: "You're not lazy — you're scared of failing publicly."

[paradigm_shift] — Does it challenge or reframe how the viewer thinks about themselves or life?
  0  : Cliché advice. Nothing new.
  3  : Slightly fresh perspective but predictable conclusion.
  5  : Interesting angle — makes you think for a moment.
  7  : Genuine reframe. Changes how you look at a common problem.
  10 : Worldview-altering. Once you hear it, you can't unhear it.
       Example: "Discipline is not about forcing yourself — it's about removing decisions."
"""

_SELF_IMPROVEMENT_FEW_SHOT = """\
INPUT EXAMPLE:
[{"id": 22, "text": "People think they need motivation to start. But that's completely backwards. Action creates motivation, not the other way around. The hardest part is always the first five minutes. After that, momentum takes over."}]

OUTPUT EXAMPLE:
{"results": [{"id": 22, "actionability": 7.0, "emotional_resonance": 7.5, "paradigm_shift": 8.0}]}
"""

_SELF_IMPROVEMENT_HOOK_FRAMEWORKS = [
    {
        "name": "harsh_truth",
        "instruction": (
            "Deliver an uncomfortable truth about personal development that most people "
            "avoid hearing. It must come from a place of genuine care, not condescension."
        ),
        "examples_id": [
            "Kamu bukan malas — kamu takut gagal di depan orang",
            "Alasan kamu stuck bukan karena kurang skill",
            "Motivasi itu jebakan — ini yang sebenarnya kamu butuh",
            "Orang sukses gak punya rahasia — kamu cuma gak mulai",
        ],
        "examples_en": [
            "You're not lazy — you're afraid of failing publicly",
            "The reason you're stuck has nothing to do with skill",
            "Motivation is a trap — here's what you actually need",
            "Successful people have no secrets — you just won't start",
        ],
    },
    {
        "name": "life_hack",
        "instruction": (
            "Share a specific, immediately actionable life hack that provides "
            "disproportionate results for minimal effort. Must feel achievable."
        ),
        "examples_id": [
            "Kebiasaan 5 menit ini ubah hidup gue total",
            "Trick sederhana biar gak pernah prokrastinasi lagi",
            "Cara bangun pagi tanpa alarm yang beneran works",
            "Rumus simpel buat keputusan sulit dalam 30 detik",
        ],
        "examples_en": [
            "This 5-minute habit completely changed my life",
            "A simple trick to never procrastinate again",
            "How to wake up early without an alarm that actually works",
            "A simple formula for making hard decisions in 30 seconds",
        ],
    },
    {
        "name": "mindset_flip",
        "instruction": (
            "Take a common mental model or assumption and flip it on its head. "
            "Show that the opposite is actually true and more useful."
        ),
        "examples_id": [
            "Berhenti cari passion — ini yang harus kamu cari",
            "Disiplin bukan soal paksaan — kamu salah paham",
            "Multitasking itu mitos — otak kamu gak di-design gitu",
            "Comfort zone bukan musuh — cara kamu keluar yang salah",
        ],
        "examples_en": [
            "Stop chasing passion — chase this instead",
            "Discipline is not about force — you're doing it wrong",
            "Multitasking is a myth — your brain isn't designed for it",
            "Your comfort zone isn't the enemy — how you leave is",
        ],
    },
    {
        "name": "wake_up_call",
        "instruction": (
            "Create urgency by showing the viewer how time is slipping away and "
            "what their future self will regret if they don't act now. Ground it in reality."
        ),
        "examples_id": [
            "Kamu punya waktu lebih sedikit dari yang kamu kira",
            "5 tahun lagi kamu bakal nyesel gak mulai hari ini",
            "Umur 30 datang lebih cepat dari yang kamu pikir",
            "Setiap hari tanpa action itu satu langkah mundur",
        ],
        "examples_en": [
            "You have less time than you think",
            "Five years from now you'll regret not starting today",
            "Thirty comes faster than you think it will",
            "Every day without action is a step backward",
        ],
    },
]

_SELF_IMPROVEMENT_SIGNALS = {
    "harsh_truth": [
        "lazy", "fail", "afraid", "excuse", "stuck", "comfort",
        "malas", "gagal", "takut", "alasan", "stuck",
    ],
    "life_hack": [
        "habit", "routine", "morning", "trick", "hack", "simple",
        "kebiasaan", "rutinitas", "pagi", "trik", "sederhana",
    ],
    "mindset_flip": [
        "mindset", "believe", "think", "perspective", "assumption",
        "mindset", "percaya", "pikir", "perspektif", "asumsi",
    ],
    "wake_up_call": [
        "time", "regret", "age", "future", "now", "today",
        "waktu", "nyesel", "umur", "masa depan", "sekarang",
    ],
}

SELF_IMPROVEMENT_PROFILE = ContentProfile(
    niche="self_improvement",
    label="Self-Improvement / Personal Development",
    audience="people who want to grow, build better habits, and level up their mindset",
    tone="raw, honest, slightly intense — NOT generic motivational speaker energy",
    scoring_rubric=_SELF_IMPROVEMENT_RUBRIC,
    scoring_weights={"actionability": 0.35, "emotional_resonance": 0.35, "paradigm_shift": 0.30},
    scoring_dimensions=["actionability", "emotional_resonance", "paradigm_shift"],
    hook_frameworks=_SELF_IMPROVEMENT_HOOK_FRAMEWORKS,
    hook_signals=_SELF_IMPROVEMENT_SIGNALS,
    few_shot_example=_SELF_IMPROVEMENT_FEW_SHOT,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRESET: ekonomi_politik
# ═══════════════════════════════════════════════════════════════════════════

_EKONPOL_RUBRIC = """\
NICHE CONTEXT: Economic Analysis, Political Commentary, Geopolitics, and Policy Critique content.
Score each segment from 0–10 using these anchor points:

[analytical_depth] — How deep and substantive is the economic or political analysis?
  0  : Zero substance. Emotional ranting with no data or logic.
  3  : Surface-level opinion anyone could form from reading headlines.
  5  : Decent analysis with some reasoning, but nothing original.
  7  : Strong analytical framework. Connects dots most people miss, uses data or historical context.
  10 : Expert-tier analysis. Reveals systemic mechanisms, hidden policy effects, or geopolitical strategy.
       Example: explaining how currency devaluation is used as a deliberate policy tool.

[public_impact] — How relevant is this to ordinary people's daily lives and well-being?
  0  : Abstract policy discussion with no tangible impact.
  3  : Vaguely relevant but hard to see the direct effect.
  5  : Clearly affects a segment of the population.
  7  : Directly impacts most viewers — wages, prices, taxes, or rights.
  10 : Urgent, immediate impact on everyday life. Makes people realize "this is happening to ME right now."
       Example: explaining why grocery prices doubled and who benefits from it.

[debate_potential] — Will this provoke strong, passionate responses from multiple sides?
  0  : Completely neutral, factual reporting with no angle.
  3  : Mild opinion that most people would agree with.
  5  : Has a clear stance — will make some people nod and others frown.
  7  : Strongly opinionated. Will split the audience and generate heated comments.
  10 : Maximum polarization. Touches identity, ideology, or deeply held beliefs.
       Example: "Democracy doesn't work the way you think it does."
"""

_EKONPOL_FEW_SHOT = """\
INPUT EXAMPLE:
[{"id": 33, "text": "Pemerintah bilang inflasi 3 persen tapi coba kamu lihat harga beras, minyak goreng, dan telur — semuanya naik 20 sampai 40 persen. Metode perhitungan inflasi itu sengaja didesain supaya angkanya kelihatan kecil. Ini bukan kebetulan, ini by design."}]

OUTPUT EXAMPLE:
{"results": [{"id": 33, "analytical_depth": 7.5, "public_impact": 9.0, "debate_potential": 8.5}]}
"""

_EKONPOL_HOOK_FRAMEWORKS = [
    {
        "name": "policy_expose",
        "instruction": (
            "Expose how a government policy, regulation, or economic decision is designed "
            "to benefit specific groups while disadvantaging ordinary citizens. Pull back "
            "the curtain on how the system actually works."
        ),
        "examples_id": [
            "Kebijakan ini diam-diam bikin rakyat makin miskin",
            "Kenapa subsidi selalu dicabut tapi pajak gak pernah turun",
            "Cara pemerintah sembunyiin angka inflasi yang sebenarnya",
            "Regulasi ini cuma nguntungin oligarki — bukan kamu",
        ],
        "examples_en": [
            "This policy is quietly making citizens poorer",
            "Why subsidies get cut but taxes never go down",
            "How the government hides real inflation numbers",
            "This regulation only benefits oligarchs — not you",
        ],
    },
    {
        "name": "geopolitical_insight",
        "instruction": (
            "Reveal a geopolitical dynamic, trade war implication, or international strategy "
            "that most people don't understand but directly affects their country's economy. "
            "Make global politics feel personally relevant."
        ),
        "examples_id": [
            "Perang dagang ini langsung ngaruh ke harga barang kamu",
            "Kenapa negara lain jual sumber daya kita lebih murah",
            "Strategi geopolitik di balik kenaikan harga BBM",
            "China dan Amerika rebutan ini — dan kamu kena dampaknya",
        ],
        "examples_en": [
            "This trade war directly affects prices you pay",
            "Why other nations sell our resources cheaper than we can",
            "The geopolitical strategy behind rising fuel prices",
            "China and America are fighting over this — and you pay the price",
        ],
    },
    {
        "name": "economic_reality",
        "instruction": (
            "Confront the audience with an economic reality that contradicts official "
            "narratives or popular belief. Use data or mechanism to back it up."
        ),
        "examples_id": [
            "Data resmi bilang ekonomi tumbuh — tapi dompet kamu bilang lain",
            "Kelas menengah sedang diam-diam dihancurkan",
            "Angka pengangguran asli jauh lebih besar dari yang dilaporkan",
            "Pertumbuhan ekonomi 5 persen tapi rakyat makin susah",
        ],
        "examples_en": [
            "Official data says economy is growing — your wallet disagrees",
            "The middle class is being silently destroyed",
            "Real unemployment is far higher than reported",
            "5 percent growth but people are worse off than ever",
        ],
    },
    {
        "name": "power_structure",
        "instruction": (
            "Reveal how power structures — political dynasties, corporate lobbying, "
            "or institutional capture — shape economic outcomes for ordinary people. "
            "Make viewers question who really makes decisions."
        ),
        "examples_id": [
            "Siapa sebenarnya yang nentuin harga kebutuhan pokok",
            "Dinasti politik dan uang rakyat — koneksinya jelas",
            "Kenapa BUMN rugi tapi direksinya tetap kaya",
            "Lobbying korporasi yang kamu gak pernah denger",
        ],
        "examples_en": [
            "Who actually decides the price of your groceries",
            "Political dynasties and public money — the connection is clear",
            "Why state companies lose money but their directors stay rich",
            "Corporate lobbying you were never meant to hear about",
        ],
    },
]

_EKONPOL_SIGNALS = {
    "policy_expose": [
        "policy", "regulation", "subsidy", "tax", "government", "law",
        "kebijakan", "regulasi", "subsidi", "pajak", "pemerintah", "undang-undang",
    ],
    "geopolitical_insight": [
        "china", "america", "trade war", "geopolitik", "export", "import",
        "perang dagang", "ekspor", "impor", "global", "dunia",
    ],
    "economic_reality": [
        "inflation", "gdp", "growth", "unemployment", "poverty",
        "inflasi", "pertumbuhan", "pengangguran", "kemiskinan", "harga",
    ],
    "power_structure": [
        "oligarch", "dynasty", "lobby", "corruption", "elite",
        "oligarki", "dinasti", "korupsi", "elite", "kekuasaan",
    ],
}

EKONPOL_PROFILE = ContentProfile(
    niche="ekonomi_politik",
    label="Ekonomi / Politik / Geopolitik",
    audience="warga yang ingin memahami bagaimana kebijakan ekonomi dan politik berdampak langsung pada kehidupan mereka",
    tone="tajam, berbasis data, kritis tapi tidak partisan — BUKAN propaganda atau teori konspirasi",
    scoring_rubric=_EKONPOL_RUBRIC,
    scoring_weights={"analytical_depth": 0.40, "public_impact": 0.35, "debate_potential": 0.25},
    scoring_dimensions=["analytical_depth", "public_impact", "debate_potential"],
    hook_frameworks=_EKONPOL_HOOK_FRAMEWORKS,
    hook_signals=_EKONPOL_SIGNALS,
    few_shot_example=_EKONPOL_FEW_SHOT,
)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

_PROFILES: dict[str, ContentProfile] = {
    "money":            MONEY_PROFILE,
    "tech":             TECH_PROFILE,
    "gaming":           GAMING_PROFILE,
    "self_improvement": SELF_IMPROVEMENT_PROFILE,
    "ekonomi_politik":  EKONPOL_PROFILE,
}


def get_profile(niche: str) -> ContentProfile:
    """
    Retrieve a content profile by niche key.

    Args:
        niche: One of "money", "tech", "gaming", "self_improvement", "ekonomi_politik".

    Returns:
        ContentProfile instance.

    Raises:
        ValueError: If niche is not recognized.
    """
    profile = _PROFILES.get(niche)
    if profile is None:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise ValueError(
            f"Unknown content niche: '{niche}'. "
            f"Available niches: {available}"
        )
    return profile


def list_profiles() -> list[str]:
    """Return all available niche keys."""
    return sorted(_PROFILES.keys())
