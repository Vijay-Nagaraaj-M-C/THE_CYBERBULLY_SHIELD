import csv
import random
import os

# ── Paths (relative to this script's location) ──────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MULTIMODAL_CSV   = os.path.join(PROJECT_ROOT, 'Datasets', 'multimodal-cyberbullying', 'cyberbully.csv')
TEXT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'Datasets', 'cyberbullying-dataset')
HTML_FILE        = os.path.join(BASE_DIR, "index.html")

os.makedirs(BASE_DIR, exist_ok=True)

# ── Dataset loader helpers ────────────────────────────────────────────────────

def load_csv_safe(path):
    """Load a CSV as a list of dicts. Falls back to latin-1 encoding."""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    print(f"  ⚠️  Could not load {path}")
    return []

def is_bully_oh(row):
    """oh_label == 1  →  bully."""
    try:
        return int(float(row.get("oh_label", 0))) == 1
    except (ValueError, TypeError):
        return False

def is_bully_twitter(row):
    """Annotation != 'none'  →  bully."""
    return row.get("Annotation", "none").strip().lower() != "none"

def clean(t, max_len=300):
    """Strip, skip blanks / very short texts, truncate long ones."""
    t = str(t).strip()
    if len(t) < 10:
        return None
    return t if len(t) <= max_len else t[:max_len].rsplit(" ", 1)[0] + "…"

# ── Load all text datasets ────────────────────────────────────────────────────
print("Loading text datasets …")

bully_pool  = []
normal_pool = []

def add(text, bully):
    t = clean(text)
    if t:
        (bully_pool if bully else normal_pool).append(t)

# Schema A — aggression / attack / toxicity  (columns: Text, oh_label)
for fname in ("aggression_parsed_dataset.csv",
              "attack_parsed_dataset.csv",
              "toxicity_parsed_dataset.csv"):
    rows = load_csv_safe(os.path.join(TEXT_DATASET_DIR, fname))
    for r in rows:
        add(r.get("Text", ""), is_bully_oh(r))
    print(f"  {fname}: {len(rows):,} rows")

# Schema B — kaggle  (try oh_label, else 'label', else assume clean)
rows = load_csv_safe(os.path.join(TEXT_DATASET_DIR, "kaggle_parsed_dataset.csv"))
for r in rows:
    text = r.get("Text") or r.get("comment_text") or r.get("text") or ""
    if "oh_label" in r:
        add(text, is_bully_oh(r))
    elif "label" in r:
        try:
            add(text, int(float(r["label"])) == 1)
        except Exception:
            add(text, False)
    else:
        add(text, False)
print(f"  kaggle_parsed_dataset.csv: {len(rows):,} rows")

# Schema C — twitter / twitter_racism / twitter_sexism  (columns: Text, Annotation)
for fname in ("twitter_parsed_dataset.csv",
              "twitter_racism_parsed_dataset.csv",
              "twitter_sexism_parsed_dataset.csv"):
    rows = load_csv_safe(os.path.join(TEXT_DATASET_DIR, fname))
    for r in rows:
        text = r.get("Text", "")
        if "Annotation" in r:
            add(text, is_bully_twitter(r))
        else:
            add(text, is_bully_oh(r))
    print(f"  {fname}: {len(rows):,} rows")

# Schema D — youtube  (columns: Text, oh_label)
rows = load_csv_safe(os.path.join(TEXT_DATASET_DIR, "youtube_parsed_dataset.csv"))
for r in rows:
    add(r.get("Text", ""), is_bully_oh(r))
print(f"  youtube_parsed_dataset.csv: {len(rows):,} rows")

# Cap & shuffle pools so we don't hog memory
random.shuffle(bully_pool)
random.shuffle(normal_pool)
bully_pool  = bully_pool[:40000]
normal_pool = normal_pool[:40000]

print(f"\n  ✅ Bully pool  : {len(bully_pool):,}")
print(f"  ✅ Normal pool : {len(normal_pool):,}")

# ── Load multimodal post dataset ──────────────────────────────────────────────
print("\nLoading multimodal posts …")
posts_data = []
with open(MULTIMODAL_CSV, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 5:
            posts_data.append((row[0], row[1], row[4]))   # img, text, label

bully_posts  = [p for p in posts_data if p[2] == "Bully"]
normal_posts = [p for p in posts_data if p[2] == "Nonbully"]
random.shuffle(bully_posts)
random.shuffle(normal_posts)
selected_posts = bully_posts[:15] + normal_posts[:15]
random.shuffle(selected_posts)
print(f"  ✅ {len(selected_posts)} posts selected (15 bully + 15 normal)")

# ── Static seed comments (deterministic baseline for extension testing) ───────
SEED_BULLY = [
    "Nobody asked for your opinion, just stop posting",
    "This is embarrassing, please delete this",
    "Why do you even have followers lol",
    "You look ridiculous, take this down",
    "Everyone is laughing at you behind your back",
    "You're so desperate for attention it's actually sad",
    "You clearly have no talent whatsoever",
    "Your friends must be so embarrassed by you",
    "Why are you even on this platform, nobody likes you here",
    "You need to seriously rethink your life choices",
    "Block me if you want but this content is genuinely awful",
    "I can't believe people actually like this garbage",
    "This made me lose brain cells, thanks a lot",
    "You're pathetic, go outside for once",
    "Imagine posting this and thinking it's good 💀",
]

SEED_NONBULLY = [
    "This is absolutely wholesome, love it! ❤️",
    "Great post! Keep it up 😊",
    "This made my day, thank you for sharing!",
    "So relatable haha 😂",
    "Incredible, where was this taken?",
    "Aww this is the sweetest thing I've seen today",
    "100% agree with this, well said!",
    "Sending love and good vibes your way 🙌",
    "This deserves way more likes fr",
    "Okay but this is actually goals 😍",
    "Finally someone said it!",
    "You're genuinely talented, keep going!",
    "This brought a smile to my face, thank you",
    "Can't stop laughing 😂😂😂",
    "Such a good vibe, we need more of this",
]

# ── Users & styling ───────────────────────────────────────────────────────────
USERS = [
    "Alice", "Bob_99", "CharlieTheG", "DankMemer", "Elisa_X",
    "FrankyBoy", "GamerDude", "Hannah_Smiles", "Ignacio", "Jenny123"
]

GRADIENTS = {
    "Alice":         "linear-gradient(135deg,#f97316,#ec4899)",
    "Bob_99":        "linear-gradient(135deg,#10b981,#3b82f6)",
    "CharlieTheG":   "linear-gradient(135deg,#f59e0b,#ef4444)",
    "DankMemer":     "linear-gradient(135deg,#6366f1,#ec4899)",
    "Elisa_X":       "linear-gradient(135deg,#14b8a6,#8b5cf6)",
    "FrankyBoy":     "linear-gradient(135deg,#f97316,#eab308)",
    "GamerDude":     "linear-gradient(135deg,#3b82f6,#06b6d4)",
    "Hannah_Smiles": "linear-gradient(135deg,#ec4899,#f97316)",
    "Ignacio":       "linear-gradient(135deg,#8b5cf6,#3b82f6)",
    "Jenny123":      "linear-gradient(135deg,#f59e0b,#ec4899)",
}

TIMESTAMPS = [
    "just now", "2 minutes ago", "15 minutes ago", "1 hour ago",
    "3 hours ago", "5 hours ago", "8 hours ago", "12 hours ago",
    "1 day ago", "2 days ago",
]

def esc(t):
    return (str(t)
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

# ── Comment picker ────────────────────────────────────────────────────────────
#
#   Comment source probability breakdown:
#     40% → bully_pool   (aggression / attack / toxicity / kaggle / twitter / youtube)
#     38% → normal_pool  (same datasets, non-bully rows)
#     12% → SEED_BULLY   (hand-crafted — always present for baseline testing)
#     10% → SEED_NONBULLY
#
def pick_comment():
    r = random.random()
    if r < 0.40:
        return random.choice(bully_pool)
    elif r < 0.78:
        return random.choice(normal_pool)
    elif r < 0.90:
        return random.choice(SEED_BULLY)
    else:
        return random.choice(SEED_NONBULLY)

# ── HTML builders ─────────────────────────────────────────────────────────────
def comment_html(c_user, c_text):
    grad = GRADIENTS.get(c_user, "linear-gradient(135deg,#8b5cf6,#3b82f6)")
    return (
        f'\n                <div class="comment-row">'
        f'<div class="c-avatar" style="background:{grad}">{c_user[0]}</div>'
        f'<div class="c-content">'
        f'<span class="c-username">{c_user}</span>'
        f'<span class="c-text message-content">{esc(c_text)}</span>'
        f'</div></div>'
    )

def post_html(img, text, label):
    user  = random.choice(USERS)
    grad  = GRADIENTS[user]
    likes = random.randint(50, 980)
    ts    = random.choice(TIMESTAMPS)

    comments = "".join(
        comment_html(random.choice(USERS), pick_comment())
        for _ in range(random.randint(7, 12))
    )

    return f"""
        <!-- [{label}] -->
        <div class="post-card">
            <div class="post-header">
                <div class="post-avatar-ring"><div class="post-avatar" style="background:{grad}">{user[0]}</div></div>
                <div class="post-user-info"><div class="post-username">{user}</div></div>
                <button class="post-more"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/></svg></button>
            </div>
            <div class="post-image-wrap">
                <img src="../../Datasets/multimodal-cyberbullying/bully_data/{img}" alt="Post"
                     onerror="this.parentElement.innerHTML='<div class=\\'no-image\\'>🖼️</div>'">
            </div>
            <div class="post-actions">
                <button class="action-icon" onclick="toggleLike(this)">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
                </button>
                <button class="action-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></button>
                <button class="action-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
                <button class="action-icon save-btn"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg></button>
            </div>
            <div class="post-likes">{likes} likes</div>
            <div class="post-caption"><span class="username">{user}</span><span class="message-content">{esc(text)}</span></div>
            <div class="comments-list">{comments}</div>
            <div class="post-timestamp">{ts}</div>
            <div class="add-comment-row">
                <button class="emoji-btn">😊</button>
                <input class="add-comment-input" type="text" placeholder="Add a comment…" oninput="togglePostBtn(this)">
                <button class="post-comment-btn">Post</button>
            </div>
        </div>"""

def stories_bar():
    items = ""
    for i, (u, g) in enumerate(GRADIENTS.items()):
        seen = " seen" if i % 3 == 0 else ""
        items += (f'<div class="story-item"><div class="story-ring{seen}">'
                  f'<div class="story-avatar" style="background:{g}">{u[0]}</div></div>'
                  f'<span class="story-name">{u}</span></div>')
    return items

def suggest_panel():
    items = ""
    for u, g in list(GRADIENTS.items())[:5]:
        items += (f'<div class="suggest-item">'
                  f'<div class="suggest-avatar" style="background:{g}">{u[0]}</div>'
                  f'<div class="suggest-info"><div class="suggest-name">{u}</div>'
                  f'<div class="suggest-sub">Suggested for you</div></div>'
                  f'<button class="suggest-follow">Follow</button></div>')
    return items

# ── Assemble document ─────────────────────────────────────────────────────────
print("\nBuilding HTML …")
all_posts = "\n".join(post_html(img, txt, lbl) for img, txt, lbl in selected_posts)

doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instagram</title>
    <link href="https://fonts.googleapis.com/css2?family=Billabong&display=swap" rel="stylesheet">
    <style>
        *{{margin:0;padding:0;box-sizing:border-box;}}
        :root{{--bg:#fafafa;--white:#fff;--border:#dbdbdb;--text:#262626;--muted:#8e8e8e;--blue:#0095f6;--red:#ed4956;}}
        body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.4;}}
        /* top nav */
        .top-nav{{display:none;position:sticky;top:0;z-index:100;background:var(--white);border-bottom:1px solid var(--border);padding:10px 16px;align-items:center;justify-content:space-between;}}
        .top-nav .ig-logo{{font-family:'Billabong',cursive;font-size:28px;color:var(--text);text-decoration:none;}}
        .top-nav-icons{{display:flex;gap:16px;}}
        .top-nav-icons svg{{width:24px;height:24px;}}
        /* layout */
        .layout{{display:flex;max-width:975px;margin:0 auto;padding-top:30px;gap:28px;}}
        /* sidebar */
        .sidebar{{width:244px;flex-shrink:0;position:sticky;top:0;height:100vh;padding:12px 12px 20px;display:flex;flex-direction:column;background:var(--white);border-right:1px solid var(--border);}}
        .sidebar-logo{{font-family:'Billabong',cursive;font-size:30px;padding:22px 12px 28px;display:block;color:var(--text);text-decoration:none;}}
        .nav-item{{display:flex;align-items:center;gap:14px;padding:12px;border-radius:8px;cursor:pointer;font-size:15px;font-weight:400;color:var(--text);text-decoration:none;transition:background .15s;margin-bottom:4px;}}
        .nav-item:hover{{background:#f0f0f0;}}
        .nav-item.active{{font-weight:700;}}
        .nav-item svg{{width:24px;height:24px;flex-shrink:0;}}
        .sidebar-bottom{{margin-top:auto;}}
        /* feed */
        .feed-wrap{{flex:1;max-width:470px;min-width:0;}}
        /* stories */
        .stories-bar{{background:var(--white);border:1px solid var(--border);border-radius:8px;padding:16px 0;margin-bottom:24px;display:flex;overflow-x:auto;scrollbar-width:none;}}
        .stories-bar::-webkit-scrollbar{{display:none;}}
        .story-item{{display:flex;flex-direction:column;align-items:center;gap:6px;padding:0 14px;flex-shrink:0;cursor:pointer;}}
        .story-ring{{width:56px;height:56px;border-radius:50%;padding:2px;background:linear-gradient(45deg,#f09433,#e6683c,#dc2743,#cc2366,#bc1888);}}
        .story-ring.seen{{background:#dbdbdb;}}
        .story-avatar{{width:100%;height:100%;border-radius:50%;border:2px solid white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:18px;color:white;}}
        .story-name{{font-size:12px;max-width:68px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
        /* post card */
        .post-card{{background:var(--white);border:1px solid var(--border);border-radius:8px;margin-bottom:24px;}}
        .post-header{{display:flex;align-items:center;padding:14px 16px;gap:10px;}}
        .post-avatar-ring{{width:32px;height:32px;border-radius:50%;padding:2px;background:linear-gradient(45deg,#f09433,#e6683c,#dc2743,#cc2366,#bc1888);flex-shrink:0;}}
        .post-avatar{{width:100%;height:100%;border-radius:50%;border:2px solid white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:white;}}
        .post-user-info{{flex:1;}}
        .post-username{{font-weight:600;font-size:14px;}}
        .post-more{{margin-left:auto;background:none;border:none;cursor:pointer;}}
        .post-more svg{{width:24px;height:24px;}}
        .post-image-wrap{{position:relative;background:black;}}
        .post-image-wrap img{{width:100%;max-height:585px;object-fit:cover;display:block;}}
        .no-image{{width:100%;height:200px;background:linear-gradient(135deg,#e0e0e0,#c0c0c0);display:flex;align-items:center;justify-content:center;font-size:40px;}}
        .post-actions{{display:flex;align-items:center;padding:8px 16px 0;gap:16px;}}
        .post-actions svg{{width:24px;height:24px;cursor:pointer;}}
        .save-btn{{margin-left:auto;}}
        .action-icon{{background:none;border:none;padding:4px 0;cursor:pointer;display:flex;align-items:center;color:var(--text);}}
        .action-icon:hover{{opacity:.6;}}
        .post-likes{{padding:8px 16px 4px;font-size:14px;font-weight:600;}}
        .post-caption{{padding:2px 16px 6px;font-size:14px;line-height:1.5;}}
        .post-caption .username{{font-weight:600;margin-right:4px;}}
        /* comments */
        .comments-list{{padding:0 16px 4px;}}
        .comment-row{{display:flex;gap:8px;margin-bottom:8px;font-size:14px;line-height:1.5;align-items:flex-start;}}
        .c-avatar{{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:10px;color:white;flex-shrink:0;margin-top:2px;}}
        .c-content{{flex:1;min-width:0;}}
        .c-username{{font-weight:600;margin-right:5px;}}
        .c-text{{color:var(--text);word-break:break-word;}}
        .post-timestamp{{padding:4px 16px 10px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.2px;}}
        .add-comment-row{{border-top:1px solid var(--border);display:flex;align-items:center;padding:8px 16px;gap:12px;}}
        .emoji-btn{{font-size:22px;background:none;border:none;cursor:pointer;}}
        .add-comment-input{{flex:1;border:none;outline:none;font-size:14px;color:var(--text);background:transparent;font-family:inherit;}}
        .add-comment-input::placeholder{{color:var(--muted);}}
        .post-comment-btn{{background:none;border:none;color:var(--blue);font-size:14px;font-weight:600;cursor:pointer;opacity:.4;}}
        .post-comment-btn.active{{opacity:1;}}
        /* right sidebar */
        .right-sidebar{{width:293px;flex-shrink:0;padding-top:16px;}}
        .rs-profile{{display:flex;align-items:center;gap:14px;margin-bottom:24px;}}
        .rs-avatar{{width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#8b5cf6,#3b82f6);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:20px;color:white;}}
        .rs-name{{font-weight:600;font-size:14px;}}
        .rs-fullname{{font-size:14px;color:var(--muted);}}
        .rs-switch-btn{{margin-left:auto;background:none;border:none;color:var(--blue);font-size:12px;font-weight:600;cursor:pointer;}}
        .rs-section-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;}}
        .rs-section-title{{font-size:14px;font-weight:600;color:var(--muted);}}
        .rs-see-all{{font-size:12px;font-weight:600;color:var(--text);cursor:pointer;text-decoration:none;}}
        .suggest-item{{display:flex;align-items:center;gap:10px;margin-bottom:12px;}}
        .suggest-avatar{{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;color:white;flex-shrink:0;}}
        .suggest-info{{flex:1;}}
        .suggest-name{{font-weight:600;font-size:12px;}}
        .suggest-sub{{font-size:12px;color:var(--muted);}}
        .suggest-follow{{background:none;border:none;color:var(--blue);font-size:12px;font-weight:600;cursor:pointer;}}
        .rs-footer{{margin-top:24px;font-size:11px;color:var(--muted);line-height:2;}}
        .rs-footer a{{color:var(--muted);text-decoration:none;}}
        .rs-footer a:hover{{text-decoration:underline;}}
        /* bottom nav */
        .bottom-nav{{display:none;position:fixed;bottom:0;left:0;right:0;background:var(--white);border-top:1px solid var(--border);padding:10px 0;justify-content:space-around;z-index:100;}}
        .bottom-nav svg{{width:24px;height:24px;}}
        .bottom-nav a{{color:var(--text);}}
        /* responsive */
        @media(max-width:935px){{.right-sidebar{{display:none;}}.layout{{max-width:614px;}}}}
        @media(max-width:768px){{
            .top-nav{{display:flex;}}.bottom-nav{{display:flex;}}
            .sidebar{{display:none;}}.layout{{padding-top:0;max-width:100%;padding-bottom:60px;}}
            .feed-wrap{{max-width:100%;}}
        }}
        @keyframes heartPop{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.3)}}}}
        .liked svg{{color:var(--red);animation:heartPop .3s ease;}}
    </style>
</head>
<body>

<nav class="top-nav">
    <a class="ig-logo" href="#">Instagram</a>
    <div class="top-nav-icons">
        <a href="#" style="color:var(--text-primary)"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="5"/><path d="M8 12h8M12 8v8"/></svg></a>
        <a href="#" style="color:var(--text-primary)"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></a>
    </div>
</nav>

<div class="layout">
    <aside class="sidebar">
        <a class="sidebar-logo" href="#">Instagram</a>
        <a href="#" class="nav-item active"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg><span>Home</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><span>Search</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg><span>Explore</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><span>Reels</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg><span>Messages</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg><span>Notifications</span></a>
        <a href="#" class="nav-item"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="5"/><path d="M8 12h8M12 8v8"/></svg><span>Create</span></a>
        <div class="sidebar-bottom">
            <a href="#" class="nav-item">
                <div style="width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#8b5cf6,#3b82f6);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;color:white;">Y</div>
                <span>Profile</span>
            </a>
        </div>
    </aside>

    <main class="feed-wrap">
        <div class="stories-bar">
            <div class="story-item">
                <div class="story-ring"><div class="story-avatar" style="background:linear-gradient(135deg,#ec4899,#8b5cf6)">Y</div></div>
                <span class="story-name">Your Story</span>
            </div>
            {stories_bar()}
        </div>
        {all_posts}
    </main>

    <aside class="right-sidebar">
        <div class="rs-profile">
            <div class="rs-avatar">Y</div>
            <div><div class="rs-name">yourusername</div><div class="rs-fullname">Your Name</div></div>
            <button class="rs-switch-btn">Switch</button>
        </div>
        <div class="rs-section-header">
            <span class="rs-section-title">Suggested for you</span>
            <a href="#" class="rs-see-all">See All</a>
        </div>
        {suggest_panel()}
        <div class="rs-footer">
            <div><a href="#">About</a> · <a href="#">Help</a> · <a href="#">Press</a> · <a href="#">API</a> · <a href="#">Jobs</a> · <a href="#">Privacy</a> · <a href="#">Terms</a></div>
            <div style="margin-top:8px;">© 2025 INSTAGRAM FROM META</div>
        </div>
    </aside>
</div>

<nav class="bottom-nav">
    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg></a>
    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg></a>
    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="5"/><path d="M8 12h8M12 8v8"/></svg></a>
    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg></a>
    <a href="#"><div style="width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#8b5cf6,#3b82f6);display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:10px;color:white;">Y</div></a>
</nav>

<script>
function toggleLike(btn) {{
    btn.classList.toggle('liked');
    const svg = btn.querySelector('svg');
    if (btn.classList.contains('liked')) {{
        svg.setAttribute('fill','#ed4956'); svg.setAttribute('stroke','#ed4956');
    }} else {{
        svg.setAttribute('fill','none'); svg.setAttribute('stroke','currentColor');
    }}
    const el = btn.closest('.post-card').querySelector('.post-likes');
    const n = parseInt(el.textContent);
    el.textContent = (btn.classList.contains('liked') ? n+1 : n-1) + ' likes';
}}
function togglePostBtn(input) {{
    input.nextElementSibling.classList.toggle('active', input.value.trim().length > 0);
}}
</script>
</body>
</html>"""

with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(doc)

print(f"\n✅  Written → {HTML_FILE}")
print(f"    Posts   : {len(selected_posts)}  (15 bully + 15 normal)")
print(f"    Comment sources: 8 text datasets + seed lists")
