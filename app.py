import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GoT Text Generator",
    page_icon="⚔️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=Inter:wght@400;500&display=swap');

  html, body, [class*="css"] {
    background-color: #0f0e0d;
    color: #d4c9b0;
  }

  /* Title */
  .got-title {
    font-family: 'IM Fell English', serif;
    font-size: 2.6rem;
    color: #c9a84c;
    letter-spacing: 0.04em;
    margin-bottom: 0.1rem;
    line-height: 1.2;
  }
  .got-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #6b6358;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* Output box */
  .output-box {
    font-family: 'IM Fell English', serif;
    font-size: 1.05rem;
    line-height: 1.9;
    color: #d4c9b0;
    background: #161410;
    border: 1px solid #2e2a22;
    border-left: 3px solid #c9a84c;
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    white-space: pre-wrap;
    min-height: 120px;
  }

  /* Divider */
  .divider {
    border: none;
    border-top: 1px solid #2e2a22;
    margin: 1.5rem 0;
  }

  /* Label override */
  label { font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; color: #8a7f6e !important; }

  /* Button */
  div.stButton > button {
    background: #c9a84c;
    color: #0f0e0d;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 3px;
    padding: 0.55rem 1.6rem;
    width: 100%;
    transition: background 0.15s;
  }
  div.stButton > button:hover { background: #e0bc62; }

  /* Textarea */
  textarea {
    background: #161410 !important;
    color: #d4c9b0 !important;
    border: 1px solid #2e2a22 !important;
    font-family: 'IM Fell English', serif !important;
    font-size: 1rem !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0c0b0a;
    border-right: 1px solid #1e1c18;
  }

  /* Stat pill */
  .stat-pill {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #6b6358;
    background: #1a1814;
    border: 1px solid #2e2a22;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    margin-right: 0.4rem;
    margin-bottom: 0.8rem;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── File check ──────────────────────────────────────────────────────────────────
REQUIRED_FILES = ["model/gpt_got.pth", "model/hparams.json", "model/tokenizer.json"]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(f"Missing file(s): `{'`, `'.join(missing)}`\n\nRun the save block in your notebook first, then place these files alongside `app.py`.")
    st.stop()


# ── Load hparams (module-level so model classes can reference them) ─────────────
with open("model/hparams.json") as f:
    _hp = json.load(f)

block_size = _hp["block_size"]
n_emb      = _hp["n_emb"]
n_head     = _hp["n_head"]
n_layer    = _hp["n_layer"]
dropout    = _hp["dropout"]
vocab_size = _hp["vocab_size"]


# ── Model architecture (identical to notebook) ──────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1  = nn.LayerNorm(n_emb)
        self.ln2  = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Load model (cached — runs once per session) ─────────────────────────────────
@st.cache_resource
def load_everything():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("model/tokenizer.json") as f:
        tok = json.load(f)
    stoi = tok["stoi"]
    itos = {int(k): v for k, v in tok["itos"].items()}

    model = GPTLanguageModel()
    model.load_state_dict(torch.load("model/gpt_got.pth", map_location=device))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, stoi, itos, device, n_params


model, stoi, itos, device, n_params = load_everything()

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    max_tokens  = st.slider("Tokens to generate", min_value=50, max_value=1000, value=300, step=50)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05,
                            help="Lower = predictable, Higher = creative / chaotic")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Model")
    st.markdown(f"""
    <span class='stat-pill'>📐 {n_params/1e6:.1f}M params</span>
    <span class='stat-pill'>🧩 {n_layer} layers</span>
    <span class='stat-pill'>👁 {n_head} heads</span>
    <span class='stat-pill'>📏 ctx {block_size}</span>
    <span class='stat-pill'>🔤 vocab {vocab_size}</span>
    <span class='stat-pill'>💻 {device.upper()}</span>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.caption("Trained on the Game of Thrones book corpus · Character-level GPT")


# ── Main ─────────────────────────────────────────────────────────────────────────
st.markdown("<div class='got-title'>⚔️ Westeros Text Forge</div>", unsafe_allow_html=True)
st.markdown("<div class='got-sub'>Mini GPT-2 · Game of Thrones · Character-level</div>", unsafe_allow_html=True)

seed_text = st.text_area(
    "Seed text (optional)",
    placeholder="The king sat upon his throne…",
    height=90,
    help="Leave empty to generate from scratch. Unknown characters are ignored."
)

col1, col2 = st.columns([3, 1])
with col1:
    generate_clicked = st.button("Generate", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if "output" not in st.session_state:
    st.session_state.output = ""

if clear_clicked:
    st.session_state.output = ""

if generate_clicked:
    with st.spinner("Writing…"):
        if seed_text.strip():
            encoded = encode(seed_text)
            if not encoded:
                st.warning("None of the seed characters are in the model's vocabulary. Generating from scratch.")
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
            else:
                context = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        output_ids = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
        st.session_state.output = decode(output_ids[0].tolist())

if st.session_state.output:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"<div class='output-box'>{st.session_state.output}</div>", unsafe_allow_html=True)
    st.download_button(
        label="Download as .txt",
        data=st.session_state.output,
        file_name="got_generated.txt",
        mime="text/plain",
    )