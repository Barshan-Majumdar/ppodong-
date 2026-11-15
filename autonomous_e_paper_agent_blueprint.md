# Autonomous 24/7 India, Kolkata & Global E-Paper News Agent — Complete Build Blueprint

## 1. Master Build Prompt for Antigravity

> **Copy and paste this entire file (or this prompt section) directly into Antigravity to scaffold and build the project.**

```
Act as a Principal Full-Stack & ML Systems Engineer. Build a complete, production-ready, 100% free, autonomous 24/7 news intelligence and E-Paper publishing system in Python that runs via GitHub Actions.

### Core Objective & Purpose
Build an automated agent that wakes up daily (and on manual dispatch), collects all current affairs and breaking news from the past 24 hours across **Kolkata, West Bengal, all major Indian States/UTs, India National Affairs, Global Breaking News, and Real-Time Trends**, processes them through an ML deduplication and ranking pipeline, synthesizes structured journalistic summaries using free LLM APIs, compiles a multi-column **Newspaper-Style E-Paper PDF** (plus a companion Markdown report), delivers it automatically via **Telegram** (and optional Email), and permanently archives every dated edition (`YYYY-MM-DD`) in the repository.

### CRITICAL NON-NEGOTIABLE RULE: ZERO NEWS EXCLUSION
- **Do NOT drop or exclude unique news stories.** Every distinct news item fetched within the 24-hour rolling window MUST appear in the final E-Paper PDF and Markdown archive.
- Deduplication must ONLY merge multiple outlets reporting the **exact same event** into a single consolidated story cluster (listing all covering sources).
- To prevent LLM output token truncation when processing 100–200+ daily stories, implement **Batch/Section-Wise Map-Reduce LLM Summarization** (processing one category/batch at a time with strict JSON schemas) so that 100% of clustered stories are summarized and rendered into the final E-Paper without omission.

### Required Directory & File Architecture
Generate the complete repository with clean, modular, type-hinted Python 3.11+ code:
1. `.github/workflows/daily_epaper.yml` — Automated cron schedule (6:00 AM IST / 00:30 UTC) + `workflow_dispatch`, system dependencies for WeasyPrint (`libpango-1.0-0`, `libharfbuzz0b`, `libpangoft2-1.0-0`), Python execution, and automated Git commit/push of dated PDF + Markdown archives.
2. `config/sources.yaml` — Centralized configuration holding all RSS URLs, Google News geo/topic/state queries, Reddit trend endpoints, and regional priority weights.
3. `src/collectors/rss_collector.py` — Asynchronous/concurrent RSS & Google News fetcher (`aiohttp` + `feedparser`) with strict 24-hour timestamp filtering (`published_parsed` / `updated_parsed`), Google News redirect URL decoding, and full-text article extraction (`trafilatura`) with graceful fallback to RSS descriptions.
4. `src/collectors/trends_collector.py` — Fetches daily viral trends for India (`geo='IN'`) via Google Trends RSS / `pytrends` and hot discussions from Indian & Kolkata subreddits (`r/kolkata`, `r/india`, `r/IndianStockMarket`, `r/kolkatacity`).
5. `src/ml/deduplicator_ranker.py` — Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute headline + lead embeddings, clusters duplicate coverage via agglomerative/cosine similarity (`threshold >= 0.80`), preserves 100% of unique clusters, and computes a deterministic **Editorial Priority Score** (Coverage Count + Regional Boost for Kolkata/WB + Source Tier + Recency).
6. `src/llm/synthesizer.py` — Uses `google-genai` SDK (`gemini-3.8-flash` primary, `gemini-3.5-flash-lite` fallback) and `groq` (`llama-3.3-70b-versatile` secondary fallback) with exponential backoff retry logic and Pydantic Structured Outputs. Processes stories in safe batches of 15–20 clusters per API call so zero stories are truncated or skipped.
7. `src/publisher/epaper_builder.py` — Renders a classic broadsheet-style multi-column HTML newspaper (`templates/epaper.html` + `templates/epaper.css`) via `Jinja2` and compiles it into a crisp, paginated PDF (`editions/YYYY/MM/Daily_Chronicle_YYYY-MM-DD.pdf`) using `WeasyPrint`, alongside a full `editions/YYYY/MM/Daily_Chronicle_YYYY-MM-DD.md` archive.
8. `src/publisher/distributor.py` — Sends the generated PDF and executive headline digest directly to Telegram via Bot API (`sendDocument` + `sendMessage`), with optional Gmail SMTP support.
9. `src/main.py` — End-to-end orchestrator with structured logging, execution metrics (total fetched, unique clusters, section counts, runtime), and fault-tolerant error handling.
10. `requirements.txt` & `README.md` — Exact pinned dependencies and 5-minute setup instructions.
11. after the built completion, mention the required keys in a .env file, so i can include them properly

Write complete, functional, non-placeholder code for every single file.

```

## 2. Zero-Cost Tech Stack (Use Cases & Why Each Tool Is Chosen)

| Layer | Technology / Library | Specific Use Case | Why This Over Paid/Heavy Alternatives | 
| ----- | ----- | ----- | ----- | 
| **Compute & 24/7 Scheduler** | **GitHub Actions** (Public Repo) | Runs the entire agent automatically every day at `06:00 AM IST` (`00:30 UTC`) and stores historical editions via Git commits. | **100% free unlimited minutes** on public repos. Unlike Render/Railway free tiers, it never sleeps, needs zero server maintenance, and natively versions every generated PDF/MD edition. | 
| **Language & Concurrency** | **Python 3.11+** (`asyncio`, `aiohttp`) | Fetches 40+ RSS feeds and decodes article URLs concurrently in \~15 seconds. | Fastest ecosystem for RSS parsing, NLP embeddings, LLM SDKs, and PDF generation. | 
| **News Ingestion** | **Google News RSS** + Direct Indian Media RSS + `feedparser` | Pulls unlimited, unblocked hyper-local (Kolkata/WB), state-level, national, and global news. | NewsAPI/GNews free tiers cap at 100 requests/day and delay articles by 24 hours. **Google News RSS is real-time, keyless, and unlimited.** | 
| **Article Body Extraction** | `trafilatura` + `beautifulsoup4` | Extracts clean article text and strips ads/navbars from decoded publisher URLs. | Outperforms `newspaper3k` in extraction accuracy and handles modern Indian news portals cleanly without headless browser overhead. | 
| **Trends Intelligence** | **Google Trends Daily RSS** + `pytrends` + Reddit Public JSON | Captures viral search spikes in India and trending civic/social topics in Kolkata & India. | X (Twitter) API costs \$100+/month. Google Trends RSS + Reddit `.json` endpoints (`r/kolkata`, `r/india`) provide real-time pulse at **\$0 cost**. | 
| **ML Deduplication & Ranking** | `sentence-transformers` (`all-MiniLM-L6-v2`) + `scikit-learn` | Converts headlines into 384-dim vectors and clusters duplicate coverage (`cosine_similarity >= 0.80`) without losing unique stories. | Runs on GitHub Actions CPU in \~3 seconds (\~90MB model), eliminating duplicate articles before LLM synthesis while preserving every distinct story. | 
| **AI Summarization & Structuring** | **Google Gemini API** (`gemini-3.8-flash` / `gemini-3.5-flash-lite`) + **Groq API** (`llama-3.3-70b-versatile`) | Categorizes clusters, writes crisp journalistic summaries, and outputs strict JSON schemas. | Gemini 3.8 Flash offers a massive context window and generous free tier with native Structured Outputs; Groq provides ultra-fast automatic failover. | 
| **E-Paper PDF Engine** | **Jinja2** + **WeasyPrint** (HTML5 + CSS3 Paged Media) | Generates a multi-column broadsheet newspaper layout with masthead, dateline, section dividers, and page numbers. | ReportLab requires painful coordinate math. WeasyPrint uses standard **CSS Multi-Column (`column-count: 3`)**, `@page` headers/footers, and Flexbox/Grid. | 
| **Delivery & Storage** | **Telegram Bot API** + **GitHub Repo Archive** (+ optional Gmail SMTP) | Pushes the daily E-Paper PDF straight to your phone/desktop via Telegram and commits `.pdf` + `.md` files to `editions/YYYY/MM/`. | WhatsApp Cloud API requires business verification and template fees. Telegram Bot API is completely free, instant, supports files up to 50MB, and acts as cloud storage. | 

## 3. API Keys, Credentials & Resource Setup (100% Free)

Configure these keys as **GitHub Repository Secrets** (`Settings -> Secrets and variables -> Actions -> New repository secret`) and locally in a `.env` file:

| Secret Name | Source URL & Exact Setup Steps | Purpose in System | Free Tier Quota | 
| ----- | ----- | ----- | ----- | 
| `GEMINI_API_KEY` | 1\. Go to [Google AI Studio](https://aistudio.google.com/)  2\. Sign in -> Click **Get API Key** -> **Create API Key** | Primary LLM engine (`gemini-3.8-flash` with fallback to `gemini-3.5-flash-lite`) for batch summarization, tagging, and headline generation. | **Free (\$0, no billing card required)** — \~10 RPM / up to 1,500 RPD on Flash/Flash-Lite models. | 
| `GROQ_API_KEY` | 1\. Go to [Groq Console](https://console.groq.com/keys)  2\. Sign in -> **API Keys** -> **Create API Key** | Secondary high-speed LLM fallback (`llama-3.3-70b-versatile`) if Gemini hits a rate limit (`HTTP 429`). | **Free (\$0)** — \~30 RPM / 14,400 RPD on open-weight models. | 
| `TELEGRAM_BOT_TOKEN` | 1\. Open Telegram -> Search `@BotFather`  2\. Send `/newbot` -> Follow prompts -> Copy HTTP API Token | Authenticates your autonomous bot to deliver the daily PDF and summary alerts. | **Unlimited & 100% Free** (up to 50 MB per PDF upload). | 
| `TELEGRAM_CHAT_ID` | 1\. Search `@userinfobot` on Telegram -> Click **Start**  2\. Copy your numeric `Id` (or channel ID `-100...`) | Targets your personal chat (or private channel) so only you receive the E-Paper. | **Unlimited & 100% Free**. | 
| `GMAIL_USER` *(Optional)* | Your Gmail address (e.g., `yourname@gmail.com`) | Optional email delivery of the E-Paper PDF alongside Telegram. | **Free** (500 emails/day limit via Gmail SMTP). | 
| `GMAIL_APP_PASSWORD` *(Optional)* | 1\. Go to [Google Account Security](https://myaccount.google.com/security)  2\. Enable **2-Step Verification** -> Search **App Passwords** -> Generate 16-char password | Authenticates `smtp.gmail.com:465` securely without exposing your main Google password. | **Free**. | 

## 4. Comprehensive News & Trends URL Registry (`config/sources.yaml`)

All Google News RSS URLs include `hl=en-IN&gl=IN&ceid=IN:en` for Indian edition localization and `when:1d` where applicable so every story from the last 24 hours is captured.

### A. Kolkata & West Bengal (Hyper-Local Priority Tier)

1. **Google News — Kolkata City & Civic:**
   `https://news.google.com/rss/search?q=Kolkata+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

2. **Google News — West Bengal State Politics, Policy & Districts:**
   `https://news.google.com/rss/search?q=%22West+Bengal%22+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

3. **Google News — Kolkata Geo Feed:**
   `https://news.google.com/rss/headlines/section/geo/Kolkata?hl=en-IN&gl=IN&ceid=IN:en`

4. **Google News — West Bengal Geo Feed:**
   `https://news.google.com/rss/headlines/section/geo/West%20Bengal?hl=en-IN&gl=IN&ceid=IN:en`

5. **The Indian Express — Kolkata City Feed:**
   `https://indianexpress.com/section/cities/kolkata/feed/`

6. **The Hindu — West Bengal State Feed:**
   `https://www.thehindu.com/news/national/west-bengal/feeder/default.rss`

7. **Times of India — Kolkata Edition RSS:**
   `https://timesofindia.indiatimes.com/rssfeeds/-2128830821.cms`

8. **ABP Ananda — Kolkata & Bengal News (Bengali/Regional cross-check):**
   `https://bengali.abplive.com/news/kolkata/feed`

### B. All Indian States & Union Territories (Comprehensive Regional Coverage)

To ensure **zero Indian state news is missed**, ingest both direct state feeds and regional cluster queries:

1. **The Hindu — All Indian States Hub:**
   `https://www.thehindu.com/news/states/feeder/default.rss`

2. **Indian Express — India & Regional States:**
   `https://indianexpress.com/section/india/feed/`

3. **North India States (Delhi NCR, UP, Punjab, Haryana, Rajasthan, Uttarakhand, HP, J&K):**
   `https://news.google.com/rss/search?q=(Delhi+OR+%22Uttar+Pradesh%22+OR+Punjab+OR+Haryana+OR+Rajasthan+OR+Uttarakhand+OR+%22Jammu+and+Kashmir%22)+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

4. **West & Central India States (Maharashtra, Gujarat, Madhya Pradesh, Chhattisgarh, Goa):**
   `https://news.google.com/rss/search?q=(Maharashtra+OR+Mumbai+OR+Gujarat+OR+%22Madhya+Pradesh%22+OR+Chhattisgarh+OR+Goa)+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

5. **South India States (Karnataka, Tamil Nadu, Telangana, Andhra Pradesh, Kerala):**
   `https://news.google.com/rss/search?q=(Karnataka+OR+Bengaluru+OR+%22Tamil+Nadu%22+OR+Chennai+OR+Telangana+OR+Hyderabad+OR+%22Andhra+Pradesh%22+OR+Kerala)+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

6. **East & North-East India States (Odisha, Bihar, Jharkhand, Assam, Sikkim, Tripura, Meghalaya, Manipur, Nagaland, Arunachal Pradesh, Mizoram):**
   `https://news.google.com/rss/search?q=(Odisha+OR+Bihar+OR+Jharkhand+OR+Assam+OR+Manipur+OR+Meghalaya+OR+Tripura+OR+Sikkim+OR+Arunachal+OR+Nagaland+OR+Mizoram)+when:1d&hl=en-IN&gl=IN&ceid=IN:en`

### C. India National Current Affairs, Governance, Economy & Tech

1. **Google News — India Top Stories:**
   `https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en`

2. **Google News — India Nation Topic:**
   `https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en`

3. **The Hindu — National News:**
   `https://www.thehindu.com/news/national/feeder/default.rss`

4. **Press Information Bureau (PIB) — Official Govt of India Releases:**
   `https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3`

5. **Indian Express — Explained & Policy Deep-Dives:**
   `https://indianexpress.com/section/explained/feed/`

6. **Google News — India Business & Economy:**
   `https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en`

7. **Google News — India Science & Technology:**
   `https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en`

### D. Global Breaking News & International Affairs

1. **Google News — World Breaking Headlines (India Edition):**
   `https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en`

2. **BBC World News RSS:**
   `http://feeds.bbci.co.uk/news/world/rss.xml`

3. **Al Jazeera English — All Breaking News:**
   `https://www.aljazeera.com/xml/rss/all.xml`

4. **The Hindu — International Affairs:**
   `https://www.thehindu.com/news/international/feeder/default.rss`

5. **Indian Express — World News:**
   `https://indianexpress.com/section/world/feed/`

### E. Viral Trends & Social Radar (India & Kolkata)

1. **Google Trends — Daily Trending Searches in India (Direct XML Feed):**
   `https://trends.google.com/trending/rss?geo=IN`

2. **Indian Express — Trending in India:**
   `https://indianexpress.com/section/trending/trending-in-india/feed/`

3. **Reddit — r/kolkata Top Daily Posts (JSON API — requires custom `User-Agent` header):**
   `https://www.reddit.com/r/kolkata/top.json?t=day&limit=15`

4. **Reddit — r/india Top Daily Posts (JSON API):**
   `https://www.reddit.com/r/india/top.json?t=day&limit=15`

5. **Reddit — r/IndianStockMarket / Business Buzz (JSON API):**
   `https://www.reddit.com/r/IndianStockMarket/top.json?t=day&limit=10`

## 5. GitHub Actions Schedule, Execution Duration & Rolling Window

### Schedule Configuration (`.github/workflows/daily_epaper.yml`)

GitHub Actions uses **UTC** time (`IST = UTC + 5:30`):

* **Primary Morning E-Paper Edition (`06:00 AM IST`):**
  `cron: '30 0 * * *'` (`00:30 UTC` every day)

* **Optional Evening Breaking Wrap (`08:00 PM IST`):**
  `cron: '30 14 * * *'` (`14:30 UTC` every day — uncomment if you want two editions daily)

* **Manual On-Demand Trigger:**
  `workflow_dispatch:` (lets you trigger an instant edition anytime from the GitHub Actions tab or via GitHub REST API).

### Execution Duration & Pipeline Timing

A complete end-to-end run takes **\~2.5 to 4.5 minutes** (well inside GitHub's 6-hour job limit):

1. **Environment & WeasyPrint Setup:** \~45 seconds (cached pip dependencies reduce this to \~15s after run #1).

2. **Concurrent RSS + Trends Fetching (40+ URLs):** \~15–25 seconds via `asyncio` + `aiohttp`.

3. **Article Text Extraction (`trafilatura`):** \~30–45 seconds (concurrent worker pool of 10 workers with 8-second per-URL timeout).

4. **ML Embedding & Cosine Clustering (`all-MiniLM-L6-v2`):** \~5–10 seconds on CPU.

5. **Batch LLM Summarization (Gemini 3.8 Flash):** \~45–75 seconds (respecting free-tier RPM via controlled `asyncio.sleep(6)` between batches of 15 clusters).

6. **WeasyPrint PDF Compilation + Telegram Push + Git Archive Commit:** \~10–15 seconds.

### How Long It Gathers Reports (Rolling 24-Hour Window)

* **Strict Rolling Window:** `WINDOW_HOURS = 24` (configurable to `26` hours to prevent boundary misses during slight cron delays).

* **Filtering Logic:**

  1. Parse `entry.published_parsed` or `entry.updated_parsed` into UTC `datetime`.

  2. Keep every item where `entry_utc >= (now_utc - timedelta(hours=26))`.

  3. Cross-check against `data/seen_urls.json` (a lightweight rolling 7-day deduplication ledger committed in the repo) so morning and evening runs never repeat an already-published story while guaranteeing **zero new stories are ever missed**.

## 6. ML Deduplication, Ranking & Zero-Exclusion Summarization Pipeline

### Step 1: Canonical URL Resolution & Cleaning

Google News RSS links use encoded redirect wrappers (`https://news.google.com/rss/articles/...`).

* Resolve redirects via lightweight HTTP `HEAD`/`GET` (or `googlenewsdecoder`) to extract the true publisher URL and clean source domain (e.g., `thehindu.com`, `telegraphindia.com`).

* Extract the full article body using `trafilatura.extract()`. If a site blocks scraping or times out, fall back gracefully to the cleaned RSS `<description>` summary so the story is **never dropped**.

### Step 2: Semantic Deduplication (Without Losing Any Story)

1. Concatenate `title + ". " + lead_paragraph[:300]` for every fetched article ($N \approx 250\text{–}400$ raw items/day).

2. Encode all texts in a single batch using `SentenceTransformer('all-MiniLM-L6-v2')`.

3. Compute the pairwise **Cosine Similarity Matrix**:
   

   $$
   \text{sim}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}
   $$

4. **Agglomerative Clustering (`distance_threshold = 0.20`, i.e., `similarity >= 0.80`):**

   * Articles covering the *exact same event* (e.g., 8 outlets reporting a Calcutta High Court ruling or ISRO launch) merge into **one Story Cluster**, preserving the richest article text as the primary body and storing all 8 outlet names in `sources[]`.

   * Articles with unique angles or distinct local events remain **singleton clusters (`size = 1`)**.

   * **Result:** 350 raw articles compress cleanly into \~120–160 unique Story Clusters with **100% topical retention**.

### Step 3: Editorial Priority Ranking Formula

Each cluster $C$ is scored and sorted within its section using:

$$
\text{Score}(C) = (W_{\text{coverage}} \times \ln(1 + \vert{}C\vert{})) + W_{\text{region}} + W_{\text{authority}} + W_{\text{recency}}
$$

* **Coverage Weight (**$W_{\text{coverage}} = 3.0$**):** Stories reported by multiple independent outlets automatically rise to the top as "Lead Breaking News".

* **Regional Priority Boost (**$W_{\text{region}}$**):** `+2.5` for Kolkata / West Bengal tags; `+1.5` for Indian States & National policy; `+1.0` for Global Breaking.

* **Source Authority (**$W_{\text{authority}}$**):** `+1.5` for primary official/national sources (PIB, The Hindu, Indian Express, BBC, Reuters/PTI); `+1.0` for standard verified outlets.

* **Recency Decay (**$W_{\text{recency}}$**):** Linear bonus from `+0.0` (24h old) to `+2.0` (less than 2 hours old).

### Step 4: Zero-Exclusion Batch LLM Summarization (Map-Reduce)

To guarantee **no news item is excluded or cut off by token limits**:

1. Group all ranked clusters into their exact E-Paper sections:

   * **Section 1:** Front Page — Top National & Global Lead Stories (Top 8 highest-scoring clusters across all categories)

   * **Section 2:** Kolkata & West Bengal Desk (100% of Kolkata/WB clusters)

   * **Section 3:** Indian States & Union Territories Digest (100% of regional state clusters, tagged with `[STATE NAME]`)

   * **Section 4:** National Affairs, Economy & Tech (100% of remaining India/Business/Tech clusters)

   * **Section 5:** World & International Desk (100% of global clusters)

   * **Section 6:** Viral & Social Trends Radar (100% of Google Trends India + Reddit Kolkata/India highlights)

2. **Chunked Batch Processing:** Split each section's clusters into chunks of **15 clusters per LLM call**.

3. Call `gemini-3.8-flash` (with `response_mime_type="application/json"` and a strict Pydantic schema):

   * Input: List of 15 clusters (`cluster_id`, `headline`, `sources`, `extracted_text[:1200]`).

   * System Prompt Rule: *"You MUST return exactly 15 summarized objects matching the 15 input `cluster_id`s. Do NOT merge, skip, or omit any `cluster_id`. Write a sharp, factual 2–3 sentence journalistic summary + 1 'Key Takeaway' line for each."*

   * Validation Check: Assert `len(output_items) == len(input_batch)`. If any ID is missing or an API error occurs, automatically retry with `gemini-3.5-flash-lite` or `groq/llama-3.3-70b-versatile`, or fall back to extractive RSS summary so **zero items are ever lost**.

### Step 5: E-Paper Broadsheet Rendering & Delivery

1. **HTML/CSS Broadsheet (`templates/epaper.html`):**

   * Classic serif masthead (**"THE DAILY CHRONICLE — INDIA & BENGAL EDITION"**), edition date, weather/day banner, and story count badge (e.g., *"142 Verified Stories | 38 Sources"*).

   * **CSS Multi-Column Layout** (`column-count: 3; column-gap: 18px; column-rule: 1px solid #d1d5db;`) with `break-inside: avoid` on individual news cards so articles never split awkwardly across columns.

   * Color-coded regional tags (`[KOLKATA]`, `[WEST BENGAL]`, `[MAHARASHTRA]`, `[GLOBAL]`, `[TRENDING]`) and clickable source links in every story card.

2. **Output Artifacts:**

   * `editions/YYYY/MM/EPaper_YYYY-MM-DD.pdf` (Compiled via `WeasyPrint`)

   * `editions/YYYY/MM/Report_YYYY-MM-DD.md` (Full searchable Markdown archive)

3. **Instant Delivery:**

   * Telegram Bot sends the PDF file directly to your chat with a pinned caption summarizing the Top 5 Front-Page Headlines.

   * GitHub Actions commits and pushes both `.pdf` and `.md` files into `editions/YYYY/MM/` for permanent cloud archiving.