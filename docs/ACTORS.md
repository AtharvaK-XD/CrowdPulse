# CrowdPulse — Apify Actor & Source Research

**Date:** 2026-11-20  
**Target:** 5 HKEX Stocks (Tencent, Alibaba, Meituan, Xiaomi, BYD) & 5 KRX Stocks (Samsung Electronics, SK Hynix, NAVER, Kakao, Hyundai Motor)  
**Author:** Antigravity (Report / QA Agent)  
**Consumer:** Claude Code (Backend Scraping Agent)  

---

## 1. Social Sources & Apify Actors Matrix

| Platform / Source | Target Exchange | Candidate Apify Actor ID | What it Scrapes | Fields Returned (Timestamp, Likes, Views, Followers) | Pricing | Maintenance & Rating | JavaScript Required? | Recommended? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naver Finance Discussion Board** | KRX | `apify/playwright-scraper` *(dedicated fallback)* | Dynamic stock discussion posts, titles, authors, and replies | **Timestamp:** Yes<br>**Likes (Up/Down):** Yes<br>**Views:** Yes<br>**Followers:** No (anonymous ID) | Platform compute ($0.25 / CU) | Official Apify Actor (Active, 4.9/5) | **Yes** *(verified: modern Next.js client-side app)* | **Recommended** (`apify/playwright-scraper` or internal JSON API `https://m.stock.naver.com/api/discussion/<ticker>`) |
| **Naver Finance Fundamental Data** | KRX | `parseforge/naver-finance-scraper` | Financial statements, PER, PBR, market cap, consensus | **Timestamp:** Yes<br>**Likes/Views:** N/A<br>**Followers:** N/A | Usage-based | Updated 2025/2026 (4.8/5) | No | Secondary for fundamentals; not for retail posts |
| **Daum Finance** | KRX | `apify/playwright-scraper` *(dedicated fallback)* | Stock discussion board posts, comments, sentiment polls | **Timestamp:** Yes<br>**Likes/Dislikes:** Yes<br>**Views:** Yes<br>**Followers:** No | Platform compute ($0.25 / CU) | Official Apify Actor (Active, 4.9/5) | **Yes** *(SPA framework requiring JS execution)* | **Recommended** (`apify/playwright-scraper` or Kakao Finance mobile feed) |
| **Futu / Moomoo Community** | HKEX | `apify/playwright-scraper` *(dedicated fallback)* | Stock discussion feeds, user posts, replies, sentiment badges | **Timestamp:** Yes<br>**Likes:** Yes<br>**Views:** Yes (impressions)<br>**Followers:** Yes (author follower count) | Platform compute ($0.25 / CU) | Official Apify Actor (Active, 4.9/5) | **Yes** *(verified: Vue/React client rendering)* | **Recommended** (`apify/playwright-scraper` targeting `moomoo.com/community`) |
| **AAStocks** | HKEX | `bovi/aastocks-scraper` | Financial news listings, stock headlines, category feeds | **Timestamp:** Yes<br>**Likes/Views:** No<br>**Followers:** No | Pay-per-event (~$0.0035 / item) | Active (5.0/5) | No (Static HTML / API) | **Recommended** for HKEX news & official disclosures |
| **AAStocks Forum** | HKEX | `apify/cheerio-scraper` *(dedicated fallback)* | Retail forum threads and comments | **Timestamp:** Yes<br>**Likes/Views:** Views only<br>**Followers:** No | Platform compute ($0.25 / CU) | Official Apify Actor (Active, 4.9/5) | **No** (Static server-rendered tables) | **Recommended** for retail discussion |
| **LIHKG (連登)** | HKEX *(backup)* | `claude_code_reviewer/lihkg-scraper` | Hong Kong's largest forum: stock threads, comments, sentiment | **Timestamp:** Yes (`created_at`)<br>**Likes:** Yes (`like_count`)<br>**Dislikes:** Yes (`dislike_count`)<br>**Views/Replies:** Yes (`reply_count`)<br>**Followers:** No | Free / standard compute | Active (5.0/5) | No (Scrapes public `api_v2` JSON directly) | **Recommended** (Primary HK retail sentiment forum) |
| **Blind (TeamBlind)** | KRX *(backup)* | `hypebridge/blind-post-scraper` | Anonymous verified employee posts, topics, comments | **Timestamp:** Yes<br>**Likes:** Yes (reactions)<br>**Views/Comments:** Yes<br>**Followers:** No (shows company + role) | Usage-based (~$1.50 / 1,000 results) | Active (4.7/5) | Handled by Actor | **Recommended** for insider / institutional employee sentiment |
| **Telegram Channels** | HKEX & KRX | `stratumics/telegram-channel-scraper` | Public Telegram stock trading and signal channels | **Timestamp:** Yes (UTC)<br>**Reactions/Likes:** Yes<br>**Views:** Yes<br>**Followers:** Yes (channel subscriber count) | Pay-per-result (~$1.00 / 1,000 posts) | Active (4.9/5) | Handled by Actor (no login needed) | **Recommended** for real-time retail signal channels |

---

## 2. Real Web Page Inspection: JavaScript Requirement Check

### Naver Finance Stock Discussion Board (`005930.KS` - Samsung Electronics)
- **URL Tested:** `https://finance.naver.com/item/board.naver?code=005930`
- **Redirect Target:** Automatically redirects to `https://stock.naver.com/domestic/stock/005930/discussion`.
- **Finding:**
  - Modern web application built with **Next.js & Turbopack** (`_next/static/chunks/turbopack-...`).
  - Raw HTML does **not** contain table rows or discussion post text.
  - Posts and comments are populated client-side by React hydration and dynamic JSON calls.
- **Scraper Verdict:**
  - **Do NOT use `apify/cheerio-scraper`** (it will yield 0 posts).
  - **Use `apify/playwright-scraper`** to wait for the discussion DOM to mount, OR Claude Code can fetch the mobile REST endpoint directly: `https://m.stock.naver.com/api/discussion/005930?page=1`.

### Futu / Moomoo Tencent Discussion Page (`0700.HK` - Tencent)
- **URLs Tested:** `https://www.futunn.com/hk/stock/0700-HK`, `https://www.moomoo.com/community/discussion/0700-HK`, `https://www.futunn.com/quote/0700-HK`.
- **Finding:**
  - The static HTML payload is a shell (empty `<div id="app">` / client hydration shell) with JavaScript bundles.
  - User discussion feeds and community comments are fetched via client-side OpenAPI requests.
- **Scraper Verdict:**
  - **Do NOT use `apify/cheerio-scraper`**.
  - **Use `apify/playwright-scraper`** with network interception or headless rendering.

---

## 3. Exa API Documentation & Limits

### Endpoint
- **Search Endpoint:** `POST https://api.exa.ai/search`
- **Recommended Request Body:**
  ```json
  {
    "query": "0700.HK Tencent stock earnings retail sentiment Hong Kong",
    "useAutoprompt": true,
    "numResults": 10,
    "contents": {
      "text": true,
      "summary": {
        "query": "Key stock price catalysts and sentiment"
      }
    }
  }
  ```
- **Contents Endpoint:** `POST https://api.exa.ai/contents` (used to pull full clean text and summaries for specific scraped URLs).

### Pricing & Free-Tier Limits
- **Free Tier Allowance:** **$10 recurring monthly credit** (resets on the 1st of every calendar month).
- **Onboarding Credit:** Additional **$10 one-time credit** upon dashboard onboarding completion.
- **Credit Card Required:** No.
- **Rate Limits:** **10 Queries Per Second (QPS)** on the free tier.
- **Cost per 1,000 Requests:**
  - Search (Lite): $4 / 1,000 requests.
  - Search (Auto / Fast): $7 / 1,000 requests.
  - Contents Extraction: $1 / 1,000 pages.
- **Capacity for CrowdPulse:** $10/month affords roughly **1,400 to 2,500 news queries**, which is more than sufficient for 10 tickers over 10 trading days.

---

## 4. Summary & Hand-off Instructions for Claude Code

1. **HKEX Social:**
   - Primary: `bovi/aastocks-scraper` + `claude_code_reviewer/lihkg-scraper` (rich sentiment & upvote metrics).
   - Secondary / Alternative: `apify/playwright-scraper` on Moomoo community.
2. **KRX Social:**
   - Primary: `apify/playwright-scraper` (or direct mobile API) on Naver Finance discussion (`stock.naver.com/domestic/stock/<ticker>/discussion`).
   - Backup: `hypebridge/blind-post-scraper` for Korean tech stock sentiment (Samsung, SK Hynix, Naver, Kakao).
3. **Cross-Exchange Telegram:**
   - `stratumics/telegram-channel-scraper` for public trader groups.
4. **News & Web Intelligence:**
   - Exa API with `contents: { summary: true, text: true }`.
