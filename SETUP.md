# Smart Notes RAG Assistant — Setup & Resume Guide

Use this file any time you come back to this project after a break and need
to get it running locally again. Follow it top to bottom, in order.

---

## 0. What you're working with

Two separate projects, two separate repos, two separate terminals when running locally:

| Project | Folder | Repo | Live URL |
|---|---|---|---|
| Backend (FastAPI) | `D:\Self learning\youtube-rag-assistant` | github.com/Abdulmoeez1010/smart-notes-rag-assistant | https://smart-notes-rag-assistant.onrender.com |
| Frontend (React) | `D:\Self learning\smart-notes-frontend` | github.com/Abdulmoeez1010/smart-notes-frontend | https://smart-notes-frontend-black.vercel.app |

The live URLs above already work — if you just want to *use* the app, visit
the Vercel URL directly, no setup needed. Everything below is for running it
**locally** (e.g. to keep developing, or to demo the YouTube ingestion path,
which only works locally — see the note at the bottom).

---

## 1. Re-cloning on a new machine (skip if the folders already exist)

```bash
cd "D:\Self learning"
git clone https://github.com/Abdulmoeez1010/smart-notes-rag-assistant.git youtube-rag-assistant
git clone https://github.com/Abdulmoeez1010/smart-notes-frontend.git
```

---

## 2. Backend: first-time setup (skip if `venv` already exists)

```bash
cd "D:\Self learning\youtube-rag-assistant"
python -m venv venv
venv\Scripts\activate          # Windows (cmd) — for Git Bash use: source venv/Scripts/activate
pip install -r requirements.txt
```

**Recreate your `.env` file** (this is never committed to GitHub, so it won't
exist after a fresh clone). Create a file named `.env` in the project root
with exactly:
```
GEMINI_API_KEY=your_real_key_here
```
No quotes around the value. Get your key from https://aistudio.google.com/apikey
if you no longer have it saved anywhere.

---

## 3. Backend: running it locally (every time)

```bash
cd "D:\Self learning\youtube-rag-assistant"
venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

Confirm it's alive: open `http://127.0.0.1:8000/docs` — you should see the
Swagger UI with all 7 endpoints listed.

**Leave this terminal running** while you work — it's your live local backend.

---

## 4. Frontend: first-time setup (skip if `node_modules` already exists)

Needs Node.js installed — check with `node -v` in a plain Command Prompt
(not Git Bash). If missing, install the LTS version from nodejs.org first.

```bash
cd "D:\Self learning\smart-notes-frontend"
npm install
```

---

## 5. Frontend: running it locally (every time)

```bash
cd "D:\Self learning\smart-notes-frontend"
npm run dev
```

Open `http://localhost:5173` in your browser.

**Leave this terminal running too** — second terminal, alongside the backend one.

---

## 6. Local vs. deployed — which `api.js` points where

`src/api.js`'s `API_BASE` constant currently points to the **live Render URL**
(`https://smart-notes-rag-assistant.onrender.com`), because that's what's
deployed on Vercel right now.

If you want your **local** frontend (`localhost:5173`) to talk to your
**local** backend (`127.0.0.1:8000`) instead — e.g. to test YouTube ingestion,
which only works locally, not on Render — temporarily change:
```js
const API_BASE = "http://127.0.0.1:8000";
```
Just remember to change it back to the Render URL before committing/pushing,
or your deployed Vercel site will break (it'll try to call your local machine,
which doesn't exist from the internet's perspective).

---

## 7. Known limitation: YouTube ingestion only works locally

YouTube blocks transcript requests coming from cloud-hosting IP ranges
(Render, AWS, GCP, etc. are all commonly blocked). This means:
- ✅ YouTube ingestion works fine when running the backend **locally**
- ❌ YouTube ingestion will fail with a 500 error on the **deployed** Render backend
- ✅ PDF and PPTX ingestion work fine in **both** local and deployed environments

This is an external YouTube anti-bot restriction, not a bug in this project.
When demoing, either show YouTube ingestion running locally, or stick to
PDF/PPTX on the live deployed link.

---

## 8. If something's not working — quick checklist

- Backend won't start → confirm `venv` is activated (`venv\Scripts\activate`)
  and `.env` exists with a valid `GEMINI_API_KEY`.
- Frontend shows a CORS error in the browser console → confirm the backend's
  `main.py` `CORSMiddleware` includes whichever origin you're calling from
  (`http://localhost:5173` for local, the Vercel URL for deployed).
- `/ingest/youtube` returns a 500 mentioning IP blocking → expected on Render,
  see section 7 above.
- `node`/`npm` not recognized in Git Bash → use plain Command Prompt instead
  (this was a known quirk during initial setup — Git Bash's PATH didn't pick
  up the Node install).

---

## 9. Redeploying after a code change

**Backend (Render):**
```bash
cd "D:\Self learning\youtube-rag-assistant"
git add .
git commit -m "your change description"
git push
```
Render auto-redeploys on push to `main`. Check progress in the Render
dashboard → your service → Events/Logs.

**Frontend (Vercel):**
```bash
cd "D:\Self learning\smart-notes-frontend"
git add .
git commit -m "your change description"
git push
```
Vercel auto-redeploys on push to `main`. Check progress in the Vercel
dashboard → your project → Deployments.
