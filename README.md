# IST Voice Assistant

Institute of Space Technology voice agent - admissions inquiry assistant with RAG, natural voice, and barge-in support.

## Deploy on Render

1. **Push code** to https://github.com/menalshahid/mylivekitagent (if not already done).
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Web Service**.
3. Connect the repo `menalshahid/mylivekitagent`.
4. Use these settings:
   - **Name:** ist-voice-agent (or any name)
   - **Region:** Choose nearest
   - **Branch:** main
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (or leave blank to use Procfile)
5. Click **Advanced** → **Add Environment Variable**:
   - `GROQ_API_KEY` = your Groq API key (required)
   - `LIVEKIT_API_KEY` = your LiveKit key (if using LiveKit)
   - `LIVEKIT_API_SECRET` = your LiveKit secret (if using LiveKit)
6. Click **Create Web Service**. Render will build and deploy.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```
