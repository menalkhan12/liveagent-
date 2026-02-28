# IST Voice Assistant

Institute of Space Technology voice agent - admissions inquiry assistant with RAG, natural voice, and barge-in support.

## Deploy on Render

1. Push to GitHub and connect the repo to Render.
2. Create a **Web Service**.
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app` (or use Procfile)
5. Add environment variables in Render dashboard:
   - `GROQ_API_KEY` - Your Groq API key
   - `LIVEKIT_API_KEY` - LiveKit API key (if using)
   - `LIVEKIT_API_SECRET` - LiveKit API secret (if using)

## Local Run

```bash
pip install -r requirements.txt
python app.py
```
