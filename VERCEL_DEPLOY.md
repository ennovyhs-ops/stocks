# Deploying to Vercel

This document shows two simple ways to deploy this Flask app to Vercel: connect your Git repository (recommended), or deploy directly with the Vercel CLI.

Prerequisites
- A Vercel account
- Your project pushed to a Git provider (GitHub, GitLab, or Bitbucket) OR local repo access
- `requirements.txt` present (this project already includes one)

Recommended (GitHub) - fast and repeatable
1. Push your repo to GitHub.
2. In the Vercel dashboard, click "Import Project" → "From Git Repository" and select your repo.
3. For Framework Preset choose "Other" (this is a Python/Flask app).
4. Vercel will detect `vercel.json` and `requirements.txt`. Confirm and deploy.

Vercel CLI - quick one-off
1. Install the Vercel CLI:

```bash
npm i -g vercel
```

2. Login:

```bash
vercel login
```

3. From your project root (where `app.py` and `vercel.json` live), run:

```bash
vercel --prod
```

4. Follow prompts to set project name and link to team/account.

Notes and gotchas
- Vercel's Python runtime will install packages from `requirements.txt` during build. Add any missing packages there.
- Vercel serverless functions are ephemeral and not intended for persistent file writes. The app currently writes outputs to `static/output/` — this may not persist across invocations. If you need persistent storage, use an external storage service (S3, Google Cloud Storage) or return files directly in responses.
- If your app writes files, consider writing into `/tmp` during request handling and streaming them back to the client, or upload them to external storage and serve from there.
- If you need environment variables (API keys, etc.), set them in the Vercel dashboard (Project → Settings → Environment Variables) or via the CLI with `vercel env add`.

Troubleshooting
- 500 errors: check function logs in Vercel dashboard to see import-time exceptions. Common causes: missing package in `requirements.txt`, code that assumes writable filesystem at import time, or network calls that require credentials.
- Static assets: Vercel will serve the `static/` folder from the deployment bundle, but files created at runtime in serverless functions are ephemeral.

Advanced
- If you prefer a dedicated WSGI server (Gunicorn + a persistent host), consider deploying to a VM/container (Render, Fly, Railway, DigitalOcean).
