# Carta Anziani — Backend Chatbot

Backend FastAPI per l'assistente AI della "Carta dei diritti delle persone anziane".

## Deploy

Servizio web Docker. Variabile d'ambiente richiesta:
- `ANTHROPIC_API_KEY` — chiave API Anthropic (formato `sk-ant-...`)

Health check: `GET /api/health` → `{"status":"ok","chunks":70}`
