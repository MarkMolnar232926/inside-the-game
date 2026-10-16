# Deploying PitchLens to Azure

PitchLens runs as one container (FastAPI API + static web UI) on **Azure Container Apps**. The AI and messaging services are optional and switched on by environment variables. Any service you leave out falls back to its local mock.

## Option A: `azd up` (recommended)

```bash
azd auth login
azd env new pitchlens-demo
azd env set AZURE_LOCATION swedencentral   # any region with Azure OpenAI + Container Apps
azd up                                     # builds the Dockerfile, provisions infra/main.bicep, deploys
```

`infra/main.bicep` provisions:

| Resource | Purpose |
|---|---|
| Log Analytics + Container Apps environment | hosting + structured logs |
| Azure Container Registry | image built by `azd` |
| Container App `ca-pitchlens-*` | API + UI, `PITCHLENS_MODE=azure`, secrets from the resources below |
| Azure OpenAI (`gpt-4o-mini` deployment) | Explainer / Recap agents |
| Azure AI Translator | translation of free-form LLM text |
| Azure AI Speech | neural TTS of narrative lines |
| Event Hubs namespace + `match-events` hub | live event ingest |
| Web PubSub | fan-out of overlay instructions to viewers |
| Storage account + `pitchlens-assets` container | rendered clips, alpha overlays, recap cards |

The demo bundle (video + timelines) is built offline with `python scripts/build_demo.py` and uploaded with `AssetStore` (`pitchlens.azure.storage`), or baked into the image by copying `outputs/demo` before `azd deploy`.

## Option B: manual (portal / CLI)

```bash
RG=rg-pitchlens; LOC=swedencentral; APP=pitchlens
az group create -n $RG -l $LOC
az acr create -g $RG -n ${APP}acr --sku Basic --admin-enabled true
az acr build -r ${APP}acr -t pitchlens:latest .
az containerapp env create -g $RG -n ${APP}-env -l $LOC
az containerapp create -g $RG -n $APP --environment ${APP}-env \
  --image ${APP}acr.azurecr.io/pitchlens:latest --registry-server ${APP}acr.azurecr.io \
  --target-port 8000 --ingress external --cpu 1 --memory 2Gi \
  --env-vars PITCHLENS_MODE=mock
# then add the AI services you want and set e.g. AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY (as secrets)
```

## Configuration

All settings are listed in `.env.example`. Verify the mode with `GET /api/health` (`{"mode": "azure"}`).

## Live ingest via Event Hubs

```python
from pathlib import Path
from pitchlens.azure.eventhubs import EventHubsPublisher
from pitchlens.synth.generate import SyntheticMatch
m = SyntheticMatch.load(Path("data/synthetic/demo"))
EventHubsPublisher().publish(m.events, partition_key=m.meta.match_id)
```

`EventHubsSource().stream()` yields validated `Event`s into the same `PitchLensEngine.step()` used by the local live feed.
