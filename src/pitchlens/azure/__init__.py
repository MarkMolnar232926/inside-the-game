"""Thin Azure adapters. Each one has a local mock so PitchLens runs fully offline.

| Concern            | Azure service                         | Local fallback            |
|--------------------|---------------------------------------|---------------------------|
| LLM agents         | Azure OpenAI / Azure AI Foundry       | grounded templates        |
| Translation        | Azure AI Translator                   | native locale templates   |
| Voice              | Azure AI Speech (neural TTS)          | SSML files only           |
| Live ingest        | Azure Event Hubs                      | file replay / LiveSource  |
| Fan-out to clients | Azure Web PubSub                      | in-process WebSocket hub  |
| Assets             | Azure Blob Storage                    | local ``outputs/`` folder |
"""
