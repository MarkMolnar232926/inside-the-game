// PitchLens on Azure: Container Apps (API + UI), Azure OpenAI, AI Translator, AI Speech,
// Event Hubs (live ingest), Web PubSub (fan-out), Blob Storage (assets), Log Analytics.
targetScope = 'resourceGroup'

@description('Environment name used as a prefix (azd sets this).')
param environmentName string
param location string = resourceGroup().location
@description('Container image (azd fills this in after building).')
param apiImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param openAiModel string = 'gpt-4o-mini'
param openAiModelVersion string = '2024-07-18'

var token = toLower(uniqueString(resourceGroup().id, environmentName))
var tags = { 'azd-env-name': environmentName, app: 'pitchlens' }

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${token}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr${token}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${token}'
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: { allowBlobPublicAccess: false, minimumTlsVersion: 'TLS1_2' }
}

resource blobSvc 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = { parent: storage, name: 'default' }
resource assets 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = { parent: blobSvc, name: 'pitchlens-assets' }

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'oai-${token}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: { customSubDomainName: 'oai-${token}' }
}

resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: openAiModel
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: { model: { format: 'OpenAI', name: openAiModel, version: openAiModelVersion } }
}

resource translator 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'trl-${token}'
  location: location
  tags: tags
  kind: 'TextTranslation'
  sku: { name: 'S1' }
  properties: {}
}

resource speech 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'spx-${token}'
  location: location
  tags: tags
  kind: 'SpeechServices'
  sku: { name: 'S0' }
  properties: {}
}

resource eventhubNs 'Microsoft.EventHub/namespaces@2024-01-01' = {
  name: 'evh-${token}'
  location: location
  tags: tags
  sku: { name: 'Standard', tier: 'Standard', capacity: 1 }
}

resource eventhub 'Microsoft.EventHub/namespaces/eventhubs@2024-01-01' = {
  parent: eventhubNs
  name: 'match-events'
  properties: { partitionCount: 4, messageRetentionInDays: 1 }
}

resource ehRule 'Microsoft.EventHub/namespaces/authorizationRules@2024-01-01' existing = {
  parent: eventhubNs
  name: 'RootManageSharedAccessKey'
}

resource pubsub 'Microsoft.SignalRService/webPubSub@2024-03-01' = {
  name: 'wps-${token}'
  location: location
  tags: tags
  sku: { name: 'Free_F1', capacity: 1 }
  properties: {}
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${token}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-pitchlens-${token}'
  location: location
  tags: union(tags, { 'azd-service-name': 'api' })
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [ { server: registry.properties.loginServer, username: registry.listCredentials().username, passwordSecretRef: 'acr' } ]
      secrets: [
        { name: 'acr', value: registry.listCredentials().passwords[0].value }
        { name: 'openai-key', value: openai.listKeys().key1 }
        { name: 'translator-key', value: translator.listKeys().key1 }
        { name: 'speech-key', value: speech.listKeys().key1 }
        { name: 'eventhub', value: ehRule.listKeys().primaryConnectionString }
        { name: 'pubsub', value: pubsub.listKeys().primaryConnectionString }
        { name: 'storage', value: 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}' }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'PITCHLENS_MODE', value: 'azure' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'openai-key' }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: openAiModel }
            { name: 'AZURE_TRANSLATOR_KEY', secretRef: 'translator-key' }
            { name: 'AZURE_TRANSLATOR_REGION', value: location }
            { name: 'AZURE_SPEECH_KEY', secretRef: 'speech-key' }
            { name: 'AZURE_SPEECH_REGION', value: location }
            { name: 'AZURE_EVENTHUB_CONNECTION_STRING', secretRef: 'eventhub' }
            { name: 'AZURE_EVENTHUB_NAME', value: eventhub.name }
            { name: 'AZURE_WEBPUBSUB_CONNECTION_STRING', secretRef: 'pubsub' }
            { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage' }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.properties.loginServer
output SERVICE_API_URI string = 'https://${api.properties.configuration.ingress.fqdn}'
