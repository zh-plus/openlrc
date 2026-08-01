import { contextBridge, ipcRenderer } from "electron";

import { channels, type OpenLRCDesktopBridge } from "../shared/contracts.js";

function subscribe<T>(channel: string, listener: (payload: T) => void): () => void {
  const wrapped = (_event: Electron.IpcRendererEvent, payload: T): void => listener(payload);
  ipcRenderer.on(channel, wrapped);
  return () => ipcRenderer.removeListener(channel, wrapped);
}

const bridge: OpenLRCDesktopBridge = {
  app: {
    info: () => ipcRenderer.invoke(channels.appInfo),
    backendStatus: () => ipcRenderer.invoke(channels.backendStatus),
    restartBackend: () => ipcRenderer.invoke(channels.backendRestart),
    openExternal: (url) => ipcRenderer.invoke(channels.appOpenExternal, { url }),
    onBackendStatus: (listener) => subscribe(channels.backendChanged, listener),
  },
  appearance: {
    get: () => ipcRenderer.invoke(channels.appearanceGet),
    setTheme: (source) => ipcRenderer.invoke(channels.appearanceSetTheme, { source }),
    setSidebarCollapsed: (collapsed) => ipcRenderer.invoke(channels.appearanceSetSidebar, { collapsed }),
    onChanged: (listener) => subscribe(channels.appearanceChanged, listener),
  },
  dialogs: {
    selectInputs: (kind) => ipcRenderer.invoke(channels.dialogSelectInputs, { kind }),
  },
  workflows: {
    preflight: (draft) => ipcRenderer.invoke(channels.workflowPreflight, { draft }),
    active: () => ipcRenderer.invoke(channels.workflowActive),
    onEvent: (listener) => subscribe(channels.workflowEvent, listener),
  },
  queue: {
    snapshot: () => ipcRenderer.invoke(channels.queueSnapshot),
    enqueue: (draft) => ipcRenderer.invoke(channels.queueEnqueue, { draft }),
    cancel: (queueId) => ipcRenderer.invoke(channels.queueCancel, { queueId }),
    reorder: (orderedQueueIds) => ipcRenderer.invoke(channels.queueReorder, { orderedQueueIds }),
    pause: () => ipcRenderer.invoke(channels.queuePause),
    resume: () => ipcRenderer.invoke(channels.queueResume),
    onChanged: (listener) => subscribe(channels.queueChanged, listener),
  },
  jobs: {
    list: () => ipcRenderer.invoke(channels.jobsList),
    get: (jobId) => ipcRenderer.invoke(channels.jobsGet, { jobId }),
    delete: (jobId) => ipcRenderer.invoke(channels.jobsDelete, { jobId }),
    resumeDraft: (jobId) => ipcRenderer.invoke(channels.jobsResumeDraft, { jobId }),
  },
  resources: {
    status: () => ipcRenderer.invoke(channels.resourcesStatus),
    refresh: () => ipcRenderer.invoke(channels.resourcesRefresh),
  },
  settings: {
    get: () => ipcRenderer.invoke(channels.settingsGet),
    update: (patch) => ipcRenderer.invoke(channels.settingsUpdate, { patch }),
  },
  credentials: {
    status: (provider) => ipcRenderer.invoke(channels.credentialsStatus, { provider }),
    set: (provider, secret) => ipcRenderer.invoke(channels.credentialsSet, { provider, secret }),
    delete: (provider) => ipcRenderer.invoke(channels.credentialsDelete, { provider }),
  },
  artifacts: {
    reveal: (jobId, artifactId) => ipcRenderer.invoke(channels.artifactReveal, { jobId, artifactId }),
    open: (jobId, artifactId) => ipcRenderer.invoke(channels.artifactOpen, { jobId, artifactId }),
  },
};

contextBridge.exposeInMainWorld("openlrc", bridge);
