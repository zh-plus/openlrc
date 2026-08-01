export const queryKeys = {
  backend: ["backend"] as const,
  appearance: ["appearance"] as const,
  queue: ["queue"] as const,
  jobs: ["jobs"] as const,
  job: (jobId: string) => ["jobs", jobId] as const,
  resources: ["resources"] as const,
  settings: ["settings"] as const,
};
