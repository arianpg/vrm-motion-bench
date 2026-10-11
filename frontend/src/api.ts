// オーケストレーターの API 型と呼び出し

export type ResultStatus = 'queued' | 'video' | 'loading' | 'running' | 'converting' | 'done' | 'failed';
export type JobStatus = 'queued' | 'loading' | 'running' | 'done' | 'failed';

export interface AssetStatus {
  id: string;
  kind: string;
  ok: boolean;
  how_to: string;
  detail: string;
}

export interface InputProperty {
  type: string;
  title?: string;
  'x-input'?: 'text' | 'duration' | 'seed' | 'file';
  'x-accept'?: 'video' | 'audio' | 'image';
}

export interface Manifest {
  id: string;
  name: string;
  version: string;
  device: 'gpu' | 'cpu';
  kind: 'motion' | 'video';
  inputs: { type: 'object'; required?: string[]; properties: Record<string, InputProperty> };
  duration: { min: number; max: number; step: number; auto_supported: boolean } | null;
  defaults: Record<string, unknown>;
  prompt: { llm_assist: boolean; guide: string; rules: Record<string, unknown> };
  output: { skeleton: string; fps: number; foot_contacts: boolean };
  resources: { vram_gb: number; load_time_s: number | null };
  license_note: string;
  asset_status: AssetStatus[];
}

export interface Result {
  id: string;
  group_id: string | null;
  adapter_id: string;
  adapter_version: string | null;
  request_text: string;
  llm_model: string | null;
  llm_prompt: string | null;
  final_prompt: string | null;
  inputs: Record<string, unknown>;
  params: Record<string, unknown>;
  seed: number | null;
  status: ResultStatus;
  error: string | null;
  load_seconds: number | null;
  generate_seconds: number | null;
  convert_seconds: number | null;
  duration: number | null;
  fps: number | null;
  files: Record<string, string>;
  created_at: string;
  finished_at: string | null;
}

export interface Job {
  id: string;
  type: 'llm_prompt' | 'video' | 'motion' | 'convert' | 'group_preview';
  resource: 'gpu' | 'cpu';
  model_key: string | null;
  result_id: string | null;
  group_id: string | null;
  status: JobStatus;
  progress: number | null;
  error: string | null;
  created: number;
  started: number | null;
  finished: number | null;
}

export interface Draft {
  id: string;
  group_id: string;
  adapter_id: string;
  llm_model: string;
  request_text: string;
  status: JobStatus;
  prompt: string | null;
  duration: number | null;
  notes: string | null;
  violations: string[];
  attempts: number;
  reasoning: string | null;
  llm_seconds: number | null;
  load_seconds: number | null;
  error: string | null;
  result_id: string | null;
  created_at: string;
}

export interface LlmModel {
  id: string;
  name: string;
  default: boolean;
  available: boolean;
  download: string;
  thinking: boolean;
  vram_gb: number;
}

export interface Group {
  id: string;
  request_text: string;
  created_at: string;
  preview: string | null;
  results: Result[];
  drafts: Draft[];
}

export interface ConfirmItem {
  adapter_id: string;
  prompt: string;
  duration: number | null;
  seed: number | null;
  draft_id: string | null;
}

export interface Health {
  status: string;
  loaded_adapter: string | null;
  gpu_used_mib: number | null;
  gpu_total_mib: number | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      const detail = body.detail ?? body;
      message = typeof detail === 'string' ? detail : (detail.message ?? JSON.stringify(detail));
    } catch {
      // 本文が JSON でない
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>('/api/health'),
  adapters: () => request<Manifest[]>('/api/adapters'),
  results: (limit = 100) => request<Result[]>(`/api/results?limit=${limit}`),
  llms: () => request<LlmModel[]>('/api/llms'),
  group: (id: string) => request<Group>(`/api/groups/${id}`),
  groups: (limit = 50) => request<Omit<Group, 'results' | 'drafts'>[]>(`/api/groups?limit=${limit}`),
  createDrafts: (body: { request_text: string; adapter_ids: string[]; llm_model?: string }) =>
    request<{ group_id: string; drafts: Draft[] }>('/api/drafts', { method: 'POST', body: JSON.stringify(body) }),
  confirm: (groupId: string, items: ConfirmItem[]) =>
    request<{ group_id: string; items: { result_id: string; job_id: string }[] }>(`/api/groups/${groupId}/confirm`, {
      method: 'POST',
      body: JSON.stringify({ items }),
    }),
  upload: async (kind: 'video' | 'audio', file: File): Promise<{ upload_id: string; name: string; size: number; duration: number | null }> => {
    const form = new FormData();
    form.append('kind', kind);
    form.append('file', file);
    const res = await fetch('/api/uploads', { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status} ${res.statusText}`);
    }
    return res.json();
  },
  retry: (resultId: string) => request<{ result_id: string; job_id: string }>(`/api/results/${resultId}/retry`, { method: 'POST' }),
  generate: (body: { adapter_id: string; inputs: Record<string, unknown>; params: Record<string, unknown>; request_text?: string }) =>
    request<{ result_id: string; job_id: string }>('/api/generate', { method: 'POST', body: JSON.stringify(body) }),
};

export const fileUrl = (result: Result, kind: string): string | null => {
  const name = result.files[kind];
  return name ? `/api/results/${result.id}/files/${encodeURIComponent(name)}` : null;
};

export const STICKMAN_URL = '/api/assets/stickman.vrm';
