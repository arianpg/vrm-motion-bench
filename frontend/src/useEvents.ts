import { useEffect, useRef, useState } from 'react';
import { api, type Draft, type Job, type Result } from './api';

// SSE でジョブと結果の更新を受け取り、最新の状態を保つ
export function useLiveState() {
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [results, setResults] = useState<Record<string, Result>>({});
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  // グループプレビューの更新（画像のキャッシュを無効にするための版数）
  const [groupPreviews, setGroupPreviews] = useState<Record<string, number>>({});
  const [connected, setConnected] = useState(false);
  const retry = useRef<number>();

  useEffect(() => {
    let source: EventSource | null = null;
    let closed = false;

    api
      .results()
      .then((list) => setResults((prev) => ({ ...Object.fromEntries(list.map((r) => [r.id, r])), ...prev })))
      .catch(() => undefined);

    const connect = () => {
      source = new EventSource('/api/events');
      source.addEventListener('open', () => setConnected(true));
      source.addEventListener('error', () => {
        setConnected(false);
        source?.close();
        if (!closed) retry.current = window.setTimeout(connect, 2000);
      });
      source.addEventListener('snapshot', (e) => {
        const { jobs: list } = JSON.parse((e as MessageEvent).data) as { jobs: Job[] };
        setJobs(Object.fromEntries(list.map((j) => [j.id, j])));
      });
      source.addEventListener('job', (e) => {
        const job = JSON.parse((e as MessageEvent).data) as Job;
        setJobs((prev) => ({ ...prev, [job.id]: job }));
      });
      source.addEventListener('draft', (e) => {
        const draft = JSON.parse((e as MessageEvent).data) as Draft;
        setDrafts((prev) => ({ ...prev, [draft.id]: draft }));
      });
      source.addEventListener('group', (e) => {
        const group = JSON.parse((e as MessageEvent).data) as { id: string };
        setGroupPreviews((prev) => ({ ...prev, [group.id]: Date.now() }));
      });
      source.addEventListener('result', (e) => {
        const result = JSON.parse((e as MessageEvent).data) as Result;
        setResults((prev) => ({ ...prev, [result.id]: result }));
      });
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry.current);
      source?.close();
    };
  }, []);

  return { jobs, results, drafts, groupPreviews, connected };
}
