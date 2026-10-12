import { useEffect, useMemo, useState } from 'react';
import { api, fileUrl, groupOrder, type Health, type Manifest } from './api';
import { BatchPanel } from './components/BatchPanel';
import { CompareTable } from './components/CompareTable';
import { GenerateForm } from './components/GenerateForm';
import { QueuePanel } from './components/QueuePanel';
import { ResultList } from './components/ResultList';
import { Viewer } from './components/Viewer';
import { useLiveState } from './useEvents';

// 比較時の色（重ね合わせの色分けと凡例）
const COLORS = ['#5b9cf5', '#ef6b6b', '#4cc38a', '#e5b450', '#b07cf5', '#4cc3c3', '#f58fc8', '#a0a8b5'];

type Tab = 'batch' | 'single';

export default function App() {
  const { jobs, results, drafts, groupPreviews, connected } = useLiveState();
  const [adapters, setAdapters] = useState<Manifest[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [tab, setTab] = useState<Tab>('batch');
  const [pendingGroup, setPendingGroup] = useState<string | null>(null);
  const [globalModel, setGlobalModel] = useState<string | null>(null);
  const [models, setModels] = useState<Record<string, string | null>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 動画生成のアダプター（kind: video）は、ほかのアダプターの前段でだけ使う
    api.adapters().then((list) => setAdapters(list.filter((a) => a.kind === 'motion'))).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const tick = () => api.health().then(setHealth).catch(() => setHealth(null));
    tick();
    const id = window.setInterval(tick, 3000);
    return () => window.clearInterval(id);
  }, []);

  const resultList = useMemo(
    () => Object.values(results).sort((a, b) => b.created_at.localeCompare(a.created_at)),
    [results],
  );
  const jobList = useMemo(() => Object.values(jobs).sort((a, b) => a.created - b.created), [jobs]);

  // 一括生成したグループは、全件が終わったら自動で比較に開く
  useEffect(() => {
    if (!pendingGroup) return;
    const members = resultList.filter((r) => r.group_id === pendingGroup);
    if (members.length > 0 && members.every((r) => r.status === 'done' || r.status === 'failed')) {
      setSelected(
        members
          .filter((r) => r.status === 'done')
          .sort(groupOrder)
          .map((r) => r.id),
      );
      setPendingGroup(null);
    }
  }, [pendingGroup, resultList]);

  const compared = selected.map((id) => results[id]).filter((r) => r && r.status === 'done');
  const clips = useMemo(
    () =>
      compared.flatMap((r, i) => {
        const url = fileUrl(r, 'vrma');
        return url
          ? [
              {
                id: r.id,
                vrmaUrl: url,
                label: `${r.adapter_id} · seed ${r.seed ?? '–'}`,
                color: COLORS[i % COLORS.length],
                modelUrl: models[r.id] ?? null,
                contactsUrl: fileUrl(r, 'contacts'),
                audioUrl: fileUrl(r, 'audio'),
              },
            ]
          : [];
      }),
    [compared.map((r) => r.id).join(','), models], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const toggle = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  return (
    <div className="app">
      <header>
        <h1>VRM Motion Bench</h1>
        {error && <span className="error">{error}</span>}
      </header>
      <main>
        <aside className="left">
          <div className="tabs">
            <button className={tab === 'batch' ? 'on' : ''} onClick={() => setTab('batch')}>依頼から一括生成</button>
            <button className={tab === 'single' ? 'on' : ''} onClick={() => setTab('single')}>単体生成</button>
          </div>
          <div hidden={tab !== 'batch'}>
            <BatchPanel adapters={adapters} drafts={drafts} onConfirmed={setPendingGroup} />
          </div>
          <div hidden={tab !== 'single'}>
            <GenerateForm adapters={adapters} onSubmitted={() => undefined} onGroupSubmitted={setPendingGroup} />
          </div>
          <QueuePanel jobs={jobList} health={health} connected={connected} />
        </aside>
        <section className="center">
          <Viewer clips={clips} modelUrl={globalModel} onModelChange={setGlobalModel} />
          {compared.length > 0 && (
            <div className="compare-bar">
              <span className="muted">{compared.length} 件を比較中</span>
              <a className="button" href={`/api/report?ids=${compared.map((r) => r.id).join(',')}`} download>
                レポートHTMLを書き出す
              </a>
            </div>
          )}
          <CompareTable
            results={compared}
            colors={COLORS}
            models={models}
            onModel={(id, url) => setModels((prev) => ({ ...prev, [id]: url }))}
            onRemove={(id) => setSelected((prev) => prev.filter((x) => x !== id))}
          />
        </section>
        <aside className="right">
          <ResultList
            results={resultList}
            jobs={jobList}
            selected={selected}
            onSelect={(id) => setSelected([id])}
            onToggle={toggle}
            onSelectGroup={setSelected}
            groupPreviews={groupPreviews}
          />
        </aside>
      </main>
    </div>
  );
}
