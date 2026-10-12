// レポート HTML（ネット接続なしで開ける比較ビュー）。データは <script id="vmb-data"> に埋め込まれる
import { StrictMode, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { Result } from '../api';
import { CompareTable } from '../components/CompareTable';
import { Viewer } from '../components/Viewer';
import '../styles.css';
import './report.css';

export interface ReportItem extends Result {
  label: string;
  color: string;
  vrma: string; // base64
  contacts: unknown | null;
}

export interface ReportData {
  title: string;
  request_text: string;
  created_at: string;
  stickman: string; // base64
  results: ReportItem[];
}

function blobUrl(base64: string, type: string): string {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type }));
}

function readData(): ReportData | null {
  const el = document.getElementById('vmb-data');
  const text = el?.textContent?.trim();
  if (!text || text === '__VMB_REPORT_DATA__') return null;
  return JSON.parse(text) as ReportData;
}

function ReportApp({ data }: { data: ReportData }) {
  const stickmanUrl = useMemo(() => blobUrl(data.stickman, 'model/gltf-binary'), [data]);
  const vrmaUrls = useMemo(() => data.results.map((r) => blobUrl(r.vrma, 'model/gltf-binary')), [data]);
  const contactsUrls = useMemo(
    () => data.results.map((r) => (r.contacts ? URL.createObjectURL(new Blob([JSON.stringify(r.contacts)], { type: 'application/json' })) : null)),
    [data],
  );
  const [visible, setVisible] = useState<string[]>(data.results.map((r) => r.id));
  const [globalModel, setGlobalModel] = useState<string | null>(null);
  const [models, setModels] = useState<Record<string, string | null>>({});

  const shown = data.results.filter((r) => visible.includes(r.id));
  const clips = shown.map((r) => ({
    id: r.id,
    vrmaUrl: vrmaUrls[data.results.indexOf(r)],
    label: r.label,
    color: r.color,
    modelUrl: models[r.id] ?? null,
    contactsUrl: contactsUrls[data.results.indexOf(r)],
  }));

  return (
    <div className="app report">
      <header>
        <h1>{data.title}</h1>
        {data.request_text && <span className="muted">依頼: {data.request_text}</span>}
        <span className="muted right">{new Date(data.created_at).toLocaleString()}</span>
      </header>
      <section className="center">
        <Viewer clips={clips} modelUrl={globalModel} onModelChange={setGlobalModel} defaultModelUrl={stickmanUrl} />
        <CompareTable
          results={shown}
          colors={shown.map((r) => r.color)}
          models={models}
          onModel={(id, url) => setModels((prev) => ({ ...prev, [id]: url }))}
          onRemove={(id) => setVisible((prev) => prev.filter((x) => x !== id))}
          vrmaUrl={(r) => vrmaUrls[data.results.findIndex((x) => x.id === r.id)] ?? null}
        />
        {visible.length < data.results.length && (
          <div className="restore">
            <button className="small" onClick={() => setVisible(data.results.map((r) => r.id))}>すべて表示に戻す</button>
          </div>
        )}
      </section>
    </div>
  );
}

const data = readData();
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {data ? <ReportApp data={data} /> : <p style={{ padding: 16 }}>レポートのデータがありません（テンプレートを直接開いています）</p>}
  </StrictMode>,
);
