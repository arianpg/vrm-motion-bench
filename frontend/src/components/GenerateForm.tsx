import { useEffect, useMemo, useState } from 'react';
import { api, type Manifest } from '../api';

// マニフェストの inputs（JSON Schema）から入力欄を作る
export function GenerateForm({ adapters, onSubmitted }: { adapters: Manifest[]; onSubmitted: (resultId: string) => void }) {
  const [adapterId, setAdapterId] = useState<string>('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [files, setFiles] = useState<Record<string, File | null>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!adapterId && adapters.length) setAdapterId(adapters[0].id);
  }, [adapters, adapterId]);

  const manifest = useMemo(() => adapters.find((a) => a.id === adapterId), [adapters, adapterId]);
  const missing = manifest?.asset_status.filter((a) => !a.ok) ?? [];

  if (!manifest) return <div className="panel">アダプターがありません</div>;

  const props = Object.entries(manifest.inputs.properties);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const inputs: Record<string, unknown> = {};
    const params: Record<string, unknown> = {};
    for (const [key, prop] of props) {
      if (prop['x-input'] === 'file') {
        const f = files[key];
        if (f) {
          try {
            inputs[key] = (await api.upload(prop['x-accept'] ?? 'video', f)).upload_id;
          } catch (err) {
            setError(`${prop.title ?? key}: ${(err as Error).message}`);
            setBusy(false);
            return;
          }
        }
        continue;
      }
      const raw = values[key];
      if (raw === undefined || raw === '') continue;
      const kind = prop['x-input'];
      if (kind === 'duration') params.duration = Number(raw);
      else if (kind === 'seed') params.seed = Number(raw);
      else inputs[key] = raw;
    }
    try {
      const res = await api.generate({ adapter_id: manifest.id, inputs, params });
      onSubmitted(res.result_id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="panel form" onSubmit={submit}>
      <h2>生成</h2>
      <label>
        モデル
        <select value={adapterId} onChange={(e) => setAdapterId(e.target.value)}>
          {adapters.map((a) => (
            <option key={a.id} value={a.id}>{a.name}（{a.device.toUpperCase()}）</option>
          ))}
        </select>
      </label>
      {props.map(([key, prop]) => {
        const kind = prop['x-input'];
        const title = prop.title ?? key;
        if (kind === 'text') {
          return (
            <label key={key}>
              {title}
              <textarea
                rows={3}
                value={values[key] ?? ''}
                placeholder="A person walks forward and waves with the right hand."
                onChange={(e) => setValues({ ...values, [key]: e.target.value })}
                required={manifest.inputs.required?.includes(key)}
              />
            </label>
          );
        }
        if (kind === 'duration' && manifest.duration) {
          const d = manifest.duration;
          return (
            <label key={key}>
              {title}（{d.min}〜{d.max} 秒）
              <input
                type="number"
                min={d.min}
                max={d.max}
                step={d.step}
                placeholder={String(manifest.defaults.duration ?? '')}
                value={values[key] ?? ''}
                onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              />
            </label>
          );
        }
        if (kind === 'seed') {
          return (
            <label key={key}>
              {title}
              <input
                type="number"
                step={1}
                placeholder="空欄ならランダム"
                value={values[key] ?? ''}
                onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              />
            </label>
          );
        }
        if (kind === 'file') {
          const accept = prop['x-accept'] === 'audio' ? 'audio/*' : 'video/*';
          return (
            <label key={key}>
              {title}（{prop['x-accept'] === 'audio' ? '音声' : '動画'}。60 秒まで）
              <input
                type="file"
                accept={accept}
                required={manifest.inputs.required?.includes(key)}
                onChange={(e) => setFiles({ ...files, [key]: e.target.files?.[0] ?? null })}
              />
            </label>
          );
        }
        return <p key={key} className="muted">{title}: この入力種別（{kind ?? prop.type}）は未対応</p>;
      })}
      {missing.length > 0 && (
        <div className="notice">
          <strong>未配置のアセットがあります</strong>
          <ul>{missing.map((a) => <li key={a.id}>{a.id}: {a.detail} — {a.how_to}</li>)}</ul>
        </div>
      )}
      {manifest.prompt.guide && (
        <details>
          <summary>プロンプトの書き方</summary>
          <pre className="guide">{manifest.prompt.guide}</pre>
        </details>
      )}
      {error && <div className="error">{error}</div>}
      <button type="submit" className="primary" disabled={busy || missing.length > 0}>
        {busy ? '投入中…' : 'キューに投入'}
      </button>
    </form>
  );
}
