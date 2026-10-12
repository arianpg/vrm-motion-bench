import { useEffect, useMemo, useState } from 'react';
import { api, type Manifest } from '../api';

/** seed 違いの比較で 1 回に投入できる件数（比較の色の数） */
export const MAX_SEED_VARIANTS = 8;
const MAX_SEED = 2 ** 31 - 1;

/**
 * seed 欄と件数から、seed 違いの比較に使う seed の一覧を作る。1 件だけなら null（通常の単体生成）。
 * - seed 欄にカンマ・空白区切りで複数並べると、その一覧を使う（件数は無視）
 * - 件数が 2 以上なら、seed を指定したときはその値からの連番、空欄ならランダム
 */
export function seedVariants(raw: string, count: number): number[] | null {
  const parts = raw.split(/[\s,、]+/).filter(Boolean);
  if (parts.length > 1) {
    const seeds = parts.map(Number);
    if (seeds.some((s) => !Number.isInteger(s) || s < 0 || s > MAX_SEED)) throw new Error(`seed は 0〜${MAX_SEED} の整数にする`);
    const unique = [...new Set(seeds)];
    if (unique.length > MAX_SEED_VARIANTS) throw new Error(`seed は ${MAX_SEED_VARIANTS} 件までにする`);
    return unique;
  }
  if (count <= 1) return null;
  if (count > MAX_SEED_VARIANTS) throw new Error(`件数は ${MAX_SEED_VARIANTS} 件までにする`);
  if (parts.length === 1) {
    const start = Number(parts[0]);
    if (!Number.isInteger(start) || start < 0 || start + count - 1 > MAX_SEED) throw new Error(`seed は 0〜${MAX_SEED} の整数にする`);
    return Array.from({ length: count }, (_, i) => start + i);
  }
  const seeds = new Set<number>();
  while (seeds.size < count) seeds.add(Math.floor(Math.random() * MAX_SEED));
  return [...seeds];
}

// マニフェストの inputs（JSON Schema）から入力欄を作る
export function GenerateForm({
  adapters,
  onSubmitted,
  onGroupSubmitted,
}: {
  adapters: Manifest[];
  onSubmitted: (resultId: string) => void;
  /** seed 違いの比較をグループとして投入したとき */
  onGroupSubmitted: (groupId: string) => void;
}) {
  const [adapterId, setAdapterId] = useState<string>('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [files, setFiles] = useState<Record<string, File | null>>({});
  const [variants, setVariants] = useState(1);
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
    const seedKey = props.find(([, p]) => p['x-input'] === 'seed')?.[0];
    let seeds: number[] | null = null;
    try {
      seeds = seedKey ? seedVariants(values[seedKey] ?? '', variants) : null;
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
      return;
    }
    for (const [key, prop] of props) {
      if (prop['x-input'] === 'file') {
        const f = files[key];
        if (f) {
          try {
            inputs[key] = (await api.upload(prop['x-accept'] === 'audio' ? 'audio' : 'video', f)).upload_id;
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
      else if (kind === 'seed') {
        if (!seeds) {
          const seed = Number(raw.trim());
          if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
            setError(`seed は 0〜${MAX_SEED} の整数にする`);
            setBusy(false);
            return;
          }
          params.seed = seed;
        }
      }
      else inputs[key] = raw;
    }
    try {
      if (seeds) {
        // 同じ入力・同じアダプターで seed だけ変えて、比較グループとしてまとめて投入する
        const subject = String(inputs.text ?? '').trim() || props.map(([k]) => files[k]?.name).find(Boolean) || '';
        const res = await api.createGroup({
          request_text: `seed 比較（${manifest.name}）${subject ? `: ${subject.slice(0, 80)}` : ''}`,
          items: seeds.map((seed) => ({ adapter_id: manifest.id, inputs, params: { ...params, seed } })),
        });
        onGroupSubmitted(res.group_id);
      } else {
        const res = await api.generate({ adapter_id: manifest.id, inputs, params });
        onSubmitted(res.result_id);
      }
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
                type="text"
                inputMode="numeric"
                placeholder="空欄ならランダム。カンマ区切りで複数指定すると seed 違いで比較"
                value={values[key] ?? ''}
                onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              />
              <span className="variants" title={`seed だけ変えて生成し、比較グループにまとめる（${MAX_SEED_VARIANTS} 件まで）。seed を指定するとその値からの連番、空欄ならランダム`}>
                seed 違いで
                <input
                  type="number"
                  min={1}
                  max={MAX_SEED_VARIANTS}
                  step={1}
                  value={variants}
                  onChange={(e) => setVariants(Math.max(1, Math.min(MAX_SEED_VARIANTS, Math.floor(Number(e.target.value) || 1))))}
                />
                件生成して比較
              </span>
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
        {busy ? '投入中…' : variants > 1 && props.some(([, p]) => p['x-input'] === 'seed') ? `${variants} 件をキューに投入` : 'キューに投入'}
      </button>
    </form>
  );
}
