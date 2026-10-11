import { useEffect, useMemo, useState } from 'react';
import { api, type Draft, type LlmModel, type Manifest } from '../api';

const STATUS_LABEL: Record<Draft['status'], string> = {
  queued: '待機中',
  loading: 'LLM 読み込み中',
  running: '生成中',
  done: '完了',
  failed: '失敗',
};

interface Edit {
  include: boolean;
  prompt: string;
  duration: string;
  seed: string;
}

/**
 * 依頼文（日本語）から、選んだアダプターごとに LLM でプロンプト案を作り、確認・手直しして一括生成する。
 */
export function BatchPanel({
  adapters,
  drafts,
  onConfirmed,
}: {
  adapters: Manifest[];
  drafts: Record<string, Draft>;
  onConfirmed: (groupId: string) => void;
}) {
  const textAdapters = useMemo(() => adapters.filter((a) => a.prompt.llm_assist), [adapters]);
  const [llms, setLlms] = useState<LlmModel[]>([]);
  const [llm, setLlm] = useState('');
  const [request, setRequest] = useState('');
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [groupId, setGroupId] = useState<string | null>(null);
  const [draftIds, setDraftIds] = useState<string[]>([]);
  const [edits, setEdits] = useState<Record<string, Edit>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  useEffect(() => {
    api.llms().then((list) => {
      setLlms(list);
      setLlm((cur) => cur || list.find((m) => m.default)?.id || list[0]?.id || '');
    }).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    setSelected((prev) => {
      const next = { ...prev };
      for (const a of textAdapters) if (!(a.id in next)) next[a.id] = true;
      return next;
    });
  }, [textAdapters]);

  const current = draftIds.map((id) => drafts[id]).filter(Boolean);

  // 完了したプロンプト案を編集欄に取り込む（ユーザーの手直しは上書きしない）
  useEffect(() => {
    setEdits((prev) => {
      const next = { ...prev };
      for (const d of current) {
        if (d.status === 'done' && !next[d.id]) {
          next[d.id] = {
            include: true,
            prompt: d.prompt ?? '',
            duration: d.duration != null ? String(d.duration) : '',
            seed: '',
          };
        }
      }
      return next;
    });
  }, [current.map((d) => `${d.id}:${d.status}`).join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const llmInfo = llms.find((m) => m.id === llm);
  const chosen = textAdapters.filter((a) => selected[a.id]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.createDrafts({ request_text: request, adapter_ids: chosen.map((a) => a.id), llm_model: llm });
      setGroupId(res.group_id);
      setDraftIds(res.drafts.map((d) => d.id));
      setEdits({});
      setConfirmed(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const confirm = async () => {
    if (!groupId) return;
    setBusy(true);
    setError(null);
    try {
      const items = current
        .filter((d) => d.status === 'done' && edits[d.id]?.include)
        .map((d) => {
          const e = edits[d.id];
          return {
            adapter_id: d.adapter_id,
            prompt: e.prompt.trim(),
            duration: e.duration ? Number(e.duration) : null,
            seed: e.seed ? Number(e.seed) : null,
            draft_id: d.id,
          };
        });
      await api.confirm(groupId, items);
      setConfirmed(true);
      onConfirmed(groupId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const allSettled = current.length > 0 && current.every((d) => d.status === 'done' || d.status === 'failed');
  const includeCount = current.filter((d) => d.status === 'done' && edits[d.id]?.include).length;
  const nameOf = (id: string) => adapters.find((a) => a.id === id)?.name ?? id;

  return (
    <div className="panel form">
      <h2>依頼から一括生成</h2>
      <label>
        作りたい動き（日本語で可）
        <textarea
          rows={3}
          value={request}
          placeholder="例: 手を振りながら前に歩いて、最後に立ち止まってお辞儀する"
          onChange={(e) => setRequest(e.target.value)}
        />
      </label>
      <fieldset className="checks">
        <legend>比較するモデル</legend>
        {textAdapters.length === 0 && <span className="muted">テキスト入力のアダプターがありません</span>}
        {textAdapters.map((a) => (
          <label key={a.id} className="check">
            <input
              type="checkbox"
              checked={!!selected[a.id]}
              onChange={(e) => setSelected({ ...selected, [a.id]: e.target.checked })}
            />
            {a.name}
          </label>
        ))}
      </fieldset>
      <label>
        LLM
        <select value={llm} onChange={(e) => setLlm(e.target.value)}>
          {llms.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.available}>
              {m.name}{m.available ? '' : '（未配置）'}
            </option>
          ))}
        </select>
      </label>
      {llmInfo && !llmInfo.available && <div className="notice">モデルファイルがありません: <code>{llmInfo.download}</code></div>}
      <button
        className="primary"
        disabled={busy || !request.trim() || chosen.length === 0 || !llmInfo?.available}
        onClick={start}
      >
        プロンプトを生成
      </button>

      {current.length > 0 && (
        <div className="drafts">
          {current.map((d) => {
            const e = edits[d.id];
            return (
              <div key={d.id} className={`draft draft-${d.status}`}>
                <div className="title">
                  <span className="badge">{nameOf(d.adapter_id)}</span>
                  <span className={`state state-${d.status === 'running' ? 'running' : d.status}`}>{STATUS_LABEL[d.status]}</span>
                  {d.status === 'done' && (
                    <label className="check right">
                      <input
                        type="checkbox"
                        checked={e?.include ?? true}
                        disabled={confirmed}
                        onChange={(ev) => setEdits({ ...edits, [d.id]: { ...e, include: ev.target.checked } })}
                      />
                      生成する
                    </label>
                  )}
                </div>
                {d.error && <div className="error small">{d.error}</div>}
                {d.status === 'done' && e && (
                  <>
                    <textarea
                      rows={3}
                      value={e.prompt}
                      disabled={confirmed}
                      onChange={(ev) => setEdits({ ...edits, [d.id]: { ...e, prompt: ev.target.value } })}
                    />
                    <div className="row">
                      <label>
                        尺（秒）
                        <input
                          type="number"
                          step={0.1}
                          value={e.duration}
                          disabled={confirmed}
                          onChange={(ev) => setEdits({ ...edits, [d.id]: { ...e, duration: ev.target.value } })}
                        />
                      </label>
                      <label>
                        seed
                        <input
                          type="number"
                          placeholder="ランダム"
                          value={e.seed}
                          disabled={confirmed}
                          onChange={(ev) => setEdits({ ...edits, [d.id]: { ...e, seed: ev.target.value } })}
                        />
                      </label>
                    </div>
                    {d.notes && <p className="notes">{d.notes}</p>}
                    {d.violations.length > 0 && (
                      <div className="notice small">
                        ルール違反が残っています（{d.attempts} 回生成）:
                        <ul>{d.violations.map((v) => <li key={v}>{v}</li>)}</ul>
                      </div>
                    )}
                    <div className="numbers">
                      LLM {d.llm_seconds?.toFixed(1) ?? '–'}s · {d.attempts} 回 · ロード {d.load_seconds != null ? `${d.load_seconds.toFixed(1)}s` : '–'}
                    </div>
                    {d.reasoning && (
                      <details>
                        <summary>思考過程</summary>
                        <pre className="guide">{d.reasoning}</pre>
                      </details>
                    )}
                  </>
                )}
              </div>
            );
          })}
          <button className="primary" disabled={busy || confirmed || !allSettled || includeCount === 0} onClick={confirm}>
            {confirmed ? '投入済み' : `確定して生成（${includeCount} 件）`}
          </button>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
