import { useEffect, useState } from 'react';
import { api, fileUrl, groupOrder, vrmaDownloadUrl, vrmaFileName, type Job, type Result } from '../api';

const STATUS_LABEL: Record<Result['status'], string> = {
  queued: '待機中',
  video: '動画生成中',
  loading: 'モデル読み込み中',
  running: '生成中',
  converting: '変換中',
  done: '完了',
  failed: '失敗',
};

function seconds(v: number | null) {
  return v == null ? '–' : `${v.toFixed(1)}s`;
}

interface Section {
  key: string;
  groupId: string | null;
  title: string;
  results: Result[];
}

// 比較グループ単位にまとめる（グループに属さない結果は 1 件ずつ）
function sections(results: Result[]): Section[] {
  const out: Section[] = [];
  const byGroup = new Map<string, Section>();
  for (const r of results) {
    if (r.group_id) {
      let s = byGroup.get(r.group_id);
      if (!s) {
        s = { key: r.group_id, groupId: r.group_id, title: r.request_text || '比較グループ', results: [] };
        byGroup.set(r.group_id, s);
        out.push(s);
      }
      s.results.push(r);
    } else {
      out.push({ key: r.id, groupId: null, title: '', results: [r] });
    }
  }
  return out;
}

export function ResultList({
  results,
  jobs,
  selected,
  onSelect,
  onToggle,
  onSelectGroup,
  groupPreviews,
}: {
  results: Result[];
  jobs: Job[];
  selected: string[];
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
  onSelectGroup: (ids: string[]) => void;
  groupPreviews: Record<string, number>;
}) {
  const [query, setQuery] = useState('');
  const q = query.trim().toLowerCase();
  const filtered = q
    ? results.filter((r) =>
        [r.adapter_id, r.final_prompt ?? '', r.request_text, r.llm_prompt ?? '', String(r.seed ?? '')].some((t) => t.toLowerCase().includes(q)),
      )
    : results;
  const progressOf = (r: Result) =>
    jobs.find((j) => j.result_id === r.id && j.status === 'running' && j.type === 'motion')?.progress;

  const item = (r: Result) => {
    const preview = fileUrl(r, 'preview');
    const progress = progressOf(r);
    const isSelected = selected.includes(r.id);
    return (
      <li
        key={r.id}
        className={`result ${isSelected ? 'selected' : ''} status-${r.status}`}
        onClick={() => r.status === 'done' && onSelect(r.id)}
      >
        <div className="thumb">{preview ? <img src={preview} alt="" loading="lazy" /> : <span>{STATUS_LABEL[r.status]}</span>}</div>
        <div className="meta">
          <div className="title">
            <span className="badge">{r.adapter_id}</span>
            <span className={`state state-${r.status}`}>
              {STATUS_LABEL[r.status]}
              {progress != null && r.status === 'running' ? ` ${Math.round(progress * 100)}%` : ''}
            </span>
            {r.status === 'done' && (
              <label className="check right" title="比較に追加" onClick={(e) => e.stopPropagation()}>
                <input type="checkbox" checked={isSelected} onChange={() => onToggle(r.id)} />
                比較
              </label>
            )}
          </div>
          <div className="prompt">{r.final_prompt ?? String(r.inputs.text ?? '')}</div>
          <div className="numbers">
            seed {r.seed ?? '–'} · 尺 {r.duration?.toFixed(1) ?? '–'}s · ロード {seconds(r.load_seconds)} · 生成 {seconds(r.generate_seconds)} · 変換 {seconds(r.convert_seconds)}
          </div>
          {r.files.video && (
            <div className="links" onClick={(e) => e.stopPropagation()}>
              <a href={fileUrl(r, 'video') ?? '#'} target="_blank" rel="noreferrer">入力動画</a>
              {r.files.start_image && <a href={fileUrl(r, 'start_image') ?? '#'} target="_blank" rel="noreferrer">開始画像</a>}
            </div>
          )}
          {r.files.vrma && (
            <div className="links" onClick={(e) => e.stopPropagation()}>
              <a href={vrmaDownloadUrl(r) ?? '#'} download={vrmaFileName(r)} title="VRMA ファイルを保存する">VRMA を保存</a>
            </div>
          )}
          {r.files.audio && (
            <div className="links" onClick={(e) => e.stopPropagation()}>
              <a href={fileUrl(r, 'audio') ?? '#'} target="_blank" rel="noreferrer">入力音声</a>
            </div>
          )}
          {r.error && <div className="error small">{r.error}</div>}
          {r.status === 'failed' && (
            <div>
              <button className="small" onClick={(e) => { e.stopPropagation(); api.retry(r.id).catch((err: Error) => alert(err.message)); }}>
                同じ条件で再実行
              </button>
            </div>
          )}
        </div>
      </li>
    );
  };

  return (
    <div className="panel results">
      <h2>結果</h2>
      <input className="search" type="search" placeholder="検索（プロンプト・依頼文・モデル・seed）" value={query} onChange={(e) => setQuery(e.target.value)} />
      {filtered.length === 0 && <p className="muted">{results.length === 0 ? 'まだ結果はありません' : '一致する結果はありません'}</p>}
      {sections(filtered).map((s) =>
        s.groupId ? (
          <div key={s.key} className="group">
            <div className="group-head">
              <span className="group-title">{s.title}</span>
              <button
                className="small"
                disabled={!s.results.some((r) => r.status === 'done')}
                onClick={() =>
                  // 作成順に並べる（グループプレビューの色の順と揃える）
                  onSelectGroup(
                    s.results
                      .filter((r) => r.status === 'done')
                      .sort(groupOrder)
                      .map((r) => r.id),
                  )
                }
              >
                グループを比較
              </button>
            </div>
            <GroupPreview groupId={s.groupId} version={groupPreviews[s.groupId] ?? 0} />
            <ul>{s.results.map(item)}</ul>
          </div>
        ) : (
          <ul key={s.key}>{s.results.map(item)}</ul>
        ),
      )}
    </div>
  );
}

// 重ね合わせプレビュー（2 件以上完了したグループで作られる）。まだなければ何も出さない
function GroupPreview({ groupId, version }: { groupId: string; version: number }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [version]);
  if (failed) return null;
  return (
    <img
      className="group-preview"
      src={`/api/groups/${groupId}/preview?v=${version}`}
      alt="重ね合わせプレビュー"
      onError={() => setFailed(true)}
    />
  );
}
