import { fileUrl, vrmaDownloadUrl, vrmaFileName, type Result } from '../api';

// 比較中の結果のメタデータを並べる。行ごとに VRM を差し替えられる
export function CompareTable({
  results,
  colors,
  models,
  onModel,
  onRemove,
  vrmaUrl = vrmaDownloadUrl,
}: {
  results: Result[];
  colors: string[];
  models: Record<string, string | null>;
  onModel: (resultId: string, url: string | null) => void;
  onRemove: (resultId: string) => void;
  /** VRMA のダウンロード先（レポートでは埋め込みデータの blob URL） */
  vrmaUrl?: (result: Result) => string | null;
}) {
  if (results.length === 0) return null;
  const sec = (v: number | null) => (v == null ? '–' : `${v.toFixed(1)}s`);
  return (
    <div className="compare-table">
      <table>
        <thead>
          <tr>
            <th />
            <th>モデル</th>
            <th>最終プロンプト</th>
            <th>seed</th>
            <th>尺</th>
            <th>ロード</th>
            <th>生成</th>
            <th>LLM</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={r.id}>
              <td><i className="chip" style={{ background: colors[i] }} /></td>
              <td>{r.adapter_id}</td>
              <td className="prompt-cell">
                {r.final_prompt ?? String(r.inputs.text ?? '')}
                {r.llm_prompt && r.llm_prompt !== r.final_prompt && (
                  <div className="muted small">LLM 案: {r.llm_prompt}</div>
                )}
                {vrmaUrl(r) && (
                  <div className="small">
                    <a href={vrmaUrl(r)!} download={vrmaFileName(r)} title="VRMA ファイルを保存する">VRMA を保存</a>
                  </div>
                )}
                {r.files.video && (
                  <div className="small">
                    <a href={fileUrl(r, 'video') ?? '#'} target="_blank" rel="noreferrer">入力動画</a>
                  </div>
                )}
                {r.files.audio && (
                  <div className="small">
                    <a href={fileUrl(r, 'audio') ?? '#'} target="_blank" rel="noreferrer">入力音声</a>
                  </div>
                )}
              </td>
              <td>{r.seed ?? '–'}</td>
              <td>{r.duration?.toFixed(1) ?? '–'}s</td>
              <td>{sec(r.load_seconds)}</td>
              <td>{sec(r.generate_seconds)}</td>
              <td className="small">{r.llm_model ?? '–'}</td>
              <td className="actions">
                <label className="model-picker small" title="この結果だけ VRM を差し替える">
                  VRM…
                  <input type="file" accept=".vrm" hidden onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onModel(r.id, URL.createObjectURL(f));
                  }} />
                </label>
                {models[r.id] && <button className="small" onClick={() => onModel(r.id, null)}>戻す</button>}
                <button className="small" title="比較から外す" onClick={() => onRemove(r.id)}>×</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
