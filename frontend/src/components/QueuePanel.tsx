import type { Health, Job } from '../api';

const TYPE_LABEL: Record<Job['type'], string> = {
  llm_prompt: 'プロンプト生成',
  video: '動画生成',
  motion: 'モーション生成',
  convert: 'VRMA変換',
  group_preview: 'グループプレビュー',
};

const STATUS_LABEL: Record<Job['status'], string> = {
  queued: '待機中',
  loading: 'モデル読み込み中',
  running: '実行中',
  done: '完了',
  failed: '失敗',
};

export function QueuePanel({ jobs, health, connected }: { jobs: Job[]; health: Health | null; connected: boolean }) {
  const active = jobs.filter((j) => j.status !== 'done' && j.status !== 'failed');
  const gpu = health?.gpu_used_mib != null && health.gpu_total_mib ? `${(health.gpu_used_mib / 1024).toFixed(1)} / ${(health.gpu_total_mib / 1024).toFixed(1)} GB` : '–';
  return (
    <div className="panel queue">
      <h2>
        キュー <span className={`dot ${connected ? 'on' : 'off'}`} title={connected ? '接続中' : '再接続中'} />
      </h2>
      <div className="numbers">ロード中のモデル: {health?.loaded_adapter ?? 'なし'} · GPU メモリ {gpu}</div>
      {active.length === 0 && <p className="muted">実行中のジョブはありません</p>}
      <ul>
        {active.map((j) => (
          <li key={j.id} className={`job job-${j.status}`}>
            <span className="badge">{j.resource.toUpperCase()}</span>
            <span>{TYPE_LABEL[j.type]}</span>
            {j.model_key && <span className="muted">{j.model_key}</span>}
            <span className="state">{STATUS_LABEL[j.status]}</span>
            {j.status === 'running' && j.progress != null && (
              <progress max={1} value={j.progress} />
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
