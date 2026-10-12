import { useEffect, useRef, useState } from 'react';
import { STICKMAN_URL } from '../api';
import { EXPORT_MAX_SECONDS, EXPORT_SIZES, exportCompareVideo, type ExportSize } from '../viewer/exportVideo';
import { VrmStage, type ClipItem, type Layout } from '../viewer/stage';

const SPEEDS = [0.25, 0.5, 1, 2];

export function Viewer({
  clips,
  modelUrl,
  onModelChange,
  defaultModelUrl = STICKMAN_URL,
  title = '',
  exportRequest = 0,
}: {
  clips: ClipItem[];
  modelUrl?: string | null;
  onModelChange?: (url: string | null) => void;
  /** 既定の VRM（アプリでは API の棒人間、レポートでは埋め込みの棒人間） */
  defaultModelUrl?: string;
  /** 比較動画の上部に出すお題 */
  title?: string;
  /** 増えるたびに比較動画（MP4）の書き出し画面を開く（ボタンは比較バーにある。サーバーでエンコードするため、アプリだけ） */
  exportRequest?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<VrmStage>();
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [follow, setFollow] = useState(true);
  const [contacts, setContacts] = useState(true);
  const [hasContacts, setHasContacts] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const [muted, setMuted] = useState(false);
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [layout, setLayout] = useState<Layout>('side');
  const [inPlace, setInPlace] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportLayout, setExportLayout] = useState<Layout>('side');
  const [exportSize, setExportSize] = useState<ExportSize>('landscape');
  const [exportAudio, setExportAudio] = useState(true);
  const [exportProgress, setExportProgress] = useState<{ done: number; total: number } | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const exportAbort = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewports, setViewports] = useState<ReturnType<VrmStage['viewports']>>([]);

  useEffect(() => {
    const s = new VrmStage(host.current!, defaultModelUrl);
    s.onTime = (t, d) => {
      setTime(t);
      setDuration(d);
    };
    s.onAudioBlocked = setAudioBlocked;
    stage.current = s;
    if (import.meta.env.DEV) (window as unknown as { __vmbStage: VrmStage }).__vmbStage = s;
    return () => s.dispose();
  }, []);

  const clipKey = clips.map((c) => `${c.vrmaUrl}#${c.modelUrl ?? ''}#${c.contactsUrl ?? ''}#${c.audioUrl ?? ''}`).join('|');
  useEffect(() => {
    const s = stage.current;
    if (!s) return;
    s.setModelUrl(modelUrl ?? defaultModelUrl);
    const reloading = s.inPlace !== inPlace && s.count > 0;
    s.inPlace = inPlace;
    setError(null);
    setLoading(true);
    s.setClips(clips, reloading)
      .then(() => {
        setViewports(s.viewports());
        setHasContacts(s.hasContacts);
        setHasAudio(s.hasAudio);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipKey, modelUrl, inPlace]);

  useEffect(() => {
    const s = stage.current;
    if (!s) return;
    s.setLayout(layout);
    setViewports(s.viewports());
  }, [layout]);

  const toggle = () => {
    const s = stage.current!;
    s.playing = !s.playing;
    if (s.playing) s.resumeAudio(); // クリックの中で再生を始めると、自動再生を止められていても音が出る
    setPlaying(s.playing);
  };

  const loadModel = (file: File | undefined) => {
    if (file && onModelChange) onModelChange(URL.createObjectURL(file));
  };

  const multi = clips.length > 1;

  useEffect(() => {
    if (exportRequest === 0) return;
    setExportLayout(layout);
    setExportError(null);
    setExportOpen(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exportRequest]);

  const startExport = async () => {
    const s = stage.current;
    if (!s) return;
    const abort = new AbortController();
    exportAbort.current = abort;
    setExportError(null);
    setExportProgress({ done: 0, total: 1 });
    s.playing = false;
    setPlaying(false);
    try {
      const url = await exportCompareVideo({
        stage: s,
        clips,
        title,
        layout: multi ? exportLayout : 'overlay',
        size: exportSize,
        audioResultId: exportAudio ? (clips.find((c) => c.audioUrl)?.id ?? null) : null,
        onProgress: (done, total) => setExportProgress({ done, total }),
        signal: abort.signal,
      });
      const a = document.createElement('a');
      a.href = url;
      a.click();
      setExportOpen(false);
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setExportError((e as Error).message);
    } finally {
      exportAbort.current = null;
      setExportProgress(null);
    }
  };

  return (
    <div className="viewer">
      <div
        className="viewer-canvas"
        ref={host}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          loadModel(e.dataTransfer.files[0]);
        }}
      >
        {clips.length === 0 && <div className="viewer-empty">結果を選ぶとここで再生します</div>}
        {loading && <div className="viewer-empty">読み込み中…</div>}
        {error && <div className="viewer-error">{error}</div>}
        {exportOpen && (
          <div className="export-dialog">
            <h3>比較動画（MP4）の書き出し</h3>
            <p className="muted small">
              お題とモーションの表示部分を 30fps で書き出します（一番長いクリップ 1 周、{EXPORT_MAX_SECONDS} 秒まで）。カメラの向きは今の表示と同じです。
            </p>
            {multi && (
              <div className="segmented" role="group" aria-label="表示形式">
                <button className={exportLayout === 'side' ? 'on' : ''} onClick={() => setExportLayout('side')} disabled={!!exportProgress}>横並び</button>
                <button className={exportLayout === 'overlay' ? 'on' : ''} onClick={() => setExportLayout('overlay')} disabled={!!exportProgress}>重ね合わせ</button>
              </div>
            )}
            <div className="segmented" role="group" aria-label="解像度">
              {(Object.keys(EXPORT_SIZES) as ExportSize[]).map((k) => (
                <button key={k} className={exportSize === k ? 'on' : ''} onClick={() => setExportSize(k)} disabled={!!exportProgress}>
                  {EXPORT_SIZES[k].label}
                </button>
              ))}
            </div>
            {hasAudio && (
              <label className="toggle">
                <input type="checkbox" checked={exportAudio} onChange={(e) => setExportAudio(e.target.checked)} disabled={!!exportProgress} />
                入力音声を入れる
              </label>
            )}
            {exportProgress && (
              <div className="export-progress">
                <progress max={exportProgress.total} value={exportProgress.done} />
                <span className="small">{exportProgress.done} / {exportProgress.total} コマ</span>
              </div>
            )}
            {exportError && <div className="error small">{exportError}</div>}
            <div className="export-actions">
              {exportProgress ? (
                <button onClick={() => exportAbort.current?.abort()}>中止</button>
              ) : (
                <>
                  <button className="primary" onClick={startExport}>書き出す</button>
                  <button onClick={() => setExportOpen(false)}>閉じる</button>
                </>
              )}
            </div>
          </div>
        )}
        {multi && layout === 'side' &&
          viewports.map((v, i) => clips[i] && (
            <div
              key={clips[i].id}
              className="vp-label"
              style={{ left: `${v.left * 100}%`, top: `${v.top * 100}%`, width: `${v.width * 100}%`, height: `${v.height * 100}%` }}
            >
              <span><i style={{ background: clips[i].color }} />{clips[i].label}</span>
            </div>
          ))}
        {multi && layout === 'overlay' && (
          <div className="legend">
            {clips.map((c) => <span key={c.id}><i style={{ background: c.color }} />{c.label}</span>)}
          </div>
        )}
      </div>
      <div className="timeline">
        <button onClick={toggle} title="再生／一時停止">{playing ? '❚❚' : '▶'}</button>
        <button onClick={() => { stage.current?.step(-1); setPlaying(false); }} title="1 フレーム戻る">◀︎|</button>
        <button onClick={() => { stage.current?.step(1); setPlaying(false); }} title="1 フレーム進む">|▶︎</button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={1 / 30}
          value={time}
          onChange={(e) => stage.current?.seek(Number(e.target.value))}
        />
        <span className="time">{time.toFixed(2)} / {duration.toFixed(2)} s</span>
        <select
          value={speed}
          onChange={(e) => {
            const v = Number(e.target.value);
            setSpeed(v);
            if (stage.current) stage.current.speed = v;
          }}
        >
          {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        {multi && (
          <div className="segmented" role="group" aria-label="表示形式">
            <button className={layout === 'side' ? 'on' : ''} onClick={() => setLayout('side')}>横並び</button>
            <button className={layout === 'overlay' ? 'on' : ''} onClick={() => setLayout('overlay')}>重ね合わせ</button>
          </div>
        )}
        {hasAudio && (
          <label className="toggle" title="入力音声をタイムラインに合わせて再生する（最初の音声付きクリップの音声）">
            <input type="checkbox" checked={!muted} onChange={(e) => {
              setMuted(!e.target.checked);
              if (stage.current) stage.current.muted = !e.target.checked;
            }} />
            音声
          </label>
        )}
        {hasAudio && audioBlocked && !muted && (
          <button onClick={() => stage.current?.resumeAudio()} title="ブラウザが自動再生を止めたため、クリックで音声を再生する">
            音声を再生
          </button>
        )}
        {hasContacts && (
          <label className="toggle" title="足が接地しているフレームで足元に印を出す（接地フラグがあるモデルのみ）">
            <input type="checkbox" checked={contacts} onChange={(e) => {
              setContacts(e.target.checked);
              if (stage.current) {
                stage.current.showContacts = e.target.checked;
                stage.current.seek(stage.current.time);
              }
            }} />
            接地
          </label>
        )}
        <label className="toggle" title="腰の水平移動を止めて再生する（上下の動きと向きの回転は残す）。固定カメラで映し続ける用途での見え方の確認用">
          <input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} />
          水平移動なし
        </label>
        <label className="toggle" title="カメラが腰を追う">
          <input type="checkbox" checked={follow} onChange={(e) => {
            setFollow(e.target.checked);
            if (stage.current) stage.current.follow = e.target.checked;
          }} />
          追従
        </label>
        {onModelChange && (
          <>
            <label className="model-picker" title="VRM 1.0 ファイルを読み込む（ドラッグ＆ドロップも可）">
              VRM…
              <input type="file" accept=".vrm" hidden onChange={(e) => loadModel(e.target.files?.[0])} />
            </label>
            {modelUrl && <button onClick={() => onModelChange(null)}>棒人間に戻す</button>}
          </>
        )}
      </div>
    </div>
  );
}
