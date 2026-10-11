import { useEffect, useRef, useState } from 'react';
import { STICKMAN_URL } from '../api';
import { VrmStage, type ClipItem, type Layout } from '../viewer/stage';

const SPEEDS = [0.25, 0.5, 1, 2];

export function Viewer({
  clips,
  modelUrl,
  onModelChange,
  defaultModelUrl = STICKMAN_URL,
}: {
  clips: ClipItem[];
  modelUrl?: string | null;
  onModelChange?: (url: string | null) => void;
  /** 既定の VRM（アプリでは API の棒人間、レポートでは埋め込みの棒人間） */
  defaultModelUrl?: string;
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
  const [layout, setLayout] = useState<Layout>('side');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewports, setViewports] = useState<ReturnType<VrmStage['viewports']>>([]);

  useEffect(() => {
    const s = new VrmStage(host.current!, defaultModelUrl);
    s.onTime = (t, d) => {
      setTime(t);
      setDuration(d);
    };
    stage.current = s;
    if (import.meta.env.DEV) (window as unknown as { __vmbStage: VrmStage }).__vmbStage = s;
    return () => s.dispose();
  }, []);

  const clipKey = clips.map((c) => `${c.vrmaUrl}#${c.modelUrl ?? ''}#${c.contactsUrl ?? ''}`).join('|');
  useEffect(() => {
    const s = stage.current;
    if (!s) return;
    s.setModelUrl(modelUrl ?? defaultModelUrl);
    setError(null);
    setLoading(true);
    s.setClips(clips)
      .then(() => {
        setViewports(s.viewports());
        setHasContacts(s.hasContacts);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipKey, modelUrl]);

  useEffect(() => {
    const s = stage.current;
    if (!s) return;
    s.setLayout(layout);
    setViewports(s.viewports());
  }, [layout]);

  const toggle = () => {
    const s = stage.current!;
    s.playing = !s.playing;
    setPlaying(s.playing);
  };

  const loadModel = (file: File | undefined) => {
    if (file && onModelChange) onModelChange(URL.createObjectURL(file));
  };

  const multi = clips.length > 1;

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
