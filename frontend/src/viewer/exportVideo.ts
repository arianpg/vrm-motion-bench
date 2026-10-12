// 比較動画（MP4）の書き出し。ビューアを 1 コマずつ描き、上部にお題と凡例を合成して JPEG でサーバーへ送る。
// エンコード（H.264 / AAC）はサーバーの ffmpeg で行う（LAN 越しの http ではブラウザの WebCodecs が使えないため）
import type { ClipItem, Layout, VrmStage } from './stage';

export const EXPORT_FPS = 30;
export const EXPORT_MAX_SECONDS = 60;
export const EXPORT_SIZES = {
  landscape: { width: 1920, height: 1080, label: '横 1920×1080' },
  portrait: { width: 1080, height: 1920, label: '縦 1080×1920' },
} as const;
export type ExportSize = keyof typeof EXPORT_SIZES;

export interface ExportOptions {
  stage: VrmStage;
  clips: ClipItem[];
  /** お題（依頼文やプロンプト）。上部に表示する */
  title: string;
  layout: Layout;
  size: ExportSize;
  /** 入れる音声の結果 ID（音声を入れないときは null） */
  audioResultId: string | null;
  onProgress: (done: number, total: number) => void;
  signal: AbortSignal;
}

const BACKGROUND = '#1d1f23';

async function post<T>(url: string, body?: BodyInit, json = true): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    body,
    headers: json ? { 'Content-Type': 'application/json' } : { 'Content-Type': 'image/jpeg' },
  });
  if (!res.ok) {
    const detail = await res.json().then((b) => b.detail).catch(() => null);
    throw new Error(typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/** 幅に収まるように文字列を折り返す（日本語は 1 文字ずつ、英語は単語ごと）。maxLines を超えたら末尾を … にする */
function wrap(ctx: CanvasRenderingContext2D, text: string, width: number, maxLines: number): string[] {
  const tokens = text.match(/[A-Za-z0-9'".,!?:;()-]+\s*|\s+|./gu) ?? [];
  const lines: string[] = [];
  let line = '';
  for (const t of tokens) {
    if (ctx.measureText(line + t).width > width && line) {
      lines.push(line.trimEnd());
      line = t.trimStart();
    } else {
      line += t;
    }
  }
  if (line.trim()) lines.push(line.trimEnd());
  if (lines.length > maxLines) {
    const kept = lines.slice(0, maxLines);
    let last = kept[maxLines - 1];
    while (last && ctx.measureText(last + '…').width > width) last = last.slice(0, -1);
    kept[maxLines - 1] = last + '…';
    return kept;
  }
  return lines;
}

function drawChip(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string, label: string) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, y - size / 2, size, size, size * 0.25);
  ctx.fill();
  ctx.fillStyle = '#e6e8eb';
  ctx.fillText(label, x + size * 1.5, y);
  return x + size * 1.5 + ctx.measureText(label).width + size * 2;
}

/**
 * 書き出しを実行し、完成した MP4 のダウンロード URL を返す。
 * 対象はお題と結果のモーションの表示部分だけ（比較表などは含めない）。
 */
export async function exportCompareVideo(opts: ExportOptions): Promise<string> {
  const { stage, clips, layout, signal } = opts;
  const { width, height } = EXPORT_SIZES[opts.size];
  const frames = Math.max(1, Math.round(Math.min(stage.duration, EXPORT_MAX_SECONDS) * EXPORT_FPS));
  const font = getComputedStyle(document.body).fontFamily;
  const unit = Math.min(width, height) / 1080; // 文字や余白の基準（短辺 1080px で 1）

  // 上部: お題（最大 2 行）と、重ね合わせなら色の凡例。横並びではビューポートごとに名前を出す
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const pad = 36 * unit;
  const titleSize = 40 * unit;
  const labelSize = 26 * unit;
  ctx.font = `600 ${titleSize}px ${font}`;
  const titleLines = opts.title ? wrap(ctx, opts.title, width - pad * 2, 2) : [];
  const overlay = layout === 'overlay' || clips.length === 1;
  const header = pad + titleLines.length * titleSize * 1.3 + (overlay ? labelSize * 1.8 : 0) + pad * 0.6;
  const viewHeight = Math.round((height - header) / 2) * 2; // 偶数にそろえる
  const top = height - viewHeight;
  const viewports = stage.viewports(width, viewHeight, layout);

  const session = await post<{ id: string }>(
    '/api/exports',
    JSON.stringify({ width, height, frames, audio_result_id: opts.audioResultId }),
  );
  try {
    await stage.exportFrames({
      width,
      height: viewHeight,
      layout,
      fps: EXPORT_FPS,
      frames,
      signal,
      onFrame: async (view, index) => {
        ctx.fillStyle = BACKGROUND;
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(view, 0, top);
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${titleSize}px ${font}`;
        ctx.fillStyle = '#f2f3f5';
        titleLines.forEach((line, i) => ctx.fillText(line, pad, pad + titleSize * (0.6 + i * 1.3)));
        ctx.font = `${labelSize}px ${font}`;
        if (overlay) {
          let x = pad;
          const y = top - pad * 0.6 - labelSize * 0.7;
          for (const c of clips) x = drawChip(ctx, x, y, labelSize * 0.8, c.color, c.label);
        } else {
          viewports.forEach((v, i) => {
            const c = clips[i];
            if (!c) return;
            // 枠の幅に収まるように文字を縮める（5 体以上並ぶと枠が狭い）
            const room = v.width * width - 32 * unit;
            const natural = ctx.measureText(c.label).width + labelSize * 2.4;
            const size = labelSize * Math.min(1, room / natural);
            ctx.font = `${size}px ${font}`;
            const x = v.left * width + 16 * unit + size * 0.4;
            const y = top + v.top * viewHeight + 16 * unit + size * 0.75;
            const w = ctx.measureText(c.label).width + size * 2.4;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.beginPath();
            ctx.roundRect(x - size * 0.4, y - size * 0.75, w, size * 1.5, size * 0.3);
            ctx.fill();
            drawChip(ctx, x, y, size * 0.8, c.color, c.label);
            ctx.font = `${labelSize}px ${font}`;
          });
        }
        const blob = await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('コマを JPEG にできませんでした'))), 'image/jpeg', 0.92),
        );
        await post(`/api/exports/${session.id}/frames`, blob, false);
        opts.onProgress(index + 1, frames);
      },
    });
    await post(`/api/exports/${session.id}/finish`);
  } catch (e) {
    await fetch(`/api/exports/${session.id}`, { method: 'DELETE' }).catch(() => undefined);
    throw e;
  }
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '_');
  const name = `vmb_compare_${layout}_${opts.size}_${stamp}.mp4`;
  return `/api/exports/${session.id}/file?download=${encodeURIComponent(name)}`;
}
