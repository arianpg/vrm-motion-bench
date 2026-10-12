import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import {
  VRMAnimationLoaderPlugin,
  VRMLookAtQuaternionProxy,
  createVRMAnimationClip,
  type VRMAnimation,
} from '@pixiv/three-vrm-animation';
import { makeInPlaceVrma } from './vrma';

export interface ClipItem {
  id: string;
  vrmaUrl: string;
  label: string;
  color: string;
  /** このクリップだけ別の VRM で表示する場合 */
  modelUrl?: string | null;
  /** 接地フラグ（contacts.json）。あるモデルだけ */
  contactsUrl?: string | null;
  /** 入力音声（EMAGE など）。最初に音声を持つクリップの音声をタイムラインに合わせて再生する */
  audioUrl?: string | null;
}

interface Contacts {
  fps: number;
  bones: Record<string, string>; // VRM ボーン → "0"/"1" の列
}

export type Layout = 'side' | 'overlay';

interface Actor {
  item: ClipItem;
  vrm: VRM;
  mixer: THREE.AnimationMixer;
  action: THREE.AnimationAction;
  duration: number;
  follow: THREE.Vector3;
  contacts: Contacts | null;
  markers: { bone: string; mesh: THREE.Mesh }[];
  materials: { material: THREE.Material; color?: THREE.Color; opacity: number; transparent: boolean; depthWrite: boolean }[];
}

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

async function loadVrm(url: string): Promise<VRM> {
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) throw new Error(`${url} は VRM ではありません`);
  if (vrm.meta.metaVersion !== '1') throw new Error('VRM 0.x には対応していません（VRM 1.0 のみ）');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  vrm.scene.traverse((o) => (o.frustumCulled = false));
  if (vrm.lookAt) {
    const proxy = new VRMLookAtQuaternionProxy(vrm.lookAt);
    proxy.name = 'VRMLookAtQuaternionProxy';
    vrm.scene.add(proxy);
  }
  return vrm;
}

async function loadVrma(url: string, inPlace: boolean): Promise<VRMAnimation> {
  let gltf;
  if (inPlace) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    gltf = await loader.parseAsync(makeInPlaceVrma(await res.arrayBuffer()), '');
  } else {
    gltf = await loader.loadAsync(url);
  }
  const anims = gltf.userData.vrmAnimations as VRMAnimation[] | undefined;
  if (!anims?.length) throw new Error(`${url} に VRM Animation がありません`);
  return anims[0];
}

const FOCUS_HEIGHT = 0.9;

/** 横並びの列数。1 体あたりの表示（縦長の人物を想定し、幅：高さ = 3：4 で比べる）が最も大きくなる列数を選ぶ */
function gridFor(n: number, w: number, h: number): { cols: number; rows: number } {
  let best = { cols: n, rows: 1, size: -1 };
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const size = Math.min(w / cols, ((h / rows) * 3) / 4);
    if (size > best.size + 1e-6) best = { cols, rows, size };
  }
  return best;
}

export interface ExportFrameOptions {
  width: number;
  height: number;
  layout: Layout;
  fps: number;
  frames: number;
  /** 描いたコマ（renderer の canvas）を受け取る。返り値の Promise を待ってから次のコマを描く */
  onFrame: (canvas: HTMLCanvasElement, index: number) => Promise<void>;
  signal?: AbortSignal;
}
const markerGeometry = new THREE.RingGeometry(0.05, 0.08, 24).rotateX(-Math.PI / 2);

async function loadContacts(url: string | null | undefined): Promise<Contacts | null> {
  if (!url) return null;
  const res = await fetch(url);
  return res.ok ? ((await res.json()) as Contacts) : null;
}

/**
 * VRM を表示し、複数の VRMA を共通タイムラインで再生する。
 * - タイムラインの長さは一番長いクリップに合わせ、短いクリップは最後のフレームで止めて待つ
 * - 横並び: 画面を分割して 1 体ずつ描く（カメラの向きは共通）
 * - 重ね合わせ: 同じ場所に半透明・色分けで重ねる
 * - 音声があるクリップでは、再生中は音声の再生位置を時刻の基準にする（映像を音声に合わせる）
 */
export class VrmStage {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private actors: Actor[] = [];
  private raf = 0;
  private last = performance.now();
  private resizeObserver: ResizeObserver;
  private loadToken = 0;
  private overlayFollow = new THREE.Vector3();
  /** 書き出し中は通常の再生・描画を止める */
  private exporting = false;
  private audio: HTMLAudioElement | null = null;

  time = 0;
  playing = true;
  speed = 1;
  /** カメラが腰の水平位置を追う（歩くモーションでも画面から外れないように） */
  follow = true;
  layout: Layout = 'side';
  /** 接地マーカー（接地フラグがあるクリップだけ） */
  showContacts = true;
  /** 腰の水平移動を止めて再生する（setClips の読み込み時に反映する） */
  inPlace = false;
  onTime?: (time: number, duration: number) => void;
  /** 音声の自動再生がブラウザに止められたとき（ユーザー操作のあとに resumeAudio() を呼ぶ） */
  onAudioBlocked?: (blocked: boolean) => void;
  private audioBlocked = false;
  private audioMuted = false;

  constructor(private container: HTMLElement, private modelUrl: string) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setScissorTest(true);
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x1d1f23);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
    this.camera.position.set(1.6, 1.5, 3.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, FOCUS_HEIGHT, 0);
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);
    this.scene.add(new THREE.GridHelper(40, 80, 0x5a5f6a, 0x363a42));

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.loop();
  }

  get duration(): number {
    return this.actors.reduce((m, a) => Math.max(m, a.duration), 0);
  }

  get count(): number {
    return this.actors.length;
  }

  setModelUrl(url: string) {
    this.modelUrl = url;
  }

  get hasAudio(): boolean {
    return this.audio !== null;
  }

  set muted(value: boolean) {
    this.audioMuted = value;
    if (this.audio) this.audio.muted = value;
  }

  private setAudio(url: string | null) {
    if (this.audio?.dataset.src === (url ?? '')) return;
    this.audio?.pause();
    this.audio = null;
    this.setAudioBlocked(false);
    if (!url) return;
    const audio = new Audio(url);
    audio.dataset.src = url;
    audio.preload = 'auto';
    audio.muted = this.audioMuted;
    this.audio = audio;
  }

  private setAudioBlocked(blocked: boolean) {
    if (this.audioBlocked === blocked) return;
    this.audioBlocked = blocked;
    this.onAudioBlocked?.(blocked);
  }

  /** ユーザー操作（クリック）の中から呼ぶと、止められていた音声を再生できる */
  resumeAudio() {
    if (!this.audio) return;
    this.setAudioBlocked(false);
    this.syncAudio(true);
  }

  /** 音声の状態（再生・一時停止・位置・速度）をタイムラインに合わせる */
  private syncAudio(force = false) {
    const audio = this.audio;
    if (!audio) return;
    audio.playbackRate = this.speed;
    const want = this.playing && this.duration > 0 && this.time < Math.min(this.duration, audio.duration || Infinity);
    if (!want) {
      if (!audio.paused) audio.pause();
      return;
    }
    if (force || Math.abs(audio.currentTime - this.time) > 0.1) audio.currentTime = this.time;
    if (audio.paused && !this.audioBlocked) {
      audio.play().catch((e: DOMException) => {
        if (e.name === 'NotAllowedError') this.setAudioBlocked(true);
      });
    }
  }

  setLayout(layout: Layout) {
    this.layout = layout;
    this.applyAppearance();
    this.resize();
  }

  /** keepTime: 読み込み直したあとも再生位置を保つ（水平移動の切り替えなど） */
  async setClips(items: ClipItem[], keepTime = false): Promise<void> {
    const token = ++this.loadToken;
    const inPlace = this.inPlace;
    const loaded = await Promise.all(
      items.map(async (item): Promise<Actor> => {
        const [vrm, anim, contacts] = await Promise.all([
          loadVrm(item.modelUrl ?? this.modelUrl),
          loadVrma(item.vrmaUrl, inPlace),
          loadContacts(item.contactsUrl),
        ]);
        const clip = createVRMAnimationClip(anim, vrm);
        const mixer = new THREE.AnimationMixer(vrm.scene);
        const action = mixer.clipAction(clip);
        action.play();
        const materials: Actor['materials'] = [];
        vrm.scene.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh) return;
          const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          const cloned = list.map((m) => m.clone());
          mesh.material = Array.isArray(mesh.material) ? cloned : cloned[0];
          for (const m of cloned) {
            const c = (m as THREE.MeshStandardMaterial).color as THREE.Color | undefined;
            materials.push({ material: m, color: c?.clone(), opacity: m.opacity, transparent: m.transparent, depthWrite: m.depthWrite });
          }
        });
        const markers = Object.keys(contacts?.bones ?? {}).map((bone) => {
          const mesh = new THREE.Mesh(
            markerGeometry,
            new THREE.MeshBasicMaterial({ color: item.color, transparent: true, opacity: 0.9, depthWrite: false }),
          );
          mesh.visible = false;
          vrm.scene.add(mesh);
          return { bone, mesh };
        });
        return { item, vrm, mixer, action, duration: clip.duration, follow: new THREE.Vector3(), contacts, markers, materials };
      }),
    );
    if (token !== this.loadToken) {
      loaded.forEach((a) => VRMUtils.deepDispose(a.vrm.scene));
      return;
    }
    this.clearActors();
    this.actors = loaded;
    loaded.forEach((a) => this.scene.add(a.vrm.scene));
    this.setAudio(items.find((i) => i.audioUrl)?.audioUrl ?? null);
    this.applyAppearance();
    this.time = keepTime ? Math.min(this.time, this.duration) : 0;
    this.apply();
    this.snapFollow();
  }

  /** 重ね合わせでは各クリップを指定色で半透明に、横並びでは元の見た目に戻す */
  private applyAppearance() {
    const overlay = this.layout === 'overlay' && this.actors.length > 1;
    for (const a of this.actors) {
      const tint = new THREE.Color(a.item.color);
      for (const m of a.materials) {
        const mat = m.material as THREE.MeshStandardMaterial;
        if (overlay) {
          if (m.color && mat.color) mat.color.copy(m.color).lerp(tint, 0.75);
          mat.transparent = true;
          mat.opacity = 0.55;
          mat.depthWrite = false;
        } else {
          if (m.color && mat.color) mat.color.copy(m.color);
          mat.transparent = m.transparent;
          mat.opacity = m.opacity;
          mat.depthWrite = m.depthWrite;
        }
        mat.needsUpdate = true;
      }
    }
  }

  seek(time: number) {
    this.time = Math.max(0, Math.min(time, this.duration));
    if (this.audio) this.audio.currentTime = this.time;
    this.apply();
    this.snapFollow();
  }

  step(frames: number, fps = 30) {
    this.playing = false;
    this.seek(this.time + frames / fps);
  }

  get hasContacts(): boolean {
    return this.actors.some((a) => a.markers.length > 0);
  }

  private apply() {
    const p = new THREE.Vector3();
    for (const a of this.actors) {
      a.action.time = Math.min(this.time, Math.max(0, a.duration - 1e-4));
      a.mixer.update(0);
      a.vrm.update(0);
      if (a.contacts && a.markers.length) {
        const frame = Math.min(Math.round(a.action.time * a.contacts.fps), Number.MAX_SAFE_INTEGER);
        a.vrm.scene.updateMatrixWorld(true);
        for (const m of a.markers) {
          const flags = a.contacts.bones[m.bone];
          const on = this.showContacts && flags[Math.min(frame, flags.length - 1)] === '1';
          m.mesh.visible = on;
          if (on) {
            a.vrm.humanoid.getRawBoneNode(m.bone as Parameters<VRM['humanoid']['getRawBoneNode']>[0])?.getWorldPosition(p);
            a.vrm.scene.worldToLocal(p);
            m.mesh.position.set(p.x, 0.005, p.z);
          }
        }
      }
    }
    this.onTime?.(this.time, this.duration);
  }

  private hipsXZ(a: Actor, out: THREE.Vector3) {
    a.vrm.humanoid.getNormalizedBoneNode('hips')?.getWorldPosition(out);
    out.y = 0;
    return out;
  }

  private snapFollow() {
    const p = new THREE.Vector3();
    for (const a of this.actors) a.follow.copy(this.hipsXZ(a, p));
    this.overlayFollow.set(0, 0, 0);
    if (this.actors.length) {
      for (const a of this.actors) this.overlayFollow.add(a.follow);
      this.overlayFollow.divideScalar(this.actors.length);
    }
  }

  private updateFollow() {
    const p = new THREE.Vector3();
    const avg = new THREE.Vector3();
    // ループで先頭に戻ったときなど、大きく離れたら追いかけずに合わせる
    const approach = (from: THREE.Vector3, to: THREE.Vector3) =>
      from.distanceTo(to) > 1.0 ? from.copy(to) : from.lerp(to, 0.1);
    for (const a of this.actors) {
      approach(a.follow, this.hipsXZ(a, p));
      avg.add(p);
    }
    if (this.actors.length) approach(this.overlayFollow, avg.divideScalar(this.actors.length));
  }

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    if (this.exporting) return;
    const now = performance.now();
    const dt = (now - this.last) / 1000;
    this.last = now;
    const duration = this.duration;
    if (this.playing && duration > 0) {
      const audio = this.audio;
      if (audio && !audio.paused && !audio.seeking && audio.currentTime < duration) {
        this.time = audio.currentTime;
      } else {
        this.time += dt * this.speed;
      }
      if (this.time >= duration) {
        this.time %= duration;
        if (audio) audio.currentTime = this.time;
      }
      this.apply();
    }
    this.syncAudio();
    this.updateFollow();
    this.controls.update();
    this.render();
  };

  private render(w = this.container.clientWidth, h = this.container.clientHeight) {
    if (w === 0 || h === 0) return;
    const base = this.camera.position.clone();
    const target = this.controls.target.clone();
    const offset = base.clone().sub(target);
    // controls.target の原点（0, FOCUS_HEIGHT, 0）からのずれ（パン操作）を追従点に足す
    const pan = target.clone().sub(new THREE.Vector3(0, FOCUS_HEIGHT, 0));

    const drawFrom = (focus: THREE.Vector3 | null) => {
      const p = (focus && this.follow ? new THREE.Vector3(focus.x, FOCUS_HEIGHT, focus.z) : new THREE.Vector3(0, FOCUS_HEIGHT, 0)).add(pan);
      this.camera.position.copy(p).add(offset);
      this.camera.lookAt(p);
      this.renderer.render(this.scene, this.camera);
    };

    const side = this.layout === 'side' && this.actors.length > 1;
    if (!side) {
      this.actors.forEach((a) => (a.vrm.scene.visible = true));
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.setScissor(0, 0, w, h);
      drawFrom(this.actors.length ? this.overlayFollow : null);
    } else {
      const { cols, rows } = gridFor(this.actors.length, w, h);
      const cw = w / cols;
      const ch = h / rows;
      this.camera.aspect = cw / ch;
      this.camera.updateProjectionMatrix();
      this.actors.forEach((a, i) => {
        this.actors.forEach((b) => (b.vrm.scene.visible = b === a));
        const x = (i % cols) * cw;
        const y = h - (Math.floor(i / cols) + 1) * ch;
        this.renderer.setViewport(x, y, cw, ch);
        this.renderer.setScissor(x, y, cw, ch);
        drawFrom(a.follow);
      });
      this.actors.forEach((a) => (a.vrm.scene.visible = true));
    }
    this.camera.position.copy(base);
    this.camera.lookAt(target);
  }

  /** 横並び時の各ビューポートの位置（0〜1、左上原点）。ラベル表示用。w・h を省くと画面の大きさを使う */
  viewports(
    w = this.container.clientWidth,
    h = this.container.clientHeight,
    layout: Layout = this.layout,
  ): { left: number; top: number; width: number; height: number }[] {
    const n = this.actors.length;
    if (!(layout === 'side' && n > 1)) return [{ left: 0, top: 0, width: 1, height: 1 }];
    const { cols, rows } = gridFor(n, w, h);
    return this.actors.map((_, i) => ({
      left: (i % cols) / cols,
      top: Math.floor(i / cols) / rows,
      width: 1 / cols,
      height: 1 / rows,
    }));
  }

  /**
   * 比較動画の書き出し用に、指定の大きさ・表示形式で 0 秒から 1 コマずつ描く（実時間ではなく fps ごとの時刻）。
   * カメラの向きは画面と同じ。描き終えたら（中断しても）画面の表示に戻す。
   */
  async exportFrames(opts: ExportFrameOptions): Promise<void> {
    const { width, height, fps, frames, onFrame, signal } = opts;
    const saved = { layout: this.layout, time: this.time, pixelRatio: this.renderer.getPixelRatio() };
    this.exporting = true;
    this.audio?.pause();
    try {
      this.layout = opts.layout;
      this.applyAppearance();
      this.renderer.setPixelRatio(1);
      this.renderer.setSize(width, height, false);
      this.time = 0;
      this.apply();
      this.snapFollow();
      for (let i = 0; i < frames; i++) {
        if (signal?.aborted) throw new DOMException('書き出しを中止しました', 'AbortError');
        this.time = Math.min(i / fps, this.duration);
        this.apply();
        // 画面の追従（毎描画で 1 割ずつ近づける）は 60fps 前提なので、30fps では 2 回分進める
        for (let k = 0; k < Math.max(1, Math.round(60 / fps)); k++) this.updateFollow();
        this.render(width, height);
        await onFrame(this.renderer.domElement, i);
      }
    } finally {
      this.layout = saved.layout;
      this.applyAppearance();
      this.renderer.setPixelRatio(saved.pixelRatio);
      this.resize();
      this.time = saved.time;
      this.apply();
      this.snapFollow();
      this.exporting = false;
      this.last = performance.now();
    }
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private clearActors() {
    for (const a of this.actors) {
      this.scene.remove(a.vrm.scene);
      VRMUtils.deepDispose(a.vrm.scene);
    }
    this.actors = [];
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.setAudio(null);
    this.resizeObserver.disconnect();
    this.clearActors();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
