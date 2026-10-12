// VRMA（GLB）のデータを直接書き換える処理

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a;
const FLOAT = 5126;

interface GltfJson {
  extensions?: { VRMC_vrm_animation?: { humanoid?: { humanBones?: Record<string, { node: number }> } } };
  animations?: { channels: { sampler: number; target: { node?: number; path: string } }[]; samplers: { output: number }[] }[];
  accessors?: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string }[];
  bufferViews?: { byteOffset?: number; byteStride?: number }[];
}

/**
 * 腰（hips）の水平移動（X・Z）を最初のフレームの値に固定した VRMA を作る。上下（Y）と回転はそのまま残す。
 * 固定カメラで顔を映し続けるアプリなど、水平に移動できない場面での見え方を確かめるのに使う。
 * 元のデータは変更せず、書き換えたコピーを返す。
 */
export function makeInPlaceVrma(src: ArrayBuffer): ArrayBuffer {
  const buf = src.slice(0);
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== GLB_MAGIC) throw new Error('VRMA（GLB）ではありません');
  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== CHUNK_JSON) throw new Error('GLB の JSON チャンクがありません');
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLength))) as GltfJson;
  // BIN チャンクは JSON チャンクの直後（8 バイトのヘッダーを挟む）
  const binStart = 20 + jsonLength + 8;

  const hips = json.extensions?.VRMC_vrm_animation?.humanoid?.humanBones?.hips?.node;
  if (hips == null) throw new Error('VRMA に hips がありません');
  for (const anim of json.animations ?? []) {
    for (const ch of anim.channels) {
      if (ch.target.node !== hips || ch.target.path !== 'translation') continue;
      const acc = json.accessors?.[anim.samplers[ch.sampler].output];
      if (!acc || acc.bufferView == null || acc.componentType !== FLOAT || acc.type !== 'VEC3') {
        throw new Error('hips の移動量の形式が想定と違います');
      }
      const bv = json.bufferViews![acc.bufferView];
      const stride = bv.byteStride ?? 12;
      const base = binStart + (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
      const x0 = view.getFloat32(base, true);
      const z0 = view.getFloat32(base + 8, true);
      for (let i = 1; i < acc.count; i++) {
        view.setFloat32(base + i * stride, x0, true);
        view.setFloat32(base + i * stride + 8, z0, true);
      }
    }
  }
  return buf;
}
