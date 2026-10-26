// 情感模块：每个 QQ 会话（群/私聊）一份会自然演化、会回归基线的心情状态。
//
// 定位：与记忆模块并列，都在桥接侧维护、随每条消息注入（见 bridge.js 的 withXxxContext）。
// 心情由「用户消息 + AI 回复」按规则式情感词典驱动（快速、免费、确定），
// 长期倾向走更慢的 EMA，给角色扮演一层「越聊越像同一个人」的连续感。
//
// 设计取舍（v1）：
//   - 情感按**会话**（QQ key）维护，不是全局一份——同一个角色在不同群里可以有不同的情绪走向。
//   - 检测是**规则式**的：一份小情感词典 + 简单否定处理。不做模型调用，不引入额外 token 成本。
//     词典是有意保持小、可读、可改的，且明确标注为启发式（不是精确情感分析）。
//   - 心情随时间向 0 回归（半衰期），没有输入就慢慢平静下来；情绪强度另算一个「活跃度」。

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export const EMOTION_HISTORY_MAX = 20;
const EMOTION_TEXT_MAX = 1000;

// 情感词典：[词, 极性(+1/-1), 权重]。刻意保持小而透明；命中词前若有「不/没/别/莫」则翻转极性。
const SENTIMENT_LEX = [
  // 正面
  ['开心', 1, 1], ['高兴', 1, 1], ['快乐', 1, 1], ['幸福', 1, 1], ['舒服', 1, 0.8], ['满意', 1, 0.8],
  ['惊喜', 1, 0.9], ['期待', 1, 0.7], ['激动', 1, 0.9], ['兴奋', 1, 0.9], ['有趣', 1, 0.7], ['好玩', 1, 0.7],
  ['可爱', 1, 0.7], ['厉害', 1, 0.7], ['太棒', 1, 1], ['太好了', 1, 1], ['不错', 1, 0.5], ['妙啊', 1, 0.9], ['绝了', 1, 0.9],
  ['哈哈', 1, 0.7], ['嘻嘻', 1, 0.7], ['嘿嘿', 1, 0.6], ['笑死', 1, 0.9], ['上头', 1, 0.8], ['爱了', 1, 0.9],
  ['喜欢', 1, 0.8], ['爱', 1, 0.7], ['谢谢', 1, 0.6], ['感谢', 1, 0.6], ['加油', 1, 0.6], ['亲亲', 1, 0.7], ['抱抱', 1, 0.7],
  ['棒', 1, 0.6], ['赞', 1, 0.6], ['好耶', 1, 0.9], ['刺激', 1, 0.6],
  // 负面
  ['难过', -1, 1], ['伤心', -1, 1], ['生气', -1, 1], ['愤怒', -1, 1], ['暴躁', -1, 1], ['烦', -1, 0.8], ['烦死', -1, 1],
  ['讨厌', -1, 1], ['恨', -1, 0.9], ['哭', -1, 0.8], ['想哭', -1, 1], ['累', -1, 0.6], ['病', -1, 0.7], ['痛', -1, 0.8],
  ['委屈', -1, 1], ['气死', -1, 1], ['气人', -1, 0.9], ['郁闷', -1, 1], ['焦虑', -1, 1], ['害怕', -1, 0.9], ['绝望', -1, 1],
  ['无语', -1, 0.7], ['心累', -1, 0.9], ['崩溃', -1, 1], ['恶心', -1, 0.9], ['失望', -1, 0.9], ['孤单', -1, 0.9], ['寂寞', -1, 0.9],
  ['难受', -1, 0.9], ['糟糕', -1, 0.8], ['惨', -1, 0.8], ['倒霉', -1, 0.8], ['滚', -1, 0.9], ['垃圾', -1, 0.9], ['别惹', -1, 0.9],
];
const NEGATION_RE = /[不没别莫]/;

function nowIso() {
  return new Date().toISOString();
}

/** 默认情感状态。 */
export function defaultEmotionState() {
  return {
    mood: 0,          // 心情 -1..1（消极..积极）
    arousal: 0,       // 活跃度 0..1（平静..激动）
    tag: '平静',       // 主导情绪标签
    tendency: 0,      // 长期倾向（更慢的 EMA）
    updatedAt: nowIso(),
    history: [],      // 最近几次的 { t, mood }，用于展示走向
  };
}

export function normalizeEmotionState(raw) {
  const e = raw && typeof raw === 'object' ? raw : {};
  const clamp = (v, lo, hi) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : 0;
  };
  return {
    mood: clamp(e.mood, -1, 1),
    arousal: clamp(e.arousal, 0, 1),
    tag: String(e.tag || labelFor(clamp(e.mood, -1, 1), clamp(e.arousal, 0, 1))),
    tendency: clamp(e.tendency, -1, 1),
    updatedAt: String(e.updatedAt || nowIso()),
    history: Array.isArray(e.history) ? e.history.slice(-EMOTION_HISTORY_MAX) : [],
  };
}

export function labelFor(mood, arousal) {
  if (mood > 0.3) return arousal > 0.5 ? '兴奋' : '开心';
  if (mood < -0.3) return arousal > 0.5 ? '生气' : '低落';
  return arousal > 0.5 ? '激动' : '平静';
}

/**
 * 规则式情感打分：返回 -1..1（积极..消极）。
 * 命中词典词累加权分，命中词前带否定（不/没/别/莫）则翻转极性；最后用 tanh 压缩。
 */
export function sentiment(text) {
  const s = String(text || '').trim();
  if (!s) return 0;
  let score = 0;
  for (const [word, sign, weight] of SENTIMENT_LEX) {
    let from = 0;
    let idx;
    while ((idx = s.indexOf(word, from)) !== -1) {
      const prev = idx > 0 ? s[idx - 1] : '';
      const negated = NEGATION_RE.test(prev);
      score += (negated ? -sign : sign) * weight;
      from = idx + word.length;
    }
  }
  return Math.tanh(score / 2);
}

/**
 * 用一条消息更新情感状态。就地更新并返回该状态。
 * @param {object} state normalizeEmotionState 后的状态
 * @param {string} text 用户消息或 AI 回复
 * @param {object} opts { now, alpha, moodHalfLifeHours }
 */
export function updateEmotion(state, text, { now = Date.now(), alpha = 0.35, moodHalfLifeHours = 6 } = {}) {
  const st = state;
  const signal = sentiment(text);

  // 心情随时间向 0 回归（没有输入就慢慢平静）。
  const prevMs = st.updatedAt ? Date.parse(st.updatedAt) : NaN;
  const hours = isFinite(prevMs) ? Math.max(0, (now - prevMs) / 3600000) : 0;
  const decay = Math.pow(2, -hours / moodHalfLifeHours);

  let mood = (st.mood || 0) * decay;
  if (signal !== 0 || Math.abs(mood) > 0.001) {
    mood = mood * (1 - alpha) + signal * alpha;
  }
  mood = Math.max(-1, Math.min(1, mood));

  // 长期倾向：更慢的 EMA（几乎不随时间回退，累积「这个会话的整体气氛」）。
  st.tendency = Math.max(-1, Math.min(1, (st.tendency || 0) * 0.95 + mood * 0.05));

  // 活跃度：信号强度的 EMA。
  st.arousal = Math.max(0, Math.min(1, (st.arousal || 0) * (1 - alpha) + Math.abs(signal) * alpha));

  st.tag = labelFor(mood, st.arousal);
  st.mood = Math.round(mood * 1000) / 1000;
  st.updatedAt = new Date(now).toISOString();
  st.history.push({ t: st.updatedAt, mood: st.mood });
  if (st.history.length > EMOTION_HISTORY_MAX) st.history = st.history.slice(-EMOTION_HISTORY_MAX);
  return st;
}

/**
 * 把情感状态渲染成注入块（'' 表示状态还没激活，不注入）。
 * 用「软提示」而不是数值，避免 AI 刻意汇报情绪。
 */
export function buildEmotionContext(state) {
  if (!state) return '';
  const st = normalizeEmotionState(state);
  if (!st.history.length) return '';
  const energy = st.arousal > 0.5 ? '活跃' : '平静';
  return [
    '【此刻状态】',
    `你现在的心情是「${st.tag}」（${st.mood > 0.02 ? '偏积极' : st.mood < -0.02 ? '偏消极' : '中性'}，精力${energy}）。`,
    '这是从最近的对话里自然演化来的，用它在语气里自然带出，不要刻意汇报「我现在心情如何」，也别让心情压过正常聊天。',
  ].join('\n');
}

// ── 存储：state/emotion.json，一个 key 一份状态 ─────────────────────────────
export function loadEmotions(file, { onCorrupt } = {}) {
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return new Map();
    throw new Error(`情感库读取失败（${error?.code || 'unknown'}）：${error?.message ?? error}`);
  }
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  if (!text.trim()) return new Map();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const quarantine = `${file}.corrupt-${Date.now()}`;
    try { fs.writeFileSync(quarantine, text, { encoding: 'utf8', mode: 0o600 }); } catch {}
    const message = `情感库 JSON 解析失败，原文已另存为 ${quarantine}：${error?.message ?? error}`;
    onCorrupt?.(message);
    throw new Error(message);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    const quarantine = `${file}.corrupt-${Date.now()}`;
    try { fs.writeFileSync(quarantine, text, { encoding: 'utf8', mode: 0o600 }); } catch {}
    const message = '情感库顶层不是对象（期望 { "<key>": { … } }）';
    onCorrupt?.(message);
    throw new Error(message);
  }
  const map = new Map();
  for (const [k, v] of Object.entries(parsed)) map.set(k, normalizeEmotionState(v));
  return map;
}

export function saveEmotions(file, map) {
  const obj = {};
  for (const [k, v] of (map instanceof Map ? map : new Map())) obj[k] = normalizeEmotionState(v);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.tmp`;
  try {
    fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), { encoding: 'utf8', mode: 0o600 });
    fs.renameSync(tmp, file);
  } catch (error) {
    try { fs.unlinkSync(tmp); } catch {}
    throw error;
  }
  return obj;
}
