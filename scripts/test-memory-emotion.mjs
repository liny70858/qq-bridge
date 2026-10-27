// 记忆/情感模块离线回归（不读真实配置、不发 QQ 消息）。
// 运行：node scripts/test-memory-emotion.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  tokenize, generalRarity, freshness, recallMemories, renderMemories,
  createMemoryEntry, normalizeMemoryEntry, collapseByFactKey, normalizeFactKey,
  isGenericFactKey, resolvePolicy, fitPolicy, buildMemoryContext,
  loadMemories, saveMemories, createSpeechStats, recordSpeech,
  buildMemoryExtractionPrompt, parseMemoryExtractionJson,
} from '../src/memory.js';
import {
  sentiment, updateEmotion, buildEmotionContext, defaultEmotionState,
  normalizeEmotionState, labelFor, loadEmotions, saveEmotions,
} from '../src/emotion.js';

let pass = 0;
function ok(cond, label) {
  assert.ok(cond, label);
  pass += 1;
}

// ── 分词 ──────────────────────────────────────────────────────────────────
{
  const t = (s) => [...tokenize(s)];
  ok(t('她的胃不好').includes('胃'), '分词：单字虚词丢弃，实词保留');
  ok(!t('她的胃不好').includes('她'), '分词：她 是单字虚词应被丢弃');
  ok(t('我们那天办婚礼').includes('婚礼'), '分词：词表最大匹配出「婚礼」');
  ok(generalRarity('表示') < generalRarity('婚礼'), '通用稀有：常用词比具体词更不值钱');
}

// ── 鲜活度衰减 ────────────────────────────────────────────────────────────
{
  const pol = resolvePolicy({});
  const m = normalizeMemoryEntry({
    text: 'x', at: new Date(Date.now() - pol.baseHalfLifeDays * 86400000).toISOString(), importance: 0,
  });
  const f = freshness(m, pol, Date.now());
  ok(f > 0.4 && f < 0.6, `鲜活度：半衰期当天约 0.5（实测 ${f.toFixed(3)}）`);
  const fact = normalizeMemoryEntry({ text: 'y', category: 'stable_profile', importance: 0 });
  ok(freshness(fact, pol, Date.now()) === pol.unknownActivation, '鲜活度：长期事实平着（=unknownActivation）');
}

// ── 事实键 ────────────────────────────────────────────────────────────────
{
  ok(normalizeFactKey('她．身高') === '她.身高', '归一键：全角点归一');
  ok(isGenericFactKey('话.内容'), '万能键：话.内容 被挡');
  ok(!isGenericFactKey('她.身高'), '万能键：她.身高 放行');
}

// ── 召回 ──────────────────────────────────────────────────────────────────
{
  const pol = resolvePolicy({});
  const entries = [
    createMemoryEntry({ text: '她胃不好，不能吃辣的', at: new Date(Date.now() - 3 * 86400000).toISOString(), importance: 1, category: 'stable_profile', factKey: '她.胃' }),
    createMemoryEntry({ text: '她说她很喜欢养猫', at: new Date(Date.now() - 86400000).toISOString(), importance: 0.5, tags: ['猫', '喜欢'] }),
    createMemoryEntry({ text: '我们约好周末一起去爬山', at: new Date().toISOString(), importance: 0.5, factKey: '约定.爬山', tags: ['爬山', '约定'] }),
  ];
  const r1 = recallMemories(entries, '她胃最近怎么样', { policy: pol });
  ok(r1.selected.length === 1 && r1.selected[0].m.text.includes('胃'), '召回：问胃命中胃那条');
  const r2 = recallMemories(entries, '周末去哪玩', { policy: pol });
  ok(r2.selected.length >= 1 && r2.selected[0].m.text.includes('爬山'), '召回：问周末命中爬山约定');
  const rendered = renderMemories(r1.selected);
  ok(rendered.includes('3 天前'), '渲染：带相对时间');
  ok(!rendered.includes('时间认不出来'), '渲染：时间戳能被正确解析');

  // 事实键收敛：同一个键只留最新
  const more = [...entries, createMemoryEntry({ text: '她胃好多了', at: new Date().toISOString(), importance: 1, category: 'stable_profile', factKey: '她.胃' })];
  const { superseded } = collapseByFactKey(more, 'test');
  ok(superseded.length === 1, '事实键收敛：同键只留最新，旧的作废');
  ok(more.find((m) => m.text.includes('胃不好')).superseded, '事实键收敛：旧条目被标作废');
}

// ── 说话尺（问句门槛）────────────────────────────────────────────────────
{
  const pol = resolvePolicy({});
  const speech = createSpeechStats();
  for (let i = 0; i < 25; i++) recordSpeech(speech, '在吗 嗯 好的 收到 哈哈 谢谢 随便 都可以 你说了算 行 知道了');
  const entries = [createMemoryEntry({ text: '她护照放桌上了', at: new Date().toISOString(), importance: 0.5 })];
  const r = recallMemories(entries, '在吗', { policy: pol, speech });
  ok(r.selected.length === 0, '说话尺：太没内容的问句被门槛挡住');
  ok(r.gate && r.gate.info < pol.queryGate, '说话尺：门槛记录了信息量');
}

// ── 存储 ──────────────────────────────────────────────────────────────────
{
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qq-mem-'));
  const file = path.join(dir, 'mem.json');
  const entries = [createMemoryEntry({ text: '第一件事', importance: 0.5 })];
  saveMemories(file, entries);
  const loaded = loadMemories(file);
  ok(loaded.length === 1 && loaded[0].text === '第一件事', '存储：保存/读取往返');
  const corrupt = file + '.bad';
  fs.writeFileSync(corrupt, '{not json', 'utf8');
  assert.throws(() => loadMemories(corrupt), /JSON 解析失败/, '存储：损坏 JSON 抛错而非当空库');
  const loaded2 = loadMemories(file);
  ok(loaded2.length === 1, '存储：损坏文件不覆盖正常库');
}

// ── 情感 ──────────────────────────────────────────────────────────────────
{
  ok(sentiment('今天好开心哈哈') > 0.3, '情感：正面词得分高');
  ok(sentiment('今天不开心') < -0.3, '情感：否定翻转极性');
  ok(sentiment('好烦气死我了') < -0.3, '情感：负面词得分低');
  ok(sentiment('') === 0, '情感：空文本得 0');
  ok(labelFor(0.5, 0.6) === '兴奋' && labelFor(-0.5, 0.2) === '低落' && labelFor(0, 0.1) === '平静', '情感：标签映射');

  let st = defaultEmotionState();
  updateEmotion(st, '好开心哈哈', { now: Date.now() });
  updateEmotion(st, '太棒了', { now: Date.now() });
  ok(st.mood > 0.3, '情感：连续正面输入拉高心情');
  ok(st.tendency > 0 && st.tendency <= st.mood, '情感：长期倾向跟着心情走但更慢');
  const ctx = buildEmotionContext(st);
  ok(ctx.includes('【此刻状态】'), '情感：注入块含状态头');
  ok(buildEmotionContext(defaultEmotionState()) === '', '情感：无历史不注入');

  // 存储往返
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qq-emo-'));
  const file = path.join(dir, 'emo.json');
  const map = new Map([['group:123', st]]);
  saveEmotions(file, map);
  const loaded = loadEmotions(file);
  ok(loaded.get('group:123').mood === st.mood, '情感：保存/读取往返');
}

// ── 模型提炼 ──────────────────────────────────────────────────────────────
{
  const prompt = buildMemoryExtractionPrompt([
    { sender: '小鲸鱼', text: '我胃不好，不能吃辣的' },
    { sender: '路人', text: '今天天气不错' },
  ]);
  ok(prompt.includes('小鲸鱼') && prompt.includes('长期记忆提炼助手'), '提炼 prompt：含语料与角色');
  ok(!prompt.includes('<script>'), '提炼 prompt：不可信文本被转义');
  const parsed = parseMemoryExtractionJson('这里有个说明\n[{"text":"她胃不好","importance":1,"factKey":"她.胃","category":"stable_profile"}]');
  ok(parsed.length === 1 && parsed[0].factKey === '她.胃' && parsed[0].category === 'stable_profile', '提炼解析：JSON 数组抠出');
  ok(parseMemoryExtractionJson('不是 JSON').length === 0, '提炼解析：非 JSON 返回空');
  ok(parseMemoryExtractionJson('[{"text":""},{"text":"ok","importance":9}]').length === 1, '提炼解析：无效项被过滤、越界 importance 归一');
}

console.log(`\n✅ memory/emotion 回归通过：${pass} 项断言`);
