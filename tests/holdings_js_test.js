// ブラウザ側の復号（static/holdings.js）と、保存先スクリプト（apps_script/Code.gs）の動作確認。 node tests/holdings_js_test.js <暗号化ファイル> <合言葉>
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const [, , encPath, pass] = process.argv;

(async () => {
  // 1) Python が暗号化したファイルを、ブラウザと同じ WebCrypto で復号できるか
  global.crypto = require('crypto').webcrypto; global.atob = s => Buffer.from(s, 'base64').toString('binary');
  require('../static/holdings.js');
  const blob = JSON.parse(fs.readFileSync(encPath, 'utf8'));
  const data = await HoldingsLib.decrypt(blob, pass);
  assert(data.holdings && Object.keys(data.holdings).length > 0, '復号したデータが空');
  let failed = false;
  try { await HoldingsLib.decrypt(blob, pass + 'x'); } catch (e) { failed = true; }
  assert(failed, '間違った合言葉で復号できてしまった');
  assert.strictEqual(HoldingsLib.normalizeCode('７２０３'), '7203');
  assert.strictEqual(HoldingsLib.normalizeCode('285a'), '285A');
  assert.strictEqual(HoldingsLib.normalizeCode('12345'), null);

  // 2) Code.gs（Google のサービスを模擬して実行）
  const sheetRows = [['code', 'added_at']]; const props = { PASSPHRASE: 'correct-horse-battery' }; const cache = {};
  const sheet = {
    getLastRow: () => sheetRows.length,
    getRange: (r, c, n) => ({ getValues: () => sheetRows.slice(r - 1, r - 1 + (n || 1)).map(x => [x[c - 1]]), getValue: () => sheetRows[r - 1][c - 1] }),
    appendRow: row => sheetRows.push(row), deleteRow: r => sheetRows.splice(r - 1, 1),
  };
  const sandbox = {
    JSON, Math, Number, String, Date, console,
    ContentService: { createTextOutput: t => ({ text: t, setMimeType() { return this; } }), MimeType: { JSON: 'json' } },
    SpreadsheetApp: { getActiveSpreadsheet: () => ({ getSheetByName: () => sheet, insertSheet: () => sheet }) },
    PropertiesService: { getScriptProperties: () => ({ getProperty: k => props[k] || null }) },
    CacheService: { getScriptCache: () => ({ get: k => cache[k] || null, put: (k, v) => { cache[k] = v; }, remove: k => { delete cache[k]; } }) },
    LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
    UrlFetchApp: { fetch: () => ({ getResponseCode: () => 204 }) },
  };
  vm.createContext(sandbox); vm.runInContext(fs.readFileSync(__dirname + '/../apps_script/Code.gs', 'utf8'), sandbox);
  const call = body => JSON.parse(sandbox.doPost({ postData: { contents: JSON.stringify(body) } }).text);
  const key = 'correct-horse-battery';
  assert.deepStrictEqual(call({ key, action: 'list' }), { ok: true, holdings: [] });
  assert.deepStrictEqual(call({ key, action: 'add', code: '7203' }).holdings, ['7203']);
  assert.deepStrictEqual(call({ key, action: 'add', code: '7203' }).holdings, ['7203'], '重複して追加された');
  assert.deepStrictEqual(call({ key, action: 'add', code: '285a' }).holdings, ['7203', '285A']);
  assert.strictEqual(call({ key, action: 'add', code: '12' }).error, 'bad_code');
  assert.strictEqual(call({ key, action: 'add', code: '7203; DROP' }).error, 'bad_code');
  assert.deepStrictEqual(call({ key, action: 'remove', code: '7203' }).holdings, ['285A']);
  assert.strictEqual(call({ key: 'wrong-wrong', action: 'list' }).error, 'unauthorized');
  assert.strictEqual(call({ action: 'list' }).error, 'unauthorized', '合言葉なしで通った');
  for (let i = 0; i < 12; i++) call({ key: 'bad' + i, action: 'list' });
  assert.strictEqual(call({ key, action: 'list' }).error, 'locked', '連続失敗でロックされない');
  props.PASSPHRASE = 'short'; for (const k in cache) delete cache[k];
  assert.strictEqual(call({ key: 'short', action: 'list' }).error, 'not_configured', '弱い合言葉を許してしまう');
  assert.strictEqual(JSON.parse(sandbox.doPost({ postData: { contents: 'not json' } }).text).error, 'bad_request');
  assert.strictEqual(JSON.parse(sandbox.doGet().text).holdings, undefined, 'doGetが保有銘柄を返している');
  console.log('JS OK');
})().catch(e => { console.error(e); process.exit(1); });
