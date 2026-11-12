// ブラウザ側の復号（static/holdings.js）と、保存先スクリプト（apps_script/Code.gs）の動作確認。 node tests/holdings_js_test.js <暗号化ファイル> <合言葉>
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const [, , encPath, pass] = process.argv;

(async () => {
  // 1) Python が暗号化したファイルを、ブラウザと同じ WebCrypto で復号できるか
  global.crypto = require('crypto').webcrypto; global.atob = s => Buffer.from(s, 'base64').toString('binary');
  require('../static/holdings.js'); require('../static/refresh.js');
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
  const gh = { calls: [], handler: () => ({ getResponseCode: () => 204 }) };
  const sandbox = {
    JSON, Math, Number, String, Date, console,
    ContentService: { createTextOutput: t => ({ text: t, setMimeType() { return this; } }), MimeType: { JSON: 'json' } },
    SpreadsheetApp: { getActiveSpreadsheet: () => ({ getSheetByName: () => sheet, insertSheet: () => sheet }) },
    PropertiesService: { getScriptProperties: () => ({ getProperty: k => props[k] || null }) },
    CacheService: { getScriptCache: () => ({ get: k => cache[k] || null, put: (k, v) => { cache[k] = v; }, remove: k => { delete cache[k]; } }) },
    LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
    UrlFetchApp: { fetch: (url, opt) => { gh.calls.push({ url, opt }); return gh.handler(url, opt); } },
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
  // 手動更新（refresh）：実行中なら起動しない・連打しない・失敗を区別する・保有銘柄シートには触れない
  const rowsBefore = sheetRows.length;
  assert.strictEqual(call({ key, action: 'refresh' }).error, 'github_not_configured');
  props.GITHUB_TOKEN = 'ghp_dummy'; props.GITHUB_REPO = 'owner/repo';
  const runs = list => ({ getResponseCode: () => 200, getContentText: () => JSON.stringify({ workflow_runs: list }) });
  const isDispatch = u => u.endsWith('/dispatches');
  gh.handler = (u) => isDispatch(u) ? { getResponseCode: () => 204 } : runs([{ status: 'completed', conclusion: 'success', created_at: '2026-11-12T00:00:00Z' }]);
  let r1 = call({ key, action: 'refresh' });
  assert.strictEqual(r1.state, 'started'); assert(r1.requested_at);
  assert.strictEqual(gh.calls.filter(c => isDispatch(c.url)).length, 1);
  assert(gh.calls.find(c => isDispatch(c.url)).url.includes('/repos/owner/repo/actions/workflows/daily.yml/dispatches'));
  assert.strictEqual(call({ key, action: 'refresh' }).state, 'too_soon', '連打で二重起動');
  assert.strictEqual(gh.calls.filter(c => isDispatch(c.url)).length, 1);
  delete cache.refresh_last;
  gh.handler = (u) => isDispatch(u) ? { getResponseCode: () => 204 } : runs([{ status: 'in_progress', created_at: '2026-11-12T22:10:00Z', html_url: 'u' }]);
  const r2 = call({ key, action: 'refresh' });
  assert.strictEqual(r2.state, 'already_running', '実行中なのに起動しようとした');
  assert.strictEqual(gh.calls.filter(c => isDispatch(c.url)).length, 1, '実行中に二重起動');
  for (const st of ['queued', 'waiting', 'pending', 'requested']) {
    gh.handler = (u) => runs([{ status: st, created_at: '2026-11-12T22:10:00Z' }]);
    assert.strictEqual(call({ key, action: 'refresh' }).state, 'already_running', st);
  }
  gh.handler = (u) => isDispatch(u) ? { getResponseCode: () => 403 } : runs([]);
  assert.strictEqual(call({ key, action: 'refresh' }).error, 'dispatch_failed');
  gh.handler = () => ({ getResponseCode: () => 500, getContentText: () => '' });
  assert.strictEqual(call({ key, action: 'refresh' }).error, 'github_error');
  assert.strictEqual(call({ key: 'wrong-wrong', action: 'refresh' }).error, 'unauthorized', '合言葉なしで更新できた');
  assert.strictEqual(sheetRows.length, rowsBefore, 'refreshが保有銘柄シートを変更した');
  delete props.GITHUB_TOKEN; delete props.GITHUB_REPO; delete cache.refresh_last;
  assert.strictEqual(call({ key: 'wrong-wrong', action: 'list' }).error, 'unauthorized');
  assert.strictEqual(call({ action: 'list' }).error, 'unauthorized', '合言葉なしで通った');
  for (let i = 0; i < 12; i++) call({ key: 'bad' + i, action: 'list' });
  assert.strictEqual(call({ key, action: 'list' }).error, 'locked', '連続失敗でロックされない');
  props.PASSPHRASE = 'short'; for (const k in cache) delete cache[k];
  assert.strictEqual(call({ key: 'short', action: 'list' }).error, 'not_configured', '弱い合言葉を許してしまう');
  assert.strictEqual(JSON.parse(sandbox.doPost({ postData: { contents: 'not json' } }).text).error, 'bad_request');
  assert.strictEqual(JSON.parse(sandbox.doGet().text).holdings, undefined, 'doGetが保有銘柄を返している');
  // 3) ブラウザ側の状態判定（static/refresh.js）
  const R = RefreshLib; const T = s => Date.parse(s);
  const run = (status, conclusion, at) => ({ status, conclusion, created_at: at });
  assert.strictEqual(R.decideRun([], 0).state, 'none');
  assert.strictEqual(R.decideRun([], T('2026-11-12T01:00:00Z')).state, 'waiting', '押した直後はまだ実行が見えない');
  assert.strictEqual(R.decideRun([run('completed', 'success', '2026-11-12T00:00:00Z')], T('2026-11-12T01:00:00Z')).state, 'waiting', '古い成功を完了と誤認');
  assert.strictEqual(R.decideRun([run('queued', null, '2026-11-12T01:00:05Z')], T('2026-11-12T01:00:00Z')).state, 'running');
  assert.strictEqual(R.decideRun([run('in_progress', null, '2026-11-12T01:00:05Z'), run('completed', 'success', '2026-11-11T01:00:05Z')], T('2026-11-12T01:00:00Z')).state, 'running');
  assert.strictEqual(R.decideRun([run('completed', 'success', '2026-11-12T01:10:00Z')], T('2026-11-12T01:00:00Z')).state, 'success');
  assert.strictEqual(R.decideRun([run('completed', 'failure', '2026-11-12T01:10:00Z')], T('2026-11-12T01:00:00Z')).state, 'failure');
  assert.strictEqual(R.decideRun([run('completed', 'cancelled', '2026-11-12T01:10:00Z')], T('2026-11-12T01:00:00Z')).state, 'cancelled');
  assert.strictEqual(R.decideRun([run('completed', 'timed_out', '2026-11-12T01:10:00Z')], 0).state, 'failure');
  assert(R.nextDelay(10000) < R.nextDelay(600000));
  assert.strictEqual(R.elapsedText(125000), '2分05秒');
  console.log('JS OK');
})().catch(e => { console.error(e); process.exit(1); });
