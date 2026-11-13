/**
 * 保有銘柄の保存用サーバー（Google スプレッドシート ＋ Apps Script）
 *
 * ・保有銘柄は、このスプレッドシートの「holdings」シートに保存されます（Googleのサーバー上）。
 *   端末・ブラウザ・キャッシュに依存しないので、スマホ変更や故障でも消えません。
 * ・操作には「合言葉」が必要です。合言葉はスクリプトプロパティ PASSPHRASE に保存します（コードには書きません）。
 * ・GitHubのトークンはフロントエンド（Webページ）には一切置きません。
 *   （任意）保有銘柄を変更した直後にサイトを再生成したい場合だけ、スクリプトプロパティに
 *   GITHUB_TOKEN（Actions の書き込み権限だけの細かい権限のトークン）と GITHUB_REPO（例: ikea56/top10）を入れます。
 *
 * ・「ランキング再生成」ボタン（action: refresh）は合言葉なしで使えます（個人用）。同じ GITHUB_TOKEN / GITHUB_REPO を使い、
 *   すでに実行中なら二重に起動せず、2分以内の連打も防ぎます。トークンはこのスクリプトの中（Google側）にだけあります。
 *   ※保有銘柄の操作（list/add/remove）は、従来どおり合言葉が必要です。
 */
var VERSION = 3;                         // Code.gs の版（ブラウザで /exec を開いたときに表示。更新できているかの確認用）
var SHEET_NAME = 'holdings';
var MAX_HOLDINGS = 50;
var CODE_RE = /^[0-9]{3}[0-9A-Z]$/;      // 4桁の証券コード（新コード 285A 形式を含む）
var MAX_FAILS = 10;                      // 合言葉の連続失敗がこの回数に達したら15分ロック
var LOCK_SECONDS = 900;

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function safeEqual_(a, b) {              // 長さと全文字を必ず比較する（途中で打ち切らない）
  var diff = a.length ^ b.length;
  for (var i = 0; i < Math.max(a.length, b.length); i++) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return diff === 0;
}

function sheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(['code', 'added_at']);
  }
  return sh;
}

function readCodes_(sh) {
  var last = sh.getLastRow();
  if (last < 2) return [];
  return sh.getRange(2, 1, last - 1, 1).getValues()
    .map(function (r) { return String(r[0]).trim().toUpperCase(); })
    .filter(function (c) { return CODE_RE.test(c); });
}

function triggerRebuild_() {              // 任意機能：サイトの再生成を依頼（トークンはここ＝サーバー側にだけ保存）
  var p = PropertiesService.getScriptProperties();
  var token = p.getProperty('GITHUB_TOKEN'), repo = p.getProperty('GITHUB_REPO');
  if (!token || !repo) return false;
  var cache = CacheService.getScriptCache();
  if (cache.get('rebuild')) return true;  // 連続操作では1分に1回だけ依頼
  try {
    var res = UrlFetchApp.fetch('https://api.github.com/repos/' + repo + '/actions/workflows/daily.yml/dispatches', {
      method: 'post',
      contentType: 'application/json',
      headers: { Authorization: 'Bearer ' + token, Accept: 'application/vnd.github+json' },
      payload: JSON.stringify({ ref: p.getProperty('GITHUB_BRANCH') || 'main' }),
      muteHttpExceptions: true
    });
    var ok = res.getResponseCode() === 204;
    if (ok) cache.put('rebuild', '1', 60);
    return ok;
  } catch (e) {
    return false;
  }
}

function activeRuns_(repo, token, workflow) {   // 実行中・待機中の更新があるか（GitHubに問い合わせる）
  var url = 'https://api.github.com/repos/' + repo + '/actions/workflows/' + workflow + '/runs?per_page=10';
  var res = UrlFetchApp.fetch(url, {
    headers: { Authorization: 'Bearer ' + token, Accept: 'application/vnd.github+json' }, muteHttpExceptions: true
  });
  if (res.getResponseCode() !== 200) throw new Error('runs_' + res.getResponseCode());
  var ACTIVE = { queued: 1, in_progress: 1, waiting: 1, requested: 1, pending: 1 };
  return (JSON.parse(res.getContentText()).workflow_runs || []).filter(function (r) { return ACTIVE[r.status]; });
}

function refresh_() {                     // 手動更新：daily-update を起動（実行中なら起動しない）
  var p = PropertiesService.getScriptProperties();
  var token = p.getProperty('GITHUB_TOKEN'), repo = p.getProperty('GITHUB_REPO');
  var workflow = p.getProperty('GITHUB_WORKFLOW') || 'daily.yml';
  if (!token || !repo) return { ok: false, error: 'github_not_configured' };
  var cache = CacheService.getScriptCache();
  var requestedAt = new Date(Date.now() - 10000).toISOString();     // 時計のずれの余裕として10秒前
  try {
    var active = activeRuns_(repo, token, workflow);
    if (active.length) return { ok: true, state: 'already_running', requested_at: active[0].created_at, run_url: active[0].html_url };
    if (cache.get('refresh_last')) return { ok: true, state: 'too_soon' };    // 連打防止（直前の起動から2分間）
    var res = UrlFetchApp.fetch('https://api.github.com/repos/' + repo + '/actions/workflows/' + workflow + '/dispatches', {
      method: 'post', contentType: 'application/json',
      headers: { Authorization: 'Bearer ' + token, Accept: 'application/vnd.github+json' },
      payload: JSON.stringify({ ref: p.getProperty('GITHUB_BRANCH') || 'main' }), muteHttpExceptions: true
    });
    if (res.getResponseCode() !== 204) return { ok: false, error: 'dispatch_failed', status: res.getResponseCode() };
    cache.put('refresh_last', '1', 120);
    return { ok: true, state: 'started', requested_at: requestedAt };
  } catch (e) {
    return { ok: false, error: 'github_error', detail: String(e).slice(0, 120) };
  }
}

function doGet() {                        // ブラウザで直接開いたとき（動作確認用。保有銘柄などのデータは返さない）
  var p = PropertiesService.getScriptProperties();
  return out_({ ok: true, service: 'holdings', version: VERSION, refresh_ready: !!(p.getProperty('GITHUB_TOKEN') && p.getProperty('GITHUB_REPO')),
                note: 'POSTで利用します' });
}

function doPost(e) {
  var req;
  try {
    req = JSON.parse(e.postData.contents);
  } catch (err) {
    return out_({ ok: false, error: 'bad_request' });
  }
  if (String(req.action || '') === 'refresh') {      // 手動更新：合言葉は不要。保有銘柄のシートには触れない。二重起動は排他ロックでも防ぐ
    try {
      var rl = LockService.getScriptLock();
      rl.waitLock(10000);
      try { return out_(refresh_()); } finally { rl.releaseLock(); }
    } catch (err) {
      return out_({ ok: false, error: 'script_error', detail: String(err).slice(0, 120) });
    }
  }
  var props = PropertiesService.getScriptProperties();
  var secret = props.getProperty('PASSPHRASE');
  if (!secret || secret.length < 8) return out_({ ok: false, error: 'not_configured' });

  var cache = CacheService.getScriptCache();
  var fails = Number(cache.get('fails') || 0);
  if (fails >= MAX_FAILS) return out_({ ok: false, error: 'locked' });
  if (!safeEqual_(String(req.key || ''), secret)) {
    cache.put('fails', String(fails + 1), LOCK_SECONDS);
    return out_({ ok: false, error: 'unauthorized' });
  }
  cache.remove('fails');

  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var sh = sheet_();
    var codes = readCodes_(sh);
    var action = String(req.action || 'list');
    if (action === 'list') {
      return out_({ ok: true, holdings: codes });
    }
    var code = String(req.code || '').trim().toUpperCase();
    if (!CODE_RE.test(code)) return out_({ ok: false, error: 'bad_code' });

    if (action === 'add') {
      if (codes.indexOf(code) >= 0) return out_({ ok: true, holdings: codes, note: 'already' });
      if (codes.length >= MAX_HOLDINGS) return out_({ ok: false, error: 'too_many' });
      sh.appendRow([code, new Date()]);
      codes.push(code);
      return out_({ ok: true, holdings: codes, rebuild: triggerRebuild_() });
    }
    if (action === 'remove') {
      var last = sh.getLastRow();
      for (var row = last; row >= 2; row--) {
        if (String(sh.getRange(row, 1).getValue()).trim().toUpperCase() === code) sh.deleteRow(row);
      }
      codes = readCodes_(sh);
      return out_({ ok: true, holdings: codes, rebuild: triggerRebuild_() });
    }
    return out_({ ok: false, error: 'bad_action' });
  } finally {
    lock.releaseLock();
  }
}
