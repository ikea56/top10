/* 手動更新（「ランキング再生成」ボタン）。
 * ・GitHubのトークンはブラウザに持ちません。更新の起動は Apps Script（保有銘柄と同じ保存先スクリプト）に依頼し、トークンはそちらにだけあります。
 * ・更新の進み具合は、GitHubの公開情報（認証なし）を読んで表示します。
 * ・すでに実行中なら二重に起動しません（Apps Script側でも確認し、GitHub Actionsも同時に1つしか走らせません）。 */
(function (g) {
  'use strict';

  // ---------- 判定ロジック（テストからも使う） ----------
  // runs：GitHub の workflow_runs（新しい順）。sinceMs：ボタンを押した時刻（それ以降に作られた実行だけを見る）
  function decideRun(runs, sinceMs) {
    let list = Array.isArray(runs) ? runs : [];
    if (sinceMs) list = list.filter(function (r) { return Date.parse(r.created_at) >= sinceMs; });
    if (!list.length) return { state: sinceMs ? 'waiting' : 'none', run: null };
    const r = list[0];
    if (r.status !== 'completed') return { state: 'running', run: r };
    if (r.conclusion === 'success') return { state: 'success', run: r };
    if (r.conclusion === 'cancelled') return { state: 'cancelled', run: r };
    return { state: 'failure', run: r };
  }
  function nextDelay(elapsedMs) { return elapsedMs < 180000 ? 30000 : 60000; }     // GitHubの無認証APIは1時間60回までなので、だんだん間隔を空ける
  function elapsedText(ms) { const s = Math.max(0, Math.round(ms / 1000)); return Math.floor(s / 60) + '分' + String(s % 60).padStart(2, '0') + '秒'; }
  g.RefreshLib = { decideRun: decideRun, nextDelay: nextDelay, elapsedText: elapsedText };

  if (typeof document === 'undefined') return;
  const cfg = g.REFRESH_CFG || {};
  const root = document.getElementById('refresh');
  if (!root) return;
  const $ = function (id) { return document.getElementById(id); };
  const btn = $('rfBtn'), stat = $('rfStatus'), msgEl = $('rfMsg'), form = $('rfPass'), input = $('rfInput');
  const ERR = {
    unauthorized: '合言葉が違います', locked: '失敗が続いたため15分ほどロックされています',
    not_configured: 'サーバー側の合言葉が未設定です（README参照）',
    github_not_configured: '手動更新の設定がまだ完了していません（README「手動更新ボタンの設定」を確認）',
    dispatch_failed: '更新を開始できませんでした（トークンの権限を確認してください）',
    github_error: 'GitHubに接続できませんでした。少し待ってからもう一度お試しください', bad_request: '送信内容が正しくありません'
  };
  let pass = null, running = false, timer = null, startedAt = 0, sinceMs = 0, watching = false;

  function setStatus(kind, text) { stat.className = 'rf-status rf-' + kind; stat.textContent = text; }
  function say(text, kind) { msgEl.className = 'hint rf-msg ' + (kind || ''); msgEl.textContent = text || ''; }
  function lock(on) { running = on; btn.disabled = on; }

  async function fetchRuns() {
    const url = 'https://api.github.com/repos/' + cfg.repo + '/actions/workflows/' + cfg.workflow + '/runs?per_page=5';
    const r = await fetch(url, { headers: { Accept: 'application/vnd.github+json' }, cache: 'no-store' });
    if (!r.ok) throw new Error('runs ' + r.status);
    return (await r.json()).workflow_runs || [];
  }
  async function latestGenerated() {
    try {
      const r = await fetch('latest.json?t=' + Date.now(), { cache: 'no-store' });
      return r.ok ? (await r.json()).generated_at : null;
    } catch (e) { return null; }
  }

  // 完了後：サイトへの反映（GitHub Pagesの公開）を待って、新しくなったら再読み込み
  async function waitForSite() {
    setStatus('run', '更新完了 — 最新データを読み込み中...');
    for (let i = 0; i < 20; i++) {
      const gen = await latestGenerated();
      if (gen && gen !== cfg.generated) { setStatus('ok', '更新完了'); say('最新のデータに切り替えます…', 'ok'); setTimeout(function () { location.reload(); }, 800); return; }
      await new Promise(function (res) { setTimeout(res, 15000); });
    }
    setStatus('ok', '更新完了');
    say('更新は完了しました。サイトへの反映に少し時間がかかっています。しばらくしてからページを再読み込みしてください。', 'ok');
    lock(false);
  }

  async function poll() {
    clearTimeout(timer);
    let d;
    try { d = decideRun(await fetchRuns(), sinceMs); }
    catch (e) { d = { state: 'unknown', run: null }; }
    const elapsed = Date.now() - startedAt;
    if (d.state === 'running' || d.state === 'waiting' || d.state === 'unknown') {
      setStatus('run', '処理中...');
      say('ランキングを再生成しています（経過 ' + elapsedText(elapsed) + '）。完了までこの画面を開いたままにするか、後でまた開いてください。', '');
      if (elapsed > 90 * 60 * 1000) { setStatus('err', 'エラー'); say('90分たっても終わりません。GitHubのActions画面で状況を確認してください。', 'err'); lock(false); return; }
      lock(true);
      timer = setTimeout(poll, nextDelay(elapsed));
    } else if (d.state === 'success') {
      await waitForSite();
    } else {
      setStatus('err', d.state === 'cancelled' ? 'エラー（中止されました）' : 'エラー');
      say('更新に失敗しました。GitHubのActions画面で原因を確認してください。表示中のランキングは前回のものです。', 'err');
      lock(false);
    }
  }

  async function request(key) {
    let res;
    try {
      const r = await fetch(cfg.endpoint, { method: 'POST', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
        body: JSON.stringify({ key: key, action: 'refresh' }) });
      res = await r.json();
    } catch (e) { res = { ok: false, error: 'github_error' }; }
    return res;
  }

  async function start(key) {
    lock(true); setStatus('run', '処理中...'); say('更新を依頼しています…', '');
    const res = await request(key);
    if (!res.ok) {
      if (res.error === 'unauthorized' || res.error === 'locked') pass = null;
      setStatus('err', 'エラー'); say(ERR[res.error] || ('エラー: ' + res.error), 'err'); lock(false); return;
    }
    pass = key;
    startedAt = Date.now();
    if (res.state === 'already_running') { sinceMs = Date.parse(res.requested_at) - 1000 || 0; say('すでに更新中です。二重には起動していません。', ''); }
    else if (res.state === 'too_soon') { sinceMs = Date.now() - 150000; say('直前に更新を開始したばかりです。その更新の完了を待ちます。', ''); }
    else { sinceMs = Date.parse(res.requested_at) || (Date.now() - 10000); }
    poll();
  }

  btn.addEventListener('click', function () {
    if (running) return say('更新中です。完了までお待ちください。', '');
    const known = pass || (g.HoldingsLib && g.HoldingsLib.getPass && g.HoldingsLib.getPass());
    if (known) return start(known);
    form.hidden = false; input.focus();
    say('更新の実行には合言葉が必要です（保有銘柄と同じ合言葉）。', '');
  });
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const v = input.value; if (!v) return;
    input.value = ''; form.hidden = true; start(v);
  });

  // 開いたとき：すでに更新中か、直近の更新が失敗していないかを表示（認証不要）
  (async function init() {
    if (!cfg.enabled) { btn.hidden = true; return; }
    setStatus('idle', '最新');
    try {
      const runs = await fetchRuns();
      const d = decideRun(runs, 0);
      if (d.state === 'running') { startedAt = Date.parse(d.run.created_at) || Date.now(); sinceMs = startedAt - 1000; lock(true); setStatus('run', '処理中...'); poll(); }
      else if ((d.state === 'failure') && Date.parse(d.run.created_at) > Date.parse(cfg.generated)) {
        setStatus('err', 'エラー'); say('直近の自動更新に失敗しています。表示中のランキングは前回のものです。', 'err');
      }
    } catch (e) { /* 状況が取れなくても、ボタンは使える */ }
  })();
})(typeof window !== 'undefined' ? window : globalThis);
