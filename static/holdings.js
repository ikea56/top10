/* 保有銘柄の管理・監視（ブラウザ側）。
 * ・保有銘柄の保存先：Google スプレッドシート＋Apps Script（サーバー側）。ブラウザには保存しません（LocalStorage不使用）。
 * ・GitHubのトークンは一切持ちません。合言葉はこのページを開いている間だけメモリに置きます。
 * ・監視データは合言葉で暗号化されたファイルで、復号はこのブラウザの中だけで行います。 */
(function (g) {
  'use strict';

  // ---------- 共通の小さな関数（テストからも使う） ----------
  const ZEN = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ';
  const HAN = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  function normalizeCode(s) {
    let t = String(s || '').trim();
    t = Array.from(t).map(function (c) { const i = ZEN.indexOf(c); return i >= 0 ? HAN[i] : c; }).join('').toUpperCase();
    return /^[0-9]{3}[0-9A-Z]$/.test(t) ? t : null;
  }
  function b64(s) { return Uint8Array.from(atob(s), function (c) { return c.charCodeAt(0); }); }
  async function decrypt(blob, pass) {          // PBKDF2-SHA256 → AES-256-GCM（Python側 src/holdings.py と同じ形式）
    const km = await crypto.subtle.importKey('raw', new TextEncoder().encode(pass), 'PBKDF2', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey({ name: 'PBKDF2', salt: b64(blob.salt), iterations: blob.iter, hash: 'SHA-256' },
      km, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64(blob.iv) }, key, b64(blob.ct));
    return JSON.parse(new TextDecoder().decode(pt));
  }
  g.HoldingsLib = { normalizeCode: normalizeCode, decrypt: decrypt };

  if (typeof document === 'undefined') return;
  const root = document.getElementById('holdings');
  const cfg = g.HOLDINGS_CFG || {};
  if (!root) return;

  // ---------- 画面部品（文字は textContent で入れるので、HTMLとして解釈されない） ----------
  function h(tag, attrs) {
    const el = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'class') el.className = attrs[k];
      else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), attrs[k]);
      else el.setAttribute(k, attrs[k]);
    });
    (function add(list) {                      // 配列（map の結果など）も受け取れるよう、平らにして追加
      list.forEach(function (c) {
        if (c == null || c === false) return;
        if (Array.isArray(c)) return add(c);
        el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      });
    })(Array.prototype.slice.call(arguments, 2));
    return el;
  }
  const yen = function (v) { return v == null ? '—' : '¥' + Number(v).toLocaleString('ja-JP', { maximumFractionDigits: v >= 1000 ? 0 : 1 }); };
  const pct = function (v) { return v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1) + '%'; };

  let pass = null, codes = [], status = null, names = {};
  g.HoldingsLib.getPass = function () { return pass; };      // 手動更新ボタンが、開いている間だけ同じ合言葉を使えるように（メモリのみ）
  const box = h('div', { class: 'h-body' });
  root.appendChild(box);

  function msg(text, kind) {
    const m = root.querySelector('.h-msg');
    if (m) { m.textContent = text || ''; m.className = 'h-msg ' + (kind || ''); }
  }

  async function api(body) {
    const r = await fetch(cfg.endpoint, { method: 'POST', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
      body: JSON.stringify(Object.assign({ key: pass }, body)) });
    return r.json();
  }
  const ERR = { unauthorized: '合言葉が違います', locked: '失敗が続いたため15分ほどロックされています', not_configured: 'サーバー側の合言葉が未設定です（README参照）',
    bad_code: '証券コードは4桁です（例 7203）', too_many: '登録できる上限に達しました', bad_request: '送信内容が正しくありません' };

  // ---------- ロック画面 ----------
  function renderLocked() {
    const input = h('input', { type: 'password', name: 'passphrase', autocomplete: 'current-password', placeholder: '合言葉', 'aria-label': '合言葉', class: 'h-input' });
    const form = h('form', { class: 'h-form', onsubmit: async function (e) {
      e.preventDefault();
      pass = input.value;
      if (!pass) return;
      msg('確認中…');
      try { await load(); } catch (err) { pass = null; msg(String(err.message || err), 'err'); }
    } }, input, h('button', { class: 'btn', type: 'submit' }, '開く'));
    box.replaceChildren(
      h('p', { class: 'hint' }, '保有銘柄は他の人に見えないよう、合言葉で保護しています。合言葉を入力すると、保有銘柄の管理と監視が表示されます。'),
      form, h('p', { class: 'h-msg' }));
  }

  // ---------- 読み込み（一覧の取得 ＋ 監視データの復号） ----------
  async function load() {
    const res = await api({ action: 'list' });
    if (!res.ok) throw new Error(ERR[res.error] || ('エラー: ' + res.error));
    codes = res.holdings || [];
    try {
      const r = await fetch(cfg.statusUrl, { cache: 'no-store' });
      status = r.ok ? await decrypt(await r.json(), pass) : null;
    } catch (e) { status = null; }
    if (!Object.keys(names).length) {
      try { names = await (await fetch(cfg.namesUrl)).json(); } catch (e) { names = {}; }
    }
    renderMain();
  }

  // ---------- 管理（追加・削除） ----------
  async function add(input) {
    const code = HoldingsLib.normalizeCode(input.value);
    if (!code) return msg('証券コードは4桁で入力してください（例 7203）', 'err');
    if (Object.keys(names).length && !names[code]) return msg('東証の上場銘柄に ' + code + ' が見つかりません', 'err');
    msg('追加中…');
    const res = await api({ action: 'add', code: code });
    if (!res.ok) return msg(ERR[res.error] || ('エラー: ' + res.error), 'err');
    codes = res.holdings; input.value = '';
    renderMain();
    msg((names[code] || code) + ' を追加しました。' + (res.rebuild ? '数分後に監視カードが更新されます。' : '監視カードは次の自動更新（平日朝7時）から表示されます。'), 'ok');
  }
  async function remove(code) {
    if (!confirm((names[code] || '') + '（' + code + '）を保有銘柄から削除しますか？')) return;
    const res = await api({ action: 'remove', code: code });
    if (!res.ok) return msg(ERR[res.error] || ('エラー: ' + res.error), 'err');
    codes = res.holdings;
    renderMain();
    msg(code + ' を削除しました', 'ok');
  }

  // ---------- 保有銘柄カード ----------
  function chip(label, state, detail) {
    return h('span', { class: 'chip ' + state }, (state === 'good' ? '✅ ' : state === 'bad' ? '⚠ ' : '－ ') + label + (detail ? '（' + detail + '）' : ''));
  }
  function card(c) {
    const a = c.action || {};
    const rank = c.rank_main ? '本日のランキング ' + c.rank_main + '位' : '本日のランキング圏外（押し目スコア順 ' + c.rank_tech + '位／' + c.n_tech + '銘柄中）';
    const e = c.earnings || {};
    const ev = (c.events || []).map(function (x) { return chip(x.label, x.state, x.detail); });
    ev.push(e.days != null && e.days <= 10 ? chip('決算接近', 'bad', e.date.slice(5).replace('-', '/') + '・あと' + e.days + '日') : chip('決算接近', 'none', e.date ? '次回 ' + e.date.slice(5).replace('-', '/') : ''));
    const s = c.stop;
    return h('article', { class: 'card stock hcard act-' + a.key },
      h('header', { class: 'stock-head' },
        h('div', { class: 'ident' }, h('h3', {}, c.name), h('p', { class: 'sub' }, c.code + (c.market ? ' ・ ' + c.market : '') + (c.sector ? ' ・ ' + c.sector : ''))),
        h('span', { class: 'pill ' + (a.css || '') }, (a.emoji || '') + ' ' + (a.text || ''))),
      c.found === false ? null : h('div', { class: 'h-kpis' },
        h('div', { class: 'kpi' }, h('span', { class: 'k' }, '現在価格'), h('span', { class: 'v' }, yen(c.close)),
          h('span', { class: 'chg ' + (c.ret1d >= 0 ? 'up' : 'down') }, '前日比 ' + pct(c.ret1d))),
        h('div', { class: 'kpi' }, h('span', { class: 'k' }, '押し目スコア'), h('span', { class: 'v' }, String(c.pb_total))),
        h('div', { class: 'kpi' }, h('span', { class: 'k' }, 'イベントスコア'), h('span', { class: 'v' }, (c.td_score > 0 ? '+' : '') + c.td_score))),
      c.found === false ? null : h('p', { class: 'formula' }, rank + '　／　総合点 ' + c.main + '（' + (c.label || '') + '）'),
      h('section', { class: 'block risk' }, h('h4', {}, '判定の理由'),
        h('ul', {}, (a.reasons || []).map(function (t) { return h('li', { class: a.key === 'hold' ? '' : 'lv-' + (a.key === 'sell' ? 'high' : 'mid') }, (a.key === 'hold' ? '✅ ' : '⚠ ') + t); }))),
      c.found === false ? null : h('div', { class: 'chips' }, ev),
      c.found === false ? null : h('section', { class: 'block events' }, h('h4', {}, 'TDnetイベント（直近180日）'),
        h('ul', {}, (c.td_events || []).length ? c.td_events.map(function (t) { return h('li', {}, t.label + '　' + t.date.slice(5).replace('-', '/') + '　(' + (t.points > 0 ? '+' : '') + t.points + '点)'); })
          : [h('li', { class: 'none' }, c.td_status === 'ok' ? '直近180日に評価対象の開示はありません' : 'TDnetを確認できていません')])),
      c.found === false || !(c.cautions || []).length ? null : h('section', { class: 'block risk' }, h('h4', {}, '【注意事項】'),
        h('ul', {}, c.cautions.map(function (t) { return h('li', { class: 'lv-' + t.level }, '⚠ ' + t.label); }))),
      s ? h('p', { class: 'stop' }, 'ATR撤退ラインの目安：' + yen(s.price) + '（' + s.name + '／ATR ' + yen(s.atr) + '／リスク幅 ' + s.risk_pct.toFixed(1) + '%）') : null);
  }

  // ---------- メイン画面（管理 ＋ 監視） ----------
  function renderMain() {
    const input = h('input', { type: 'text', inputmode: 'text', autocapitalize: 'characters', maxlength: '8', placeholder: '例 7203', 'aria-label': '銘柄コード', class: 'h-input' });
    const list = h('ul', { class: 'h-list' }, codes.length ? codes.map(function (c) {
      const st = status && status.holdings && status.holdings[c];
      return h('li', {}, h('span', { class: 'h-name' }, c + ' ' + (names[c] || (st && st.name) || '')),
        st ? h('span', { class: 'pill ' + st.action.css }, st.action.emoji + ' ' + st.action.text) : h('span', { class: 'hint' }, '次回更新で監視開始'),
        h('button', { class: 'btn danger', type: 'button', onclick: function () { remove(c); } }, '削除'));
    }) : [h('li', { class: 'none' }, 'まだ登録されていません')]);
    const cards = codes.map(function (c) { return status && status.holdings && status.holdings[c]; }).filter(Boolean);
    const pending = codes.filter(function (c) { return !(status && status.holdings && status.holdings[c]); });
    const order = { sell: 0, watch: 1, hold: 2 };
    cards.sort(function (x, y) { return order[x.action.key] - order[y.action.key]; });
    const n = function (k) { return cards.filter(function (c) { return c.action.key === k; }).length; };
    box.replaceChildren.apply(box, [
      h('h3', { class: 'h-title' }, '保有銘柄管理'),
      h('form', { class: 'h-form', onsubmit: function (e) { e.preventDefault(); add(input); } },
        h('label', { class: 'k' }, '銘柄コード'), input, h('button', { class: 'btn', type: 'submit' }, '追加')),
      h('p', { class: 'h-msg' }),
      h('h3', { class: 'h-title' }, '保有銘柄一覧'), list,
      h('h3', { class: 'h-title' }, '保有銘柄監視'),
      status ? h('p', { class: 'hint' }, '🔴 売却検討 ' + n('sell') + '　🟡 要監視 ' + n('watch') + '　🟢 保有継続 ' + n('hold') + '　（データ基準日 ' + status.data_date + '。ランキング順位に関係なく常に監視しています）')
        : h('p', { class: 'hint' }, '監視データはまだありません。次の自動更新（平日朝7時）のあとに表示されます。'),
      pending.length && status ? h('p', { class: 'hint' }, '追加したばかりの銘柄（' + pending.join('・') + '）は、次の更新から監視カードが表示されます。') : null,
      h('div', {}, cards.map(card))].filter(Boolean));      // replaceChildren は null を文字の「null」にしてしまうため除く
  }

  // ---------- 起動 ----------
  if (!cfg.endpoint) {
    box.appendChild(h('p', { class: 'hint' }, '保有銘柄管理はまだ設定されていません。設定方法は README の「保有銘柄管理のセットアップ」を見てください。'));
  } else {
    renderLocked();
  }
})(typeof window !== 'undefined' ? window : globalThis);
