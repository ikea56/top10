/* 10時エントリー判定の表示。entry.json（entry10.yml が毎朝10:30までに更新）を読み、各銘柄カードに後から追記する。
   順位・点数は一切変えない。状態の保存（LocalStorage等）もしない。 */
(function () {
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); };
  var sgn = function (v, d) { return v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(d == null ? 1 : d) + '%'; };
  var jstToday = function () {
    var p = new Intl.DateTimeFormat('en-CA', {timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit'}).format(new Date());
    return p;                                   // YYYY-MM-DD
  };
  function volText(it) {
    if (it.vr == null) return '出来高進捗 不明';
    return '出来高進捗 ' + it.vol_label + '（普段の' + it.vr.toFixed(1) + '倍）';
  }
  function cardHtml(it) {
    var head = '<span class="e10-title">⏰ 10時判定</span> <span class="e10-badge e10-' + it.key + '">' + it.emoji + ' ' + esc(it.label) + '</span>';
    if (it.key === 'na') return head + '<div class="e10-lines"><span>' + esc((it.reasons || [''])[0]) + '</span></div>';
    var lines = '<span>ギャップ率 <b>' + sgn(it.gap) + '</b></span><span>寄付き後騰落率 <b>' + sgn(it.ret) + '</b></span><span>' + esc(volText(it)) + '</span>';
    var why = (it.reasons || []).map(function (r) { return '<li>' + esc(r) + '</li>'; }).join('');
    return head + '<div class="e10-lines">' + lines + '</div><ul class="e10-why">' + why + '</ul>';
  }
  function stripHtml(d) {
    var c = d.counts || {}, m = d.market || {};
    var items = (m.items || []).map(function (i) {
      return '<span class="e10-mi ' + esc(i.status || '') + '">' + esc(i.label) + ' ' + (i.pct == null ? '—' : sgn(i.pct)) + '</span>';
    }).join('');
    return '<b>⏰ 10時エントリー判定</b>（基準 ' + esc(d.asof || '10:00') + '）' +
      '<div class="e10-counts"><span>🟢 エントリー可 ' + (c.go || 0) + '</span><span>🟡 様子見 ' + (c.wait || 0) + '</span><span>🔴 見送り ' + (c.skip || 0) + '</span></div>' +
      '<div class="e10-mkt">市場環境：<b>' + esc(m.label || '—') + '</b></div><div class="e10-mis">' + items + '</div>' +
      '<p class="hint">順位は朝のまま変わりません（中長期の有望度）。10時判定は「いま入って高値掴みにならないか」だけを見ます。投資判断はご自身で。</p>';
  }
  function setPlaceholder(text) {
    document.querySelectorAll('.e10 .e10-body').forEach(function (el) { el.textContent = text; });
  }
  function show(d) {
    var strip = document.getElementById('e10Strip');
    var rd = strip ? strip.getAttribute('data-ranking-date') : null;
    if (d.ranking_data_date && rd && d.ranking_data_date.slice(0, 10) !== rd.slice(0, 10)) {
      setPlaceholder('今日のランキング用の10時判定は、まだ作られていません（10:30までに追記されます）'); return;
    }
    if (d.trade_date !== jstToday()) {
      setPlaceholder('本日（' + jstToday() + '）の10時判定は、まだありません（10:30までに追記されます）'); return;
    }
    if (d.status !== 'ok') {
      setPlaceholder('本日は10時判定のデータを取得できませんでした（休場日・取得失敗など）'); return;
    }
    if (strip) { strip.innerHTML = stripHtml(d); strip.hidden = false; }
    document.querySelectorAll('.e10').forEach(function (el) {
      var it = (d.items || {})[el.getAttribute('data-code')];
      if (!it) return;
      el.className = 'e10 e10-card-' + it.key;
      el.innerHTML = cardHtml(it);
    });
  }
  function load() {
    if (!document.querySelector('.e10')) return;
    fetch('entry.json?_=' + Date.now(), {cache: 'no-store'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(show)
      .catch(function () { setPlaceholder('10時判定は、まだ作られていません（10:30までに追記されます）'); });
  }
  function timings() {
    var box = document.getElementById('timings'); if (!box) return;
    fetch('timings.json?_=' + Date.now(), {cache: 'no-store'}).then(function (r) { return r.json(); }).then(function (t) {
      var sec = t.seconds || {}, rows = Object.keys(sec).filter(function (k) { return k !== '合計'; })
        .sort(function (a, b) { return sec[b] - sec[a]; })
        .map(function (k) { return '<li>' + esc(k) + '：' + (sec[k] >= 60 ? (sec[k] / 60).toFixed(1) + '分' : sec[k].toFixed(0) + '秒') + '</li>'; }).join('');
      document.getElementById('timingsBody').innerHTML = '<ul>' + rows + '</ul><p class="hint">合計 ' + ((sec['合計'] || 0) / 60).toFixed(1) + '分</p>';
      box.hidden = false;
    }).catch(function () {});
  }
  timings();
  // 開いたまま待っている場合は、10:45まで2分ごとに自動で読み直す（まだ出ていないときだけ）
  var polls = 0;
  var timer = setInterval(function () {
    var pending = document.querySelector('.e10 .e10-body');
    var hm = new Intl.DateTimeFormat('en-GB', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit'}).format(new Date());
    if (!pending || ++polls > 60 || hm > '10:45' && hm < '23:59' && polls > 3) { clearInterval(timer); return; }
    load();
  }, 120000);
  if (document.readyState !== 'loading') load(); else document.addEventListener('DOMContentLoaded', load);
})();
