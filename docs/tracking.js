/* 追跡・検証ページ：絞り込みと、行を開いたときの毎日の表。文字は textContent で入れる（HTMLとして解釈されない）。 */
(function () {
  'use strict';
  const table = document.getElementById('trkTable');
  if (!table) return;
  let data = null;
  const el = function (tag, text, cls) { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  const num = function (v, d) { return v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(d == null ? 1 : d) + '%'; };

  const LIMIT = 30, moreBtn = document.getElementById('moreBtn');
  let filter = 'all', expanded = false;
  function apply() {
    let shown = 0, total = 0;
    table.querySelectorAll('tr.entry').forEach(function (tr) {
      const st = tr.getAttribute('data-status'), r = tr.getAttribute('data-ret'), f = filter;
      const match = f === 'all' || (f === 'listed' && st === '掲載中') || (f === 'out' && st === '圏外へ') ||
        (f === 'plus' && r !== '' && Number(r) >= 0) || (f === 'minus' && r !== '' && Number(r) < 0);
      if (match) total++;
      const show = match && (expanded || shown < LIMIT);
      if (show) shown++;
      tr.hidden = !show;
      const d = tr.nextElementSibling; if (d && d.classList.contains('detail')) d.hidden = !show || d.getAttribute('data-open') !== '1';
    });
    moreBtn.hidden = total <= LIMIT;
    moreBtn.textContent = expanded ? '新しい' + LIMIT + '件だけ表示' : 'すべて表示（' + total + '件）';
  }
  document.querySelectorAll('.chip[data-f]').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('.chip[data-f]').forEach(function (x) { x.classList.toggle('on', x === b); });
      filter = b.getAttribute('data-f'); apply();
    });
  });
  moreBtn.addEventListener('click', function () { expanded = !expanded; apply(); });
  apply();

  async function toggle(tr) {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains('detail')) { const open = next.getAttribute('data-open') === '1'; next.setAttribute('data-open', open ? '0' : '1'); next.hidden = open; return; }
    if (!data) {
      try { data = (await (await fetch('tracking.json', { cache: 'no-store' })).json()).entries; } catch (e) { data = {}; }
    }
    const rows = data[tr.getAttribute('data-key')];
    const td = el('td'); td.colSpan = 8;
    if (!rows || !rows.length) { td.appendChild(el('p', '日ごとの記録はありません（追跡が終わった銘柄は、主な日の結果だけを残しています）。', 'hint')); }
    else {
      const t = el('table', null, 'daily');
      const hr = el('tr'); ['営業日', '日付', '終値', '騰落率', 'TOPIX', 'その日の順位'].forEach(function (h) { hr.appendChild(el('th', h)); });
      t.appendChild(el('thead')).appendChild(hr);
      const body = el('tbody');
      rows.slice().reverse().forEach(function (r) {
        const x = el('tr');
        x.appendChild(el('td', r[0] === 0 ? '載った日' : r[0] + '日後'));
        x.appendChild(el('td', r[1].slice(5)));
        x.appendChild(el('td', r[2] == null ? '—' : Number(r[2]).toLocaleString('ja-JP', { maximumFractionDigits: 1 })));
        x.appendChild(el('td', r[3] == null ? '買値待ち' : num(r[3]), r[3] == null ? '' : (r[3] >= 0 ? 'pos' : 'neg')));
        x.appendChild(el('td', num(r[4])));
        x.appendChild(el('td', r[5] == null ? (r[0] === 0 ? '—' : '圏外') : r[5] + '位'));
        body.appendChild(x);
      });
      t.appendChild(body);
      const wrap = el('div', null, 'table-wrap'); wrap.appendChild(t); td.appendChild(wrap);
    }
    const d = el('tr', null, 'detail'); d.setAttribute('data-open', '1'); d.appendChild(td);
    tr.parentNode.insertBefore(d, tr.nextSibling);
  }
  table.querySelectorAll('tr.entry').forEach(function (tr) {
    tr.addEventListener('click', function () { toggle(tr); });
    tr.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(tr); } });
  });
})();
