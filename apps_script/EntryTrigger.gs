/**
 * 10時エントリー判定を、GitHub Actionsの自動実行（約1時間遅れることがある）に頼らず、
 * Google Apps Script の時刻トリガーで「平日 10:10〜10:40 のあいだに1回」起動する。
 *
 * 【設定（1回だけ）】
 *  1. https://script.google.com で「新しいプロジェクト」を作り、このファイルの中身を貼り付ける（保持用の Code.gs とは別のプロジェクトでOK）
 *  2. 左の ⚙「プロジェクトの設定」→「スクリプト プロパティ」に GITHUB_TOKEN を追加し、値に GitHub のトークンを入れる
 *       トークンの作り方：GitHub → 右上アイコン → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate
 *         Repository access: Only select repositories → ikea56/top10
 *         Permissions → Repository permissions → Actions: Read and write（これだけ）
 *       ※ トークンはこの Apps Script の「スクリプト プロパティ」にだけ入れる（サイトのJavaScriptやリポジトリには絶対に入れない）
 *  3. 上の「プロジェクトの設定」で タイムゾーンを (GMT+09:00) 東京 にする
 *  4. エディタで setup を選んで「実行」→ 権限を許可（5分ごとに tick が動くトリガーが作られる）
 *  5. 動作確認：テスト用に runNow を実行 → GitHub の Actions に entry10 が出れば成功
 */
var OWNER = 'ikea56';
var REPO = 'top10';
var WORKFLOW = 'entry10.yml';
var REF = 'main';
var START_HHMM = 1010;   // この時刻以降に起動（Yahooの株価は約20分遅れ。entry10側が10:27まで足の到着を待つ）
var END_HHMM = 1040;     // この時刻を過ぎたら、その日は起動しない

function tick() {
  var now = new Date();
  var dow = Number(Utilities.formatDate(now, 'Asia/Tokyo', 'u'));        // 1=月 … 7=日
  var hhmm = Number(Utilities.formatDate(now, 'Asia/Tokyo', 'HHmm'));
  if (dow >= 6 || hhmm < START_HHMM || hhmm > END_HHMM) return;
  var today = Utilities.formatDate(now, 'Asia/Tokyo', 'yyyy-MM-dd');
  var props = PropertiesService.getScriptProperties();
  if (props.getProperty('LAST_FIRED') === today) return;                 // その日は起動済み
  if (dispatch_()) props.setProperty('LAST_FIRED', today);
}

function runNow() { Logger.log(dispatch_() ? '起動しました（GitHubのActionsを確認）' : '失敗しました（ログを確認）'); }

function dispatch_() {
  var token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) { Logger.log('GITHUB_TOKEN が未設定です'); return false; }
  var url = 'https://api.github.com/repos/' + OWNER + '/' + REPO + '/actions/workflows/' + WORKFLOW + '/dispatches';
  var res = UrlFetchApp.fetch(url, {
    method: 'post', contentType: 'application/json', muteHttpExceptions: true,
    headers: { Authorization: 'Bearer ' + token, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' },
    payload: JSON.stringify({ ref: REF })
  });
  Logger.log('GitHub応答: ' + res.getResponseCode() + ' ' + res.getContentText());
  return res.getResponseCode() === 204;
}

function setup() {
  ScriptApp.getProjectTriggers().forEach(function (t) { if (t.getHandlerFunction() === 'tick') ScriptApp.deleteTrigger(t); });
  ScriptApp.newTrigger('tick').timeBased().everyMinutes(5).create();
  Logger.log('5分ごとのトリガーを作りました');
}
