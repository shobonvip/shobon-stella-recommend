import html
from dataclasses import dataclass

# --- データクラス定義 (前回と同じ) ---
@dataclass
class BMSChartData:
    title: str
    level: str
    difficulty_value: float
    current_lamp: str
    min_bp: int
    score_rate: float
    next_goal: str
    pass_prob: float

    @property
    def lamp_color_class(self):
        if "HARD" in self.current_lamp or "EX-HARD" in self.current_lamp:
            return "lamp-hard"
        elif "EASY" in self.current_lamp:
            return "lamp-easy"
        elif "FULLCOMBO" in self.current_lamp:
            return "lamp-fc"
        elif "FAILED" in self.current_lamp:
            return "lamp-failed"
        return "lamp-noplay"

def generate_bms_tabs(chart_list, filename="my_bms_tabs.html"):
    # 1. 難易度リストの作成 (数値順にソート)
    unique_levels = sorted(
        list(set(c.level for c in chart_list)),
        key=lambda x: float(x.replace("★", "").replace("st", "").replace("▼", ""))
    )
    # 先頭に "ALL" を追加して、タブのリストを作る
    tabs = ["ALL"] + unique_levels

    # HTMLヘッダー & CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>My BMS Difficulty Table</title>
        <style>
            body {{ font-family: sans-serif; background-color: #222; color: #eee; padding: 20px; }}
            
            /* タブ部分のスタイル */
            .tab {{ overflow: hidden; border: 1px solid #444; background-color: #333; border-radius: 5px 5px 0 0; }}
            .tab button {{
                background-color: inherit; float: left; border: none; outline: none;
                cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold;
            }}
            .tab button:hover {{ background-color: #555; }}
            .tab button.active {{ background-color: #007bff; color: white; }}

            /* タブの中身 */
            .tabcontent {{
                display: none; padding: 6px 12px; border: 1px solid #444; border-top: none;
                animation: fadeEffect 0.5s; /* フェードインアニメーション */
            }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}

            /* テーブル装飾 */
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #444; text-align: left; }}
            th {{ background-color: #333; cursor: pointer; user-select: none; }}
            th:hover {{ background-color: #555; }}
            th::after {{ content: ' ⇅'; font-size: 0.8em; color: #888; }}
            
            /* ランプ色 */
            .lamp-hard {{ color: #ff5555; font-weight: bold; text-shadow: 0 0 5px #ff5555; }}
            .lamp-easy {{ color: #55ff55; font-weight: bold; text-shadow: 0 0 5px #55ff55; }}
            .lamp-fc   {{ color: #55ffff; font-weight: bold; text-shadow: 0 0 5px #55ffff; }}
            .lamp-failed {{ color: #aaaaaa; }}
            .lamp-noplay {{ color: #666666; }}
            .prob-high {{ color: #ffcc00; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>BMS 攻略目標リスト</h1>
        
        <div class="tab">
    """

    # 2. タブボタンの生成 (IDは tab-btn-0, tab-btn-1... とする)
    for i, level in enumerate(tabs):
        # 最初のタブ(ALL)だけデフォルトでアクティブにする
        active_class = " active" if i == 0 else ""
        html_content += f"""
            <button class="tablinks{active_class}" onclick="openTab(event, 'tab-content-{i}')">{html.escape(level)}</button>
        """
    
    html_content += "</div>" # tab close

    # 3. タブごとのコンテンツ(テーブル)生成
    for i, level in enumerate(tabs):
        # 表示するデータリストをフィルタリング
        if level == "ALL":
            target_data = chart_list
        else:
            target_data = [c for c in chart_list if c.level == level]
        
        # 最初のタブだけ表示(display: block)、他は非表示
        display_style = "display: block;" if i == 0 else ""
        
        html_content += f"""
        <div id="tab-content-{i}" class="tabcontent" style="{display_style}">
            <table id="table-{i}">
                <thead>
                    <tr>
                        <th onclick="sortTable({i}, 0, 'text')">タイトル</th>
                        <th onclick="sortTable({i}, 1, 'number')">難易度</th>
                        <th onclick="sortTable({i}, 2, 'text')">現状</th>
                        <th onclick="sortTable({i}, 3, 'number')">最小BP</th>
                        <th onclick="sortTable({i}, 4, 'number')">スコア率</th>
                        <th onclick="sortTable({i}, 5, 'text')">次の目標</th>
                        <th onclick="sortTable({i}, 6, 'number')">達成確率</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for chart in target_data:
            prob_class = "prob-high" if chart.pass_prob >= 50.0 else ""
            html_content += f"""
                <tr>
                    <td data-value="{html.escape(chart.title)}">{html.escape(chart.title)}</td>
                    <td data-value="{chart.difficulty_value}">{html.escape(chart.level)}</td>
                    <td data-value="{chart.current_lamp}" class="{chart.lamp_color_class}">{html.escape(chart.current_lamp)}</td>
                    <td data-value="{chart.min_bp}">{chart.min_bp}</td>
                    <td data-value="{chart.score_rate}">{chart.score_rate:.2f}%</td>
                    <td data-value="{chart.next_goal}">{html.escape(chart.next_goal)}</td>
                    <td data-value="{chart.pass_prob}" class="{prob_class}">{chart.pass_prob:.1f}%</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        """

    # 4. JavaScript (タブ切り替え & ソート)
    html_content += """
        <script>
            // --- タブ切り替え関数 ---
            function openTab(evt, tabId) {
                var i, tabcontent, tablinks;
                
                // すべてのコンテンツを隠す
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                
                // すべてのボタンから active クラスを削除
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                
                // 指定されたタブを表示し、ボタンを active にする
                document.getElementById(tabId).style.display = "block";
                evt.currentTarget.className += " active";
            }

            // --- ソート関数 (テーブルIDを指定して実行) ---
            function sortTable(tableIndex, col, type) {
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                // タブごとにテーブルIDが違うので、引数で受け取る
                table = document.getElementById("table-" + tableIndex);
                switching = true;
                dir = "asc"; 
                
                while (switching) {
                    switching = false;
                    rows = table.rows;
                    
                    for (i = 1; i < (rows.length - 1); i++) {
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[col].getAttribute("data-value");
                        y = rows[i + 1].getElementsByTagName("TD")[col].getAttribute("data-value");
                        
                        if (type === 'number') {
                            x = parseFloat(x);
                            y = parseFloat(y);
                        } else {
                            x = x.toLowerCase();
                            y = y.toLowerCase();
                        }

                        if (dir == "asc") {
                            if (x > y) { shouldSwitch = true; break; }
                        } else if (dir == "desc") {
                            if (x < y) { shouldSwitch = true; break; }
                        }
                    }
                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount ++; 
                    } else {
                        if (switchcount == 0 && dir == "asc") {
                            dir = "desc";
                            switching = true;
                        }
                    }
                }
            }
        </script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"タブ付きファイルを作成しました: {filename}")

# --- 実行用サンプル ---
if __name__ == "__main__":
    sample_data = [
        BMSChartData("FREEDOM DiVE", "★24", 24.5, "FAILED", 150, 78.5, "EASY", 12.4),
        BMSChartData("Air", "★24", 24.2, "EASY", 80, 82.1, "HARD", 5.2),
        BMSChartData("conflict", "★14", 14.0, "HARD", 12, 94.5, "EX-HARD", 60.8),
        BMSChartData("Absurd Gaff", "★12", 12.0, "FAILED", 45, 88.0, "EASY", 85.3),
        BMSChartData("L9", "★1", 1.0, "FULLCOMBO", 0, 100.0, "MAX", 99.9),
    ]
    generate_bms_tabs(sample_data)