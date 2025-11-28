"""

Shobon Stella Recommend v1.1
by Shobon

"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import sqlite3
import csv
import numpy as np
import html
import json
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple

# score のデータから必要な情報を取得
def refine_score_data(score: dict) -> dict:
	ret = dict()
	ret['sha256'] = score['sha256']
	ret['clear'] = score['clear']
	ret['score_rate'] = (score['epg'] * 2 + score['lpg'] * 2 + score['egr'] + score['lgr']) / (2 * score['notes'])
	ret['minbp'] = score['minbp']
	return ret

# score.db から情報を取得
def get_score_list(directory: str) -> List[dict]:
	with sqlite3.connect(directory) as con:
		con.row_factory = sqlite3.Row
		cur = con.cursor()
		cur.execute("SELECT * FROM score")
		score_tables = cur.fetchall()
		score_list = []
		score_map = dict()
		for score_row in score_tables:
			dat = refine_score_data(dict(score_row))
			if score_row['sha256'] not in score_map:
				score_map[score_row['sha256']] = dat
			else:
				score_map[score_row['sha256']]['clear']\
					= str(max(int(score_map[score_row['sha256']]['clear']), int(dat['clear'])))
				score_map[score_row['sha256']]['minbp']\
					= str(min(int(score_map[score_row['sha256']]['minbp']), int(dat['minbp'])))
				score_map[score_row['sha256']]['score_rate']\
					= str(max(float(score_map[score_row['sha256']]['score_rate']), float(dat['score_rate'])))				
		for sha256, dat in score_map.items():
			score_list.append(dat)
		return score_list

# mocha_sl/st.csv から情報を取得 
def get_song_list(directory: str) -> List[dict]:
	with open(directory, encoding='utf-8') as file:
		reader = csv.reader(file)
		row_name = ["title","display_level","md5","sha256","beta_easy","beta_hard","alpha","has_data"]
		song_list = []
		for song in list(reader)[1:]:
			tmp = dict()
			for x in range(len(song)):
				tmp[row_name[x]] = song[x]
			be = float(tmp["beta_easy"])
			bh = float(tmp["beta_hard"])
			if bh - be < 0.01:
				print(f"TOO NEAR between EASY AND HARD!! {tmp}")
			assert tmp["has_data"] == "True"
			song_list.append(tmp)
		return song_list

# クリア状況によって No Play / Failed / Easy / Hard に分ける
def get_clear_type(c: int) -> str:
	if c >= 6:
		return "Hard"
	elif c >= 4:
		return "Easy"
	elif c >= 1:
		return "Failed"
	else:
		return "No Play"

# 詳細なクリア状況に分ける
def get_detailed_clear_type(c: int) -> str:
	if c >= 8:
		return "FullCombo"
	elif c == 7:
		return "ExHard"
	elif c >= 6:
		return "Hard"
	elif c >= 5:
		return "Clear"
	elif c >= 4:
		return "Easy"
	elif c >= 3:
		return "L-Assist"
	elif c >= 2:
		return "Assist"
	elif c >= 1:
		return "Failed"
	else:
		return "No Play"

# 次に目指すべきクリア状況
def get_next_clear_type(c: int) -> str:
	if c >= 6:
		return ""
	if c >= 4:
		return "Hard"
	return "Easy"

# 各難易度の平均 beta を計算する
def get_average_list(song_list: List[dict], mode_slst: str) -> List[float]:
	sum_beta = []
	nums = []
	for song in song_list:
		if song["display_level"][:2] != mode_slst:
			continue
		level = int(song["display_level"][2:])
		while level >= len(nums):
			nums.append(0)
			sum_beta.append(0.0)
		nums[level] += 1
		sum_beta[level] += float(song["beta_easy"])
	ret = [sum_beta[i] / nums[i] for i in range(len(nums))]
	return ret

# beta の値から sl に変換
def beta_to_stella(average_list: List[float], c: float) -> float:
	now_st = -1
	for level in range(len(average_list) - 1):
		assert average_list[level] < average_list[level+1]
		if average_list[level] <= c:
			now_st = level
	if now_st == -1:
		now_st = 0
	elif now_st == len(average_list):
		now_st = len(average_list) - 1
	return (c - average_list[now_st]) / (average_list[now_st + 1] - average_list[now_st]) + now_st

# クリアできる確率
def prob_grm(theta: float, beta: float, alpha: float) -> float:
	return 1.0 / (1.0 + np.exp(- alpha * (theta - beta)))

# 負の対数尤度を計算
def negative_log_likelihood(theta: float, score_list: List[dict], song_list: List[dict]) -> float:
	log_likelihood = 0.0
	epsilon = 1e-9
	sha256_dict = dict()
	for song in song_list:
		sha256_dict[song["sha256"]] = song
	for score in score_list:
		clear_result = get_clear_type(int(score["clear"]))
		if clear_result == "No Play":
			continue 
		if score["sha256"] not in sha256_dict:
			continue
		prob = 0.0
		song = sha256_dict[score["sha256"]]
		beta_easy = float(song["beta_easy"])
		beta_hard = float(song["beta_hard"])
		alpha = float(song["alpha"])
		p1 = prob_grm(theta, beta_easy, alpha)
		p2 = prob_grm(theta, beta_hard, alpha)
		if clear_result == "Failed":
			prob = 1.0 - p1
		elif clear_result == "Easy":
			prob = p1 - p2
		elif clear_result == "Hard":
			prob = p2
		else:
			assert False
		log_likelihood += np.log(max(prob, epsilon))
	return - log_likelihood

# minimize_scalar で最尤推定
def max_likelihood_estimation(score_list: List[dict], song_list: List[dict]) -> float:
	result = minimize_scalar(
		negative_log_likelihood, 
		bounds = (-20, 10), # この範囲に解があると仮定
		args = (score_list, song_list),
		method = 'bounded'
	)
	return result

# ppを計算する
def pp_value(average_list: List[float], beta: float) -> float:
	return (beta_to_stella(average_list, beta) + 2) * 40

# ppリストを取得
def get_sorted_pp_data(average_list: List[float], score_list: List[dict], song_list: List[dict], max_num: int) -> List[dict]:
	ret = []

	sha256_dict = dict()
	for song in song_list:
		sha256_dict[song["sha256"]] = song
	
	for score in score_list:
		clear_result = get_clear_type(int(score["clear"]))
		if clear_result == "No Play":
			continue 
		if score["sha256"] not in sha256_dict:
			continue

		song = sha256_dict[score["sha256"]]
		tmp = dict()
		tmp["title"] = song["title"]
		tmp["level"] = song["display_level"]
		tmp["sha256"] = song["sha256"]
		tmp["minbp"] = score["minbp"]
		tmp["score_rate"] = score["score_rate"]
		tmp["display_level"] = song["display_level"]
		tmp["alpha"] = float(song["alpha"])
		tmp["real_clear"] = score["clear"]

		if clear_result == "Failed":
			continue
		elif clear_result == "Easy":
			beta_easy = float(song["beta_easy"])
			pp = pp_value(average_list, beta_easy)
			tmp["beta"] = float(song["beta_easy"])
			tmp["pp"] = pp
			tmp["clear"] = "Easy"
		elif clear_result == "Hard":
			beta_hard = float(song["beta_hard"])
			pp = pp_value(average_list, beta_hard)
			tmp["beta"] = float(song["beta_hard"])
			tmp["pp"] = pp
			tmp["clear"] = "Hard"
		else:
			assert False
	
		ret.append(tmp)
	
	ret.sort(key = lambda x: -x["pp"])
	if len(ret) > max_num:
		ret = ret[:max_num]
	
	print(*ret, sep='\n')
	return ret


def generate_html_top100(
	score_list: List[dict],
	song_list: List[dict],
	mode_slst: str,
	average_list: List[float],
	estimated_theta: float,
	filename_top100: str
):
	dictLamp = {
		"FullCombo": 8,
		"ExHard": 7,
		"Hard": 6,
		"Clear": 5,
		"Easy": 4,
		"L-Assist": 3,
		"Assist": 2,
		"Failed": 1,
		"No Play": "NaN",
		"": "NaN"
	}

	lamp_order = ["No Play", "Failed", "Assist", "L-Assist", "Easy", "Clear", "Hard", "ExHard", "FullCombo"]
	lamp_filter_html = ""
	
	for lamp in lamp_order:
		safe_lamp = html.escape(lamp)
		lamp_filter_html += f"""
			<label class="filter-item">
				<input type="checkbox" class="lamp-checkbox" value="{dictLamp[safe_lamp]}" checked onchange="applyLampFilter()">
				{safe_lamp}
			</label>
		"""


	top100_list = get_sorted_pp_data(average_list, score_list, song_list, 100)
	pp_sum = 0
	pp_raw_sum = 0
	for song in top100_list[::-1]:
		pp_sum *= 0.97
		pp_sum += song['pp']
		pp_raw_sum += song['pp']


	# Prefix Template
	html_content = f"""
	<!DOCTYPE html>
	<html lang="ja">
	<head>
		<meta charset="UTF-8">
		<title>Shobon Stella Recommend</title>
		<style>
			body {{ font-family: sans-serif; background-color: #222; color: #eee; padding: 20px; }}
			
			.header-container {{
				display: flex;
				justify-content: space-between; /* タイトルは左、ボタンは右に配置 */
				align_items: center;
				margin-bottom: 20px;
				border-bottom: 1px solid #444;
				padding-bottom: 10px;
			}}
			.nav-btn {{
				background-color: #004488;
				color: white;
				padding: 10px 20px;
				text-decoration: none;
				border-radius: 5px;
				font-weight: bold;
				transition: background-color 0.3s;
				box-shadow: 0 2px 4px rgba(0,0,0,0.3);
			}}
			.nav-btn:hover {{
				background-color: #003366;
				transform: translateY(-1px);
			}}			

			/* --- ランプフィルタエリア --- */
			.filter-container {{
				background-color: #333; padding: 10px 15px; border-radius: 5px;
				margin-bottom: 15px; border: 1px solid #444;
			}}
			.filter-label {{ font-weight: bold; margin-right: 10px; font-size: 0.9em; color: #aaa; }}
			.filter-item {{ 
				display: inline-block; margin-right: 15px; cursor: pointer; user-select: none; font-weight: bold;
			}}
			.filter-buttons {{ margin-top: 5px; }}
			.filter-buttons button {{
				font-size: 0.8em; padding: 2px 8px; margin-right: 5px; cursor: pointer;
				background: #555; color: #fff; border: 1px solid #666; border-radius: 3px;
			}}
			.filter-buttons button:hover {{ background: #666; }}
			
			/* テーブル装飾 */
			table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
			th, td {{ padding: 10px; border-bottom: 1px solid #444; text-align: left; }}
			th {{ background-color: #333; cursor: pointer; user-select: none; }}
			th:hover {{ background-color: #555; }}
			th::after {{ content: ' ⇅'; font-size: 0.8em; color: #888; }}
			
			/* ランプの色分け */
			.lamp-fc {{ color: #55ffff; font-weight: bold; text-shadow: 0 0 5px #55ffff; }}
			.lamp-exh {{ color: #ffff55; font-weight: bold; text-shadow: 0 0 5px #ffff55; }}
			.lamp-hard {{ color: #ff5555; font-weight: bold; text-shadow: 0 0 5px #ff5555; }}
			.lamp-clear {{ color: #ffbb55; font-weight: bold; text-shadow: 0 0 5px #ffbb55; }}
			.lamp-easy {{ color: #55ff55; font-weight: bold; text-shadow: 0 0 5px #55ff55; }}
			.lamp-assist {{ color: #ff55ff; font-weight: bold; text-shadow: 0 0 5px #ff55ff; }}
			.lamp-failed {{ color: #cccccc; }}
			.lamp-noplay {{ color: #666666; }}

			a{{
				text-decoration: none;
				color: #eee;
			}}
		</style>
	</head>
	<body>
		<h1>Shobon Stella Recommend - Performance Top 100</h1>
		<a href="result_table.html" class="nav-btn">難易度表 ページへ ➜</a>
		<h2><font color="#55ffff">{pp_sum:.0f}pp</font> (Raw: {pp_raw_sum:.0f}pp)</h2>
		<h3></h3>

		<div class="filter-container">
			<div style="margin-bottom:5px;">
				<span class="filter-label">Now Lamp:</span>
				{lamp_filter_html}
			</div>
			<div class="filter-buttons">
				<button onclick="toggleLampAll(true)">全選択</button>
				<button onclick="toggleLampAll(false)">全解除</button>
			</div>
		</div>
	"""

	html_content += f"""
	<div id="tab-content" class="tabcontent">
		<table id="table">
			<thead>
				<tr>
					<th onclick="sortTable(0, 'number')">順位</th>
					<th onclick="sortTable(1, 'text')">タイトル</th>
					<th onclick="sortTable(2, 'smart-number')">表</th>
					<th onclick="sortTable(3, 'number')">ランプ</th>
					<th onclick="sortTable(4, 'smart-number')">推定</th>
					<th onclick="sortTable(5, 'number')">地力度</th>
					<th onclick="sortTable(6, 'smart-number')">達成確率</th>
					<th onclick="sortTable(7, 'smart-number')">pp</th>
				</tr>
			</thead>

			<tbody>
	"""

	for ret_num, song in enumerate(top100_list):
		ret_title = song["title"]
		if len(ret_title) >= 50:
			ret_title = ret_title[:47]+'...'
		ret_title = html.escape(ret_title)
		ret_sha256 = song["sha256"]


		ret_jiriki = f"{float(song['alpha'])/2:.2f}"
		ret_lamp = get_detailed_clear_type(int(song["real_clear"]))
		ret_ok_lamp = song["clear"]
		ret_dislv = song["display_level"]
		ret_lv = -1
		ret_pp = f"{song['pp']:.0f}pp"
		ret_prob = -1
		
		if ret_ok_lamp == "Easy":
			ret_prob = f"{prob_grm(estimated_theta, float(song['beta']), float(song['alpha'])) * 100:.2f} %"
			ret_lv = f"{mode_slst}{beta_to_stella(average_list, float(song['beta'])):.2f}"
		elif ret_ok_lamp == "Hard":
			ret_prob = f"{prob_grm(estimated_theta, float(song['beta']), float(song['alpha'])) * 100:.2f} %"
			ret_lv = f"{mode_slst}{beta_to_stella(average_list, float(song['beta'])):.2f}"
		else:
			assert(False)

		ret_colorclass = ""

		if ret_lamp == "No Play":
			ret_colorclass = "lamp-noplay"
		elif ret_lamp == "Failed":
			ret_colorclass = "lamp-failed"
		elif ret_lamp == "Assist":
			ret_colorclass = "lamp-assist"
		elif ret_lamp == "L-Assist":
			ret_colorclass = "lamp-assist"
		elif ret_lamp == "Easy":
			ret_colorclass = "lamp-easy"
		elif ret_lamp == "Clear":
			ret_colorclass = "lamp-clear"
		elif ret_lamp == "Hard":
			ret_colorclass = "lamp-hard"
		elif ret_lamp == "ExHard":
			ret_colorclass = "lamp-exh"
		elif ret_lamp == "FullCombo":
			ret_colorclass = "lamp-fc"

		ret_lampnum = dictLamp[ret_lamp]		

		html_content += f"""
				<tr class="chart-row">
					<td data-value="{ret_num + 1}">
						{ret_num + 1}
					</td>
					<td data-value="{ret_title}">
						<a href="https://mocha-repository.info/song.php?sha256={ret_sha256}">{ret_title}</a>
					</td>
					<td data-value="{ret_dislv}">
						{ret_dislv}
					</td>
					<td data-value="{ret_lampnum}" class="{ret_colorclass}">
						{ret_lamp}
					</td>
					<td data-value="{ret_lv}">
						{ret_lv}
					</td>
					<td data-value="{ret_jiriki}">
						{ret_jiriki}
					</td>
					<td data-value="{ret_prob}">
						{ret_prob}
					</td>
					<td data-value="{ret_pp}">
						{ret_pp}
					</td>
				</tr>
		"""
		
	html_content += "</tbody></table></div>"

	# Suffix Template
	html_content += """
		<script>
			function openTab(evt, tabId) {
				var i, tabcontent, tablinks;
				tabcontent = document.getElementsByClassName("tabcontent");
				for (i = 0; i < tabcontent.length; i++) {
					tabcontent[i].style.display = "none";
				}
				tablinks = document.getElementsByClassName("tablinks");
				for (i = 0; i < tablinks.length; i++) {
					tablinks[i].className = tablinks[i].className.replace(" active", "");
				}
				document.getElementById(tabId).style.display = "block";
				evt.currentTarget.className += " active";
			}
		
			function applyLampFilter() {
				const checkboxes = document.querySelectorAll('.lamp-checkbox');
				const checkedLamps = Array.from(checkboxes)
										  .filter(cb => cb.checked)
										  .map(cb => cb.value);

				const rows = document.querySelectorAll('tr.chart-row');
				
				rows.forEach(row => {
					const lampCell = row.cells[3]; 
					const lampValue = lampCell.getAttribute('data-value');
					if (checkedLamps.includes(lampValue)) {
						row.style.display = ""; 
					} else {
						row.style.display = "none"; 
					}
				});
			}

			// 全選択/全解除ボタン
			function toggleLampAll(checked) {
				const checkboxes = document.querySelectorAll('.lamp-checkbox');
				checkboxes.forEach(cb => cb.checked = checked);
				applyLampFilter();
			}

			var sortState = {};

			function extractNumber(str) {
				if (!str) return NaN;
				var cleaned = str.replace(/[^-0-9.]/g, '');
				var num = parseFloat(cleaned);
				return num;
			}

			function sortTable(col, type) {
				var table = document.getElementById("table");
				var tbody = table.tBodies[0];
				var rows = Array.from(tbody.rows);

				var dir = 'asc';
				if (sortState && sortState.col === col && sortState.dir === 'asc') {
					dir = 'desc';
				}
				sortState = { col: col, dir: dir };

				function getSortValue(row) {
					var val = row.cells[col].getAttribute("data-value");
					var isEmpty = (val === null || val === undefined || val.trim() === "");
					if (type === 'number' || type === 'smart-number') {
						var num;
						if (isEmpty) {
							num = NaN;
						} else if (type === 'smart-number') {
							num = extractNumber(val);
						} else {
							num = parseFloat(val);
						}
						if (isNaN(num)) {
							return dir === 'asc' ? Number.MAX_VALUE : -Number.MAX_VALUE;
						}
						return num;
					} else {
						if (isEmpty) return dir === 'asc' ? "\uFFFF" : ""; // 文字列のソートで最後尾に行くような文字
						return val.toLowerCase();
					}
				}

				// ソート実行
				rows.sort(function(a, b) {
					var valA = getSortValue(a);
					var valB = getSortValue(b);

					if (valA < valB) return dir === 'asc' ? -1 : 1;
					if (valA > valB) return dir === 'asc' ? 1 : -1;
					return 0;
				});

				tbody.append(...rows);
			}
		</script>
	</body>
	</html>
	"""

	with open(filename_top100, "w", encoding="utf-8") as f:
		f.write(html_content)
	print(f"ファイルを作成しました: {filename_top100}")
	return

def generate_html_table(
	score_list: List[dict],
	song_list: List[dict],
	mode_slst: str,
	average_list: List[float],
	estimated_theta: float,
	filename_table: str
):
	dictLamp = {
		"FullCombo": 8,
		"ExHard": 7,
		"Hard": 6,
		"Clear": 5,
		"Easy": 4,
		"L-Assist": 3,
		"Assist": 2,
		"Failed": 1,
		"No Play": "NaN",
		"": "NaN"
	}

	level_list = set()
	for song in song_list:
		#if song["display_level"][:2] != mode_slst:
		#	continue
		level = song["display_level"]
		level_list.add(level)

	level_list = list(level_list)
	level_list.sort(key=lambda x:(x[:2],int(x[2:])))
	
	tabs = level_list
	lamp_order = ["No Play", "Failed", "Assist", "L-Assist", "Easy", "Clear", "Hard", "ExHard", "FullCombo"]
	lamp_filter_html = ""
	
	for lamp in lamp_order:
		safe_lamp = html.escape(lamp)
		lamp_filter_html += f"""
			<label class="filter-item">
				<input type="checkbox" class="lamp-checkbox" value="{dictLamp[safe_lamp]}" checked onchange="applyLampFilter()">
				{safe_lamp}
			</label>
		"""
	# Prefix Template
	html_content = f"""
	<!DOCTYPE html>
	<html lang="ja">
	<head>
		<meta charset="UTF-8">
		<title>Shobon Stella Recommend</title>
		<style>
			body {{ font-family: sans-serif; background-color: #222; color: #eee; padding: 20px; }}
			
			.header-container {{
				display: flex;
				justify-content: space-between; /* タイトルは左、ボタンは右に配置 */
				align_items: center;
				margin-bottom: 20px;
				border-bottom: 1px solid #444;
				padding-bottom: 10px;
			}}
			.nav-btn {{
				background-color: #004488;
				color: white;
				padding: 10px 20px;
				text-decoration: none;
				border-radius: 5px;
				font-weight: bold;
				transition: background-color 0.3s;
				box-shadow: 0 2px 4px rgba(0,0,0,0.3);
			}}
			.nav-btn:hover {{
				background-color: #003366;
				transform: translateY(-1px);
			}}			

			/* --- ランプフィルタエリア --- */
			.filter-container {{
				background-color: #333; padding: 10px 15px; border-radius: 5px;
				margin-bottom: 15px; border: 1px solid #444;
			}}
			.filter-label {{ font-weight: bold; margin-right: 10px; font-size: 0.9em; color: #aaa; }}
			.filter-item {{ 
				display: inline-block; margin-right: 15px; cursor: pointer; user-select: none; font-weight: bold;
			}}
			.filter-buttons {{ margin-top: 5px; }}
			.filter-buttons button {{
				font-size: 0.8em; padding: 2px 8px; margin-right: 5px; cursor: pointer;
				background: #555; color: #fff; border: 1px solid #666; border-radius: 3px;
			}}
			.filter-buttons button:hover {{ background: #666; }}

			/* タブ部分 */
			.tab {{ overflow: hidden; border: 1px solid #444; background-color: #333; border-radius: 5px 5px 0 0; }}
			.tab button {{
				background-color: inherit; float: left; border: none; outline: none;
				cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold;
			}}
			.tab button:hover {{ background-color: #555; }}
			.tab button.active {{ background-color: #007bff; color: white; }}

			/* タブコンテンツ */
			.tabcontent {{
				display: none; padding: 6px 12px; border: 1px solid #444; border-top: none;
			}}
			
			/* テーブル装飾 */
			table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
			th, td {{ padding: 10px; border-bottom: 1px solid #444; text-align: left; }}
			th {{ background-color: #333; cursor: pointer; user-select: none; }}
			th:hover {{ background-color: #555; }}
			th::after {{ content: ' ⇅'; font-size: 0.8em; color: #888; }}
			
			/* ランプの色分け */
			.lamp-fc {{ color: #55ffff; font-weight: bold; text-shadow: 0 0 5px #55ffff; }}
			.lamp-exh {{ color: #ffff55; font-weight: bold; text-shadow: 0 0 5px #ffff55; }}
			.lamp-hard {{ color: #ff5555; font-weight: bold; text-shadow: 0 0 5px #ff5555; }}
			.lamp-clear {{ color: #ffbb55; font-weight: bold; text-shadow: 0 0 5px #ffbb55; }}
			.lamp-easy {{ color: #55ff55; font-weight: bold; text-shadow: 0 0 5px #55ff55; }}
			.lamp-assist {{ color: #ff55ff; font-weight: bold; text-shadow: 0 0 5px #ff55ff; }}
			.lamp-failed {{ color: #cccccc; }}
			.lamp-noplay {{ color: #666666; }}

			a{{
				text-decoration: none;
				color: #eee;
			}}
		</style>
	</head>
	<body>
		<h1>Shobon Stella Recommend</h1>
		<a href="result_top100.html" class="nav-btn">TOP100 ページへ ➜</a>
		<h2>あなたの推定実力: <font color="#55ffff">{mode_slst}{beta_to_stella(average_list, estimated_theta):.2f}</font></h2>

		<div class="filter-container">
			<div style="margin-bottom:5px;">
				<span class="filter-label">Now Lamp:</span>
				{lamp_filter_html}
			</div>
			<div class="filter-buttons">
				<button onclick="toggleLampAll(true)">全選択</button>
				<button onclick="toggleLampAll(false)">全解除</button>
			</div>
		</div>
		<div class="tab">
	"""

	sha256_dict = dict()
	for score in score_list:
		sha256_dict[score["sha256"]] = score

	for i, level in enumerate(tabs):
		active_class = " active" if i == 0 else ""
		html_content += f"""
			<button class="tablinks{active_class}" onclick="openTab(event, 'tab-content-{i}')">{html.escape(level)}</button>
		"""

	html_content += "</div>"

	for i, level in enumerate(tabs):
		if level == "ALL":
			target_song = song_list
		else:
			target_song = [c for c in song_list if c["display_level"] == level]
		
		display_style = "display: block;" if i == 0 else ""
		
		html_content += f"""
		<div id="tab-content-{i}" class="tabcontent" style="{display_style}">
			<table id="table-{i}">
				<thead>
					<tr>
						<th onclick="sortTable({i}, 0, 'text')">タイトル</th>
						<th onclick="sortTable({i}, 1, 'smart-number')">推定(E)</th>
						<th onclick="sortTable({i}, 2, 'smart-number')">推定(H)</th>
						<th onclick="sortTable({i}, 3, 'number')">地力度</th>
						<th onclick="sortTable({i}, 4, 'number')">ランプ</th>
						<th onclick="sortTable({i}, 5, 'number')">最小BP</th>
						<th onclick="sortTable({i}, 6, 'smart-number')">スコア</th>
						<th onclick="sortTable({i}, 7, 'number')">次の目標</th>
						<th onclick="sortTable({i}, 8, 'smart-number')">達成確率</th>
					</tr>
				</thead>

				<tbody>
		"""

		for song in target_song:
			#if song["display_level"][:2] != mode_slst:
			#	continue
			#level = int(song["display_level"][2:])

			ret_title = song["title"]
			if len(ret_title) >= 50:
				ret_title = ret_title[:47]+'...'
			ret_title = html.escape(ret_title)
			ret_sha256 = song["sha256"]

			ret_lvec = f"{mode_slst}{beta_to_stella(average_list, float(song['beta_easy'])):.2f}"
			ret_lvhc = f"{mode_slst}{beta_to_stella(average_list, float(song['beta_hard'])):.2f}"

			ret_jiriki = f"{float(song['alpha'])/2:.2f}"
			ret_lamp = "No Play"
			ret_minbp = ""
			ret_score = ""
			ret_nextlamp = "Easy"
			ret_prob = f"{prob_grm(estimated_theta, float(song['beta_easy']), float(song['alpha'])) * 100:.2f} %"

			if song["sha256"] in sha256_dict:
				ret_lamp = get_detailed_clear_type(int(sha256_dict[song["sha256"]]["clear"]))
				ret_minbp = sha256_dict[song["sha256"]]["minbp"]
				ret_score = f"{float(sha256_dict[song['sha256']]['score_rate'])*100:.2f} %"
				ret_nextlamp = get_next_clear_type(int(sha256_dict[song["sha256"]]["clear"]))
				if ret_nextlamp == "Easy":
					ret_prob = f"{prob_grm(estimated_theta, float(song['beta_easy']), float(song['alpha'])) * 100:.2f} %"
				elif ret_nextlamp == "Hard":
					ret_prob = f"{prob_grm(estimated_theta, float(song['beta_hard']), float(song['alpha'])) * 100:.2f} %"
				else:
					ret_prob = ""

			ret_colorclass = ""
			ret_nextcolorclass = ""

			if ret_lamp == "No Play":
				ret_colorclass = "lamp-noplay"
			elif ret_lamp == "Failed":
				ret_colorclass = "lamp-failed"
			elif ret_lamp == "Assist":
				ret_colorclass = "lamp-assist"
			elif ret_lamp == "L-Assist":
				ret_colorclass = "lamp-assist"
			elif ret_lamp == "Easy":
				ret_colorclass = "lamp-easy"
			elif ret_lamp == "Clear":
				ret_colorclass = "lamp-clear"
			elif ret_lamp == "Hard":
				ret_colorclass = "lamp-hard"
			elif ret_lamp == "ExHard":
				ret_colorclass = "lamp-exh"
			elif ret_lamp == "FullCombo":
				ret_colorclass = "lamp-fc"

			if ret_nextlamp == "Easy":
				ret_nextcolorclass = "lamp-easy"
			elif ret_nextlamp == "Hard":
				ret_nextcolorclass = "lamp-hard"

			ret_lampnum = dictLamp[ret_lamp]		
			ret_nextlampnum = dictLamp[ret_nextlamp]		
			
			html_content += f"""
					<tr class="chart-row">
						<td data-value="{ret_title}">
							<a href="https://mocha-repository.info/song.php?sha256={ret_sha256}">{ret_title}</a>
						</td>
						<td data-value="{ret_lvec}">
							{ret_lvec}
						</td>
						<td data-value="{ret_lvhc}">
							{ret_lvhc}
						</td>
						<td data-value="{ret_jiriki}">
							{ret_jiriki}
						</td>
						<td data-value="{ret_lampnum}" class="{ret_colorclass}">
							{ret_lamp}
						</td>
						<td data-value="{ret_minbp}">
							{ret_minbp}
						</td>
						<td data-value="{ret_score}">
							{ret_score}
						</td>
						<td data-value="{ret_nextlampnum}">
							{ret_nextlamp}
						</td>
						<td data-value="{ret_prob}">
							{ret_prob}
						</td>
					</tr>
			"""
		html_content += "</tbody></table></div>"

	# Suffix Template
	html_content += """
		<script>
			function openTab(evt, tabId) {
				var i, tabcontent, tablinks;
				tabcontent = document.getElementsByClassName("tabcontent");
				for (i = 0; i < tabcontent.length; i++) {
					tabcontent[i].style.display = "none";
				}
				tablinks = document.getElementsByClassName("tablinks");
				for (i = 0; i < tablinks.length; i++) {
					tablinks[i].className = tablinks[i].className.replace(" active", "");
				}
				document.getElementById(tabId).style.display = "block";
				evt.currentTarget.className += " active";
			}
		
			function applyLampFilter() {
				const checkboxes = document.querySelectorAll('.lamp-checkbox');
				const checkedLamps = Array.from(checkboxes)
										  .filter(cb => cb.checked)
										  .map(cb => cb.value);

				const rows = document.querySelectorAll('tr.chart-row');
				
				rows.forEach(row => {
					const lampCell = row.cells[4]; 
					const lampValue = lampCell.getAttribute('data-value');
					if (checkedLamps.includes(lampValue)) {
						row.style.display = ""; 
					} else {
						row.style.display = "none"; 
					}
				});
			}

			// 全選択/全解除ボタン
			function toggleLampAll(checked) {
				const checkboxes = document.querySelectorAll('.lamp-checkbox');
				checkboxes.forEach(cb => cb.checked = checked);
				applyLampFilter();
			}

			var sortState = {};

			function extractNumber(str) {
				if (!str) return NaN;
				var cleaned = str.replace(/[^-0-9.]/g, '');
				var num = parseFloat(cleaned);
				return num;
			}

			function sortTable(tableIndex, col, type) {
				var table = document.getElementById("table-" + tableIndex);
				var tbody = table.tBodies[0];
				var rows = Array.from(tbody.rows);

				var dir = 'asc';
				if (sortState[tableIndex] && sortState[tableIndex].col === col && sortState[tableIndex].dir === 'asc') {
					dir = 'desc';
				}
				sortState[tableIndex] = { col: col, dir: dir };

				function getSortValue(row) {
					var val = row.cells[col].getAttribute("data-value");
					var isEmpty = (val === null || val === undefined || val.trim() === "");
					if (type === 'number' || type === 'smart-number') {
						var num;
						if (isEmpty) {
							num = NaN;
						} else if (type === 'smart-number') {
							num = extractNumber(val);
						} else {
							num = parseFloat(val);
						}
						if (isNaN(num)) {
							return dir === 'asc' ? Number.MAX_VALUE : -Number.MAX_VALUE;
						}
						return num;
					} else {
						if (isEmpty) return dir === 'asc' ? "\uFFFF" : ""; // 文字列のソートで最後尾に行くような文字
						return val.toLowerCase();
					}
				}

				// ソート実行
				rows.sort(function(a, b) {
					var valA = getSortValue(a);
					var valB = getSortValue(b);

					if (valA < valB) return dir === 'asc' ? -1 : 1;
					if (valA > valB) return dir === 'asc' ? 1 : -1;
					return 0;
				});

				tbody.append(...rows);
			}
		</script>
	</body>
	</html>
	"""

	with open(filename_table, "w", encoding="utf-8") as f:
		f.write(html_content)
	print(f"ファイルを作成しました: {filename_table}")
	return


def generate_html(
	score_list: List[dict],
	song_list: List[dict],
	mode_slst: str,
	filename_table = "result_table.html",
	filename_top100 = "result_top100.html"
):

	average_list = get_average_list(song_list, mode_slst)

	result = max_likelihood_estimation(score_list, song_list)
	if not result.success:
		raise RuntimeError(f"最尤推定に失敗しました. {result.message}")

	estimated_theta = result.x
	print(f"Estimated: {mode_slst}{beta_to_stella(average_list, estimated_theta):.2f}")

	generate_html_table(score_list, song_list, mode_slst, average_list, estimated_theta, filename_table)
	generate_html_top100(score_list, song_list, mode_slst, average_list, estimated_theta, filename_top100)

class BMSApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Shobon Stella Recommend v1.1")
		self.root.geometry("600x450")
		self.config_file = "config.json"

		# --- DB選択 ---
		tk.Label(root, text="1. score.db:").pack(anchor="w", padx=10, pady=(10, 0))
		self.frame_db = tk.Frame(root)
		self.frame_db.pack(fill="x", padx=10)
		self.entry_db = tk.Entry(self.frame_db)
		self.entry_db.pack(side="left", fill="x", expand=True)
		tk.Button(self.frame_db, text="Browse", command=self.browse_db).pack(side="right", padx=5)
		
		# --- CSV選択 ---
		tk.Label(root, text="2. sl_mocha.csv:").pack(anchor="w", padx=10, pady=(10, 0))
		tk.Label(root, text="このソフトの data フォルダの中に入っています", font=("", 8), fg="gray").pack(anchor="w", padx=10)
		self.frame_csv = tk.Frame(root)
		self.frame_csv.pack(fill="x", padx=10)
		self.entry_csv = tk.Entry(self.frame_csv)
		self.entry_csv.pack(side="left", fill="x", expand=True)
		tk.Button(self.frame_csv, text="Browse", command=self.browse_csv).pack(side="right", padx=5)

		tk.Button(root, text="実行", command=self.run_process, bg="#ddddff", height=2).pack(pady=20, fill="x", padx=50)
		
		# --- ログ出力エリア ---
		tk.Label(root, text="Log:").pack(anchor="w", padx=10)
		self.log_area = scrolledtext.ScrolledText(root, height=10)
		self.log_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))

		self.load_config()

	def log(self, message):
		self.log_area.insert(tk.END, message + "\n")
		self.log_area.see(tk.END)
		self.root.update()

	def load_config(self):
		if os.path.exists(self.config_file):
			try:
				with open(self.config_file, "r", encoding="utf-8") as f:
					config = json.load(f)
					db_path = config.get("db_path", "")
					csv_path = config.get("csv_path", "")
					if db_path: self.entry_db.insert(0, db_path)
					if csv_path: self.entry_csv.insert(0, csv_path)
					self.log("設定ファイルを読み込みました。")
			except:
				self.log("設定ファイルの読み込みに失敗しました。")
	
	def save_config(self):
		config = {
			"db_path": self.entry_db.get(),
			"csv_path": self.entry_csv.get()
		}
		try:
			with open(self.config_file, "w", encoding="utf-8") as f:
				json.dump(config, f)
			self.log("設定ファイルを保存しました。")
		except:
			pass
	
	def browse_db(self):
			filename = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db"), ("All Files", "*.*")])
			if filename:
				self.entry_db.delete(0, tk.END)
				self.entry_db.insert(0, filename)
				
	def browse_csv(self):
		filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
		if filename:
			self.entry_csv.delete(0, tk.END)
			self.entry_csv.insert(0, filename)
	
	def run_process(self):
		mode_slst = 'sl'
		score_dir = self.entry_db.get()
		song_dir = self.entry_csv.get()

		if not os.path.exists(score_dir) or not os.path.exists(song_dir):
			messagebox.showerror("Error", "ファイルが見つかりません。パスを確認してください。")
			return
		
		self.save_config()
		
		try:
			self.log("--- 処理開始 ---")

			self.log("DBを読み込んでいます...")
			score_list = get_score_list(score_dir)
			self.log(f"DB読み込み完了: {len(score_list)} 件のスコアデータ")

			self.log("CSVを解析しています...")
			song_list = get_song_list(song_dir)
			self.log(f"CSV解析完了: 全 {len(song_list)} 曲")

			generate_html(score_list, song_list, mode_slst)
			
			self.log(f"完了！")
			messagebox.showinfo("Success", f"HTMLを作成しました！")
			
			import webbrowser
			webbrowser.open("result_table.html")

			self.root.destroy()

		except Exception as e:
			import traceback
			err_msg = traceback.format_exc()
			self.log("エラーが発生しました:\n" + err_msg)
			messagebox.showerror("Error", f"エラーが発生しました:\n{e}")
if __name__ == "__main__":
	root = tk.Tk()
	app = BMSApp(root)
	root.mainloop()