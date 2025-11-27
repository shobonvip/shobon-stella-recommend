"""

Shobon Stella Recommend v1
by Shobon

"""

import sqlite3
import csv
import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple

# score のデータから必要な情報を取得
def refine_score_data(score: dict) -> dict:
	ret = dict()
	ret['sha256'] = score['sha256']
	ret['clear'] = score['clear']
	ret['score'] = score['epg'] * 2 + score['lpg'] * 2 + score['egr'] + score['lgr']
	ret['minbp'] = score['minbp']
	return ret

# score.db から情報を取得
def get_score_list(directory: str) -> List[dict]:
	with sqlite3.connect(directory) as con:
		con.row_factory = sqlite3.Row
		cur = con.cursor()
		cur.execute("SELECT * FROM score")
		score_tables = cur.fetchall()
		score_list = [refine_score_data(dict(score_row)) for score_row in score_tables]
		return score_list

# mocha_sl/st.csv から情報を取得 
def get_song_list(directory: str) -> List[dict]:
	with open(directory) as file:
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

# 各難易度の平均 beta を計算する
def get_average_list(song_list: List[dict], filter_slst: str) -> List[float]:
	sum_beta = []
	nums = []
	for song in song_list:
		if song["display_level"][:2] != filter_slst:
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
	return ret

mode_slst = 'sl'
score_dir = "data/score.db"
score_list = get_score_list(score_dir)
song_dir = "data/sl_mocha.csv"
song_list = get_song_list(song_dir)
average_list = get_average_list(song_list, mode_slst)

#print(average_list)

result = max_likelihood_estimation(score_list, song_list)
if not result.success:
	raise RuntimeError(f"最尤推定に失敗しました. {result.message}")

estimated_theta = result.x
print(f"Estimated: {mode_slst}{beta_to_stella(average_list, estimated_theta):.2f}")

pp_list = get_sorted_pp_data(average_list, score_list, song_list, 100)
for num, song in enumerate(pp_list):
	print(f"{num+1:>3}: {song['pp']:>4.0f}pp {song['clear']} {mode_slst}{beta_to_stella(average_list,song['beta']):>5.2f} {song['title']}")