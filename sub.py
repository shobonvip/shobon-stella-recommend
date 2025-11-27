import sqlite3
con = sqlite3.connect("data/score.db")
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute("SELECT * FROM score")
score_tables = cur.fetchall()
#col_names = [d[0] for d in cur.description]
#print(col_names)
score_data = [dict(score_row) for score_row in score_tables] 
#for c, v in score_data[512].items():
#	print(f"| {c} | {v} |")