Shobon Stella Recommend v1.1
s#vip

現時点では satellite しか対応してません
beatoraja の score.db と sl_mocha.csv (デフォルトで選ばれているものがおすすめ) を選ぶと推定難易度表と自分の top 100 を作ってくれます

st_mocha.csv　を選ぶとバグるのでご注意！

選ぶ score.db は多分壊れることはないと思いますが (beatoraja で譜面クリアなどファイル更新が行われる瞬間と同時にやってしまうと良くない現象が起きるかも) できればバックアップしてください

main.exe と main.py は全く同じですが main.exe は pythonの環境がなくても実行できます

2025/11/28 v1
2025/11/29 v1.1
- 重複スコアデータの処理を追加 (複数あった場合, score/minbp/クリアランプのそれぞれについて、複数データの中で最大/最小/最良のものが採用されます)