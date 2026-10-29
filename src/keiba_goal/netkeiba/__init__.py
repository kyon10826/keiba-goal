"""netkeiba の公開ページの取得・解析・record_data 形式への変換 (design §3.3, §3.4)。

http (取得・manifest・キャッシュ) → parse_* (ページ → 型付きの表) → validate (ページの「完全」の判定)
→ convert (record 47 列 + 派生列) → crawl (窓の取得) / rebuild (キャッシュからの作り直し・まとめ)。
race_days は窓 1 の開催日の一覧と取りこぼしの検査、files は一時ファイル経由の書き込みと作り直しのロック。
record_data は records (P1 の読み込みをそのまま使う) から読み、course_table・horse_master・gate (合格ゲート) が使う。
post_times は開催日の一覧から発走時刻を集める (rebuild.combine がまとめに含める)。
"""
