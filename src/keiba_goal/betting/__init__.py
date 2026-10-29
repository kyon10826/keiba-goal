"""目標到達型 (Goal-reaching) の投票 (design §9)。ネットワーク・ファイル I/O から分離した純粋関数が中心。

- theory: o*_k・P*_k・bold stake・到達確率の上限
- joint: 1 着は p_win、2〜m 着は別の強さの Plackett–Luce という 2 段モデルの順序表と P_hit
- allocate: どれが当たっても目標に届く 100 円単位の配分
- basket: 候補・除外・探索・ゲート、表示前の安全ゲート
- settle: 精算 (返還 → 払戻表 → 0)
- session: イベントソース型の 1 日のセッション (追記のみの events.jsonl)
"""
