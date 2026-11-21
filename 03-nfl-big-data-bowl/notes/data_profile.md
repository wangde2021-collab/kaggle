# NFL Big Data Bowl 2027 — 数据画像

生成时间：2026-11-21（北京时间）；工具：本地云端 VM（pandas 2.1.4 / numpy 1.26.4），大文件分块读取。
数据来源：Kaggle CLI 下载 nfl-big-data-bowl-2027，9 个 CSV，共 2.31 GB（压缩包 489 MB）。
**行数核对异常（2026-11-21 复核）**：player_play.csv 用 csv 模块按引号解析为 314,197 条记录，但物理行数 wc -l 为 327,020 行——多出的 12,823 行是 play_description 字段内嵌的换行符造成的；Kaggle Data 页标注的 N=316,338 与实测 314,197 相差 2,141，原因不明（页面数字可能有误或口径不同），以实测 314,197 为准。

## players.csv

- 行数：510（不含表头）；列数：10；文件大小：0.05 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 样例/备注 |
|---|---|---|---|---|---|
| nfl_id | int64 | 0 | 0.0% | 510 | 55867 |
| display_name | object | 0 | 0.0% | 508 | Will Anderson |
| draft_year | int64 | 0 | 0.0% | 3 | 2023 |
| nfl_position | object | 0 | 0.0% | 17 | DE |
| birth_date | object | 0 | 0.0% | 441 | 2001-09-02 |
| college_name | object | 0 | 0.0% | 112 | Alabama |
| college_conference | object | 0 | 0.0% | 21 | Southeastern Conference |
| draft_round | float64 | 127 | 24.9% | 7 | 1.0 |
| draft_pick_within_round | float64 | 127 | 24.9% | 42 | 3.0 |
| draft_overall_pick | float64 | 127 | 24.9% | 222 | 3.0 |

样例行（前 3 行）：
```
   nfl_id    display_name  draft_year nfl_position  birth_date college_name       college_conference  draft_round  draft_pick_within_round  draft_overall_pick
0   55867   Will Anderson        2023           DE  2001-09-02      Alabama  Southeastern Conference          1.0                      3.0                 3.0
1   55870   Paris Johnson        2023            T  2001-07-31   Ohio State       Big Ten Conference          1.0                      6.0                 6.0
2   55874  Darnell Wright        2023            T  2001-08-10    Tennessee  Southeastern Conference          1.0                     10.0                10.0
```

## combine_results.csv

- 行数：510（不含表头）；列数：18；文件大小：0.04 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 样例/备注 |
|---|---|---|---|---|---|
| draft_year | int64 | 0 | 0.0% | 3 | 2023 |
| nfl_id | int64 | 0 | 0.0% | 510 | 56040 |
| combine_position | object | 0 | 0.0% | 5 | TE |
| combine_height | float64 | 0 | 0.0% | 95 | 77.875 |
| combine_weight | int64 | 0 | 0.0% | 149 | 245 |
| hand_size | float64 | 0 | 0.0% | 27 | 10.0 |
| arm_length | float64 | 0 | 0.0% | 60 | 32.25 |
| wing_span | float64 | 0 | 0.0% | 114 | 79.75 |
| ten_yd_split | float64 | 85 | 16.7% | 46 | 1.6 |
| forty | float64 | 86 | 16.9% | 106 | 4.84 |
| vertical | float64 | 72 | 14.1% | 41 | 38.5 |
| broad_jump | float64 | 87 | 17.1% | 42 | 125.0 |
| three_cone | float64 | 330 | 64.7% | 95 | 6.87 |
| short_shuttle | float64 | 309 | 60.6% | 88 | 4.12 |
| bench_reps | float64 | 318 | 62.4% | 25 | 23.0 |
| ngs_athleticism_score | int64 | 0 | 0.0% | 48 | 60 |
| ngs_college_production_score | float64 | 1 | 0.2% | 44 | 77.0 |
| ngs_final_score | float64 | 1 | 0.2% | 42 | 74.0 |

样例行（前 3 行）：
```
   draft_year  nfl_id combine_position  combine_height  combine_weight  hand_size  arm_length  wing_span  ten_yd_split  forty  vertical  broad_jump  three_cone  short_shuttle  bench_reps  ngs_athleticism_score  ngs_college_production_score  ngs_final_score
0        2023   56040               TE          77.875             245      10.00      32.250     79.750          1.60   4.84      38.5       125.0         NaN            NaN         NaN                     60                           NaN              NaN
1        2023   56010               DB          70.625             191       8.75      29.625     72.250          1.58   4.59      33.5       116.0         NaN            NaN        23.0                     60                          77.0             74.0
2        2023   56085               TE          79.375             255      10.25      34.000     83.125          1.57   4.55      40.0       128.0        6.87           4.12        23.0                     97                          65.0             79.0
```

## games.csv

- 行数：1,002（不含表头）；列数：7；文件大小：0.04 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 样例/备注 |
|---|---|---|---|---|---|
| game_id | int64 | 0 | 0.0% | 1,002 | 2023090700 |
| game_key | int64 | 0 | 0.0% | 1,002 | 59173 |
| season | int64 | 0 | 0.0% | 3 | 2023 |
| season_type | object | 0 | 0.0% | 3 | REG |
| week | int64 | 0 | 0.0% | 23 | 1 |
| home_team_abbr | object | 0 | 0.0% | 33 | KC |
| visitor_team_abbr | object | 0 | 0.0% | 33 | DET |

样例行（前 3 行）：
```
      game_id  game_key  season season_type  week home_team_abbr visitor_team_abbr
0  2023090700     59173    2023         REG     1             KC               DET
1  2023091000     59174    2023         REG     1            ATL               CAR
2  2023091001     59175    2023         REG     1            BAL               HOU
```

## player_career_successes.csv

- 行数：510（不含表头）；列数：9；文件大小：0.01 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 样例/备注 |
|---|---|---|---|---|---|
| nfl_id | int64 | 0 | 0.0% | 510 | 55867 |
| career_offensive_snaps | int64 | 0 | 0.0% | 195 | 0 |
| career_defensive_snaps | int64 | 0 | 0.0% | 183 | 1884 |
| career_special_teams_snaps | int64 | 0 | 0.0% | 224 | 92 |
| career_games_active | int64 | 0 | 0.0% | 51 | 51 |
| career_games_started | int64 | 0 | 0.0% | 46 | 44 |
| ap_all_pro_1st_team | int64 | 0 | 0.0% | 2 | 1 |
| ap_all_pro_2nd_team | int64 | 0 | 0.0% | 2 | 0 |
| pro_bowl_original_ballot | int64 | 0 | 0.0% | 3 | 1 |

样例行（前 3 行）：
```
   nfl_id  career_offensive_snaps  career_defensive_snaps  career_special_teams_snaps  career_games_active  career_games_started  ap_all_pro_1st_team  ap_all_pro_2nd_team  pro_bowl_original_ballot
0   55867                       0                    1884                          92                   51                    44                    1                    0                         1
1   55870                    2802                       0                         169                   47                    43                    0                    0                         0
2   55874                    3222                       0                         167                   51                    49                    0                    1                         0
```

## player_play.csv

- 行数：314,197（不含表头）；列数：64；文件大小：140.07 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 均值 | 最小值 | 最大值 |
|---|---|---|---|---|---|---|---|
| game_id | int64 | 0 | 0.0% | 1,001 | 2024519475.1424 | 2023080351.0000 | 2026020800.0000 |
| play_id | int64 | 0 | 0.0% | 4,931 | 2182.1633 | 54.0000 | 5463.0000 |
| team_abbr | object | 0 | 0.0% | 33 | — | — | — |
| nfl_id | int64 | 0 | 0.0% | 497 | 56831.3935 | 55867.0000 | 59278.0000 |
| lined_up_position | object | 1,383 | 0.4% | 17 | — | — | — |
| play_description | object | 0 | 0.0% | 114,899 | — | — | — |
| quarter | int64 | 0 | 0.0% | 5 | 2.5767 | 1.0000 | 5.0000 |
| down | int64 | 0 | 0.0% | 5 | 1.8551 | 0.0000 | 4.0000 |
| yards_to_go | int64 | 0 | 0.0% | 40 | 8.5374 | 0.0000 | 40.0000 |
| possession_team | object | 0 | 0.0% | 33 | — | — | — |
| yardline_side | object | 4,102 | 1.3% | 33 | — | — | — |
| yardline_number | int64 | 0 | 0.0% | 50 | 29.9788 | 1.0000 | 50.0000 |
| game_clock | object | 0 | 0.0% | 901 | — | — | — |
| pre_snap_home_score | int64 | 0 | 0.0% | 52 | 11.3678 | 0.0000 | 70.0000 |
| pre_snap_visitor_score | int64 | 0 | 0.0% | 46 | 10.5364 | 0.0000 | 51.0000 |
| play_direction | object | 50 | 0.0% | 2 | — | — | — |
| pre_snap_home_team_win_probability | float64 | 3 | 0.0% | 113,328 | 0.5372 | 0.0008 | 1.0000 |
| pre_snap_visitor_team_win_probability | float64 | 3 | 0.0% | 113,328 | 0.4628 | 0.0000 | 0.9992 |
| home_team_win_probability_added | float64 | 6 | 0.0% | 114,047 | -0.0001 | -0.9929 | 0.9931 |
| visitor_team_win_probility_added | float64 | 6 | 0.0% | 114,047 | 0.0001 | -0.9931 | 0.9929 |
| expected_points_added | float64 | 0 | 0.0% | 112,620 | -0.0530 | -12.8964 | 8.9688 |
| expected_points | float64 | 0 | 0.0% | 104,329 | 2.3143 | -2.9000 | 6.6567 |
| offense_formation | object | 5,335 | 1.7% | 7 | — | — | — |
| receiver_alignment | object | 4,597 | 1.5% | 15 | — | — | — |
| pass_result | object | 103,636 | 33.0% | 5 | — | — | — |
| pass_length | float64 | 129,458 | 41.2% | 82 | 8.1043 | -70.0000 | 65.0000 |
| play_nullified_by_penalty | object | 0 | 0.0% | 2 | — | — | — |
| penalty_yards | float64 | 293,447 | 93.4% | 72 | 0.6055 | -25.0000 | 52.0000 |
| pre_penalty_yards_gained | int64 | 0 | 0.0% | 120 | 5.3409 | -28.0000 | 98.0000 |
| yards_gained | int64 | 0 | 0.0% | 146 | 5.1830 | -99.0000 | 98.0000 |
| team_coverage_man_zone | object | 38,919 | 12.4% | 2 | — | — | — |
| team_coverage_type | object | 38,919 | 12.4% | 8 | — | — | — |
| in_motion_at_ball_snap | object | 158,231 | 50.4% | 2 | — | — | — |
| target | bool | 0 | 0.0% | 2 | 0.0313 | 0.0000 | 1.0000 |
| rec_yards | float64 | 307,937 | 98.0% | 88 | 12.0096 | -11.0000 | 93.0000 |
| rush_yards | float64 | 314,175 | 100.0% | 12 | 4.5000 | -2.0000 | 15.0000 |
| yards_after_catch | float64 | 307,937 | 98.0% | 62 | 4.6177 | -8.0000 | 76.0000 |
| fumble | float64 | 314,124 | 100.0% | 1 | 1.0000 | 1.0000 | 1.0000 |
| fumble_lost | float64 | 314,124 | 100.0% | 2 | 0.3973 | 0.0000 | 1.0000 |
| route_ran | object | 260,889 | 83.0% | 12 | — | — | — |
| separation_at_pass_forward | float64 | 267,460 | 85.1% | 30,351 | 3.1153 | 0.0922 | 22.7941 |
| cushion | float64 | 210,445 | 67.0% | 1,461 | 6.0006 | 0.5000 | 20.2800 |
| expected_yards_after_catch | float64 | 305,221 | 97.1% | 8,947 | 4.1782 | 0.0100 | 37.0200 |
| dropback_duration | float64 | 253,394 | 80.6% | 5,852 | 3.0072 | 0.2570 | 14.2450 |
| extended_sack_allowed | float64 | 253,394 | 80.6% | 3 | 0.0017 | 0.0000 | 1.0000 |
| pass_rushers_encountered | float64 | 253,394 | 80.6% | 4 | 0.9794 | 0.0000 | 3.0000 |
| peak_pressure_probability_allowed | float64 | 256,169 | 81.5% | 1,566 | 0.2781 | 0.0000 | 1.0000 |
| pressure_allowed | object | 253,394 | 80.6% | 2 | — | — | — |
| sack_allowed | float64 | 253,394 | 80.6% | 3 | 0.0096 | 0.0000 | 1.0000 |
| time_to_pressure_allowed | float64 | 309,227 | 98.4% | 80 | 3.0300 | 0.9000 | 10.7000 |
| quick_pressure | object | 310,434 | 98.8% | 2 | — | — | — |
| unblocked_pressure | object | 227,359 | 72.4% | 2 | — | — | — |
| blitzing | object | 113,470 | 36.1% | 2 | — | — | — |
| player_get_off | float64 | 270,195 | 86.0% | 1,662 | 1.0097 | 0.0040 | 2.0000 |
| sack | float64 | 227,362 | 72.4% | 3 | 0.0057 | 0.0000 | 1.0000 |
| tackle | float64 | 302,301 | 96.2% | 3 | 0.6201 | 0.0000 | 2.0000 |
| assist | float64 | 302,301 | 96.2% | 2 | 0.3797 | 0.0000 | 1.0000 |
| caused_forced_fumble | float64 | 302,301 | 96.2% | 2 | 0.0122 | 0.0000 | 1.0000 |
| recovered_fumble | object | 311,983 | 99.3% | 2 | — | — | — |
| tackle_for_loss | object | 302,301 | 96.2% | 2 | — | — | — |
| time_to_pressure | float64 | 310,393 | 98.8% | 78 | 2.9923 | 0.8000 | 9.6000 |
| time_to_qb_hurry | float64 | 310,318 | 98.8% | 79 | 3.0015 | 0.4000 | 9.6000 |
| coverage_assignment | object | 241,545 | 76.9% | 20 | — | — | — |
| coverage_assignment_at_snap | object | 242,072 | 77.0% | 20 | — | — | — |

样例行（前 3 行）：
```
      game_id  play_id team_abbr  nfl_id lined_up_position                                                                                                         play_description  quarter  down  yards_to_go possession_team yardline_side  yardline_number game_clock  pre_snap_home_score  pre_snap_visitor_score play_direction  pre_snap_home_team_win_probability  pre_snap_visitor_team_win_probability  home_team_win_probability_added  visitor_team_win_probility_added  expected_points_added  expected_points offense_formation receiver_alignment pass_result  pass_length play_nullified_by_penalty  penalty_yards  pre_penalty_yards_gained  yards_gained team_coverage_man_zone team_coverage_type in_motion_at_ball_snap  target  rec_yards  rush_yards  yards_after_catch  fumble  fumble_lost route_ran  separation_at_pass_forward  cushion  expected_yards_after_catch  dropback_duration  extended_sack_allowed  pass_rushers_encountered  peak_pressure_probability_allowed pressure_allowed  sack_allowed  time_to_pressure_allowed quick_pressure unblocked_pressure blitzing  player_get_off  sack  tackle  assist  caused_forced_fumble recovered_fumble tackle_for_loss  time_to_pressure  time_to_qb_hurry coverage_assignment coverage_assignment_at_snap
0  2025111610     2279       BAL   57183                 T                                                       (8:42) (Shotgun) L.Jackson pass incomplete deep left to D.Hopkins.        3     1           10             BAL           BAL               45      08:42                   16                      10          right                            0.488722                               0.511278                         0.030960                         -0.030960              -0.510282         2.716763           SHOTGUN                3x1           I         22.0                         N            NaN                         0             0           MAN_COVERAGE        COVER_1_MAN                  False   False        NaN         NaN                NaN     NaN          NaN       NaN                         NaN      NaN                         NaN              2.079                    0.0                       1.0                           0.153333            False           0.0                       NaN            NaN                NaN    False             NaN   NaN     NaN     NaN                   NaN              NaN             NaN               NaN               NaN                 NaN                         NaN
1  2026011701     1626        SF   58248     INTERIOR_LINE                                        (3:48) (Shotgun) K.Walker left end to SEA 28 for 5 yards (S.Okuayinonu; M.Sigle).        2     2            7             SEA           SEA               23      03:48                   17                       6           left                            0.925716                               0.074284                        -0.002361                          0.002361              -0.048963         0.357381           SHOTGUN                2x1         NaN          NaN                         N            NaN                         5             5          ZONE_COVERAGE       COVER_2_ZONE                    NaN   False        NaN         NaN                NaN     NaN          NaN       NaN                         NaN      NaN                         NaN                NaN                    NaN                       NaN                                NaN              NaN           NaN                       NaN            NaN                NaN      NaN             NaN   NaN     NaN     NaN                   NaN              NaN             NaN               NaN               NaN                 NaN                         NaN
2  2025101900     2838       LAR   57140              EDGE  (7:01) C.Van Lanen reported in as eligible.  T.Lawrence scrambles left tackle pushed ob at LA 30 for 14 yards (K.Curl).        3     2            2             JAX           LAR               44      07:01                    0                      21           left                            0.051319                               0.948681                         0.005685                         -0.005685               0.927307         3.313706        SINGLEBACK                3x1           R          NaN                         N            NaN                        14            14          ZONE_COVERAGE       COVER_3_ZONE                    NaN   False        NaN         NaN                NaN     NaN          NaN       NaN                         NaN      NaN                         NaN                NaN                    NaN                       NaN                                NaN              NaN           NaN                       NaN            NaN              False     True           0.906   0.0     NaN     NaN                   NaN              NaN             NaN               NaN               NaN       NO_ASSIGNMENT               NO_ASSIGNMENT
```

## combine_tracking.csv

- 行数：463,189（不含表头）；列数：14；文件大小：79.91 MB

| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 均值 | 最小值 | 最大值 |
|---|---|---|---|---|---|---|---|
| draft_year | int64 | 0 | 0.0% | 3 | 2024.0932 | 2023.0000 | 2025.0000 |
| event_id | object | 0 | 0.0% | 6,310 | — | — | — |
| nfl_id | int64 | 0 | 0.0% | 510 | 57415.5787 | 55867.0000 | 59278.0000 |
| entity_type | object | 0 | 0.0% | 2 | — | — | — |
| time | object | 0 | 0.0% | 429,596 | — | — | — |
| drill_type | object | 0 | 0.0% | 9 | — | — | — |
| drill_name | object | 0 | 0.0% | 56 | — | — | — |
| attempt | int64 | 0 | 0.0% | 17 | 3.0492 | 1.0000 | 17.0000 |
| x | float64 | 0 | 0.0% | 178,985 | 44.2625 | 1.3000 | 103.9000 |
| y | float64 | 0 | 0.0% | 179,841 | 24.0396 | -11.4598 | 66.2828 |
| s | float64 | 0 | 0.0% | 182,182 | 5.1308 | 0.0000 | 25.7942 |
| a | float64 | 0 | 0.0% | 371,251 | 2.8131 | 0.0000 | 44.6073 |
| dis | float64 | 0 | 0.0% | 179,472 | 0.3529 | 0.0000 | 45.2853 |
| dir | float64 | 0 | 0.0% | 429,809 | 149.6679 | 0.0000 | 359.9997 |

样例行（前 3 行）：
```
   draft_year                          event_id  nfl_id entity_type                     time       drill_type                      drill_name  attempt          x          y         s         a       dis         dir
0        2025  20250302100900_1740941802777_000   58286      PLAYER  2025-03-02T18:56:38.200  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT        4  44.785924  28.285528  0.220828  0.682443  0.011085  250.278576
1        2025  20250302100900_1740941802777_000   58286      PLAYER  2025-03-02T18:56:38.300  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT        4  44.723780  28.297391  0.332931  0.619551  0.003864  257.319206
2        2025  20250302100900_1740941802777_000   58286      PLAYER  2025-03-02T18:56:38.400  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT        4  44.632680  28.279528  0.481838  0.596455  0.008305  253.176813
```

## game_tracking_2023.csv

- 行数：3,665,443（不含表头）；列数：12；文件大小：319.92 MB（分块读取，每块 500,000 行）

| 列名 | 类型 | 缺失数 | 缺失率 | 均值 | 最小值 | 最大值 |
|---|---|---|---|---|---|---|
| game_id | int64 | 0 | 0.0% | 2023180779.2099 | 2023080351.0000 | 2024021100.0000 |
| play_id | int64 | 0 | 0.0% | 2206.5511 | 54.0000 | 5196.0000 |
| nfl_id | int64 | 0 | 0.0% | 55990.4031 | 55867.0000 | 56662.0000 |
| time | object | 0 | 0.0% | — | — | — |
| x | float64 | 0 | 0.0% | 60.1777 | -5.8900 | 126.7300 |
| y | float64 | 0 | 0.0% | 26.7934 | -7.2900 | 61.2400 |
| s | float64 | 0 | 0.0% | 2.5308 | 0.0000 | 12.6700 |
| a | float64 | 0 | 0.0% | 1.6355 | 0.0000 | 18.3300 |
| dis | float64 | 0 | 0.0% | 0.2568 | 0.0000 | 3.4400 |
| o | float64 | 0 | 0.0% | 177.6835 | 0.0000 | 360.0000 |
| dir | float64 | 0 | 0.0% | 179.6210 | 0.0000 | 360.0000 |
| event | object | 3,422,066 | 93.4% | — | — | — |

event 分布（Top 10）：
```
event
NaN                        3422066.0
ball_snap                    50813.0
pass_forward                 31422.0
tackle                       29011.0
first_contact                28345.0
pass_arrived                 23967.0
pass_outcome_caught          18845.0
handoff                      14053.0
pass_outcome_incomplete      10862.0
play_action                   8274.0
```

- nfl_id 唯一值数：156
- game_id 唯一值数：333
- play_id 唯一值数：4,649

样例行（前 5 行）：
```
      game_id  play_id  nfl_id                     time      x      y    s    a  dis       o     dir event
0  2024010707      570   56450  2024-01-07T21:46:07.700  42.17  25.29  0.0  0.0  0.0  290.97  213.26   NaN
1  2024010707      570   56450  2024-01-07T21:46:07.800  42.17  25.29  0.0  0.0  0.0  290.97  210.36   NaN
2  2024010707      570   56450  2024-01-07T21:46:07.900  42.18  25.29  0.0  0.0  0.0  290.97  205.45   NaN
3  2024010707      570   56450  2024-01-07T21:46:08.000  42.18  25.29  0.0  0.0  0.0  290.97  200.73   NaN
4  2024010707      570   56450  2024-01-07T21:46:08.100  42.18  25.29  0.0  0.0  0.0  290.97  199.27   NaN
```

## game_tracking_2024.csv

- 行数：7,827,944（不含表头）；列数：12；文件大小：684.69 MB（分块读取，每块 500,000 行）

| 列名 | 类型 | 缺失数 | 缺失率 | 均值 | 最小值 | 最大值 |
|---|---|---|---|---|---|---|
| game_id | int64 | 0 | 0.0% | 2024188612.6894 | 2024080151.0000 | 2025020900.0000 |
| play_id | int64 | 0 | 0.0% | 2175.2753 | 55.0000 | 5220.0000 |
| nfl_id | int64 | 0 | 0.0% | 56647.9152 | 55867.0000 | 57911.0000 |
| time | object | 0 | 0.0% | — | — | — |
| x | float64 | 0 | 0.0% | 59.6641 | -6.3300 | 125.6600 |
| y | float64 | 0 | 0.0% | 26.7739 | -8.1300 | 63.9600 |
| s | float64 | 0 | 0.0% | 2.5181 | 0.0000 | 13.8700 |
| a | float64 | 0 | 0.0% | 1.5901 | 0.0000 | 22.0200 |
| dis | float64 | 0 | 0.0% | 0.2557 | 0.0000 | 16.8600 |
| o | float64 | 0 | 0.0% | 178.8819 | 0.0000 | 360.0000 |
| dir | float64 | 0 | 0.0% | 180.1098 | 0.0000 | 360.0000 |
| event | object | 7,306,249 | 93.3% | — | — | — |

event 分布（Top 10）：
```
event
NaN                        7306249.0
ball_snap                   108626.0
pass_forward                 64826.0
first_contact                64745.0
tackle                       64072.0
pass_arrived                 48877.0
pass_outcome_caught          39397.0
handoff                      31977.0
pass_outcome_incomplete      22030.0
play_action                  17841.0
```

- nfl_id 唯一值数：318
- game_id 唯一值数：334
- play_id 唯一值数：4,719

样例行（前 5 行）：
```
      game_id  play_id  nfl_id                     time      x      y     s     a   dis      o     dir event
0  2024080952     3379   57367  2024-08-10T01:30:31.800  87.09  22.52  0.28  0.11  0.03  98.51  126.08   NaN
1  2024080952     3379   57367  2024-08-10T01:30:31.900  87.11  22.51  0.24  0.18  0.02  97.33  125.81   NaN
2  2024080952     3379   57367  2024-08-10T01:30:32.000  87.12  22.50  0.17  0.24  0.01  96.65  124.99   NaN
3  2024080952     3379   57367  2024-08-10T01:30:32.100  87.13  22.49  0.15  0.21  0.02  95.42  120.75   NaN
4  2024080952     3379   57367  2024-08-10T01:30:32.200  87.14  22.48  0.11  0.19  0.01  95.42  136.32   NaN
```

## game_tracking_2025.csv

- 行数：11,160,903（不含表头）；列数：12；文件大小：975.20 MB（分块读取，每块 500,000 行）

| 列名 | 类型 | 缺失数 | 缺失率 | 均值 | 最小值 | 最大值 |
|---|---|---|---|---|---|---|
| game_id | int64 | 0 | 0.0% | 2025190196.6021 | 2025073151.0000 | 2026020800.0000 |
| play_id | int64 | 0 | 0.0% | 2168.3552 | 55.0000 | 5463.0000 |
| nfl_id | int64 | 0 | 0.0% | 57230.2658 | 55867.0000 | 59278.0000 |
| time | object | 0 | 0.0% | — | — | — |
| x | float64 | 0 | 0.0% | 59.9876 | -5.8600 | 127.4000 |
| y | float64 | 0 | 0.0% | 26.7379 | -8.7800 | 63.0500 |
| s | float64 | 0 | 0.0% | 2.4712 | 0.0000 | 16.8800 |
| a | float64 | 0 | 0.0% | 1.5532 | -14.3960 | 33.7590 |
| dis | float64 | 0 | 0.0% | 0.2516 | 0.0000 | 4.2000 |
| o | float64 | 0 | 0.0% | 178.9688 | 0.0000 | 360.0000 |
| dir | float64 | 0 | 0.0% | 180.0851 | 0.0000 | 360.0000 |
| event | object | 10,425,238 | 93.4% | — | — | — |

event 分布（Top 10）：
```
event
NaN                        10425238.0
ball_snap                    154839.0
first_contact                 91278.0
pass_forward                  91231.0
tackle                        91186.0
pass_arrived                  68478.0
pass_outcome_caught           54929.0
handoff                       46564.0
pass_outcome_incomplete       31977.0
play_action                   25740.0
```

- nfl_id 唯一值数：448
- game_id 唯一值数：334
- play_id 唯一值数：4,746

样例行（前 5 行）：
```
      game_id  play_id  nfl_id                     time      x      y     s     a   dis       o    dir event
0  2025090701      290   57347  2025-09-07T17:09:39.500  83.06  32.06  0.10  0.06  0.04    8.00  18.86   NaN
1  2025090701      290   57347  2025-09-07T17:09:39.600  83.07  32.07  0.10  0.07  0.02    2.61  21.95   NaN
2  2025090701      290   57347  2025-09-07T17:09:39.700  83.06  32.06  0.07  0.05  0.02  357.42  16.24   NaN
3  2025090701      290   57347  2025-09-07T17:09:39.800  83.07  32.07  0.07  0.05  0.01  352.73  18.93   NaN
4  2025090701      290   57347  2025-09-07T17:09:39.900  83.08  32.07  0.06  0.05  0.01  349.52  20.80   NaN
```

## 关键分布与跨表链接检查

### players.csv
```
draft_year 分布：
draft_year
2023    160
2024    177
2025    173

nfl_position 分布：
nfl_position
WR     106
CB      60
G       51
T       50
DT      47
TE      42
DE      36
OLB     30
SS      28
FS      25
C       20
NT       5
DB       5
ILB      2
RB       1
MLB      1
FB       1

draft_round 缺失（UDFA）数：127 / 510
college 前 10：
college_name
Georgia            22
Alabama            16
Louisiana State    15
Texas              14
South Carolina     14
Ohio State         14
Michigan           12
Miami              11
Florida            11
Mississippi        11
```

### combine_results.csv
```
三项 drill 缺失（opt-out）数：three_cone=330, short_shuttle=309, bench_reps=318 / 510
forty 描述统计：
count    424.000000
mean       4.733514
std        0.309449
min        4.280000
25%        4.470000
50%        4.620000
75%        5.010000
max        5.480000
ngs_final_score 描述统计：
count    509.000000
mean      70.333988
std        7.530414
min       52.000000
25%       65.000000
50%       70.000000
75%       75.000000
max       94.000000
```

### player_career_successes.csv（目标变量候选）
```
             nfl_id  career_offensive_snaps  career_defensive_snaps  career_special_teams_snaps  career_games_active  career_games_started  ap_all_pro_1st_team  ap_all_pro_2nd_team  pro_bowl_original_ballot
count    510.000000              510.000000              510.000000                  510.000000           510.000000            510.000000           510.000000           510.000000                510.000000
mean   57331.319608              327.415686              234.307843                  113.503922            21.721569              7.852941             0.011765             0.003922                  0.027451
std      979.506689              632.400084              492.121894                  156.194148            16.285063             11.466015             0.107931             0.062561                  0.196310
min    55867.000000                0.000000                0.000000                    0.000000             0.000000              0.000000             0.000000             0.000000                  0.000000
25%    56273.500000                0.000000                0.000000                    5.000000             6.250000              0.000000             0.000000             0.000000                  0.000000
50%    57290.000000                0.000000                0.000000                   53.000000            17.000000              2.000000             0.000000             0.000000                  0.000000
75%    58302.750000              361.750000              220.500000                  147.750000            34.000000             12.000000             0.000000             0.000000                  0.000000
max    59278.000000             3303.000000             2666.000000                  957.000000            51.000000             50.000000             1.000000             1.000000                  2.000000

pro_bowl_original_ballot > 0 人数：11
ap_all_pro_1st_team > 0 人数：6
career_games_started == 0 人数：208
career_offensive_snaps + 防守 snaps 均为 0 人数：105
```

### games.csv
```
season 分布：
season
2023    334
2024    334
2025    334
season_type 分布：
season_type
REG     816
PRE     147
POST     39
```

### 跨表 nfl_id 链接检查
```
players nfl_id：510；combine_results nfl_id：510，与 players 交集：510，combine_results 多出：0，players 缺失：0
career_successes nfl_id：510，与 players 交集：510
player_play nfl_id 唯一值：497，与 players 交集：497
combine_tracking nfl_id 唯一值（不含空）：510，与 players 交集：510
player_play 行数：314,197；game_id 唯一值：1001；play_id 唯一值：4931
combine_tracking event_id 唯一值：6310；drill_type 分布：
drill_type
SKILL_DRILLS_WR     120383
SKILL_DRILLS_DB      94623
SKILL_DRILLS_DL      77428
SKILL_DRILLS_OL      61129
FORTY_YARD_DASH      41500
SKILL_DRILLS_TE      37222
THREE_CONE_DRILL     15951
SHORT_SHUTTLE        11690
SKILL_DRILLS_LB       3263
player_play 中 target=True 行数：9842 / 314197
```
