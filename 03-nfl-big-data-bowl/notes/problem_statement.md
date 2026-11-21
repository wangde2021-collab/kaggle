# NFL Big Data Bowl 2027 — 题目原文（全文存档）

来源：Kaggle 比赛页面，2026-11-21 02:06（北京时间）浏览器实读转录。
比赛地址：https://www.kaggle.com/competitions/nfl-big-data-bowl-2027
账号 dewang2025 已于同日接受规则并加入比赛。

> **重要发现：本届不是预测类比赛，没有 Public/Private leaderboard、没有提交文件格式、没有每日提交次数限制。**
> 比赛形式是 **writeup（研究报告）评审制**：提交一份不超过 15 页（含标题、正文、附录）的英文研究报告，由 NFL 评审团按 100 分制打分。
> 截止：**2027-01-07 07:59（GMT+8）**提交 writeup（页面提示 "Your submission is due by Jan 7, 2027 at 7:59 AM GMT+8"）。
> 任务原定的“验证分数、提交文件、五轮跑分”流程需按评审制调整：以统计结果（相关性、R²、显著性、交叉验证）代替 leaderboard 分数；最终交付物是研究报告而非预测 CSV。
> 另：二大爷是在校大学生（教育邮箱已验证），符合 Collegiate Prize（$10,000，含在校大学生的队伍中得分最高者）的参评条件，且可与总奖金兼得。

---

## 1. Overview

主办方：The National Football League · Featured Hackathon · 2 months to go。
标题：NFL Big Data Bowl 2027。
副标题：Uncover non-obvious linkages between Combine sensor tracking and regular-season NFL game performance。

提示：You haven't created a writeup yet. Your submission is due by Jan 7, 2027 at 7:59 AM GMT+8.

Teams of up to 5 members can submit a writeup presenting novel findings, data-driven insights, and models that quantify the relationship between pre-draft athleticism and professional performance. You can include videos, images, and code in your writeup.

### Prizes（奖金总计 $100,000）

- 1st Place: $60,000
- 2nd Place: $20,000
- 3rd Place: $10,000
- A $10,000 prize will be awarded to the top-scoring team among those that include a college student as a member. Teams may be eligible for both the overall prizes and the Collegiate prize.
- Each team (including all members) that makes the top 50 will receive an invite to present their work at the 2027 NFL Scouting Combine. We will cover reasonable travel costs to attend this event.
- Cash prizes will be paid out only once winners are confirmed, verified, and have followed all procedures.

### Timeline（时间线）

- Dec 17, 2025 - Competition begins
- Dec 23, 2025 - Entry deadline
- Jan 7, 2027 - Final submission deadline
- Jan 7, 2027 - Competition ends

The Host will review the entries and, based on the judging criteria, will announce the winners on or about the week of February 22, 2027. However, this date may change at the Host's sole discretion.

Competition Host：The National Football League

Eligibility requirements：To be eligible for the Collegiate prize, teams must include a college student. If selected as a Collegiate winner, teams will be required to present proof of student status at the time of the competition to the Host, in the Host's sole discretion.

### Rules（规则要点）

- Individual competitors are limited to one account and one team; team size is capped at five (5) members. Any competitor on multiple teams or with multiple accounts will be disqualified. Competitors can form teams from any location worldwide, with one (1) registered Kaggle member designated as the Team Captain. Submissions may only be made through Kaggle. For complete information on the rules, please refer to the Rules section of this competition.
- All submissions must be original work created by the team and not previously published.
- Writeups must be made in English.
- The writeup should present novel findings, data-driven insights, and models that quantify the relationship between pre-draft athleticism and professional performance. Submissions should be supported by rigorous analysis, sound methodology, and clear presentation.
- The competition is open to Kaggle users worldwide, except where prohibited by law. Competitors under 18 years old must have permission from a parent or guardian. Competitors must comply with Kaggle's Terms of Use. See the Rules page for full details.
- For more information about the competition, please contact: bigdatabowl@nfl.com

### Evaluation

The following table lists the requirements for each submission component, and the relative contributions to the final score (total of 100 points). A team may not submit a writeup with more than 15 pages, including title, main body, and appendices. A Judges Panel will review each writeup and award points as follows: Four components are assessed:

- Quality of Research (40 points): Submissions should have a clear, well-defined research question, use appropriate datasets, methodologies, and analyses, and communicate results clearly and effectively. Strong submissions will incorporate robust statistical techniques and be based on reliable, reproducible analysis. Submissions should demonstrate a strong understanding of the problem and data, while using rigorous, innovative, and well-supported analytical approaches.
- Innovation (30 points): Creativity and originality in the approach, methodology, and techniques used in the analysis, or in new applications to previously unexplored areas. Innovation can be expressed through novel approaches to the topic, new methods of data analysis, or fresh perspectives on the relationship between athleticism and on-field performance.
- Presentation (20 points): Clarity, organization, and presentation of the research question, research design, and results. Submissions should be well-organized, easy to follow, and clearly present the analytical process and findings.
- Data Visualization (10 points): Effective use of charts, graphs, and other visuals to communicate results and enhance understanding of the key findings and contributions of the research.

Please note: the writeup page limit is 15 pages, including the title, main body, and appendices. Submissions that exceed 15 pages will be disqualified. Submissions must be written in English.

### Tracks and Awards

- The competition offers a total of $100,000 in prizes across four awards: First Place ($60,000), Second Place ($20,000), Third Place ($10,000), and a Collegiate Prize ($10,000). The Collegiate prize will be awarded to the top-scoring team that includes a college student as a member. Teams may be eligible for both the overall prizes and the Collegiate prize.
- Each team (including all members) that makes the top 50 will receive an invite to present their work at the 2027 NFL Scouting Combine. We will cover reasonable travel costs to attend this event.

### Judges Panel（评审团）

- The panel will consist of the following judges: John T. (NFL, Head of Data Science), Jane L. (NFL, Director of Analytics), Chris M. (NFL, Senior Data Scientist), Sarah P. (NFL, Lead Data Scientist), Alex K. (NFL, Data Scientist), David W. (NFL, Manager of Data Science), and Rachel S. (NFL, Data Scientist). (原文如此)

### Citation

- Please cite the NFL Big Data Bowl 2027 dataset as follows: NFL Big Data Bowl 2027. Kaggle. https://www.kaggle.com/competitions/nfl-big-data-bowl-2027
- For more information, please contact bigdatabowl@nfl.com.

---

## 2. Evaluation（单独页面，与 Overview 内 Evaluation 章节核对一致）

该地址（https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview/evaluation）在 Kaggle 上实际打开的是 Overview 页内的 Evaluation 章节，内容与上文 Overview 中的 Evaluation 部分完全相同（已两次读取核对一致）。本届比赛是 writeup（研究报告）评审制，不是预测类比赛，因此页面上没有 Public/Private leaderboard 划分、没有提交文件格式要求、也没有每日提交次数限制。评分细则如下（总分 100 分，writeup 限 15 页，含标题、正文和附录，超页取消资格，须用英文撰写）：

- Quality of Research（研究质量）：40 分。研究问题清晰明确，使用合适的数据集、方法和分析，清晰有效地传达结果；采用稳健的统计技术，分析可靠、可复现；对问题和数据理解深入，分析方法严谨、创新、有充分支撑。
- Innovation（创新性）：30 分。分析方法、技术或应用领域的创造性和原创性；可体现在新颖的研究思路、新的数据分析方法，或对运动能力与场上表现关系的新视角。
- Presentation（呈现）：20 分。研究问题、研究设计和结果的清晰度、组织性和呈现质量；条理清晰、易于跟随，分析过程和发现表达清楚。
- Data Visualization（数据可视化）：10 分。有效使用图表等视觉手段传达结果，增强对关键发现和贡献的理解。

其余内容（奖项、评审团、引用方式）与 Overview 页相同，见上。

---

## 3. Data

提示：You haven't created a writeup yet. Your submission is due by Jan 7, 2027 at 7:59 AM GMT+8.

### Dataset Description / Overview of Datasets

The competition dataset contains high-frequency optical tracking, standardized physical testing results, and multi-season NFL game performance metrics for 510 rookie prospects drafted or signed between 2023 and 2025.
The dataset is partitioned into 9 CSV files (~2.2 GiB total) linking pre-draft physical traits directly to regular-season on-field outcomes.

文件总览：9 files，Size 2.31 GB，Type csv，License Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)，158 columns。

各文件大小：combine_results.csv 41.02 kB；combine_tracking.csv 83.79 MB；game_tracking_2023.csv 335.46 MB；game_tracking_2024.csv 717.95 MB；game_tracking_2025.csv 1.02 GB；games.csv 43.17 kB；player_career_successes.csv 13.8 kB；player_play.csv 146.88 MB；players.csv 47.3 kB。

### File 1: players.csv — Contains demographic, collegiate, and draft metadata for the 510 rookie prospects.

字段：nfl_id int64 例 55867 — Unique identifier assigned to each NFL player. Primary key across all tracking and performance tables.；display_name str 例 Will Anderson — Player's full legal name as registered with the NFL.；draft_year int64 例 2023 — Year the prospect entered the NFL Draft (2023, 2024, or 2025).；nfl_position str 例 DE — Official NFL roster position (e.g., WR, CB, T, G, C, DT, DE, OLB, TE, SS, FS, NT, DB, ILB, MLB, RB, FB).；birth_date str 例 2001-09-02 — Player date of birth in YYYY-MM-DD format.；college_name str 例 Alabama — College or university where the prospect played prior to entering the draft.；college_conference str 例 Southeastern Conference — NCAA athletic conference of the prospect's collegiate team.；draft_round int64 / null 例 1 — Round in which the player was selected (1–7). null for Undrafted Free Agents (UDFAs).；draft_pick_within_round int64 / null 例 3 — Pick number within the specific draft round. null for UDFAs.；draft_overall_pick int64 / null 例 3 — Overall pick selection number across the entire draft (1–259+). null for UDFAs.

### File 2: combine_results.csv — Contains official anthropometric measurements, standardized physical testing splits, and Next Gen Stats draft grades from the NFL Scouting Combine.

字段：draft_year int64 例 2023 — Year of the Scouting Combine (2023, 2024, 2025).；nfl_id int64 例 56040 — Unique player identifier matching players.csv.；combine_position str 例 TE — Position group the prospect participated with during Combine drills.；combine_height float64 例 77.875 — Official prospect height measured in inches (e.g., 77.875 = 6'5.875").；combine_weight int64 例 245 — Official prospect body weight measured in pounds (lbs).；hand_size float64 例 10.0 — Hand span measured from tip of thumb to tip of pinky in inches.；arm_length float64 例 32.25 — Arm length measured from acromion to tip of middle finger in inches.；wing_span float64 例 79.75 — Total fingertip-to-fingertip wingspan in inches.；ten_yd_split float64 例 1.60 — Official electronic 10-yard split time during the 40-yard dash in seconds.；forty float64 例 4.84 — Official electronic 40-yard dash time in seconds.；vertical float64 例 38.5 — Standing vertical jump height in inches.；broad_jump float64 例 125.0 — Standing broad jump distance in inches.；three_cone float64 例 7.12 — 3-cone (L-drill) agility time in seconds (contains null for drill opt-outs).；short_shuttle float64 例 4.25 — 20-yard short shuttle (5-10-5) agility time in seconds (contains null for opt-outs).；bench_reps float64 例 22.0 — Number of 225 lb bench press repetitions completed (contains null for opt-outs).；ngs_athleticism_score int64 例 60 — Next Gen Stats composite athleticism score (0–100 percentile scale).；ngs_college_production_score float64 例 78.2 — Next Gen Stats college production score based on college statistics.；ngs_final_score float64 例 74.5 — Next Gen Stats overall prospect score combining athleticism and college production.

### File 3: combine_tracking.csv — Contains 10 Hz high-frequency optical sensor tracking across 6,310 Combine drill attempts, including 40-yard dash sprints and position-specific agility/route drills.

字段：draft_year int64 例 2025 — Year of Combine testing (2023, 2024, 2025).；event_id str 例 2025030210... — Unique identifier linking all tracking frames within a single continuous drill attempt.；nfl_id int64 / null 例 58286 — Player identifier matching players.csv. For ball, it represents the nfl_id for player in given event_id.；entity_type str 例 PLAYER — Entity tracked in the frame: PLAYER or BALL.；time str 例 2025-03-02T18:56:38.200 — ISO 8601 timestamp for the optical tracking frame (sampled every 0.10 seconds).；drill_type str 例 SKILL_DRILLS_OL — Broad drill category (e.g., FORTY_YARD_DASH, THREE_CONE_DRILL, SHORT_SHUTTLE, SKILL_DRILLS_WR, SKILL_DRILLS_DB, SKILL_DRILLS_OL, SKILL_DRILLS_DL, SKILL_DRILLS_TE).；drill_name str 例 OL_PULL_DRILL_FOLD... — Specific drill name (e.g., GAUNTLET_DRILL, RUN_THE_HOOP_DRILL, BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION, SLOT_STRIKE_ROUTE_LEFT, PASS_PRO_MIRROR_DRILL).；attempt int64 例 4 — Attempt number index for the prospect within the given drill.；x float64 例 87.09 — Player horizontal position along field length in yards (0-120).；y float64 例 22.52 — Player vertical position across field width in yards (0-53.33).；s float64 例 0.28 — Instantaneous player speed in yards per second (yd/s).；a float64 例 0.11 — Instantaneous player acceleration in yards per second squared (yds / sec^2).；dis float64 例 0.01 — Distance traveled from the previous tracking frame in yards.；dir float64 例 126.08 — Player motion direction angle in degrees (0-360).

### File 4: player_career_successes.csv — Contains cumulative career volume statistics, snap counts, and accolades earned by the 510 prospects during their rookie and subsequent seasons.

字段：nfl_id int64 例 55867 — Unique player identifier matching players.csv.；career_offensive_snaps int64 例 0 — Cumulative career offensive snaps played across regular-season and playoff games.；career_defensive_snaps int64 例 1884 — Cumulative career defensive snaps played across regular-season and playoff games.；career_special_teams_snaps int64 例 92 — Cumulative career special teams snaps played.；career_games_active int64 例 51 — Total number of regular-season games the player was on the active roster.；career_games_started int64 例 44 — Total number of regular-season games started by the player.；ap_all_pro_1st_team int64 例 1 — Total number of Associated Press (AP) First-Team All-Pro selections earned in career.；ap_all_pro_2nd_team int64 例 0 — Total number of Associated Press (AP) Second-Team All-Pro selections earned in career.；pro_bowl_original_ballot int64 例 1 — Total number of Pro Bowl selections on the original ballot (excluding alternate injury replacements).

### File 5: player_play.csv — Contains granular in-game Next Gen Stats and play-by-play metrics for every snap involving the rookie cohort across the 2023, 2024, and 2025 seasons (N=316,338 player-play records).

A. Play Context & Identifiers：game_id int64 例 2023091801 — Unique 10-digit game identifier matching games.csv.；play_id int64 例 56 — Unique play identifier within the game.；team_abbr str 例 BAL — 3-letter abbreviation of the team the player is on.；nfl_id int64 例 55976 — Unique player identifier matching players.csv.；lined_up_position str 例 T — Position where the player aligned on this specific play (e.g., WR, CB, T, G, C, DE, DT, TE, FS, SS, OLB).；play_description str 例 (15:00) P.Mahomes pass... — Official textual play-by-play description of the play.；quarter int64 例 1 — Quarter of the game (1, 2, 3, 4, or 5 for overtime).；down int64 例 1 — Current down (1, 2, 3, 4).；yards_to_go int64 例 10 — Yards needed to achieve a first down or touchdown.；possession_team str 例 CLE — 3-letter abbreviation of the offensive team.；yardline_side str 例 CLE — 3-letter abbreviation of the team side of the field where the ball is spotted.；yardline_number int64 例 25 — Distance in yards to the nearest goal line (1 ≤ yardline ≤ 50).；game_clock str 例 15:00 — Game clock remaining in the quarter in MM:SS format.；pre_snap_home_score int64 例 0 — Home team score prior to the snap.；pre_snap_visitor_score int64 例 0 — Visiting team score prior to the snap.；play_direction str 例 left — Direction of offensive play progression (left or right).

B. Win Probability & Expected Points：pre_snap_home_team_win_probability float64 例 0.4237 — Home team estimated win probability prior to snap.；pre_snap_visitor_team_win_probability float64 例 0.5763 — Visiting team estimated win probability prior to snap.；home_team_win_probability_added float64 例 0.2210 — Change in home team win probability produced by the play.；visitor_team_win_probility_added float64 例 -0.2210 — Change in visiting team win probability produced by the play.（字段名原文如此拼写）；expected_points_added float64 例 -8.459 — Net Expected Points Added (EPA) generated on the play.；expected_points float64 例 1.459 — Expected points value of the offensive drive state prior to snap.

C. Schemes, Formations & Play Results：offense_formation str 例 EMPTY — Offensive pre-snap personnel formation (e.g., SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL, JUMBO).；receiver_alignment str 例 3x2 — Wide receiver alignment distribution (e.g., 3x1, 2x2, 3x2, 1x1, 4x1).；pass_result str 例 IN — Outcome of the pass play: C (Complete), I (Incomplete), S (Sack), IN (Interception), R (Scramble).；pass_length float64 例 5.0 — Air yards traveled by the pass past the line of scrimmage (negative for passes behind the LOS).；play_nullified_by_penalty str 例 N — Y if the play outcome was wiped out by penalty, N otherwise.；penalty_yards float64 例 null — Total penalty yardage enforced on the play.；pre_penalty_yards_gained int64 例 0 — Gross yards gained before penalty assessment.；yards_gained int64 例 0 — Official net yards gained on the play.；team_coverage_man_zone str 例 ZONE_COVERAGE — Primary defensive coverage scheme classification (MAN_COVERAGE, ZONE_COVERAGE).；team_coverage_type str 例 COVER_6_ZONE — Specific defensive coverage shell (e.g., COVER_1, COVER_2, COVER_3, COVER_4, COVER_6_ZONE, COVER_0, PREVENT).；in_motion_at_ball_snap bool 例 False — True if the player was in active pre-snap motion when the ball was snapped.

D. Receiver & Ball Carrier Metrics：target bool 例 False — True if the player was the intended target of the pass.；rec_yards float64 例 14.0 — Receiving yards gained by the player on the play.；rush_yards float64 例 null — Rushing yards gained by the player on the play.；yards_after_catch float64 例 6.2 — Yards gained by the receiver after securing the catch (YAC).；fumble float64 例 0.0 — 1.0 if the player fumbled the ball on the play, 0.0 otherwise.；fumble_lost float64 例 0.0 — 1.0 if the player's fumble was recovered by the opponent.；route_ran str 例 SLANT — Specific route run by the receiver (e.g., GO, OUT, SLANT, CROSS, POST, CORNER, HITCH, SCREEN, FLAT, WHEEL).；separation_at_pass_forward float64 例 2.84 — Distance in yards to the nearest defender at the moment the ball is thrown forward.；cushion float64 例 6.15 — Pre-snap distance in yards between the receiver and their primary coverage defender.；expected_yards_after_catch float64 例 4.1 — Next Gen Stats expected yards after catch model prediction at catch moment.

E. Offensive Line & Pass Protection Metrics：dropback_duration float64 例 1.835 — Time from snap to pass release, sack, or pressure in seconds.；extended_sack_allowed float64 例 0.0 — Sacks allowed on extended dropbacks exceeding 4.0 seconds.；pass_rushers_encountered float64 例 1.0 — Number of distinct pass rushers engaged by the offensive lineman on the play.；peak_pressure_probability_allowed float64 例 0.1975 — Peak estimated pressure probability allowed by the offensive lineman.；pressure_allowed bool 例 False — True if the offensive lineman allowed a pressure (QB hurry, hit, or sack).；sack_allowed float64 例 0.0 — Sacks charged to the offensive lineman (1.0 for solo, 0.5 for shared).；time_to_pressure_allowed float64 例 2.45 — Time in seconds from snap until the offensive lineman allowed pressure.

F. Defensive Pass Rush & Tackle Metrics：blitzing bool 例 False — True if the defender was rushing the passer on a blitz.；player_get_off float64 例 0.742 — Time in seconds from ball snap until defensive rusher moves forward across the line of scrimmage.；sack float64 例 1.0 — 1.0 if the defender recorded a sack on the play (0.5 for shared sack).；tackle float64 例 1.0 — 1.0 if the defender made a solo tackle on the ball carrier.；assist float64 例 0.0 — 1.0 if the defender assisted on a tackle.；caused_forced_fumble float64 例 0.0 — 1.0 if the defender forced a fumble on the play.；recovered_fumble float64 例 0.0 — 1.0 if the player recovered a loose fumble.；tackle_for_loss float64 例 1.0 — 1.0 if the tackle stopped the play behind the line of scrimmage for negative yards.；time_to_pressure float64 例 2.18 — Time elapsed from snap until this defender generated pressure on the quarterback.；time_to_qb_hurry float64 例 2.80 — Time elapsed from snap until this defender forced a QB hurry.；coverage_assignment str 例 MAN — Pre-snap coverage assignment (e.g., MAN, DEEP_THIRD, FLAT, HOOK_CURL).；coverage_assignment_at_snap str 例 MAN — Coverage responsibility updated at ball snap following pre-snap motion.；quick_pressure float64 例 null — 1.0 if a pass rusher pressures within 2.50 seconds of the snap.；unblocked_pressure float64 例 null — 1.0 if a pass rusher pressured the QB through this lineman's gap without being blocked.

### File 6: games.csv — Contains schedule, season, and team matchup metadata for all 1,002 NFL regular-season and playoff games across 2023, 2024, and 2025.

字段：game_id int64 例 2023090700 — Unique 10-digit NFL game identifier (YYYYMMDD##). Primary key.；game_key int64 例 59173 — Historical NFL game key identifier.；season int64 例 2023 — NFL season year (2023, 2024, 2025).；season_type str 例 REG — Type of season: REG (Regular Season) or POST (Playoffs).；week int64 例 1 — Week number within the season (Weeks 1–18).；home_team_abbr str 例 KC — 3-letter abbreviation of the home team.；visitor_team_abbr str 例 DET — 3-letter abbreviation of the visiting team.

### Files 7, 8, 9: game_tracking_2023.csv, game_tracking_2024.csv, game_tracking_2025.csv — Contain 10 Hz high-frequency in-game optical tracking frames for all rookie cohort prospects across each respective NFL season.

字段：game_id int64 例 2024080952 — Game identifier matching games.csv.；play_id int64 例 3379 — Play identifier matching player_play.csv.；nfl_id int64 例 57367 — Player identifier matching players.csv (tracks all 510 rookie cohort prospects).；time str 例 2024-08-10T01:30:31.800 — ISO 8601 timestamp for the in-game optical tracking frame (sampled every 0.10s).；x float64 例 87.09 — Player horizontal position along field length in yards (0-120).；y float64 例 22.52 — Player vertical position across field width in yards (0-53.33).；s float64 例 0.28 — Instantaneous player speed in yards per second (yd/s).；a float64 例 0.11 — Instantaneous player acceleration in yards per second squared (yds / sec^2).；dis float64 例 0.03 — Distance traveled from the previous tracking frame in yards.；o float64 例 98.51 — Player body orientation angle in degrees (0-360).；dir float64 例 126.08 — Player motion direction angle in degrees (0-360).；event str / null 例 pass_forward — Game play milestone event tag (e.g., ball_snap, pass_forward, pass_arrived, tackle, touchdown, first_contact, out_of_bounds, or null for regular frames).

（注：Data 页正文为“~2.2 GiB total”，右侧文件总览显示为 2.31 GB，两处均按页面原样记录。）
