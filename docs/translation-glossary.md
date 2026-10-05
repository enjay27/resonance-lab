# Translation glossary — Japanese chat → Korean

**Version:** 1.0.0 · **Season:** S1 (2026-10)

How Japanese *Blue Protocol: Star Resonance* chat is translated into Korean for resonance-lab's data. Read by people and given
as the brief to translation agents. It holds terms and rules only — never chat lines (no data in git, `tests/test_docs.py`).

**Authority** — when two entries disagree, the higher one wins:

1. **Official** — the game's own Korean names, supplied by the maintainer. Never varied, never abbreviated.
2. **Maintainer-fixed** — decided by the maintainer for the app's Korean players.
3. **Convention** — decided while translating season 1; stable, but open to change (a changelog row says when).
4. **Observed** — the generated table at the end: what the first pass used, with counts. A guess until promoted.

Companion: [`labeling-guide.md`](labeling-guide.md) (which category a line belongs to). Season procedure: [Updating for a new season](#updating-for-a-new-season).

## 1. The setting every translator is told

Every line is a real chat message from the Japanese server of the game, written by Japanese players for other players. Channels: `WORLD`
(everyone), `PARTY` (the players in your party, usually inside a raid), `LOCAL` (nearby), `BEGINNER`. Almost everything is about game content:
raids and difficulties, dungeons with tiers (`M6`, `M15`), party recruitment in slot notation, boss and channel calls, raid mechanics, ordinary
player talk. The readers of the translation are **Korean players of the same game** — write what they would type in their own chat.
Guild messages are not translated (the maintainer's guild is Korean); see the labeling guide.

## 2. Style rules

1. **Meaning-faithful, natural chat.** Not literal, not formal prose. Keep the register: polite Japanese (です/ます) → 해요체 or 합쇼체; casual → 반말.
   Keep the tone and the emoticons. Add nothing, drop nothing, explain nothing (no 구함/모집 if the source has no such word).
2. **Laughter and reactions:** `ｗ` `w` `草` `笑` → `ㅋㅋ` (as many as the source suggests); `泣` → `ㅠㅠ`; short reactions (すご, えぐ, やば) → 대박, 미쳤다, 헐…
   Greetings and farewells → the Korean game-chat formula (안녕하세요, 수고하셨습니다, 잘 부탁드립니다).
3. **Kept unchanged:** numbers, IDs, channel numbers (`48ch` → `48채널`, number kept), `@` `＠` `↑` `～`, Latin symbols (`M6`, `S3`, `NM`, `EH`, `BS`, `MS`,
   `k`, `T`/`H`/`D`, `PT`), kaomoji (including `・` and `ー` inside them), placeholders `[이모지]` `[스티커]`. Counts are written in Arabic numerals (一周 → 1회).
4. **Slot notation stays as letters:** `@T1`, `@D2H1`, `T1H3D12`, `@TDDH` keep `T` `H` `D` and the numbers; words around them are translated
   (`@たくさん/いっぱい/沢山` → `@많이`, `@誰でも` → `@누구나`, `@T〆` → `@T 마감`, `自認D` → `딜러 자처`). The *word* DPS becomes 딜러 (`@DPS2` → `@딜러2`).
5. **No Japanese left:** no kana, no kanji in the Korean. A katakana name that is not listed is transliterated by sound; a kanji term that is not listed
   gets its Sino-Korean reading or a natural descriptive word, and is reported as a term. Player nicknames inside a line are transliterated by sound.
6. **Nothing to translate** (symbols, emoticons, Latin letters, numbers only) → copied unchanged.
7. **Garbled, cut off or ambiguous** → best reading plus a short English `flag`; never skip a line.
8. **Typos of the poster are read correctly:** `MN` → `NM`; `NA始` → `NM 시작`.

## 3. Official: classes and subclasses

Players often write only the subclass, or a one-kanji short form.

| Class (Korean) | Class (Japanese) | Subclass (Korean) | Subclass (Japanese) |
|---|---|---|---|
| 트윈 스트라이커 | ツインストライカー | 무상 / 적홍 | 双炎 / 炎舞 |
| 스톰 블레이드 | ストームブレイド | 발도 / 월광 | 雷刃 / 月影 |
| 윈드 나이트 | ゲイルランサー | 질풍 / 난무 | 烈風 / 乱風 |
| 프로스트 메이지 | フロストメイジ | 얼음창 / 얼음빔 | 氷牙 / 霜天 |
| 디바인 아처 | ディバインアーチャー | 늑대활 / 매활 | 狼弓 / 鷹弓 |
| 헤비 가디언 | ヘヴィガーディアン | 방패 / 가드 | 剛身 / 剛守 |
| 실드 나이트 | シールドファイター | 방패 / 광휘 | 光砕 / 光盾 |
| 실반 오라클 | ヴァーダントオラクル | **심판** / **치유** | 威咲 (= イサキ = いさき) / 森癒 |
| 비트 퍼포머 | ビートパフォーマー | **음파** / **협주** | 狂音 / 響奏 |

Short forms seen in chat:

| Written | Korean |
|---|---|
| 威咲, イサキ, いさき, 威 | 심판 |
| 森癒, 森 (as a healer subclass) | 치유 |
| 狂音, 狂 | 음파 |
| 響奏, 協奏, 響, 奏, きょうそう | 협주 |
| 鷹弓, 鷹 | 매활 |
| 狼弓, 狼 | 늑대활 |
| 威狂, 威咲か狂音 | 심판 / 음파 (both names) |
| 咲狂 | 심판·음파 |
| フロスト, オラクル alone | the full class name (프로스트 메이지, 실반 오라클) |

`森` inside `迷妄の森` / `迷森` is the dungeon (미망의 숲), never the healer subclass. `ギター` ("guitar healer") is probably 음파 or 협주: with only
`ギター` written, keep 기타 and flag it (open question below).

## 4. Official: content

| Korean | Japanese | Note |
|---|---|---|
| 개척 | 開拓 | 開拓局 → 개척국, 開拓依頼 → 개척 의뢰 (개척국 is a convention, see open questions) |
| 불안정 | 不安定 | 不安定な空間 → 불안정한 공간 |
| 월드 보스 | ワールドレイド, ワルレ | |
| 극한공간 | 極限空間, 極限 | |
| 우리보 | ウリボ, ウリ坊 | ウリボ箱 → 우리보 상자 |
| 미망의 숲 | 迷妄の森, 迷森, 迷妄 | 迷妄クエスト → 미망 퀘스트 |
| 환상꿈 레이드 | 幻夢レイド | stages below |

**환상꿈 레이드 stages:** 시작 = 始 · **계속 = 継** (not 계승) · 종결 = 終. 継NM → `계속 NM`, 始継NM → `시작·계속 NM`, 終NM → `종결 NM`, 始継終 → `시작·계속·종결`.

**극한공간 dungeons** — the full official name every time:

| Korean | Japanese (all the ways it is written) |
|---|---|
| 저주받은 무덤 | 呪われし煌墓, 墓, お墓, 煌墓 |
| 기계화 처리소 | 機械化工場, 工場 |
| 침식 티나 | 蝕・ティナの精神領域, ティナ **when it is the dungeon** (ティナM6, ティナ周回, ティナ14強度) |
| 안개 속 사냥터 | 霧海の猟場, 霧海, 猟場, 狩場 |
| 침식 거탑 | 蝕・巨塔の遺跡, 巨塔, 塔 (as a dungeon short form) |
| 환해 암초 | 珊瑚岩の谷, 珊瑚, 珊瑚岩; nickname ナッポ → 나뽀 (ナッポM6 → 나뽀 M6) |

Plain `티나` is the character / Imagine ティナ (`T1:ティナアル`, `ティナ5凸`, `ティナしかない`).

## 5. Maintainer-fixed

| Japanese | Korean | | Japanese | Korean |
|---|---|---|---|---|
| 消化 | 숙제 | | リキャスト | 쿨타임 |
| 完凸 | 풀돌 | | 盾 | 탱커 |
| ファスト | 속공 | | 杖 | 법사 |
| 器用 | 숙련 | | 弓 | 궁수 |
| ウルト | 궁 | | イマジン | 이매진 |
| ガシャ | 뽑기 | | ばんわ | 존밤 |
| ヒグマ | 산적 두목 | | ムークボス | 무크 두목 |

火力 → 딜러 when it names the role or a person (火力募集); for damage output use 딜 / 딜러형 — never "딜러 화력".

## 6. Convention: roles, recruitment, raid and party talk

| Japanese | Korean | | Japanese | Korean |
|---|---|---|---|---|
| タンク | 탱커 | | レイド | 레이드 |
| ヒーラー / ヒラ / 回復 (the role) | 힐러 | | 回帰レイド | 복귀 레이드 |
| 回復 (an action) | 힐 | | ナイトメア / NM / Nm | NM |
| ダメヒラ | 딜힐러 | | イージー / ハード | 이지 / 하드 (イージーのみ → `이지 난이도만`, never `이지만`) |
| 純H / 純ヒラ | 순수 힐러 | | S1 / S2 / S3 | unchanged |
| 遠距離D / レンジD | 원딜 | | 連戦 | 연전 |
| 近接D | 근딜 | | ギミック | 기믹 |
| 若葉 | 새싹 | | ギミック理解者 / 予習済み | 기믹 숙지자 / 예습 완료 |
| 不動若葉 | 가만히 있는 새싹 | | 説明なし / 説明無 | 설명 없음 |
| メンター | 멘토 | | N滅解散 | N전멸 해산 |
| 〆 | 마감 | | 自動承認 | 자동 승인 |
| 抜け自由 | 중도 이탈 자유 | | スキップ | 스킵 |
| 初回(報酬) | 첫 클리어(보상) | | 周回 | 돌기 (5周 → 5회) |
| 配信中 | 방송 중 | | 強度 | 강도 |
| ヘルプ | 도움 | | 前半 / 後半 | 전반 / 후반 |
| ID (instance) | 던전 | | VC | 보이스 챗 |
| 無言OK | 무언 OK | | 運ゲ | 운빨겜 |
| チェイス | 체이스 | | 麻雀 | 마작 |
| 刻印 | 각인 | | 護符 | 부적 |
| ピン | 핀 | | レールガン | 레일건 |
| ロボ | 로봇 | | 距離減衰 | 거리 감쇠 |
| 予兆 | 전조 | | 床抜け | 바닥 빠짐 |
| 頭割り | 쉐어 | | 覇者 | 패자 |
| 峠(の牙) | 고개(의 송곳니) | | バハマール | 바하마르 |

Raid mechanics and party talk:

| Japanese | Korean | | Japanese | Korean |
|---|---|---|---|---|
| ワイプ / わいぷ | 전멸 (**not** 와이프 = wife) | | 退出 | 퇴장 |
| インタラクト | 상호작용 | | リログ | 재접속 |
| タンクバスター | 탱크버스터 | | 挑発 | 도발 |
| クリスタル | 크리스탈 | | ポータル | 포털 |
| ワープ | 워프 | | 反転 | 반전 |
| 召喚 | 소환 | | 床 (the mechanic) | 바닥 |
| 身投げ / 飛び降り | 뛰어내리기 | | ファントム | 팬텀 |
| カウント(ダウン) | 카운트(다운) | | リーダー (party) | 파티장 |
| ゲージ | 게이지 | | マーカー | 마커 |
| MT / ST | unchanged | | 1死 | 1데스 |
| 誤爆 | 오발송 (a message posted to the wrong chat) | | 席外す / 離席 | 자리 비움 |
| お花摘み / トイレ | 화장실 | | 了解 | 알겠습니다 / 알겠어요 / 알겠어 (match the register) |
| おつ / お疲れ | 수고하셨습니다 / 수고~ (match the register) | | | |

Places, bosses and characters — transliterate by sound unless listed: 浮島 → 부유섬, 千夢 → 천몽, 幻夢 → 환상꿈, アルーナ → 아루나, ファルファラ → 팔파라 (not
official), バジ → 바지, プレデター → 프레데터, クロックゲイザー → 클락 게이저, ブラッドバウル → 블러드 바울, 毛玉 → 털뭉치, 金ぽ → 금포.

## 7. Open questions (answer, then record the answer in the changelog)

| Term | Current handling | Question |
|---|---|---|
| ギター | 기타, flagged | The 음파 or 협주 Beat Performer? |
| 遺跡 alone | 유적 (85 uses; 2 as 침식 거탑) | Does a bare 遺跡 in a dungeon list mean 침식 거탑 (遺跡46F kept as 유적)? |
| 開拓局 | 개척국 | The glossary only has 開拓; is 개척국 right? |
| 放置若葉 | 방치 새싹 | Official wording? |
| フロスト, オラクル alone | expanded to the class name | Is the short form a known 별명? |
| ファルファラ | 팔파라 | An Imagine; official Korean name? |
| もり, なっぽ in hiragana | 미망의 숲, 나뽀 | Right reading? |
| 継続 vs 継 | 継続 → 계속 | Always the raid stage? |
| Guild names | transliterated | Keep, or leave guild lines untranslated entirely? |

## Updating for a new season

1. Branch from `main`; edit this file only (and `labeling-guide.md` when the categories change).
2. New official names from the game → section 3/4 (**Official**); a new maintainer decision → section 5; a settled translating choice → section 6.
3. Run the new season's chat through the translators with this file as the brief; the term report (Japanese, Korean used, count) goes into the *Observed*
   table: replace it, do not append. A term with more than one rendering is either legitimate (say why in a note) or a mistake to fix.
4. Bump the version: patch for a fix or a new term, minor for a new section or a changed rule, major for a new season. Add a changelog row.
5. `just check` (`tests/test_docs.py` pins the header, the changelog and the official class names).

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-10-05 | First versioned glossary. Official classes and content names from the maintainer; 継 → 계속 (was 계승); dungeon short forms use the full official names; 2,169 non-guild lines translated with it (0 violations of the strict check). |

## Observed terms, season 1 (generated, provisional)

Terms the first pass reported on 2,169 lines, used at least twice; the sections above win over any row here. A `/` between renderings means the pass used
both (see section 7 and the style notes for the legitimate ones: 火力 role vs damage, ティナ dungeon vs character, 迷妄 forest vs quest).

| Japanese | Uses | Korean (count) |
|---|---|---|
| レイド | 281 | 레이드 (281) |
| NM | 238 | NM (238) |
| 継 | 218 | 계속 (218) |
| 終 | 154 | 종결 (154) |
| 始 | 101 | 시작 (101) |
| 周回 | 101 | 돌기 (100) / 돕니다 (1) |
| ナイトメア | 101 | NM (101) |
| 巨塔 | 89 | 침식 거탑 (89) |
| 遺跡 | 85 | 유적 (83) / 침식 거탑 (2) |
| 強度 | 71 | 강도 (71) |
| VC | 70 | 보이스 챗 (70) |
| ティナ | 70 | 침식 티나 (62) / 티나 (8) |
| 墓 | 63 | 저주받은 무덤 (63) |
| ギミック | 63 | 기믹 (63) |
| 自動承認 | 57 | 자동 승인 (57) |
| 若葉 | 48 | 새싹 (48) |
| DPS | 44 | 딜러 (39) / 딜 (5) |
| 工場 | 43 | 기계화 처리소 (43) |
| 配信中 | 42 | 방송 중 (42) |
| 説明なし | 42 | 설명 없음 (42) |
| ギミック理解者 | 42 | 기믹 숙지자 (42) |
| 不安定 | 39 | 불안정 (39) |
| 消化 | 39 | 숙제 (39) |
| 後半 | 39 | 후반 (39) |
| 森 | 38 | 치유 (38) |
| Nm | 34 | NM (34) |
| 響 | 32 | 협주 (32) |
| 〆 | 32 | 마감 (32) |
| ピン | 31 | 핀 (31) |
| 霧海 | 30 | 안개 속 사냥터 (30) |
| ナッポ | 30 | 나뽀 (30) |
| 千夢 | 28 | 천몽 (28) |
| 自認D | 26 | 딜러 자처 (26) |
| 連戦 | 24 | 연전 (24) |
| ハード | 24 | 하드 (24) |
| 響奏 | 24 | 협주 (24) |
| 3滅解散 | 23 | 3전멸 해산 (23) |
| 迷妄 | 22 | 미망 (19) / 미망의 숲 (3) |
| 猟場 | 22 | 안개 속 사냥터 (22) |
| 説明無し | 22 | 설명 없음 (22) |
| 抜け自由 | 21 | 중도 이탈 자유 (21) |
| ヘルプ | 21 | 도움 (21) |
| 開拓 | 20 | 개척 (20) |
| 初回 | 19 | 첫 클리어 (19) |
| イージー | 19 | 이지 (19) |
| 威咲 | 19 | 심판 (19) |
| 幻夢レイド | 19 | 환상꿈 레이드 (19) |
| 極限 | 18 | 극한공간 (18) |
| 火力 | 18 | 딜 (9) / 딜러 (8) / 딜러형 (1) |
| 床 | 18 | 바닥 (18) |
| タンク | 17 | 탱커 (17) |
| 狂音 | 17 | 음파 (17) |
| 開拓局 | 17 | 개척국 (17) |
| PT | 17 | PT (17) |
| スキップ | 16 | 스킵 (16) |
| 依頼 | 15 | 의뢰 (15) |
| 回帰レイド | 15 | 복귀 레이드 (15) |
| 駆け込み | 15 | 막차 (15) |
| レールガン | 15 | 레일건 (15) |
| ロボ | 14 | 로봇 (14) |
| イサキ | 14 | 심판 (14) |
| 前半 | 14 | 전반 (14) |
| 珊瑚 | 14 | 환해 암초 (14) |
| 迷妄の森 | 13 | 미망의 숲 (13) |
| 狩場 | 13 | 안개 속 사냥터 (13) |
| メア | 12 | NM (12) |
| ツアー | 12 | 투어 (12) |
| ウリボ箱 | 12 | 우리보 상자 (12) |
| 主 | 12 | 파티장 (12) |
| dps | 12 | 딜러 (12) |
| いっぱい | 12 | 많이 (12) |
| 麻雀 | 11 | 마작 (11) |
| バハ | 11 | 바하 (11) |
| BS | 11 | BS (11) |
| 退出 | 11 | 퇴장 (11) |
| かかし | 10 | 허수아비 (10) |
| レグ | 10 | 레그 (10) |
| 刻印 | 10 | 각인 (10) |
| クリスタル | 10 | 크리스탈 (10) |
| 初回報酬 | 9 | 첫 클리어 보상 (9) |
| 滅 | 9 | 전멸 (9) |
| 鷹 | 9 | 매활 (9) |
| フロスト | 9 | 프로스트 메이지 (7) / 프로스트 (2) |
| プレデター | 9 | 프레데터 (9) |
| 距離減 | 9 | 거리 감쇠 (9) |
| 予習済み | 9 | 예습 완료 (9) |
| 狂 | 8 | 음파 (8) |
| ID | 8 | 던전 (8) |
| 推奨強度 | 8 | 권장 강도 (8) |
| 距離減衰 | 8 | 거리 감쇠 (8) |
| 運ゲ | 8 | 운빨겜 (8) |
| 継続 | 8 | 계속 (8) |
| @いっぱい | 8 | @많이 (8) |
| 1周 | 7 | 1회 (7) |
| 英雄 | 7 | 영웅 (7) |
| 覇者 | 7 | 패자 (7) |
| クエスト | 7 | 퀘스트 (7) |
| 協奏 | 7 | 협주 (7) |
| ノーマル | 7 | 노멀 (7) |
| 弓 | 7 | 궁수 (7) |
| クロックゲイザー | 7 | 클락 게이저 (7) |
| チェイス | 7 | 체이스 (7) |
| カウント | 7 | 카운트 (7) |
| 迷森 | 7 | 미망의 숲 (7) |
| 始継 | 7 | 시작·계속 (7) |
| 浮島 | 6 | 부유섬 (6) |
| ダンジョン | 6 | 던전 (6) |
| カカシ | 6 | 허수아비 (6) |
| なっぽ | 6 | 나뽀 (6) |
| お墓 | 6 | 저주받은 무덤 (6) |
| 金ぽ | 6 | 금포 (6) |
| 護符 | 6 | 부적 (6) |
| 塔 | 6 | 침식 거탑 (6) |
| 継NM | 6 | 계속 NM (6) |
| 自由抜け | 6 | 중도 이탈 자유 (6) |
| ファルファラ | 6 | 팔파라 (6) |
| ヒーラー | 5 | 힐러 (5) |
| 峠の牙 | 5 | 고개의 송곳니 (5) |
| ヒラ | 5 | 힐러 (5) |
| 森癒 | 5 | 치유 (5) |
| 威 | 5 | 심판 (5) |
| ふろすと | 5 | 프로스트 (3) / 프로스트 메이지 (2) |
| フルスコア | 5 | 풀스코어 (5) |
| ワルレ | 5 | 월드 보스 (5) |
| MT | 5 | MT (5) |
| ワイプ | 5 | 전멸 (5) |
| たくさん | 5 | 많이 (5) |
| タンメン | 5 | 탕멘 (3) / 탕면 (2) |
| 床抜け | 5 | 바닥 빠짐 (5) |
| ワープ | 5 | 워프 (5) |
| インタラクト | 5 | 상호작용 (5) |
| 幻夢 | 5 | 환상꿈 (5) |
| 解説無し | 5 | 설명 없음 (5) |
| 周 | 5 | 회 (5) |
| バハマール | 4 | 바하마르 (4) |
| デイリー | 4 | 일일 (3) / 일일 퀘스트 (1) |
| 個別 | 4 | 개별 (4) |
| 幻花 | 4 | 환화 (4) |
| 回帰 | 4 | 복귀 (4) |
| スターリー島 | 4 | 스타리 섬 (4) |
| 活動所 | 4 | 활동소 (4) |
| お手伝い | 4 | 도움 (4) |
| 峠 | 4 | 고개 (4) |
| 5周 | 4 | 5회 (4) |
| 回復 | 4 | 힐 (4) |
| 不動若葉 | 4 | 가만히 있는 새싹 (4) |
| ギター | 4 | 기타 (4) |
| 奏 | 4 | 협주 (4) |
| ウリ箱 | 4 | 우리보 상자 (4) |
| 釣り | 4 | 낚시 (4) |
| 爆弾 | 4 | 폭탄 (4) |
| クリ目 | 4 | 클리어 목적 (4) |
| ツモ切り | 4 | 쯔모기리 (4) |
| 無言OK | 4 | 무언 OK (4) |
| ダスト | 4 | 더스트 (4) |
| 仮面 | 4 | 가면 (4) |
| スコア | 4 | 스코어 (4) |
| ギミック予習済み | 4 | 기믹 예습 완료 (4) |
| イージ | 4 | 이지 (4) |
| ヒグマ | 4 | 산적 두목 (4) |
| ルンバ | 4 | 룸바 (4) |
| 厳選 | 4 | 옵션 작 (4) |
| 巨塔の遺跡 | 3 | 침식 거탑 (3) |
| 運動会 | 3 | 운동회 (3) |
| バウル | 3 | 바울 (3) |
| 単品 | 3 | 단품 (3) |
| 初回クリア | 3 | 첫 클리어 (3) |
| むーく | 3 | 무크 (3) |
| ギルハン | 3 | 길드 사냥 (3) |
| 純H | 3 | 순수 힐러 (3) |
| 鍵 | 3 | 열쇠 (3) |
| キョウオン | 3 | 음파 (3) |
| ST | 3 | ST (3) |
| S1 | 3 | S1 (3) |
| vc | 3 | 보이스 챗 (3) |
| 威狂 | 3 | 심판 / 음파 (3) |
| レグディニス | 3 | 레그디니스 (3) |
| 一回 | 3 | 1회 (3) |
| ごばく | 3 | 오발송 (3) |
| ms | 3 | ms (3) |
| ギミック予習者 | 3 | 기믹 예습자 (3) |
| 予兆 | 3 | 전조 (3) |
| パテ | 3 | 파티 (3) |
| 予習済 | 3 | 예습 완료 (3) |
| ボイス指揮 | 3 | 보이스 지휘 (3) |
| 便 | 3 | 차 (3) |
| 始・継 | 3 | 시작·계속 (3) |
| 深層 | 3 | 심층 (3) |
| 近接D | 2 | 근딜 (2) |
| 説明無 | 2 | 설명 없음 (2) |
| 5滅解散 | 2 | 5전멸 해산 (2) |
| ボード | 2 | 보드 (2) |
| ３滅解散 | 2 | 3전멸 해산 (2) |
| 毛玉 | 2 | 털뭉치 (2) |
| ひーらー | 2 | 힐러 (2) |
| 極限空間 | 2 | 극한공간 (2) |
| しめ | 2 | 마감 (2) |
| サブ | 2 | 부캐 (2) |
| ＠T〆 | 2 | @T 마감 (2) |
| 石集め | 2 | 돌 모으기 (2) |
| 途中抜け | 2 | 중도 이탈 (2) |
| ゲイザー | 2 | 게이저 (2) |
| ボイチャ | 2 | 보이스 챗 (2) |
| ＮＭ | 2 | NM (2) |
| きょとう | 2 | 거탑 (2) |
| すきっぷ | 2 | 스킵 (2) |
| 反転 | 2 | 반전 (2) |
| デバフ | 2 | 디버프 (2) |
| ヘイト | 2 | 어그로 (2) |
| 召喚 | 2 | 소환 (2) |
| 身投げ | 2 | 뛰어내려 (1) / 뛰어내리기 (1) |
| わいぷ | 2 | 전멸 (2) |
| かいたくきょく | 2 | 개척국 (2) |
| ふあんてい | 2 | 불안정 (2) |
| 牙 | 2 | 송곳니 (2) |
| 2周 | 2 | 2회 (2) |
| ﾅｯﾎﾟ | 2 | 나뽀 (2) |
| タンクバスター | 2 | 탱크버스터 (2) |
| 不安 | 2 | 불안 (2) |
| リーダー | 2 | 파티장 (2) |
| ダメージヒーラー | 2 | 딜힐러 (2) |
| 無凸 | 2 | 무돌 (1) / 노돌 (1) |
| れいど | 2 | 레이드 (2) |
| EH | 2 | EH (2) |
| メンター | 2 | 멘토 (2) |
| MN | 2 | NM (2) |
| 即死 | 2 | 즉사 (1) / 즉사기 (1) |
| 頭割り | 2 | 쉐어 (2) |
| アルーナ | 2 | 아루나 (2) |
| スーモ | 2 | 스모 (2) |
| 近距離D | 2 | 근딜 (2) |
| 職不問 | 2 | 직업 무관 (2) |
| きり | 2 | 안개 바다 (2) |
| 初回特典 | 2 | 첫 클리어 특전 (2) |
| キョウソウ | 2 | 협주 (1) / 쿄소 (1) |
| 床破壊 | 2 | 바닥 파괴 (2) |
| アル | 2 | 아루 (2) |
| 遠距離DPS | 2 | 원딜 (2) |
| 純ひら | 2 | 순수 힐러 (2) |
| S3 | 2 | S3 (2) |
| アタッカー | 2 | 딜러 (2) |
