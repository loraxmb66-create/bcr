
import sys
import math
import random
import time
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- Core Baccarat Logic ----------

CARD_VALUES = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 0, 11: 0, 12: 0, 13: 0  # 10, J, Q, K -> 0
}

RANK_LABEL = {
    1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "10", 11: "J", 12: "Q", 13: "K"
}

@dataclass
class Shoe:
    decks: int = 8
    counts: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.counts:
            # Each deck: 4 of each rank
            self.counts = {rank: 4 * self.decks for rank in range(1, 14)}

    def remaining(self) -> int:
        return sum(self.counts.values())

    def remove(self, rank: int):
        if self.counts.get(rank, 0) <= 0:
            raise ValueError(f"No {RANK_LABEL[rank]} left to draw.")
        self.counts[rank] -= 1

    def add_back(self, rank: int):
        self.counts[rank] += 1

    def draw_random(self) -> int:
        total = self.remaining()
        if total <= 0:
            raise ValueError("Shoe is empty.")
        r = random.randint(1, total)
        cum = 0
        for rank, cnt in self.counts.items():
            if cnt <= 0:
                continue
            cum += cnt
            if r <= cum:
                self.counts[rank] -= 1
                return rank
        # Fallback (shouldn't happen)
        for rank, cnt in self.counts.items():
            if cnt > 0:
                self.counts[rank] -= 1
                return rank
        raise RuntimeError("Failed to draw a card.")

def hand_total(cards: List[int]) -> int:
    return sum(CARD_VALUES[c] for c in cards) % 10

def player_draws_third(player_cards: List[int], banker_cards: List[int]) -> bool:
    # Natural check (handled outside)
    total = hand_total(player_cards)
    if total <= 5:
        return True
    return False

def banker_draw_rule(banker_cards: List[int], player_third: int) -> bool:
    # Banker third-card rule given player's third card value (0..9, or None if player stands)
    b = hand_total(banker_cards)
    if player_third is None:
        return b <= 5
    # Detailed tableau:
    if b <= 2:
        return True
    if b == 3:
        return player_third != 8
    if b == 4:
        return player_third in [2,3,4,5,6,7]
    if b == 5:
        return player_third in [4,5,6,7]
    if b == 6:
        return player_third in [6,7]
    return False  # b == 7 stands; 8-9 would be naturals handled earlier

def deal_hand_from_shoe(shoe: Shoe) -> Tuple[str, List[int], List[int]]:
    # Returns (result, player_cards, banker_cards) where result in {"Player","Banker","Tie"}
    # Draw initial two cards each
    pc = [shoe.draw_random(), shoe.draw_random()]
    bc = [shoe.draw_random(), shoe.draw_random()]

    pt = hand_total(pc)
    bt = hand_total(bc)

    # Check natural
    if pt in (8,9) or bt in (8,9):
        if pt > bt:
            return "Player", pc, bc
        elif bt > pt:
            return "Banker", pc, bc
        else:
            return "Tie", pc, bc

    # Player's action
    player_third_rank = None
    if pt <= 5:
        player_third_rank = shoe.draw_random()
        pc.append(player_third_rank)

    # Banker's action depends on player's third card (value or None)
    player_third_val = CARD_VALUES[player_third_rank] if player_third_rank is not None else None
    if banker_draw_rule(bc, player_third_val):
        bc.append(shoe.draw_random())

    # Decide outcome
    pt = hand_total(pc)
    bt = hand_total(bc)
    if pt > bt:
        return "Player", pc, bc
    elif bt > pt:
        return "Banker", pc, bc
    else:
        return "Tie", pc, bc

def simulate_next_hand_probabilities(shoe: Shoe, n_sims: int = 20000, seed: int = 42) -> Dict[str, float]:
    # Monte Carlo simulation given current shoe composition
    random.seed(seed)
    win_counts = {"Player": 0, "Banker": 0, "Tie": 0}
    # Copy shoe counts once to avoid mutation
    base_counts = dict(shoe.counts)
    for _ in range(n_sims):
        # create a temp shoe (fast shallow copy of counts)
        temp_shoe = Shoe(decks=shoe.decks, counts=dict(base_counts))
        try:
            result, _, _ = deal_hand_from_shoe(temp_shoe)
        except ValueError:
            # If shoe empty -> break early
            break
        win_counts[result] += 1
    total = sum(win_counts.values())
    if total == 0:
        return {"Player": 0.0, "Banker": 0.0, "Tie": 0.0}
    return {k: v / total for k, v in win_counts.items()}

def expected_value(prob: Dict[str, float], payout_player=1.0, payout_banker=0.95, payout_tie=8.0) -> Dict[str, float]:
    # EV per 1 unit bet with given payouts/commission
    # EV = p(win)*payout - p(lose)*1; for ties on Player/Banker, usually push (0 EV contribution)
    pP = prob.get("Player", 0.0)
    pB = prob.get("Banker", 0.0)
    pT = prob.get("Tie", 0.0)

    # Player bet: wins when Player; loses when Banker; pushes on Tie
    ev_player = pP * payout_player - pB * 1.0

    # Banker bet: wins when Banker; loses when Player; pushes on Tie
    ev_banker = pB * payout_banker - pP * 1.0

    # Tie bet: wins when Tie; loses when not tie
    ev_tie = pT * payout_tie - (1.0 - pT) * 1.0

    return {"Player": ev_player, "Banker": ev_banker, "Tie": ev_tie}

def kelly_fraction(edge: float, payout: float) -> float:
    # Kelly for a single-outcome bet with net odds 'b' (payout) and edge (EV) per 1 stake
    # edge = p*b - (1-p), where b is net odds; solving for p:
    # p = (edge + 1) / (b + 1)
    # Kelly fraction f* = (b*p - (1-p)) / b = edge / b
    if payout <= 0:
        return 0.0
    return max(0.0, edge / payout)

# ---------- Streamlit App (GUI) ----------

def try_streamlit():
    try:
        import streamlit as st
        import pandas as pd
    except Exception as e:
        return False

    st.set_page_config(page_title="Baccarat Toolkit (Educational)", page_icon="🎴", layout="wide")

    st.title("🎴 Baccarat Toolkit — Mô phỏng & Quản lý Vốn (Giáo dục)")
    st.caption("Công cụ này chỉ phục vụ mục đích giáo dục/xác suất. Không có đảm bảo thắng. Casino luôn có lợi thế.")

    with st.sidebar:
        st.header("Thiết lập Shoe")
        decks = st.selectbox("Số bộ bài (decks)", [6, 8], index=1)
        init_counts = {rank: 4 * decks for rank in range(1, 14)}

        st.subheader("Ghi nhận bài đã ra (tùy chọn)")
        cols = st.columns(13)
        remove_counts = {}
        for i, rank in enumerate(range(1,14)):
            with cols[i]:
                remove_counts[rank] = st.number_input(f"{RANK_LABEL[rank]}", min_value=0, max_value=4*decks, value=0, step=1)

        # Build shoe
        shoe = Shoe(decks=decks, counts={r: max(0, init_counts[r] - remove_counts[r]) for r in init_counts})

        st.markdown("---")
        st.subheader("Thông số payout/commission")
        payout_player = st.number_input("Player trả (net) 1 = 1:1", min_value=0.0, value=1.0, step=0.05, format="%.2f")
        payout_banker = st.number_input("Banker trả (net) 0.95 = 0.95:1 (5% commission)", min_value=0.0, value=0.95, step=0.05, format="%.2f")
        payout_tie = st.number_input("Tie trả (net) (thường 8:1 hoặc 9:1)", min_value=0.0, value=8.0, step=0.5, format="%.2f")

        st.markdown("---")
        st.subheader("Mô phỏng")
        n_sims = st.slider("Số lần mô phỏng", min_value=5000, max_value=100000, value=30000, step=5000)
        seed = st.number_input("Seed (ngẫu nhiên có thể tái lập)", min_value=0, value=42, step=1)

    tab1, tab2, tab3, tab4 = st.tabs(["🔢 Xác suất ván kế", "💹 EV & Gợi ý cược", "💼 Quản lý vốn", "🧾 Ghi nhật ký phiên"])

    with tab1:
        st.subheader("Xác suất ván kế tiếp (Monte Carlo)")
        st.write(f"Bài còn lại: **{shoe.remaining()}** lá")
        run = st.button("Chạy mô phỏng")
        if run:
            start = time.time()
            prob = simulate_next_hand_probabilities(shoe, n_sims=n_sims, seed=seed)
            dur = time.time() - start
            df = pd.DataFrame([{
                "Kết quả": "Player", "Xác suất": prob["Player"]
            },{
                "Kết quả": "Banker", "Xác suất": prob["Banker"]
            },{
                "Kết quả": "Tie", "Xác suất": prob["Tie"]
            }])
            st.dataframe(df.style.format({"Xác suất": "{:.4%}"}), use_container_width=True)
            st.caption(f"Thời gian chạy: {dur:.2f}s")

            st.info("Lưu ý: Đây là mô phỏng xấp xỉ dựa trên số bài còn lại bạn nhập. Không đảm bảo chính xác tuyệt đối, và không đảo ngược lợi thế nhà cái.")

    with tab2:
        st.subheader("EV theo cấu hình payout hiện tại")
        run_ev = st.button("Tính EV")
        if run_ev:
            prob = simulate_next_hand_probabilities(shoe, n_sims=max(10000, n_sims//3), seed=seed)
            ev = expected_value(prob, payout_player=payout_player, payout_banker=payout_banker, payout_tie=payout_tie)
            st.write("**Xác suất ước lượng**:", {k: f"{v:.2%}" for k, v in prob.items()})
            st.write("**EV trên mỗi 1 đơn vị cược** (giá trị kỳ vọng):",
                     {k: f"{v:+.4f}" for k, v in ev.items()})
            best = max(ev, key=ev.get)
            st.success(f"Phương án có EV cao nhất hiện tại: **{best}** ({ev[best]:+.4f} / đơn vị). "
                       "Nếu tất cả âm, nghĩa là không có kèo +EV (thường xuyên xảy ra).")

            st.caption("Banker trả 0.95:1 (có commission) thường có nhà lợi thế ~1.06%, Player ~1.24%, Tie 8:1 ~14% (với shoe đầy). Khi còn ít bài, EV có thể dao động nhỏ nhưng nhà cái vẫn lợi.")

    with tab3:
        st.subheader("Quản lý vốn")
        bankroll = st.number_input("Vốn hiện có", min_value=0.0, value=100.0, step=10.0, format="%.2f")
        unit = st.number_input("Đơn vị cược cơ bản (flat)", min_value=0.0, value=1.0, step=1.0, format="%.2f")
        frac_kelly = st.slider("Tỷ lệ Kelly (0 = không dùng)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        option = st.selectbox("Kiểu cược", ["Flat (cố định)", "Kelly theo EV tốt nhất", "Fractional Kelly"])
        run_bank = st.button("Gợi ý mức cược")
        if run_bank:
            prob = simulate_next_hand_probabilities(shoe, n_sims=max(10000, n_sims//3), seed=seed)
            ev = expected_value(prob, payout_player=payout_player, payout_banker=payout_banker, payout_tie=payout_tie)
            best = max(ev, key=ev.get)
            best_ev = ev[best]

            if option == "Flat (cố định)":
                stake = unit
                method = "Flat"
            elif option == "Kelly theo EV tốt nhất":
                b = {"Player": payout_player, "Banker": payout_banker, "Tie": payout_tie}[best]
                f = kelly_fraction(best_ev, b)
                stake = bankroll * f
                method = f"Kelly ({best})"
            else:
                b = {"Player": payout_player, "Banker": payout_banker, "Tie": payout_tie}[best]
                f = kelly_fraction(best_ev, b) * frac_kelly
                stake = bankroll * f
                method = f"Fractional Kelly x{frac_kelly:.2f} ({best})"

            stake = max(0.0, round(stake, 2))
            st.info(f"**Gợi ý cược**: {method} → **{stake}** (từ vốn {bankroll}).")
            if best_ev <= 0:
                st.warning("EV tốt nhất ≤ 0 → Không nên tăng cược. Cược nhỏ hoặc bỏ qua ván.")

            st.caption("Kelly chỉ phù hợp khi có lợi thế dương và biết chính xác xác suất. Trong Baccarat thực tế, lợi thế dương là rất hiếm.")

    with tab4:
        st.subheader("Nhật ký phiên chơi")
        st.write("Ghi lại kết quả và biến động vốn để tự đánh giá rủi ro.")
        import pandas as pd
        csv_file = st.text_input("Tên file CSV để lưu/đọc", value="baccarat_sessions.csv")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ts = st.text_input("Thời gian (auto nếu để trống)", value="")
        with col2:
            result = st.selectbox("Kết quả", ["Player", "Banker", "Tie"])
        with col3:
            stake_in = st.number_input("Cược", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        with col4:
            pnl = st.number_input("Lãi/Lỗ (+/-)", value=0.0, step=1.0, format="%.2f")
        add = st.button("Thêm dòng")
        if add:
            row = {
                "time": ts if ts.strip() else time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": result,
                "stake": stake_in,
                "pnl": pnl
            }
            try:
                exists = False
                try:
                    with open(csv_file, "r", newline="", encoding="utf-8") as f:
                        exists = True
                except FileNotFoundError:
                    pass
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["time","result","stake","pnl"])
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                st.success(f"Đã ghi vào {csv_file}")
            except Exception as e:
                st.error(f"Lỗi ghi file: {e}")

        load = st.button("Mở file CSV")
        if load:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                st.dataframe(df, use_container_width=True)
                st.write("Tổng PnL:", round(df["pnl"].sum(), 2))
                st.write("Số ván:", len(df))
            except Exception as e:
                st.error(f"Không mở được file: {e}")

    with st.expander("❗ Khuyến cáo rủi ro"):
        st.write("""
- Đây là công cụ **giáo dục** để hiểu xác suất/EV. Không có hệ thống nào đảm bảo **thắng lâu dài**.
- Baccarat tiêu chuẩn (shoe đầy): **Banker** có lợi thế nhà cái ~1.06%, **Player** ~1.24%, **Tie** (8:1) ~14%.
- Mô phỏng Monte Carlo chỉ xấp xỉ; nhập sai số bài còn lại sẽ làm kết quả sai lệch.
- Chỉ sử dụng tiền có thể mất; xác lập giới hạn và tuân thủ.
        """)

    return True

# ---------- CLI Fallback ----------

def run_cli():
    print("=== Baccarat Toolkit (CLI) — Educational Use Only ===")
    try:
        decks = int(input("Number of decks [6 or 8, default 8]: ") or "8")
        if decks not in (6,8):
            decks = 8
    except Exception:
        decks = 8
    shoe = Shoe(decks=decks)
    print("You can remove cards (e.g., 'A 3' removes three Aces). Enter blank to stop.")
    while True:
        s = input("Remove (Rank count) e.g. 'A 2' or '10 1' (blank to finish): ").strip()
        if not s:
            break
        try:
            rnk, cnt = s.split()
            # Map input to rank number
            rnk = rnk.upper()
            to_rank = None
            for k,v in RANK_LABEL.items():
                if v == rnk:
                    to_rank = k
                    break
            if to_rank is None:
                print("Unknown rank.")
                continue
            cnt = int(cnt)
            for _ in range(cnt):
                shoe.remove(to_rank)
            print(f"Removed {cnt} x {rnk}. Remaining cards: {shoe.remaining()}")
        except Exception as e:
            print("Failed:", e)

    try:
        sims = int(input("Monte Carlo simulations [default 30000]: ") or "30000")
    except Exception:
        sims = 30000
    prob = simulate_next_hand_probabilities(shoe, n_sims=sims, seed=42)
    print("Estimated next-hand probabilities:", {k: f"{v:.2%}" for k,v in prob.items()})
    try:
        pP = float(input("Player payout (net, default 1.0): ") or "1.0")
        pB = float(input("Banker payout (net, default 0.95): ") or "0.95")
        pT = float(input("Tie payout (net, default 8.0): ") or "8.0")
    except Exception:
        pP, pB, pT = 1.0, 0.95, 8.0
    ev = expected_value(prob, payout_player=pP, payout_banker=pB, payout_tie=pT)
    print("EV per 1 unit:", {k: f"{v:+.4f}" for k,v in ev.items()})
    best = max(ev, key=ev.get)
    print(f"Best EV option now: {best} ({ev[best]:+.4f} per unit). If all negative, consider skipping.")

if __name__ == "__main__":
    if not try_streamlit():
        run_cli()
