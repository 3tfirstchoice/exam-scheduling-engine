"""
model.py
========
Lõi tối ưu hóa lịch coi thi sử dụng NSGA-II + Robin Hood Post-processing.

BẢNG ÁNH XẠ RÀNG BUỘC (Requirement Traceability Matrix)
─────────────────────────────────────────────────────────────────────────────
  RC #  │ Nội dung nghiệp vụ                           │ Vị trí trong code
─────────────────────────────────────────────────────────────────────────────
  RC1   │ Không gác 2 ca trùng giờ                     │ G  ← _build_conflict_matrices()
  RC2   │ Tổng yêu cầu ≤ nguồn lực hiện có             │ Pre-check trong run_nsga2_scheduler()
  RC3   │ Mỗi người gác ≥ 1 ca                         │ F1a ← NO_ASSIGNMENT_PENALTY
  RC4   │ Mỗi ca đủ số lượng giám thị                  │ Slot expansion trong run_nsga2_scheduler()
  RC6   │ Bảo vệ cán bộ >45t (ca muộn + quá tải)      │ F2b ← elderly_night_violation_matrix
  RC7   │ Tối ưu tổng quãng đường di chuyển            │ F2a ← travel_distance_per_slot
  RC8   │ Công bằng số ca (tiến về μ)                  │ F1a, F1b, F1c
  RC9   │ Gác nhiều ca/ngày → ưu tiên cùng cơ sở       │ F1d ← soft_cross_campus_pairs
  RC10  │ Không gác liên tiếp 2 CS khác nhau/ngày      │ G  ← _build_conflict_matrices()
  RC11  │ Hạn chế 2 ca liên tiếp cùng CS (mềm)        │ F2c ← soft_consecutive_same_cs_pairs
  RC14  │ Hạn chế trực cuối tuần (T7, CN)              │ F3  ← is_weekend_slot
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from numba import njit, prange

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX                   
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

import config


# ═══════════════════════════════════════════════════════════════════
# PHẦN 1 ─ ĐỊNH NGHĨA BÀI TOÁN TỐI ƯU HÓA
# ═══════════════════════════════════════════════════════════════════

class ExamSchedulingProblem(Problem):

    def __init__(
        self,
        num_slots  : int,
        num_staff  : int,
        shift_data : pd.DataFrame,
        staff_data : pd.DataFrame,
    ) -> None:
        super().__init__(
            n_var    = num_slots,
            n_obj    = 3,
            n_constr = 1,
            xl       = 0,
            xu       = num_staff - 1,
            vtype    = int,
        )

        self.num_slots   = num_slots
        self.num_staff   = num_staff
        self.avg_shifts  = num_slots / num_staff
        self.min_shifts_allowed = max(1, int(np.floor(self.avg_shifts)) - config.ALLOWED_SHIFT_DEVIATION)
        self.max_shifts_allowed = int(np.ceil(self.avg_shifts)) + config.ALLOWED_SHIFT_DEVIATION

        _print_init_banner(num_slots, num_staff, self.avg_shifts,
                           self.min_shifts_allowed, self.max_shifts_allowed)

        self._preprocess_staff_data(staff_data)
        self._preprocess_shift_data(shift_data)
        self._build_conflict_matrices(num_slots)

    # ───────────────────────────────────────────────────────────────
    # 1.1  TIỀN XỬ LÝ DỮ LIỆU CÁN BỘ
    # ───────────────────────────────────────────────────────────────

    def _preprocess_staff_data(self, staff_data: pd.DataFrame) -> None:
        self.staff_age = (
            pd.to_numeric(staff_data["Tuổi"], errors="coerce").fillna(0).values
        )

        col_dist_cs1 = next(
            (c for c in staff_data.columns if c in ["CS1", "Cơ sở 1", "KC CS1 (km)"]), None
        )
        col_dist_cs2 = next(
            (c for c in staff_data.columns if c in ["CS2", "Cơ sở 2", "KC CS2 (km)"]), None
        )
        self.distance_to_cs1 = (
            pd.to_numeric(staff_data[col_dist_cs1], errors="coerce").fillna(0).values
            if col_dist_cs1 else np.zeros(self.num_staff)
        )
        self.distance_to_cs2 = (
            pd.to_numeric(staff_data[col_dist_cs2], errors="coerce").fillna(0).values
            if col_dist_cs2 else np.zeros(self.num_staff)
        )

        self.total_commute_distance = self.distance_to_cs1 + self.distance_to_cs2
        self.is_further_than        = (
            self.total_commute_distance[:, None] > self.total_commute_distance[None, :]
        )                                          

    # ───────────────────────────────────────────────────────────────
    # 1.2  TIỀN XỬ LÝ DỮ LIỆU CA THI
    # ───────────────────────────────────────────────────────────────

    def _preprocess_shift_data(self, shift_data: pd.DataFrame) -> None:
        self.shift_order_in_day = (
            shift_data["Ca thi"].astype(str)
            .str.extract(r"Ca\s*(\d)")[0]
            .fillna(0).astype(int).values
        )

        self.exam_date, _ = pd.factorize(shift_data["Ngày"])

      
        self.campus, _ = pd.factorize(shift_data["Cơ sở"])
        campus_str = shift_data["Cơ sở"].astype(str).values
        is_cs1 = np.array(["1" in str(cs) for cs in campus_str], dtype=bool)
        
        # Broadcasting: (num_staff,) + (num_slots,) → (num_staff, num_slots)
        self.travel_distance_matrix = np.where(
            is_cs1[None, :],  # Expand to (1, num_slots)
            self.distance_to_cs1[:, None],  # Expand to (num_staff, 1)
            self.distance_to_cs2[:, None]   # Expand to (num_staff, 1)
        ).astype(np.float32)

        self.is_late_shift = (
            shift_data["Ca thi"].astype(str)
            .str.contains(r"Ca\s*[45]", case=False).values
        )

        self.is_weekend_slot = shift_data["Thứ"].isin(["Thứ 7", "Chủ Nhật"]).values

        self.elderly_night_violation_matrix = (
            (self.staff_age > config.ELDERLY_AGE_THRESHOLD)[:, None]
            & self.is_late_shift[None, :]
        )                                     

    # ───────────────────────────────────────────────────────────────
    # 1.3  XÂY DỰNG MA TRẬN RÀNG BUỘC
    # ───────────────────────────────────────────────────────────────

   
    def _build_conflict_matrices(self, num_slots: int) -> None:
        # BƯỚC 1: Ép kiểu dữ liệu 
        dates = self.exam_date.astype(np.int64)
        shifts = self.shift_order_in_day.astype(np.int64)
        campus_arr = self.campus.astype(np.int64)

        # BƯỚC 2: Sắp xếp theo ngày để gom các slot cùng ngày lại
        sort_idx = np.argsort(dates)
        
        sorted_dates = dates[sort_idx]
        sorted_shifts = shifts[sort_idx]
        sorted_campus = campus_arr[sort_idx]
        # Giữ index gốc để map lại kết quả y như code cũ
        sorted_orig_idx = np.arange(num_slots, dtype=np.int32)[sort_idx]

        # BƯỚC 3: Tìm ranh giới của từng ngày (chia mảng)
        change_mask = sorted_dates[1:] != sorted_dates[:-1]
        start_indices = np.zeros(np.sum(change_mask) + 1, dtype=np.int32)
        start_indices[1:] = np.where(change_mask)[0] + 1
        end_indices = np.append(start_indices[1:], num_slots)

        @njit(fastmath=True, parallel=True, cache=True)
        def _count_conflicts_segmented(start_arr, end_arr, s_arr, c_arr):
            hard = soft_cross = soft_consec = same_shift = 0
            num_days = len(start_arr)

            for d in prange(num_days):
                start = start_arr[d]
                end = end_arr[d]
                
                # Numba giờ chỉ duyệt vòng lặp NGẮN bên trong từng ngày
                for j in range(start, end):
                    s_j = s_arr[j]
                    c_j = c_arr[j]
                    for k in range(j + 1, end):
                        gap = abs(s_j - s_arr[k])
                        same_campus = (c_j == c_arr[k])

                        if gap == 0 or (gap == 1 and not same_campus):
                            hard += 1
                        elif gap >= 2 and not same_campus:
                            soft_cross += 1
                        elif gap == 1 and same_campus:
                            soft_consec += 1
                        if gap == 0 and same_campus:
                            same_shift += 1

            return hard, soft_cross, soft_consec, same_shift

        @njit(fastmath=True, cache=True)
        def _fill_conflicts_segmented(start_arr, end_arr, s_arr, c_arr, orig_idx, 
                                      h_count, sc_count, scc_count, ss_count):
            hard_pairs        = np.empty((h_count, 2), dtype=np.int32)
            soft_cross_pairs  = np.empty((sc_count, 2), dtype=np.int32)
            soft_consec_pairs = np.empty((scc_count, 2), dtype=np.int32)
            same_shift_pairs  = np.empty((ss_count, 2), dtype=np.int32)

            h_idx = sc_idx = scc_idx = ss_idx = 0
            num_days = len(start_arr)

            for d in range(num_days):
                start = start_arr[d]
                end = end_arr[d]

                for j in range(start, end):
                    s_j = s_arr[j]
                    c_j = c_arr[j]
                    idx_j = orig_idx[j]
                    for k in range(j + 1, end):
                        gap = abs(s_j - s_arr[k])
                        same_campus = (c_j == c_arr[k])
                        idx_k = orig_idx[k]

                        p0, p1 = (idx_j, idx_k) if idx_j < idx_k else (idx_k, idx_j)

                        if gap == 0 or (gap == 1 and not same_campus):
                            hard_pairs[h_idx, 0] = p0
                            hard_pairs[h_idx, 1] = p1
                            h_idx += 1
                        elif gap >= 2 and not same_campus:
                            soft_cross_pairs[sc_idx, 0] = p0
                            soft_cross_pairs[sc_idx, 1] = p1
                            sc_idx += 1
                        elif gap == 1 and same_campus:
                            soft_consec_pairs[scc_idx, 0] = p0
                            soft_consec_pairs[scc_idx, 1] = p1
                            scc_idx += 1
                        if gap == 0 and same_campus:
                            same_shift_pairs[ss_idx, 0] = p0
                            same_shift_pairs[ss_idx, 1] = p1
                            ss_idx += 1

            return hard_pairs, soft_cross_pairs, soft_consec_pairs, same_shift_pairs

        # ====================== THỰC THI ======================
        hard_c, sc_c, scc_c, ss_c = _count_conflicts_segmented(
            start_indices, end_indices, sorted_shifts, sorted_campus
        )

        hard, soft_cross, soft_consec, same_shift = _fill_conflicts_segmented(
            start_indices, end_indices, sorted_shifts, sorted_campus, sorted_orig_idx,
            hard_c, sc_c, scc_c, ss_c
        )

        # Sắp xếp lại để trả về kết quả 
        if len(hard) > 0: hard = hard[np.lexsort((hard[:, 1], hard[:, 0]))]
        if len(soft_cross) > 0: soft_cross = soft_cross[np.lexsort((soft_cross[:, 1], soft_cross[:, 0]))]
        if len(soft_consec) > 0: soft_consec = soft_consec[np.lexsort((soft_consec[:, 1], soft_consec[:, 0]))]
        if len(same_shift) > 0: same_shift = same_shift[np.lexsort((same_shift[:, 1], same_shift[:, 0]))]

        self.hard_conflict_pairs            = hard
        self.soft_cross_campus_pairs        = soft_cross
        self.soft_consecutive_same_cs_pairs = soft_consec
        self.same_shift_pairs               = same_shift

        # Pre-compute conflict_map từ hard_conflict_pairs
        self.conflict_map = {}
        if len(hard) > 0:
            for slot_j, slot_k in hard:
                if slot_j not in self.conflict_map:
                    self.conflict_map[slot_j] = []
                if slot_k not in self.conflict_map:
                    self.conflict_map[slot_k] = []
                self.conflict_map[slot_j].append(slot_k)
                self.conflict_map[slot_k].append(slot_j)

        print(f"  Cặp xung đột cứng     : {len(hard):>5}")
        print(f"  Cặp soft khác CS      : {len(soft_cross):>5}")
        print(f"  Cặp soft liền ca CS   : {len(soft_consec):>5}")
        print(f"  Cặp gác chung         : {len(same_shift):>5}\n")
    # ───────────────────────────────────────────────────────────────
    # 1.4  HÀM ĐÁNH GIÁ QUẦN THỂ (VECTORIZED FITNESS EVALUATION)
    # ───────────────────────────────────────────────────────────────

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        pop_size       = X.shape[0]
        all_staff_idx  = np.arange(self.num_staff)

        shift_count_per_staff = (X[..., np.newaxis] == all_staff_idx).sum(axis=1)
        travel_km_per_slot = self.travel_distance_matrix[X, np.arange(self.num_slots)]

        # ── G │ RÀNG BUỘC CỨNG ────────────────────────────────────────────────
        if len(self.hard_conflict_pairs) > 0:
            slots_j = self.hard_conflict_pairs[:, 0]
            slots_k = self.hard_conflict_pairs[:, 1]
            num_hard_violations  = (X[:, slots_j] == X[:, slots_k]).sum(axis=1)
            hard_constraint_penalty = num_hard_violations.astype(float) * config.HARD_CONFLICT_PENALTY
        else:
            hard_constraint_penalty = np.zeros(pop_size)

        # ── F1a │ PHÂN BỔ CƠ BẢN ───────────────────────────────────────────────
        no_assignment_violations  = np.sum(shift_count_per_staff == 0, axis=1)
        out_of_range_violations   = np.sum(
            (shift_count_per_staff < self.min_shifts_allowed) |
            (shift_count_per_staff > self.max_shifts_allowed),
            axis=1,
        )
        f1_basic_allocation = (
            no_assignment_violations * config.NO_ASSIGNMENT_PENALTY
            + out_of_range_violations * config.OUT_OF_RANGE_PENALTY
        )

        # ── F1b │ CÔNG BẰNG PHÂN PHỐI ──────────────────────────────────────────
        max_min_gap             = shift_count_per_staff.max(axis=1) - shift_count_per_staff.min(axis=1)
        gap_above_ideal         = np.maximum(0, max_min_gap - 1)
        std_deviation_of_shifts = np.std(shift_count_per_staff, axis=1)

        f1_distribution_fairness = (
            gap_above_ideal ** 2          * config.MAX_MIN_GAP_PENALTY
            + std_deviation_of_shifts ** 2 * config.STD_DEVIATION_PENALTY
        )

        # ── F1c │ CÔNG BẰNG LIÊN THẾ HỆ & ĐỊA LÝ ───────────────────────────────
        # Dùng np.add.at() 
        total_km_per_staff = np.zeros((pop_size, self.num_staff), dtype=np.float32)
        for p_idx in range(pop_size):
            # np.add.at accumulates values at specified indices
            # Tương đương: total_km_per_staff[p_idx, X[p_idx]] += travel_km_per_slot[p_idx]
            np.add.at(total_km_per_staff[p_idx], X[p_idx], travel_km_per_slot[p_idx])

        count_gte = shift_count_per_staff[:, :, None] >= shift_count_per_staff[:, None, :]

        is_elderly = self.staff_age > config.ELDERLY_AGE_THRESHOLD
        is_young   = ~is_elderly

        f1_intergenerational_fairness = np.zeros(pop_size)
        # Chỉ tính comparison cho elderly vs young subset 
        if is_elderly.any() and is_young.any():
            elderly_idx = np.where(is_elderly)[0]
            young_idx = np.where(is_young)[0]
            
            # Giảm chiều so sánh từ (N_elderly, N_young) ứng với mỗi pop
            elderly_km = total_km_per_staff[:, elderly_idx]  # (pop, N_elderly)
            young_km = total_km_per_staff[:, young_idx]      # (pop, N_young)
            
            # Broadcasting comparison: (pop, N_elderly, 1) > (pop, 1, N_young)
            elderly_km_gt_young_km = elderly_km[:, :, None] > young_km[:, None, :]
            
            # Count shift comparisons cho elderly
            elderly_count = shift_count_per_staff[:, elderly_idx]  # (pop, N_elderly)
            young_count = shift_count_per_staff[:, young_idx]      # (pop, N_young)
            elderly_count_gte_young = elderly_count[:, :, None] >= young_count[:, None, :]
            
            num_age_violations = np.sum(
                elderly_km_gt_young_km & elderly_count_gte_young, axis=(1, 2)
            )
            f1_intergenerational_fairness = num_age_violations * config.ELDERLY_HEAVIER_LOAD_PENALTY

        count_gt = shift_count_per_staff[:, :, None] > shift_count_per_staff[:, None, :]
        num_geo_violations = np.sum(
            self.is_further_than[None, :, :] & count_gt, axis=(1, 2)
        )
        f1_geographic_fairness = num_geo_violations * config.DISTANT_HEAVIER_LOAD_PENALTY

        # ── F1d │ ƯU TIÊN CÙNG CƠ SỞ TRONG NGÀY ───────────────────────────────
        if len(self.soft_cross_campus_pairs) > 0:
            slots_j = self.soft_cross_campus_pairs[:, 0]
            slots_k = self.soft_cross_campus_pairs[:, 1]
            num_cross_campus_same_day = (X[:, slots_j] == X[:, slots_k]).sum(axis=1)
            f1_same_campus_preference = (
                num_cross_campus_same_day * config.SAME_DAY_CAMPUS_SWITCH_PENALTY
            )
        else:
            f1_same_campus_preference = np.zeros(pop_size)

        fairness_score = (
            f1_basic_allocation
            + f1_distribution_fairness
            + f1_intergenerational_fairness
            + f1_geographic_fairness
            + f1_same_campus_preference
        )

        # ── F2x │ HẠN CHẾ MỘT CÁN BỘ GÁC NHIỀU CA CÙNG NGÀY ──────────────────
        num_days = int(self.exam_date.max() + 1)
        if num_days > 1:
            staff_day_index = X * num_days + self.exam_date[None, :]
            staff_day_counts = np.zeros(
                (pop_size, self.num_staff * num_days),
                dtype=np.int32,
            )
            np.add.at(
                staff_day_counts,
                (np.arange(pop_size)[:, None], staff_day_index),
                1,
            )
            staff_day_counts = staff_day_counts.reshape(
                pop_size, self.num_staff, num_days
            )
            extra_same_day_shifts = np.maximum(0, staff_day_counts - 1)
            f2_multishift_same_day_penalty = (
                np.sum(extra_same_day_shifts, axis=(1, 2))
                * config.MULTI_SHIFT_PER_DAY_PENALTY
            )
        else:
            f2_multishift_same_day_penalty = np.zeros(pop_size)

        # ── F2x2 │ HẠN CHẾ ĐỔI CƠ SỞ TRONG NGÀY KHI GÁC NHIỀU CA ─────────────
        if len(self.soft_cross_campus_pairs) > 0:
            slots_j = self.soft_cross_campus_pairs[:, 0]
            slots_k = self.soft_cross_campus_pairs[:, 1]
            num_cross_campus_same_day = (X[:, slots_j] == X[:, slots_k]).sum(axis=1)
            f2_cross_campus_same_day_penalty = (
                num_cross_campus_same_day * config.CROSS_CAMPUS_SAME_DAY_PENALTY
            )
        else:
            f2_cross_campus_same_day_penalty = np.zeros(pop_size)

        # ── F2a │ TỐI ƯU QUÃNG ĐƯỜNG DI CHUYỂN ────────────────────────────────
        f2_total_travel_distance = (
            travel_km_per_slot.sum(axis=1) * config.TRAVEL_DISTANCE_WEIGHT
        )

        # ── F2b │ BẢO VỆ CÁN BỘ CAO TUỔI ──────────────────────────────────────
        num_elderly_on_late_shift = (
            self.elderly_night_violation_matrix[X, np.arange(self.num_slots)]
            .sum(axis=1)
        )
        f2_elderly_late_shift_penalty = (
            num_elderly_on_late_shift * config.ELDERLY_LATE_SHIFT_PENALTY
        )

        f2_elderly_overload_penalty = np.zeros(pop_size)
        if is_elderly.any():
            elderly_shifts_above_avg = np.maximum(
                0, shift_count_per_staff[:, is_elderly] - self.avg_shifts
            )
            f2_elderly_overload_penalty = (
                elderly_shifts_above_avg.sum(axis=1) * config.ELDERLY_SHIFT_OVERLOAD_PENALTY
            )

        # ── F2c │ HẠN CHẾ CA LIÊN TIẾP CÙNG CƠ SỞ ──────────────────────────────
        if len(self.soft_consecutive_same_cs_pairs) > 0:
            slots_j = self.soft_consecutive_same_cs_pairs[:, 0]
            slots_k = self.soft_consecutive_same_cs_pairs[:, 1]
            num_consecutive_same_cs = (X[:, slots_j] == X[:, slots_k]).sum(axis=1)
            f2_consecutive_fatigue_penalty = (
                num_consecutive_same_cs * config.CONSECUTIVE_SAME_CAMPUS_PENALTY
            )
        else:
            f2_consecutive_fatigue_penalty = np.zeros(pop_size)

            
        # ── F2d │ HẠN CHẾ LẶP CẶP GÁC CHUNG (DIVERSITY OF PAIRS) ───────────────
        if len(self.same_shift_pairs) > 0:
            slots_u = self.same_shift_pairs[:, 0]
            slots_v = self.same_shift_pairs[:, 1]

            # Bóc xuất mã cán bộ đang gác tại các cặp slot này
            staff_u = X[:, slots_u]
            staff_v = X[:, slots_v]

            # Quy chuẩn (min, max) để cặp (A, B) hay (B, A) đều cho ra 1 ID duy nhất
            min_staff = np.minimum(staff_u, staff_v)
            max_staff = np.maximum(staff_u, staff_v)

            # Hàm Băm (Hash): Ép mỗi cặp cán bộ thành 1 con số nguyên duy nhất
            pair_hash_id = min_staff * self.num_staff + max_staff

            # Count repeated pair occurrences beyond the first shared shift
            f2_repeat_pair_penalty = np.zeros(pop_size, dtype=np.float32)
            for p_idx in range(pop_size):
                unique_ids, counts = np.unique(pair_hash_id[p_idx], return_counts=True)
                repeated_pairs = np.maximum(0, counts - 1).sum()
                f2_repeat_pair_penalty[p_idx] = repeated_pairs * config.REPEAT_PAIR_PENALTY
        else:
            f2_repeat_pair_penalty = np.zeros(pop_size)

        quality_score = (
            f2_multishift_same_day_penalty
            + f2_cross_campus_same_day_penalty
            + f2_total_travel_distance
            + f2_elderly_late_shift_penalty
            + f2_elderly_overload_penalty
            + f2_consecutive_fatigue_penalty
            + f2_repeat_pair_penalty
        )

        # ── F3 │ CÂN BẰNG CA CUỐI TUẦN ────────────────────────────────────────
        weekend_shift_count_per_staff = (
            (X[:, self.is_weekend_slot, None] == all_staff_idx).sum(axis=1)
        )
        shifts_above_avg_per_staff = np.maximum(
            0, shift_count_per_staff - self.avg_shifts
        )
        weekend_score = (
            np.sum(weekend_shift_count_per_staff * shifts_above_avg_per_staff, axis=1)
            * config.WEEKEND_OVERLOAD_PENALTY
        )

        out["F"] = np.column_stack([fairness_score, quality_score, weekend_score])
        out["G"] = hard_constraint_penalty.reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════════
# PHẦN 2 ─ HÀM ĐIỀU PHỐI CHÍNH (ORCHESTRATOR)
# ═══════════════════════════════════════════════════════════════════

def run_nsga2_scheduler(
    shift_df: pd.DataFrame,
    staff_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:

    total_required_slots = int(shift_df["Số lượng cán bộ cần thiết"].sum())
    total_available_staff = len(staff_df)
    max_safe_capacity = total_available_staff * (
        int(np.ceil(total_required_slots / total_available_staff))
        + config.ALLOWED_SHIFT_DEVIATION
    )

    if total_required_slots > max_safe_capacity:
        raise ValueError(
            f"[RC2] KHÔNG KHẢ THI: Tổng nhu cầu ({total_required_slots} lượt gác) "
            f"vượt quá khả năng cung ({total_available_staff} cán bộ × "
            f"{max_safe_capacity // total_available_staff} ca tối đa).\n"
            f"         Gợi ý: Thêm cán bộ hoặc tăng ALLOWED_SHIFT_DEVIATION."
        )

    # Vectorization 
    expand_counts = shift_df["Số lượng cán bộ cần thiết"].astype(int).values
    
    # Chỉ số row cần repeat (mỗi row repeat expand_counts[i] lần)
    repeat_indices = np.repeat(np.arange(len(shift_df)), expand_counts)
    
    # Expand DataFrame bằng fancy indexing 
    slots_df = shift_df.iloc[repeat_indices].reset_index(drop=True)
    num_slots = len(slots_df)
    num_staff = len(staff_df)

    print(f"  [RC2] Tổng lượt gác cần thiết : {total_required_slots}Khả thi")
    print(f"  [RC4] Tổng slot sau mở rộng   : {num_slots}")

    problem = ExamSchedulingProblem(num_slots, num_staff, slots_df, staff_df)

    algorithm = NSGA2(
        pop_size             = config.POPULATION_SIZE,
        sampling             = IntegerRandomSampling(),
        crossover            = SBX(prob=0.9, eta=20),
        mutation             = PM(
                                   prob    = config.MUTATION_RATE,
                                   eta     = 10,
                                   repair  = RoundingRepair(),
                               ),
        eliminate_duplicates = True,
    )

    print(
        f"\n[NSGA-II] Bắt đầu tiến hóa "
        f"({config.NUM_GENERATIONS} thế hệ × {config.POPULATION_SIZE} cá thể)..."
    )
    optimization_result = minimize(
        problem,
        algorithm,
        termination = ("n_gen", config.NUM_GENERATIONS),
        seed        = config.RANDOM_SEED,
        verbose     = True,
    )

    is_feasible_solution = optimization_result.G.flatten() <= 0

    #Từ chối xử lý Robin Hood nếu nghiệm Infeasible để tránh sinh ra "Rác"
    if not np.any(is_feasible_solution):
        print(
            "\n[CẢNH BÁO] Không tìm được nghiệm thỏa mãn 100% ràng buộc cứng [RC1, RC10]!\n"
            "           Lịch xuất ra có thể chứa xung đột.\n"
            "           [BỎ QUA] Tạm dừng Robin Hood để tránh làm hỏng cấu trúc nghiệm.\n"
        )
        candidate_chromosomes = optimization_result.X
        candidate_objectives  = optimization_result.F
        
        # Chọn nghiệm ít rác nhất dù vẫn có rác
        weights = np.array([config.WEIGHT_FAIRNESS_F1, config.WEIGHT_QUALITY_F2, config.WEIGHT_WEEKEND_F3])
        obj_min, obj_max = candidate_objectives.min(axis=0), candidate_objectives.max(axis=0)
        obj_norm = (candidate_objectives - obj_min) / np.maximum(obj_max - obj_min, 1e-8)
        tcheby_scores = np.max(weights * obj_norm, axis=1) + config.TCHEBYCHEFF_AUGMENTATION_COEFF * np.sum(weights * obj_norm, axis=1)
        best_chromosome = candidate_chromosomes[np.argmin(tcheby_scores)].astype(int)

        _print_final_summary(best_chromosome, is_feasible_solution, num_staff)
        return best_chromosome, slots_df

    # Nếu Feasible, tiến hành bình thường
    candidate_chromosomes = optimization_result.X[is_feasible_solution]
    candidate_objectives  = optimization_result.F[is_feasible_solution]
    print(f"\n[OK] Tìm được {np.sum(is_feasible_solution)} nghiệm feasible.")

    obj_min  = candidate_objectives.min(axis=0)
    obj_max  = candidate_objectives.max(axis=0)
    obj_norm = (candidate_objectives - obj_min) / np.maximum(obj_max - obj_min, 1e-8)

    weights        = np.array([config.WEIGHT_FAIRNESS_F1, config.WEIGHT_QUALITY_F2, config.WEIGHT_WEEKEND_F3])
    tcheby_scores  = (
        np.max(weights * obj_norm, axis=1)
        + config.TCHEBYCHEFF_AUGMENTATION_COEFF * np.sum(weights * obj_norm, axis=1)
    )
    best_chromosome = candidate_chromosomes[np.argmin(tcheby_scores)].astype(int)

    best_chromosome = _robin_hood_gap_reducer(
        best_chromosome, problem, num_slots, num_staff
    )

    _print_final_summary(best_chromosome, is_feasible_solution, num_staff)

    return best_chromosome, slots_df


# ═══════════════════════════════════════════════════════════════════
# PHẦN 3 ─ ROBIN HOOD POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _robin_hood_gap_reducer(
    chromosome: np.ndarray,
    problem: ExamSchedulingProblem,
    num_slots: int,
    num_staff: int,
    max_iterations: int = 800
) -> np.ndarray:
    target_gap = config.ALLOWED_SHIFT_DEVIATION
    
    #Dùng problem.conflict_map (pre-computed)
    conflict_map = problem.conflict_map if hasattr(problem, 'conflict_map') else {}

    # An toàn thuộc tính
    staff_ages = getattr(problem, 'staff_ages', getattr(problem, 'staff_age', np.zeros(num_staff)))
    is_elderly = staff_ages > getattr(config, 'ELDERLY_AGE_THRESHOLD', 55)
    is_late = getattr(problem, 'is_late_shift', getattr(problem, 'is_late', np.zeros(num_slots, dtype=bool)))

    day_by_slot = getattr(problem, 'exam_date', np.zeros(num_slots, dtype=np.int32))
    campus_by_slot = getattr(problem, 'campus', np.zeros(num_slots, dtype=np.int32))
    shift_order_by_slot = getattr(problem, 'shift_order_in_day', np.zeros(num_slots, dtype=np.int32))

    print(f"\n[Robin Hood] Bắt đầu tinh chỉnh Gap (mục tiêu ≤ {target_gap})...")

    # Top-K động
    k = max(3, min(8, num_staff // 8))

    # Pre-build staff→slots mapping để tránh np.where() liên tục
    staff_to_slots = [[] for _ in range(num_staff)]
    for slot_idx, staff_idx in enumerate(chromosome):
        staff_to_slots[staff_idx].append(slot_idx)

    for iteration in range(max_iterations):
        # Dùng bincount 1 lần rồi maintain thay vì tính lại
        shift_counts = np.bincount(chromosome, minlength=num_staff)
        current_gap = shift_counts.max() - shift_counts.min()

        if current_gap <= target_gap:
            break

        rich_indices = np.argsort(shift_counts)[-k:][::-1]
        poor_indices = np.argsort(shift_counts)[:k]

        moved_in_iter = 0
        max_moves = max(1, min(5, current_gap // 2))

        for rich_staff in rich_indices:
            if moved_in_iter >= max_moves:
                break
            if shift_counts[rich_staff] <= 1:        # Bảo vệ không cướp hết
                continue

            # Dùng pre-built mapping thay vì np.where()
            slots_of_rich = staff_to_slots[rich_staff].copy()
            if len(slots_of_rich) == 0:
                continue

            np.random.shuffle(slots_of_rich)

            for slot in slots_of_rich:
                if moved_in_iter >= max_moves:
                    break

                for poor_staff in poor_indices:
                    if shift_counts[rich_staff] - shift_counts[poor_staff] <= target_gap:
                        break

                    # 1. Ràng buộc người già - ca muộn
                    if is_elderly[poor_staff] and is_late[slot]:
                        continue

                    # 2. Kiểm tra hard conflict dùng pre-computed conflict_map
                    if slot in conflict_map:
                        conflicting = any(chromosome[c_slot] == poor_staff 
                                        for c_slot in conflict_map[slot])
                        if conflicting:
                            continue

                    # 3. Tránh làm xấu hơn chất lượng F2 bằng cách thêm ca cùng ngày khác cơ sở
                    slot_day = day_by_slot[slot]
                    slot_campus = campus_by_slot[slot]
                    slot_shift = shift_order_by_slot[slot]
                    conflict_quality = False
                    for existing_slot in staff_to_slots[poor_staff]:
                        if day_by_slot[existing_slot] != slot_day:
                            continue
                        if campus_by_slot[existing_slot] != slot_campus:
                            conflict_quality = True
                            break
                        if abs(shift_order_by_slot[existing_slot] - slot_shift) == 1:
                            conflict_quality = True
                            break
                    if conflict_quality:
                        continue

                    # === THỰC HIỆN CHUYỂN ===
                    old_staff = chromosome[slot]
                    chromosome[slot] = poor_staff
                    
                    # Update staff→slots mapping 
                    staff_to_slots[old_staff].remove(slot)
                    staff_to_slots[poor_staff].append(slot)
                    
                    shift_counts[old_staff] -= 1
                    shift_counts[poor_staff] += 1
                    moved_in_iter += 1
                    break   # Chuyển xong 1 ca → thử slot tiếp theo của rich_staff

        # Nếu iteration này không chuyển được ca nào → tối ưu cục bộ
        if moved_in_iter == 0:
            print(f"[Robin Hood] Đạt tối ưu cục bộ sau {iteration + 1} iterations.")
            break

    final_counts = np.bincount(chromosome, minlength=num_staff)
    final_gap = final_counts.max() - final_counts.min()

    print(f"[Robin Hood] Hoàn tất sau {iteration + 1} iterations. "
          f"Gap cuối: {final_gap} | Max={final_counts.max()} | Min={final_counts.min()}")

    return chromosome

# ═══════════════════════════════════════════════════════════════════
# PHẦN 4 ─ TIỆN ÍCH HỖ TRỢ
# ═══════════════════════════════════════════════════════════════════

def _print_init_banner(
    num_slots        : int,
    num_staff        : int,
    avg_shifts       : float,
    min_shifts       : int,
    max_shifts       : int,
) -> None:
    print("=" * 65)
    print("  HỆ THỐNG TỐI ƯU HÓA XẾP LỊCH COI THI")
    print("=" * 65)
    print(f"  Tổng slot cần gán        : {num_slots}")
    print(f"  Tổng số cán bộ           : {num_staff}")
    print(f"  Trung bình lý tưởng  (μ) : {avg_shifts:.3f} ca/người")
    print(f"  Dải ca hợp lệ            : [{min_shifts}, {max_shifts}]"
          f"  (±{config.ALLOWED_SHIFT_DEVIATION})")
    print(f"  Ngưỡng cao tuổi          : >{config.ELDERLY_AGE_THRESHOLD} tuổi [RC6]")
    print("-" * 65)


def _print_final_summary(
    best_chromosome     : np.ndarray,
    is_feasible_solution: np.ndarray,
    num_staff           : int,
) -> None:
    counts     = np.bincount(best_chromosome, minlength=num_staff)
    feasibility_status = (
        "Feasible — không vi phạm ràng buộc cứng"
        if np.any(is_feasible_solution)
        else "CẢNH BÁO — lịch có thể chứa vi phạm RC1/RC10"
    )

    print("=" * 65)
    print("  BÁO CÁO NGHIỆM THU CUỐI CÙNG")
    print("=" * 65)
    print(f"  Ca nhiều nhất       : {counts.max()}")
    print(f"  Ca ít nhất          : {counts.min()}")
    print(f"  Chênh lệch Gap      : {counts.max() - counts.min()} ca")
    print(f"  Độ lệch chuẩn  (σ)  : {np.std(counts.astype(float)):.4f}")
    print(f"  Trung bình     (μ)  : {counts.mean():.4f}")
    print(f"  Trạng thái         : {feasibility_status}")
    print("=" * 65 + "\n")