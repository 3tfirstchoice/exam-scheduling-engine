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
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
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

        # Để giữ an toàn cho travel matrix với dữ liệu chỉ có CS1/CS2 hiện tại:
        self.travel_distance_matrix = np.zeros((self.num_staff, self.num_slots))
        for i, campus_name in enumerate(shift_data["Cơ sở"].astype(str)):
            if "1" in campus_name:
                self.travel_distance_matrix[:, i] = self.distance_to_cs1
            else:
                self.travel_distance_matrix[:, i] = self.distance_to_cs2

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
        hard_conflict_list        = []
        cross_campus_soft_list    = []
        consecutive_same_cs_list  = []
        same_shift_list           = []

        for slot_j in range(num_slots):
            for slot_k in range(slot_j + 1, num_slots):

                if self.exam_date[slot_j] != self.exam_date[slot_k]:
                    continue    

                shift_gap      = abs(self.shift_order_in_day[slot_j] - self.shift_order_in_day[slot_k])
                is_diff_campus = self.campus[slot_j] != self.campus[slot_k]
                is_same_campus = not is_diff_campus

                if shift_gap == 0 or (shift_gap == 1 and is_diff_campus):
                    hard_conflict_list.append((slot_j, slot_k))
                elif shift_gap >= 2 and is_diff_campus:
                    cross_campus_soft_list.append((slot_j, slot_k))
                elif shift_gap == 1 and is_same_campus:
                    consecutive_same_cs_list.append((slot_j, slot_k))
                if shift_gap == 0 and is_same_campus:
                    same_shift_list.append((slot_j, slot_k))

        def _to_numpy_pair_array(pair_list: list) -> np.ndarray:
            return (
                np.array(pair_list, dtype=np.int32)
                if pair_list
                else np.empty((0, 2), dtype=np.int32)
            )

        self.hard_conflict_pairs            = _to_numpy_pair_array(hard_conflict_list)
        self.soft_cross_campus_pairs        = _to_numpy_pair_array(cross_campus_soft_list)
        self.soft_consecutive_same_cs_pairs = _to_numpy_pair_array(consecutive_same_cs_list)
        self.same_shift_pairs               = _to_numpy_pair_array(same_shift_list)

        print(f"  Cặp xung đột cứng  [RC1, RC10]: {len(self.hard_conflict_pairs):>5}")
        print(f"  Cặp soft khác CS   [RC9]      : {len(self.soft_cross_campus_pairs):>5}")
        print(f"  Cặp soft liền ca   [RC11]     : {len(self.soft_consecutive_same_cs_pairs):>5}\n")
        print(f"  Cặp slot gác chung      : {len(self.same_shift_pairs):>5}\n")
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
        total_km_per_staff = np.zeros((pop_size, self.num_staff))
        for staff_idx in range(self.num_staff):
            total_km_per_staff[:, staff_idx] = (
                travel_km_per_slot * (X == staff_idx)
            ).sum(axis=1)

        count_gte = shift_count_per_staff[:, :, None] >= shift_count_per_staff[:, None, :]

        is_elderly = self.staff_age > config.ELDERLY_AGE_THRESHOLD
        is_young   = ~is_elderly

        f1_intergenerational_fairness = np.zeros(pop_size)
        if is_elderly.any() and is_young.any():
            elderly_km_gt_young_km = (
                total_km_per_staff[:, is_elderly, None]
                > total_km_per_staff[:, None, is_young]
            )
            elderly_count_gte_young = count_gte[:, is_elderly, :][:, :, is_young]
            num_age_violations = np.sum(
                elderly_km_gt_young_km & elderly_count_gte_young, axis=(1, 2)
            )
            f1_intergenerational_fairness += num_age_violations * config.ELDERLY_HEAVIER_LOAD_PENALTY

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

            #Sort theo chiều ngang và đếm số phần tử liền kề giống nhau
            sorted_hash = np.sort(pair_hash_id, axis=1)
            repeated_pairs_count = np.sum(sorted_hash[:, 1:] == sorted_hash[:, :-1], axis=1)

            f2_repeat_pair_penalty = repeated_pairs_count * config.REPEAT_PAIR_PENALTY
        else:
            f2_repeat_pair_penalty = np.zeros(pop_size)

        quality_score = (
            f2_total_travel_distance
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

    expanded_slots = [
        row.to_dict()
        for _, row in shift_df.iterrows()
        for _ in range(int(row["Số lượng cán bộ cần thiết"]))
    ]
    slots_df  = pd.DataFrame(expanded_slots)
    num_slots = len(slots_df)
    num_staff = len(staff_df)

    print(f"  [RC2] Tổng lượt gác cần thiết : {total_required_slots}Khả thi")
    print(f"  [RC4] Tổng slot sau mở rộng   : {num_slots}")

    problem = ExamSchedulingProblem(num_slots, num_staff, slots_df, staff_df)

    algorithm = NSGA2(
        pop_size             = config.POPULATION_SIZE,
        sampling             = IntegerRandomSampling(),
        crossover            = UniformCrossover(prob=0.9),
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
    chromosome : np.ndarray,
    problem    : ExamSchedulingProblem,
    num_slots  : int,
    num_staff  : int,
) -> np.ndarray:
    
    target_gap   = config.ALLOWED_SHIFT_DEVIATION
    ceiling_avg  = int(np.ceil(num_slots  / num_staff))
    floor_avg    = int(np.floor(num_slots / num_staff))
    conflict_arr = problem.hard_conflict_pairs

    # Lấy thông tin tuổi tác và ca muộn để Robin Hood tôn trọng RC6
    is_elderly = problem.staff_age > config.ELDERLY_AGE_THRESHOLD
    is_late    = problem.is_late_shift

    print(f"\n[Robin Hood] Bắt đầu tinh chỉnh Gap (mục tiêu ≤ {target_gap})...")
    keep_iterating = True

    while keep_iterating:
        keep_iterating = False
        shift_counts = np.bincount(chromosome, minlength=num_staff)

        current_gap = shift_counts.max() - shift_counts.min()
        if current_gap <= target_gap:
            break   

        staff_ranked_rich_to_poor = np.argsort(shift_counts)[::-1]
        staff_ranked_poor_to_rich = np.argsort(shift_counts)

        for rich_staff in staff_ranked_rich_to_poor:
            if shift_counts[rich_staff] <= ceiling_avg:
                continue   

            slots_of_rich_staff = np.where(chromosome == rich_staff)[0]

            for poor_staff in staff_ranked_poor_to_rich:
                if shift_counts[poor_staff] >= floor_avg + (target_gap - 1):
                    continue   

                for candidate_slot in slots_of_rich_staff:
                    #Bảo vệ người già: Khước từ gán nếu đó là ca đêm
                    if is_elderly[poor_staff] and is_late[candidate_slot]:
                        continue
                    
                    if len(problem.same_shift_pairs) > 0:
                        # 1. Tìm các slot đang diễn ra CÙNG CA với candidate_slot này
                        mask_0 = problem.same_shift_pairs[:, 0] == candidate_slot
                        mask_1 = problem.same_shift_pairs[:, 1] == candidate_slot
                        partner_slots = np.concatenate([
                            problem.same_shift_pairs[mask_0, 1],
                            problem.same_shift_pairs[mask_1, 0]
                        ])
                        
                        # 2. Lấy danh sách ID cán bộ (partners) đang gác ở các slot đó
                        partners = chromosome[partner_slots]
                        
                        # 3. Quét xem poor_staff ĐÃ TỪNG gác chung với các partner này mấy lần rồi
                        creates_pair_violation = False
                        c1_arr = chromosome[problem.same_shift_pairs[:, 0]]
                        c2_arr = chromosome[problem.same_shift_pairs[:, 1]]
                        
                        for partner in partners:
                            overlap_count = np.sum(
                                ((c1_arr == poor_staff) & (c2_arr == partner)) |
                                ((c1_arr == partner) & (c2_arr == poor_staff))
                            )
                            # Nếu gán vào đây mà đụng mặt nhau từ lần thứ 2 trở lên -> Hủy kèo!
                            if overlap_count >= 2:  
                                creates_pair_violation = True
                                break
                                
                        if creates_pair_violation:
                            continue  # Bỏ qua slot này, Robin Hood đi tìm slot khác

                    chromosome[candidate_slot] = poor_staff   

                    related_conflicts = conflict_arr[
                        (conflict_arr[:, 0] == candidate_slot) |
                        (conflict_arr[:, 1] == candidate_slot)
                    ]
                    creates_hard_violation = any(
                        chromosome[pair[1] if pair[0] == candidate_slot else pair[0]] == poor_staff
                        for pair in related_conflicts
                    )

                    if not creates_hard_violation:
                        keep_iterating = True   
                        break
                    else:
                        chromosome[candidate_slot] = rich_staff   

                if keep_iterating:
                    break
            if keep_iterating:
                break

    final_counts = np.bincount(chromosome, minlength=num_staff)
    print(
        f"[Robin Hood] Hoàn tất — "
        f"Gap cuối: {final_counts.max() - final_counts.min()} ca\n"
    )
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