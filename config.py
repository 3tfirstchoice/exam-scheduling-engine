"""
config.py
=========
Cấu hình toàn cục — tập trung MỌI tham số của hệ thống tại một nơi.
Mỗi hằng số đều được gắn nhãn [RC#] để tra cứu ngược về ràng buộc
nghiệp vụ gốc trong tài liệu đặc tả.

BẢNG ÁNH XẠ RÀNG BUỘC → OBJECTIVE:
─────────────────────────────────────────────────────────────────
  G   [RC1,  RC10]  Ràng buộc cứng  — loại bỏ nghiệm infeasible
  F1  [RC3,  RC8, RC9]  Công bằng   — trục chính của bài toán
  F2  [RC6,  RC7, RC11] Chất lượng  — ưu tiên phụ về vận hành
  F3  [RC14]             Cuối tuần  — ưu tiên phụ về nghỉ ngơi
─────────────────────────────────────────────────────────────────
  Thứ tự ưu tiên tổng quát:   G  >  F1  >  F2  >  F3
"""

import os

# ═══════════════════════════════════════════════════════════════════
# PHẦN 1 ─ ĐƯỜNG DẪN FILE
# ═══════════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_STAFF_FILE     = os.path.join(DATA_DIR,   "can_bo.xlsx")
INPUT_SHIFT_FILE     = os.path.join(DATA_DIR,   "ca_thi.xlsx")
OUTPUT_SCHEDULE_FILE = os.path.join(OUTPUT_DIR, "Ket_Qua_Xep_Lich.xlsx")


# ═══════════════════════════════════════════════════════════════════
# PHẦN 2 ─ CẤU HÌNH THUẬT TOÁN NSGA-II
# ═══════════════════════════════════════════════════════════════════

POPULATION_SIZE = 300   
NUM_GENERATIONS = 1000  
MUTATION_RATE   = 0.45  
RANDOM_SEED     = 42    

ALLOWED_SHIFT_DEVIATION = 2
ELDERLY_AGE_THRESHOLD = 45   


# ═══════════════════════════════════════════════════════════════════
# PHẦN 3 ─ TRỌNG SỐ CHỌN NGHIỆM (TCHEBYCHEFF SCALARIZATION)
# ═══════════════════════════════════════════════════════════════════

WEIGHT_FAIRNESS_F1 = 0.70   
WEIGHT_QUALITY_F2  = 0.20   
WEIGHT_WEEKEND_F3  = 0.10   

TCHEBYCHEFF_AUGMENTATION_COEFF = 0.05   


# ═══════════════════════════════════════════════════════════════════
# PHẦN 4 ─ HỆ THỐNG ĐIỂM PHẠT
# ═══════════════════════════════════════════════════════════════════

# ── G │ Ràng buộc cứng ─────────────────────────────────── [RC1, RC10] ──────
HARD_CONFLICT_PENALTY = 10_000_000_000.0    # 1e10 — "Án tử hình" (Death Penalty)

# ── F1a │ Phân bổ cơ bản ───────────────────────────────── [RC3, RC8] ───────
NO_ASSIGNMENT_PENALTY   = 150_000_000.0   
OUT_OF_RANGE_PENALTY    =  60_000_000.0   

# ── F1b │ Công bằng phân phối ──────────────────────────── [RC8] ────────────
MAX_MIN_GAP_PENALTY     =  15_000_000.0   
STD_DEVIATION_PENALTY   =   5_000_000.0   

# ── F1c │ Công bằng liên thế hệ & địa lý ─────────────── [RC8+] ─────────────
ELDERLY_HEAVIER_LOAD_PENALTY   = 30_000_000.0  
DISTANT_HEAVIER_LOAD_PENALTY   = 50_000_000.0  

# ── F1d │ Ưu tiên cùng cơ sở trong ngày ──────────────── [RC9] ──────────────
SAME_DAY_CAMPUS_SWITCH_PENALTY =  8_000_000.0  

# ── F2a │ Tối ưu quãng đường ──────────────────────────── [RC7] ──────────────
# Nâng Scale của Quãng đường để cân bằng với các Penalty khác trong F2
TRAVEL_DISTANCE_WEIGHT  = 500.0   # [RC7] Hệ số nhân km — Đã tăng từ 2.0 lên 500.0

# ── F2b │ Bảo vệ cán bộ cao tuổi ─────────────────────── [RC6] ──────────────
ELDERLY_LATE_SHIFT_PENALTY   = 15_000.0   
ELDERLY_SHIFT_OVERLOAD_PENALTY = 20_000.0 

# ── F2c │ Hạn chế ca liên tiếp cùng cơ sở ────────────── [RC11] ─────────────
CONSECUTIVE_SAME_CAMPUS_PENALTY = 5_000.0 

# ── F2d │ Hạn chế lặp cặp gác chung (Diversity of Pairs) ────────────────────
REPEAT_PAIR_PENALTY = 12_000.0 

# ── F3 │ Cân bằng ca cuối tuần ────────────────────────── [RC14] ─────────────
WEEKEND_OVERLOAD_PENALTY = 7_000.0