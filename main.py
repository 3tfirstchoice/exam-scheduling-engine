import warnings
import config
from src.loader import load_data
from src.model import run_nsga2_scheduler
from src.exporter import export_results

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("[INFO] EXAM SCHEDULING SYSTEM ")
    print("[INFO] Algorithm: NSGA-II (Multi-Objective Optimization)")
    print("=" * 60)

    # 1. Nạp dữ liệu
    shift_df, staff_df = load_data(config.INPUT_SHIFT_FILE, config.INPUT_STAFF_FILE)

    # 2. Chạy thuật toán tối ưu
    best_assignments, flattened_slots = run_nsga2_scheduler(shift_df, staff_df)

    # 3. Xuất kết quả
    export_results(best_assignments, flattened_slots, staff_df, config.OUTPUT_SCHEDULE_FILE)

    print("\n" + "=" * 60)
    print("[RESULT] PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()