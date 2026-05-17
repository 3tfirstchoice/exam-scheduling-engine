import pandas as pd
import numpy as np

# ===========================================================================
# CÁC HÀM XÂY DỰNG SHEET DỮ LIỆU (HELPER FUNCTIONS)
# ===========================================================================

def _build_sheet_ca_thi(slots_df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 1: Gom nhóm danh sách cán bộ theo từng ca thi cụ thể."""
    group_cols = [c for c in slots_df.columns if c not in ['MS_CB_Phan_Cong', 'Số lượng cán bộ cần thiết']]
    
    ca_thi_df = slots_df.groupby(group_cols, dropna=False).agg(
        Số_lượng_thực_tế=('MS_CB_Phan_Cong', 'count'),
        Danh_sách_CB_Phân_công=('MS_CB_Phan_Cong', lambda x: ', '.join(x.dropna()))
    ).reset_index()
    
    return ca_thi_df


def _build_sheet_can_bo(slots_df: pd.DataFrame, staff_df: pd.DataFrame, staff_id_col: str) -> pd.DataFrame:
    """Sheet 2: Thống kê lịch trực chi tiết cho từng cán bộ."""
    # 1. Xử lý định dạng ngày tháng an toàn
    if pd.api.types.is_datetime64_any_dtype(slots_df['Ngày']):
        date_str = slots_df['Ngày'].dt.strftime('%d/%m/%Y')
    else:
        date_str = slots_df['Ngày'].astype(str)

    # 2. Tạo chuỗi mô tả ca thi
    slots_df['Mô_tả_ca'] = date_str + " - " + slots_df['Ca thi'].astype(str) + " (" + slots_df['Cơ sở'].astype(str) + ")"

    # 3. Gom nhóm theo mã cán bộ
    staff_shifts = slots_df.groupby('MS_CB_Phan_Cong').agg(
        Số_ca_trực=('Mô_tả_ca', 'count'),
        Danh_sách_ca=('Mô_tả_ca', lambda x: '; '.join(x))
    ).reset_index()

    # 4. Ghép với thông tin gốc của cán bộ
    can_bo_df = staff_df.merge(staff_shifts, left_on=staff_id_col, right_on='MS_CB_Phan_Cong', how='left')
    
    # 5. Xử lý người không có lịch
    can_bo_df['Số_ca_trực'] = can_bo_df['Số_ca_trực'].fillna(0).astype(int)
    can_bo_df['Danh_sách_ca'] = can_bo_df['Danh_sách_ca'].fillna("Không có lịch")
    
    # 6. Dọn dẹp cột thừa
    can_bo_df = can_bo_df.drop(columns=['MS_CB_Phan_Cong'], errors='ignore')
    can_bo_df = can_bo_df.rename(columns={'Số_ca_trực': 'Số ca trực', 'Danh_sách_ca': 'Danh sách ca'})
    
    return can_bo_df


def _build_sheet_thong_ke(ca_thi_df: pd.DataFrame, ca_counts: np.ndarray, 
                          total_assigns: int, total_staff: int) -> pd.DataFrame:
    """Sheet 3: Tính toán các chỉ số thống kê tổng thể."""
    max_ca = ca_counts.max()
    min_ca = ca_counts.min()

    stats_dict = {
        'Chỉ số': [
            'Tổng số ca thi',
            'Tổng số cán bộ',
            'Tổng số lượt phân công',
            'Trung bình ca/người',
            'Ca nhiều nhất (Max)',
            'Ca ít nhất (Min)',
            'Chênh lệch Max - Min (Gap)',
            'Độ lệch chuẩn (Std Dev)'
        ],
        'Giá trị': [
            len(ca_thi_df),
            total_staff,
            total_assigns,
            round(total_assigns / total_staff, 4),
            max_ca,
            min_ca,
            max_ca - min_ca,
            round(np.std(ca_counts.astype(float)), 4)
        ]
    }
    return pd.DataFrame(stats_dict)


# ===========================================================================
# HÀM XUẤT FILE CHÍNH (GỌI TỪ FILE MAIN)
# ===========================================================================

def export_results(best_chromosome: np.ndarray, slots_df: pd.DataFrame, 
                   staff_df: pd.DataFrame, output_file: str) -> None:
    """
    Điểm vào chính: Nhận kết quả từ thuật toán và xuất ra file Excel gồm đúng 3 Sheets.
    """
    # 1. Gán mã cán bộ dựa trên index của best_chromosome
    staff_id_col = 'MS của CÁN BỘ COI THI'  # Đổi tên cột này nếu file gốc của ông đặt khác
    staff_ids = staff_df[staff_id_col].values
    
    working_slots_df = slots_df.copy()
    working_slots_df['MS_CB_Phan_Cong'] = staff_ids[best_chromosome]

    # 2. Xây dựng 3 DataFrames
    df_ca_thi   = _build_sheet_ca_thi(working_slots_df)
    df_can_bo   = _build_sheet_can_bo(working_slots_df, staff_df, staff_id_col)
    
    ca_counts   = df_can_bo['Số ca trực'].values
    df_thong_ke = _build_sheet_thong_ke(df_ca_thi, ca_counts, len(working_slots_df), len(staff_df))

    # 3. Ghi ra file Excel với tên Sheet tiếng Việt chuẩn
    print(f"\n[EXPORT] Đang ghi dữ liệu ra file: {output_file} ...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_ca_thi.to_excel(writer, sheet_name='Ca Thi', index=False)
        df_can_bo.to_excel(writer, sheet_name='Cán Bộ', index=False)
        df_thong_ke.to_excel(writer, sheet_name='Thống Kê', index=False)

    print("-" * 65)
    print(f"[OK] Đã xuất thành công 3 Sheet: 'Ca Thi', 'Cán Bộ', 'Thống Kê'")
    print("-" * 65)