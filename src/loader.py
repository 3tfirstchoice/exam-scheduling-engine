import sys
import pandas as pd

def load_data(shift_file, staff_file):
    try:
        shift_df = pd.read_excel(shift_file)
        staff_df = pd.read_excel(staff_file)

        # Chuẩn hóa chuỗi: loại bỏ khoảng trắng thừa ở hai đầu
        for df in [shift_df, staff_df]:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()

        # Đảm bảo kiểu dữ liệu cần thiết cho việc tính toán
        shift_df['Số lượng cán bộ cần thiết'] = (
            pd.to_numeric(shift_df['Số lượng cán bộ cần thiết'], errors='coerce')
            .fillna(1)
            .astype(int)
        )
        shift_df['MS Ca thi'] = shift_df['MS Ca thi'].astype(str)

        # Kiểm tra tính toàn vẹn của dữ liệu đầu vào (Cột bắt buộc)
        required_shift_cols = ['MS Ca thi', 'Ca thi', 'Thứ', 'Ngày', 'Cơ sở', 'Số lượng cán bộ cần thiết']
        required_staff_cols = ['MS của CÁN BỘ COI THI', 'Tuổi']
        
        for col in required_shift_cols:
            if col not in shift_df.columns:
                print(f"[WARNING] Missing column '{col}' in shift data file.")
                
        for col in required_staff_cols:
            if col not in staff_df.columns:
                print(f"[WARNING] Missing column '{col}' in staff data file.")

        print(f"[INFO] Data loaded successfully: {len(shift_df)} shifts, {len(staff_df)} staff members.")
        return shift_df, staff_df

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read data files: {e}")
        sys.exit(1)