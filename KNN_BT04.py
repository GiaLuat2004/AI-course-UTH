import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Dữ liệu từ bảng
data = {
    'Experience': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    'Salary': [0.0, 0.0, 0.0, 0.0, 60.0, 64.0, 55.0, 61.0, 66.0, 83.0, 93.0, 91.0, 98.0, 101.0]
}
# Test với experience = 6.3
test_experience = 6.3
k_value = 5

X = np.array(data['Experience']).reshape(-1, 1)
y = np.array(data['Salary'])

print("=" * 60)
print("BÀI TẬP KNN - DỰ ĐOÁN LƯƠNG THEO KINH NGHIỆM")
print("=" * 60)

# Hàm in bảng chi tiết với ký tự ASCII đơn giản (tương thích Windows)
def print_neighbors_table(neighbors_data, title="K láng giềng gần nhất", use_unicode=False):
    """
    In bảng chi tiết các láng giềng với ký tự ASCII
    
    Parameters:
    - neighbors_data: list of tuples (experience, salary, distance)
    - title: tiêu đề của bảng
    - use_unicode: True = dùng box-drawing Unicode, False = dùng ASCII đơn giản
    """
    print(f"\n{title}:")
    
    if use_unicode:
        # Box-drawing characters (đẹp hơn nhưng có thể lệch trên Windows)
        print("┌─────────────────┬─────────────────┬─────────────────┐")
        print("│   Experience    │     Salary      │   Khoảng cách   │")
        print("├─────────────────┼─────────────────┼─────────────────┤")
        for exp, salary, dist in neighbors_data:
            print(f"│ {exp:^15.1f} │ {salary:^15.1f} │ {dist:^15.4f} │")
        print("└─────────────────┴─────────────────┴─────────────────┘")
    else:
        # ASCII đơn giản (tương thích tốt với mọi console)
        print("+" + "-" * 17 + "+" + "-" * 17 + "+" + "-" * 17 + "+")
        print("|".ljust(1) + "   Experience    " + "|".ljust(1) + "     Salary      " + "|".ljust(1) + "   Khoảng cách   " + "|")
        print("+" + "=" * 17 + "+" + "=" * 17 + "+" + "=" * 17 + "+")
        for exp, salary, dist in neighbors_data:
            print(f"| {exp:^15.1f} | {salary:^15.1f} | {dist:^15.4f} |")
        print("+" + "-" * 17 + "+" + "-" * 17 + "+" + "-" * 17 + "+")

# Câu 1: Tạo hàm knn_predictor tự cài đặt
def knn_predictor(X_train, y_train, x_test, k=3, verbose=True, use_unicode=False):
    """
    Hàm dự đoán KNN tự cài đặt
    
    Parameters:
    - X_train: mảng các giá trị experience huấn luyện
    - y_train: mảng các giá trị salary tương ứng
    - x_test: giá trị experience cần dự đoán
    - k: số lượng láng giềng gần nhất
    - verbose: True = in chi tiết, False = không in
    - use_unicode: True = dùng box-drawing Unicode, False = ASCII đơn giản
    
    Returns:
    - Giá trị salary dự đoán
    """
    # Tính khoảng cách Euclidean từ điểm test đến tất cả điểm train
    distances = []
    for i in range(len(X_train)):
        # Khoảng cách Euclidean
        dist = np.sqrt((X_train[i][0] - x_test) ** 2)
        distances.append((dist, y_train[i], X_train[i][0]))
    
    # Sắp xếp theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[0])
    
    # Lấy k láng giềng gần nhất
    k_nearest = distances[:k]
    
    # In thông tin chi tiết về k láng giềng gần nhất (nếu verbose=True)
    if verbose:
        neighbors_data = [(exp, salary, dist) for dist, salary, exp in k_nearest]
        print_neighbors_table(neighbors_data, f"{k} láng giềng gần nhất với experience = {x_test}", use_unicode)
    
    # Tính trung bình salary của k láng giềng (cho bài toán hồi quy)
    predicted_salary = np.mean([salary for _, salary, _ in k_nearest])
    
    return predicted_salary


print("\n" + "=" * 60)
print("CÂU 1: SỬ DỤNG HÀM TỰ CÀI ĐẶT knn_predictor")
print("=" * 60)

# In chi tiết với ASCII đơn giản (use_unicode=False để tránh lệch trên Windows)
predicted_salary_custom = knn_predictor(X, y, test_experience, k=k_value, verbose=True, use_unicode=False)
print(f"\nDự đoán salary với experience = {test_experience} (k={k_value}):")
print(f"Salary dự đoán = {predicted_salary_custom:.2f}")

# Câu 2: So sánh với sklearn
print("\n" + "=" * 60)
print("CÂU 2: SO SÁNH VỚI THƯ VIỆN SKLEARN")
print("=" * 60)

# Tạo mô hình KNN với sklearn
knn_sklearn = KNeighborsRegressor(n_neighbors=k_value)
knn_sklearn.fit(X, y)

# Dự đoán với sklearn
X_test = np.array([[test_experience]])
predicted_salary_sklearn = knn_sklearn.predict(X_test)[0]

print(f"\nDự đoán salary với experience = {test_experience} (k={k_value}):")
print(f"Salary dự đoán (sklearn) = {predicted_salary_sklearn:.2f}")

# Tìm k láng giềng gần nhất từ sklearn
distances_sklearn, indices_sklearn = knn_sklearn.kneighbors(X_test)

neighbors_sklearn = [(X[idx][0], y[idx], dist) for idx, dist in zip(indices_sklearn[0], distances_sklearn[0])]
print_neighbors_table(neighbors_sklearn, f"{k_value} láng giềng gần nhất (theo sklearn)", use_unicode=False)

# So sánh kết quả
print("\n" + "=" * 60)
print("SO SÁNH KẾT QUẢ")
print("=" * 60)
print(f"Dự đoán từ hàm tự cài đặt: {predicted_salary_custom:.2f}")
print(f"Dự đoán từ sklearn:         {predicted_salary_sklearn:.2f}")
print(f"Chênh lệch:                 {abs(predicted_salary_custom - predicted_salary_sklearn):.6f}")

if abs(predicted_salary_custom - predicted_salary_sklearn) < 0.001:
    print("\n✓ Kết quả KHỚP HOÀN TOÀN! Hàm tự cài đặt chính xác.")
else:
    print("\n✗ Có sự khác biệt nhỏ giữa hai phương pháp.")

# Thử nghiệm với các giá trị k khác nhau
print("\n" + "=" * 60)
print("THỰC NGHIỆM VỚI CÁC GIÁ TRỊ K KHÁC NHAU")
print("=" * 60)
print("\n+" + "-" * 10 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
print("|    K     |  Dự đoán (custom)  |  Dự đoán (sklearn) |  Chênh lệch   |")
print("+" + "=" * 10 + "+" + "=" * 20 + "+" + "=" * 20 + "+" + "=" * 15 + "+")

for k in [1, 3, 5, 7]:
    # KHÔNG in chi tiết (verbose=False) để tránh output dài
    pred_custom = knn_predictor(X, y, test_experience, k=k, verbose=False)
    
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X, y)
    pred_sklearn = knn_temp.predict(X_test)[0]
    
    diff = abs(pred_custom - pred_sklearn)
    print(f"| {k:^8} | {pred_custom:^18.2f} | {pred_sklearn:^18.2f} | {diff:^13.6f} |")

print("+" + "-" * 10 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")

print("\n" + "=" * 60)
print("KẾT LUẬN")
print("=" * 60)
print("""
1. Hàm knn_predictor tự cài đặt hoạt động chính xác, cho kết quả
   giống hệt với thư viện sklearn.

2. Thuật toán KNN hoạt động bằng cách:
   - Tính khoảng cách từ điểm cần dự đoán đến tất cả điểm trong tập train
   - Chọn k điểm gần nhất
   - Lấy trung bình giá trị salary của k điểm đó

3. Với experience = 6.3 và k=3:
   - Các láng giềng gần nhất là: 6.0, 6.5, 7.0
   - Salary tương ứng: 93.0, 91.0, 98.0
   - Trung bình: (93.0 + 91.0 + 98.0) / 3 = 94.0
""")
