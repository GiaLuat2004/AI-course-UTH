import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from collections import Counter

# Dữ liệu từ bảng
data = {
    'Experience': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    'Salary': [0.0, 0.0, 0.0, 0.0, 60.0, 64.0, 55.0, 61.0, 66.0, 83.0, 93.0, 91.0, 98.0, 101.0]
}

X = np.array(data['Experience']).reshape(-1, 1)
y = np.array(data['Salary'])

print("=" * 60)
print("BÀI TẬP KNN - DỰ ĐOÁN LƯƠNG THEO KINH NGHIỆM")
print("=" * 60)

# Câu 1: Tạo hàm knn_predictor tự cài đặt
def knn_predictor(X_train, y_train, x_test, k=3):
    """
    Hàm dự đoán KNN tự cài đặt
    
    Parameters:
    - X_train: mảng các giá trị experience huấn luyện
    - y_train: mảng các giá trị salary tương ứng
    - x_test: giá trị experience cần dự đoán
    - k: số lượng láng giềng gần nhất
    
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
    
    # In thông tin chi tiết về k láng giềng gần nhất
    print(f"\n{k} láng giềng gần nhất với experience = {x_test}:")
    print(f"{'Experience':<15} {'Salary':<15} {'Khoảng cách':<15}")
    print("-" * 45)
    for dist, salary, exp in k_nearest:
        print(f"{exp:<15.1f} {salary:<15.1f} {dist:<15.4f}")
    
    # Tính trung bình salary của k láng giềng (cho bài toán hồi quy)
    predicted_salary = np.mean([salary for _, salary, _ in k_nearest])
    
    return predicted_salary

# Test với experience = 6.3
test_experience = 6.3
k_value = 3

print("\n" + "=" * 60)
print("CÂU 1: SỬ DỤNG HÀM TỰ CÀI ĐẶT knn_predictor")
print("=" * 60)

predicted_salary_custom = knn_predictor(X, y, test_experience, k=k_value)
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

print(f"\n{k_value} láng giềng gần nhất (theo sklearn):")
print(f"{'Experience':<15} {'Salary':<15} {'Khoảng cách':<15}")
print("-" * 45)
for idx, dist in zip(indices_sklearn[0], distances_sklearn[0]):
    print(f"{X[idx][0]:<15.1f} {y[idx]:<15.1f} {dist:<15.4f}")

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
print(f"\n{'K':<10} {'Dự đoán (custom)':<20} {'Dự đoán (sklearn)':<20} {'Chênh lệch':<15}")
print("-" * 65)

for k in [1, 3, 5, 7]:
    pred_custom = knn_predictor(X, y, test_experience, k=k)
    
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X, y)
    pred_sklearn = knn_temp.predict(X_test)[0]
    
    diff = abs(pred_custom - pred_sklearn)
    print(f"{k:<10} {pred_custom:<20.2f} {pred_sklearn:<20.2f} {diff:<15.6f}")

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
