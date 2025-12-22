'''
Bài toàn đong nước (Water Jug Problem) với hai bình có dung tích khác nhau.
Mục tiêu là đong được một lượng nước cụ thể bằng cách sử dụng hai bình này và các hành động đổ đầy, đổ hết, và rót nước giữa hai bình.
Giải pháp được thực hiện bằng hai thuật toán tìm kiếm: BFS (Tìm kiếm theo chiều rộng) và DFS (Tìm kiếm theo chiều sâu).
'''
from collections import deque

# Cấu hình bài toán
CAPACITY_A = 3  # Bình A: 3 lít
CAPACITY_B = 8  # Bình B: 8 lít
GOAL = 7        # Mục tiêu: 7 lít

def get_action_name(old_state, new_state):
    """Trả về tên hành động từ trạng thái cũ sang trạng thái mới"""
    old_a, old_b = old_state
    new_a, new_b = new_state
    
    # Ưu tiên kiểm tra hành động rót nước giữa các bình trước
    # Đổ B sang A (B giảm, A tăng)
    if new_a > old_a and new_b < old_b:
        amount = new_a - old_a
        return f"Đổ ({amount} lít) từ B sang A "
    # Đổ A sang B (A giảm, B tăng)
    elif new_b > old_b and new_a < old_a:
        amount = new_b - old_b
        return f"Đổ ({amount} lít) từ A sang B "
    # Đổ đầy bình A từ nguồn
    elif new_a == CAPACITY_A and old_a != CAPACITY_A and new_b == old_b:
        return "Đổ đầy bình A"
    # Đổ đầy bình B từ nguồn
    elif new_b == CAPACITY_B and old_b != CAPACITY_B and new_a == old_a:
        return "Đổ đầy bình B"
    # Đổ hết bình A
    elif new_a == 0 and old_a != 0:
        return "Đổ hết bình A"
    # Đổ hết bình B
    elif new_b == 0 and old_b != 0:
        return "Đổ hết bình B"
    
    return f"Không rõ: {old_state} -> {new_state}"

def get_successors(state, m, n):
    """Sinh ra các trạng thái kế tiếp từ trạng thái hiện tại"""
    jug1, jug2 = state
    successors = []
    # 1: Fill jug1 (Đổ đầy bình A)
    successors.append((m, jug2))

    # 2: Fill jug2 (Đổ đầy bình B)
    successors.append((jug1, n))
    
    # 3: Empty jug1 (Đổ hết bình A)
    successors.append((0, jug2))
    
    # 4: Empty jug2 (Đổ hết bình B)
    successors.append((jug1, 0))
    
    # 5: Pour jug1 into jug2 (Đổ A sang B)
    pour1to2 = min(jug1, n - jug2)
    successors.append((jug1 - pour1to2, jug2 + pour1to2))
    
    # 6: Pour jug2 into jug1 (Đổ B sang A)
    pour2to1 = min(jug2, m - jug1)
    successors.append((jug1 + pour2to1, jug2 - pour2to1))
    
    return successors

def print_table(path, jug1_final, jug2_final, d):
    """In kết quả dưới dạng bảng"""
    print("\n┌──────┬─────────────────────────────────────┬─────────┬─────────┐")
    print("│ Bước │           Hành động                 │  Bình A │  Bình B │")
    print("├──────┼─────────────────────────────────────┼─────────┼─────────┤")
    print(f"│  0   │ Trạng thái ban đầu                  │    0    │    0    │")
    
    for step, (action, state) in enumerate(path, 1):
        a, b = state
        print(f"│  {step:<2}  │ {action:<35} │    {a}    │    {b}    │")
    
    print("└──────┴─────────────────────────────────────┴─────────┴─────────┘")
    
    if jug1_final == d:
        print(f"\n✓ Kết quả: Bình A chứa {d} lít (Mục tiêu đạt được!)")
    else:
        print(f"\n✓ Kết quả: Bình B chứa {d} lít (Mục tiêu đạt được!)")

def solve_bfs(m, n, d):
    """Giải bài toán bằng thuật toán BFS - Tìm kiếm theo chiều rộng"""
    print("\n" + "="*70)
    print("GIẢI BẰNG BFS (Breadth-First Search - Tìm kiếm theo chiều rộng)")
    print("="*70)
    if d > max(m, n):
        print("Không thể đong được lượng nước lớn hơn bình lớn nhất!")
        return -1
    # Queue: (state, path)
    # path là danh sách các (state, action) đã đi qua
    q = deque([((0, 0), [])])
    visited = set([(0, 0)])
    while q:
        state, path = q.popleft()
        jug1, jug2 = state
        # Kiểm tra đã đạt mục tiêu chưa
        if jug1 == d or jug2 == d:
            print(f"\n✓ Đã tìm thấy giải pháp để đong được {d} lít!")
            print(f"✓ Tổng số bước: {len(path)}")
            # In đường đi dạng (a, b) -> (a, b) -> ...
            path_str = "(0, 0)"
            for _, state in path:
                path_str += f" -> {state}"
            print(f"\n📍 Giải: {path_str}")
            # In bảng kết quả
            print_table(path, jug1, jug2, d)
            return len(path)
        # Duyệt các trạng thái kế tiếp
        for next_state in get_successors(state, m, n):
            if next_state not in visited:
                visited.add(next_state)
                action = get_action_name(state, next_state)
                new_path = path + [(action, next_state)]
                q.append((next_state, new_path))
    print("Không tìm thấy giải pháp!")
    return -1

def solve_dfs(m, n, d):
    """Giải bài toán bằng thuật toán DFS - Tìm kiếm theo chiều sâu"""
    print("\n" + "="*70)
    print("GIẢI BẰNG DFS (Depth-First Search - Tìm kiếm theo chiều sâu)")
    print("="*70)
    if d > max(m, n):
        print("Không thể đong được lượng nước lớn hơn bình lớn nhất!")
        return -1
    # Stack: (state, path)
    stack = [((0, 0), [])]
    visited = set([(0, 0)])
    while stack:
        state, path = stack.pop()
        jug1, jug2 = state
        # Kiểm tra đã đạt mục tiêu chưa
        if jug1 == d or jug2 == d:
            print(f"\n✓ Đã tìm thấy giải pháp để đong được {d} lít!")
            print(f"✓ Tổng số bước: {len(path)}")
            # In đường đi dạng (a, b) -> (a, b) -> ...
            path_str = "(0, 0)"
            for _, state in path:
                path_str += f" -> {state}"
            print(f"\n📍 Giải: {path_str}")
            # In bảng kết quả
            print_table(path, jug1, jug2, d)
            return len(path)
        # Duyệt các trạng thái kế tiếp
        for next_state in get_successors(state, m, n):
            if next_state not in visited:
                visited.add(next_state)
                action = get_action_name(state, next_state)
                new_path = path + [(action, next_state)]
                stack.append((next_state, new_path))
    print("Không tìm thấy giải pháp!")
    return -1

if __name__ == "__main__":
    # Bình A = 3 lít, Bình B = 8 lít, Mục tiêu = 7 lít
    m, n, d = CAPACITY_A, CAPACITY_B, GOAL
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "BÀI TOÁN ĐONG NƯỚC (WATER JUG PROBLEM)" + " "*10 + "║")
    print("╚" + "="*58 + "╝")
    print(f"\n📋 Điều kiện bài toán:")
    print(f"   • Bình A: {m} lít")
    print(f"   • Bình B: {n} lít")
    print(f"   • Mục tiêu: Đong được {d} lít")
    print(f"   • Trạng thái ban đầu: (0, 0) - cả hai bình đều rỗng")
    # Giải bằng BFS
    steps_bfs = solve_bfs(m, n, d)
    # Giải bằng DFS
    steps_dfs = solve_dfs(m, n, d)
