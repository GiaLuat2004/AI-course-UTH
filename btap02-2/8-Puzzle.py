import heapq
from typing import List, Tuple, Dict, Set

# ==================================================
# TRẠNG THÁI ĐÍCH
# ==================================================
GOAL = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Pre-compute vị trí đích của mỗi số để tối ưu hóa manhattan_distance
GOAL_POS: Dict[int, Tuple[int, int]] = {}
for i in range(3):
    for j in range(3):
        GOAL_POS[GOAL[i][j]] = (i, j)

# ==================================================
# HEURISTIC: MANHATTAN DISTANCE
# ==================================================
def manhattan_distance(state: List[List[int]]) -> int:
    """
    Tính khoảng cách Manhattan từ trạng thái hiện tại đến trạng thái đích.
    Manhattan distance = tổng khoảng cách hàng + tổng khoảng cách cột của mỗi ô.
    """
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:  # Bỏ qua ô trống
                gi, gj = GOAL_POS[val]  # Vị trí đích của giá trị val
                dist += abs(i - gi) + abs(j - gj)
    return dist

def print_manhattan_detail(state: List[List[int]]) -> str:
    """
    In chi tiết cách tính Manhattan distance cho mỗi ô.
    Trả về chuỗi mô tả chi tiết.
    """
    lines = []
    lines.append("   Chi tiết tính Manhattan distance:")
    total = 0
    
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:  # Bỏ qua ô trống
                gi, gj = GOAL_POS[val]  # Vị trí đích của giá trị val
                dist = abs(i - gi) + abs(j - gj)
                total += dist
                lines.append(f"      • Ô {val}: từ ({i},{j}) → ({gi},{gj}) = |{i}-{gi}| + |{j}-{gj}| = {dist}")
    
    lines.append(f"      → TỔNG h(n) = {total}")
    return "\n".join(lines)

# ==================================================
# SINH TRẠNG THÁI KỀ
# ==================================================
def get_neighbors(state: List[List[int]]) -> List[List[List[int]]]:
    """
    Tạo các trạng thái kề bằng cách di chuyển ô trống lên/xuống/trái/phải.
    """
    neighbors = []
    
    # Tìm vị trí ô trống (0)
    x, y = 0, 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                x, y = i, j
                break

    # Thử 4 hướng: lên, xuống, trái, phải
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            # Tạo trạng thái mới bằng cách copy và hoán đổi
            new_state = [row[:] for row in state]  # Deep copy
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)

    return neighbors

# ==================================================
# TIỆN ÍCH
# ==================================================
def to_tuple(state: List[List[int]]) -> Tuple:
    """Chuyển mảng 2D thành tuple để dùng làm key trong set/dict."""
    return tuple(tuple(row) for row in state)

def states_equal(state1: List[List[int]], state2: List[List[int]]) -> bool:
    """Kiểm tra hai trạng thái có bằng nhau không."""
    for i in range(3):
        for j in range(3):
            if state1[i][j] != state2[i][j]:
                return False
    return True

def print_puzzle_inline(state: List[List[int]]) -> List[str]:
    """Trả về các dòng của puzzle để in inline trong bảng."""
    lines = []
    lines.append("┌───┬───┬───┐")
    for i in range(3):
        row = state[i]
        lines.append("│ " + " │ ".join('_' if x == 0 else str(x) for x in row) + " │")
        if i < 2:
            lines.append("├───┼───┼───┤")
    lines.append("└───┴───┴───┘")
    return lines

def print_puzzle_box(state: List[List[int]]):
    """In trạng thái dạng bảng đẹp."""
    for line in print_puzzle_inline(state):
        print(line)

# ==================================================
# THUẬT TOÁN TÌM KIẾM
# ==================================================
def greedy_bfs(start: List[List[int]]) -> Tuple[List[List[List[int]]], int, int]:
    """
    Thuật toán Greedy Best-First Search.
    
    Chiến lược: Priority = h(n) = Manhattan distance
    - Chỉ xem xét heuristic, bỏ qua chi phí đã đi
    - Chọn trạng thái gần đích nhất theo heuristic
    
    Args:
        start: Trạng thái ban đầu (mảng 2D)
    
    Returns:
        (path, nodes_expanded, nodes_generated)
    """
    pq = []
    visited: Set[Tuple] = set()
    g_score: Dict[Tuple, int] = {}
    
    # Tính heuristic ban đầu
    h0 = manhattan_distance(start)
    start_t = to_tuple(start)
    
    # Counter để đảm bảo thứ tự khi priority bằng nhau
    counter = 0
    
    # Priority chỉ dựa vào h(n)
    priority = h0
    
    heapq.heappush(pq, (priority, counter, 0, start, []))
    g_score[start_t] = 0
    
    nodes_expanded = 0
    nodes_generated = 1
    
    while pq:
        _, _, g, current, path = heapq.heappop(pq)
        current_t = to_tuple(current)
        
        # Kiểm tra đã thăm chưa
        if current_t in visited:
            continue
        
        # Kiểm tra đạt đích chưa
        if states_equal(current, GOAL):
            return path + [current], nodes_expanded, nodes_generated
        
        visited.add(current_t)
        nodes_expanded += 1
        
        # Mở rộng các trạng thái kề
        for neighbor in get_neighbors(current):
            n_t = to_tuple(neighbor)
            new_g = g + 1
            
            if n_t in visited:
                continue
            
            # Kiểm tra xem có tìm được đường đi tốt hơn không
            if n_t in g_score and new_g >= g_score[n_t]:
                continue
            
            g_score[n_t] = new_g
            h = manhattan_distance(neighbor)
            
            # Greedy BFS: Priority chỉ dùng h(n)
            priority = h
            
            counter += 1
            heapq.heappush(
                pq,
                (priority, counter, new_g, neighbor, path + [current])
            )
            nodes_generated += 1
    
    return [], nodes_expanded, nodes_generated


def astar_search(start: List[List[int]]) -> Tuple[List[List[List[int]]], int, int]:
    """
    Thuật toán A* Search.
    
    Chiến lược: Priority = f(n) = g(n) + h(n)
    - g(n): Chi phí thực tế từ điểm bắt đầu (số bước đã đi)
    - h(n): Ước lượng chi phí đến đích (Manhattan distance)
    - Kết hợp cả chi phí đã đi và ước lượng còn lại
    
    Args:
        start: Trạng thái ban đầu (mảng 2D)
    
    Returns:
        (path, nodes_expanded, nodes_generated)
    """
    pq = []
    visited: Set[Tuple] = set()
    g_score: Dict[Tuple, int] = {}
    
    # Tính heuristic ban đầu
    h0 = manhattan_distance(start)
    start_t = to_tuple(start)
    
    # Counter để đảm bảo thứ tự khi priority bằng nhau
    counter = 0
    
    # Priority dựa vào f(n) = g(n) + h(n)
    priority = 0 + h0  # f = g + h
    
    heapq.heappush(pq, (priority, counter, 0, start, []))
    g_score[start_t] = 0
    
    nodes_expanded = 0
    nodes_generated = 1
    
    while pq:
        _, _, g, current, path = heapq.heappop(pq)
        current_t = to_tuple(current)
        
        # Kiểm tra đã thăm chưa
        if current_t in visited:
            continue
        
        # Kiểm tra đạt đích chưa
        if states_equal(current, GOAL):
            return path + [current], nodes_expanded, nodes_generated
        
        visited.add(current_t)
        nodes_expanded += 1
        
        # Mở rộng các trạng thái kề
        for neighbor in get_neighbors(current):
            n_t = to_tuple(neighbor)
            new_g = g + 1
            
            if n_t in visited:
                continue
            
            # Kiểm tra xem có tìm được đường đi tốt hơn không
            if n_t in g_score and new_g >= g_score[n_t]:
                continue
            
            g_score[n_t] = new_g
            h = manhattan_distance(neighbor)
            
            # A*: Priority = f(n) = g(n) + h(n)
            priority = new_g + h
            
            counter += 1
            heapq.heappush(
                pq,
                (priority, counter, new_g, neighbor, path + [current])
            )
            nodes_generated += 1
    
    return [], nodes_expanded, nodes_generated

# ==================================================
# IN LỜI GIẢI
# ==================================================
def print_solution(path: List[List[List[int]]], title: str, nodes_expanded: int, nodes_generated: int):
    """In chi tiết lời giải dạng bảng thống kê."""
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)
    
    if not path:
        print("❌ Không tìm thấy lời giải!")
        return
    
    steps = len(path) - 1
    print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
    print(f"   • Số bước di chuyển: {steps}")
    print(f"   • Số nút được mở rộng (explored): {nodes_expanded}")
    print(f"   • Số nút được sinh ra (generated): {nodes_generated}")
    print(f"   • Branching factor trung bình: {nodes_generated / max(nodes_expanded, 1):.2f}")
    
    print("\n" + "=" * 100)
    print("BẢNG THỐNG KÊ CHI TIẾT CÁC BƯỚC (với chi tiết tính Manhattan distance)".center(100))
    print("=" * 100)
    
    # In header của bảng
    print("\n┌────────┬────────┬────────┬────────┬─────────────────────────────────────────┐")
    print(f"│{'Bước':^8}│ {'h(n)':^6} │ {'g(n)':^6} │ {'f(n)':^6} │{'Trạng thái Puzzle':<40} │")
    print("├"+ "─" * 7 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * 40 + "┤")
    
    # In từng bước
    for step, state in enumerate(path):
        h = manhattan_distance(state)
        g = step
        f = g + h
        
        # Lấy các dòng của puzzle
        puzzle_lines = print_puzzle_inline(state)
        
        # In dòng đầu tiên với thông tin bước
        if step == 0:
            step_label = "Đầu"
        elif step == len(path) - 1:
            step_label = "Đích"
        else:
            step_label = str(step)
        
        print(f"│{step_label:^7} │ {h:^6} │ {g:^6} │ {f:^6} │ {puzzle_lines[0]:<40}│")
        
        # In các dòng còn lại của puzzle
        for i in range(1, len(puzzle_lines)):
            print(f"│{'':^7} │ {'':^6} │ {'':^6} │ {'':^6} │ {puzzle_lines[i] :<40}│")
        
        # In dòng phân cách giữa các bước (trừ bước cuối)
        if step < len(path) - 1:
            print("├"+ "─" * 7 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * 40 + "┤")
    
    print("└────────┴────────┴────────┴────────┴─────────────────────────────────────────┘")
    
    print("\n" + "=" * 100)

def verify_solution(path: List[List[List[int]]]) -> bool:
    """Kiểm tra tính hợp lệ của lời giải."""
    if not path:
        print("❌ Không có lời giải!")
        return False
    
    # Kiểm tra trạng thái cuối có phải là đích không
    if not states_equal(path[-1], GOAL):
        print("❌ Lỗi: Trạng thái cuối không phải là đích!")
        return False
    
    # Kiểm tra mỗi bước có hợp lệ không
    for i in range(len(path) - 1):
        current = path[i]
        next_state = path[i + 1]
        
        # Đếm số ô khác nhau
        diff_count = 0
        for row in range(3):
            for col in range(3):
                if current[row][col] != next_state[row][col]:
                    diff_count += 1
        
        # Phải có đúng 2 ô khác nhau (ô trống và ô bị hoán đổi)
        if diff_count != 2:
            print(f"❌ Lỗi: Bước {i+1} không hợp lệ! Có {diff_count} ô thay đổi.")
            return False
        
        # Kiểm tra ô trống di chuyển đúng cách (chỉ 1 ô kề)
        x1, y1, x2, y2 = 0, 0, 0, 0
        for row in range(3):
            for col in range(3):
                if current[row][col] == 0:
                    x1, y1 = row, col
                if next_state[row][col] == 0:
                    x2, y2 = row, col
        
        manhattan_move = abs(x1 - x2) + abs(y1 - y2)
        if manhattan_move != 1:
            print(f"❌ Lỗi: Bước {i+1} - ô trống di chuyển không hợp lệ!")
            return False
    
    print("✓ Lời giải hợp lệ!")
    return True

# ==================================================
# MAIN
# ==================================================
def main():
    # Trạng thái ban đầu A: 7 2 4 / 5 _ 6 / 8 3 1
    start = [
        [7, 2, 4],
        [5, 0, 6],
        [8, 3, 1]
    ]
    
    print("=" * 70)
    print("BÀI TOÁN 8 Ô TRƯỢT (8-PUZZLE)")
    print("=" * 70)
    
    print("\n📌 TRẠNG THÁI BAN ĐẦU (A):")
    print_puzzle_box(start)
    print(print_manhattan_detail(start))
    print(f"Manhattan distance đến đích: {manhattan_distance(start)}")
    
    print("\n📌 TRẠNG THÁI ĐÍCH (B):")
    print_puzzle_box(GOAL)
    
    # =====================================================================
    # THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH
    # =====================================================================
    print("\n" + "=" * 70)
    print("THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH (Greedy BeFS)")
    print("=" * 70)
    print("📖 Chiến lược: Priority = h(n) = Manhattan distance")
    print("   - Chỉ xem xét heuristic, bỏ qua chi phí đã đi")
    print("   - Chọn trạng thái gần đích nhất theo heuristic")
    print("✓ Ưu điểm: Tìm kiếm nhanh, mở rộng ít nút")
    print("✗ Nhược điểm: Không đảm bảo tìm được lời giải tối ưu")
    
    path_greedy, nodes_greedy, gen_greedy = greedy_bfs(start)
    print_solution(path_greedy, "KẾT QUẢ: Greedy Best-First Search", nodes_greedy, gen_greedy)
    
    print("\n" + "─" * 70)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI GREEDY BeFS:")
    print("─" * 70)
    verify_solution(path_greedy)
    
    # =====================================================================
    # THUẬT TOÁN 2: A* SEARCH
    # =====================================================================
    print("\n" + "=" * 70)
    print("THUẬT TOÁN 2: A* SEARCH")
    print("=" * 70)
    print("📖 Chiến lược: Priority = f(n) = g(n) + h(n)")
    print("   - g(n): Chi phí thực tế từ điểm bắt đầu (số bước đã đi)")
    print("   - h(n): Ước lượng chi phí đến đích (Manhattan distance)")
    print("   - Kết hợp cả chi phí đã đi và ước lượng còn lại")
    print("✓ Ưu điểm: Đảm bảo tìm được lời giải tối ưu (admissible heuristic)")
    print("✗ Nhược điểm: Có thể mở rộng nhiều nút hơn Greedy BeFS")
    
    path_astar, nodes_astar, gen_astar = astar_search(start)
    print_solution(path_astar, "KẾT QUẢ: A* Search", nodes_astar, gen_astar)
    
    print("\n" + "─" * 70)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI A*:")
    print("─" * 70)
    verify_solution(path_astar)
    
    # =====================================================================
    # SO SÁNH KẾT QUẢ
    # =====================================================================
    print("\n" + "=" * 70)
    print("SO SÁNH VÀ ĐÁNH GIÁ KẾT QUẢ")
    print("=" * 70)
    
    if path_greedy and path_astar:
        steps_greedy = len(path_greedy) - 1
        steps_astar = len(path_astar) - 1
        
        print(f"\n{'Tiêu chí':<35} {'Greedy BeFS':>15} {'A* Search':>15}")
        print("─" * 70)
        print(f"{'Số bước di chuyển':<35} {steps_greedy:>15} {steps_astar:>15}")
        print(f"{'Số nút mở rộng (explored)':<35} {nodes_greedy:>15} {nodes_astar:>15}")
        print(f"{'Số nút sinh ra (generated)':<35} {gen_greedy:>15} {gen_astar:>15}")
        
        print("\n" + "─" * 70)
        print("PHÂN TÍCH:")
        print("─" * 70)
        
        if steps_astar < steps_greedy:
            improvement = ((steps_greedy - steps_astar) / steps_greedy) * 100
            print(f"\n📊 Độ dài đường đi:")
            print(f"   ✓ A* tìm được lời giải TỐI ƯU với {steps_astar} bước")
            print(f"   ✓ A* ngắn hơn Greedy BeFS {steps_greedy - steps_astar} bước ({improvement:.1f}% tốt hơn)")
            print(f"   ✗ Greedy BeFS không tối ưu: {steps_greedy} bước")
        elif steps_astar == steps_greedy:
            print(f"\n📊 Độ dài đường đi:")
            print(f"   ✓ Cả hai đều tìm được lời giải tối ưu: {steps_astar} bước")
        
        if nodes_greedy < nodes_astar:
            reduction = ((nodes_astar - nodes_greedy) / nodes_astar) * 100
            print(f"\n📊 Hiệu quả không gian tìm kiếm:")
            print(f"   ✓ Greedy BeFS hiệu quả hơn, mở rộng ít hơn {nodes_astar - nodes_greedy} nút ({reduction:.1f}%)")
            print(f"   ✗ A* phải khám phá nhiều nút hơn để đảm bảo tối ưu")
        
        print(f"\n💡 Kết luận:")
        print(f"   1. A* đảm bảo tìm đường đi NGẮN NHẤT ({steps_astar} bước)")
        print(f"   2. Greedy BeFS nhanh hơn nhưng không đảm bảo tối ưu ({steps_greedy} bước)")
        print(f"   3. Heuristic Manhattan không bao giờ overestimate khoảng cách thực")
        print(f"   4. A* sử dụng f(n) = g(n) + h(n) để cân bằng giữa chi phí và heuristic")
        print(f"   5. Greedy BeFS chỉ dùng h(n), có thể bị lạc vào local minimum")

if __name__ == "__main__":
    main()
