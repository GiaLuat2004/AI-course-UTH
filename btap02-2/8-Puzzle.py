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

    # Thử 4 hướng theo ưu tiên: trái, phải, lên, xuống
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
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

def print_detailed_path(path: List[List[List[int]]], algorithm: str):
    """
    In chi tiết 4 bước đầu và 4 bước cuối trong path với các trạng thái kề được sinh ra.
    
    LƯU Ý: Với A*, neighbor được CHỌN không nhất thiết có f(n) nhỏ nhất trong số
    các neighbors của state hiện tại. A* chọn NODE có f(n) nhỏ nhất trong TOÀN BỘ
    priority queue (gồm cả các node từ các state khác). Đánh dấu "← CHỌN" chỉ 
    để chỉ ra neighbor nào nằm trong đường đi solution cuối cùng.
    """
    print(f"\n{'='* 80}")
    print(f"CHI TIẾT CÁC BƯỚC TÌM KIẾM - {algorithm}".center(80))
    print(f"{'='* 80}")
  
    total_steps = len(path) - 1  # Số bước (không tính trạng thái đầu)
    
    # Xác định các bước cần in
    if total_steps <= 8:
        # Nếu tổng số bước <= 8, in tất cả
        steps_to_print = list(range(total_steps))
    else:
        # In 4 bước đầu và 4 bước cuối
        steps_to_print = list(range(4)) + list(range(total_steps - 4, total_steps))
    
    for step in range(total_steps):
        if step not in steps_to_print:
            # In dấu ... cho phần bị bỏ qua
            if step == 4:
                print(f"\n{'='* 80}")
                print(f"... (BỎ QUA {total_steps - 8} BƯỚC GIỮA) ...".center(80))
                print(f"{'='* 80}\n")
            continue
        
        state = path[step]
        g = step
        h = manhattan_distance(state)
        f = g + h
        
        print(f"\n{'='* 80}")
        print(f"BƯỚC {step + 1}/{total_steps}: g(n)={g}, h(n)={h}, f(n)={f}")
        print(f"{'='* 80}")
        
        print("\nTrạng thái hiện tại:")
        for row in state:
            print(f"  {row}")
        
        # Sinh các trạng thái kề
        neighbors = get_neighbors(state)
        next_state = path[step + 1]
        
        print(f"\n→ Các trạng thái kề được sinh ra ({len(neighbors)} trạng thái):")
        print("  (Tất cả được thêm vào priority queue, chờ được chọn theo f(n) min)\n")
        
        # Tìm neighbor có f(n) min trong neighbors hiện tại
        min_f_in_neighbors = float('inf')
        for neighbor in neighbors:
            new_g = g + 1
            new_h = manhattan_distance(neighbor)
            new_f = new_g + new_h
            if new_f < min_f_in_neighbors:
                min_f_in_neighbors = new_f
        
        for idx, neighbor in enumerate(neighbors, 1):
            new_g = g + 1
            new_h = manhattan_distance(neighbor)
            new_f = new_g + new_h
            
            # Đánh dấu trạng thái được chọn và neighbor có f min
            is_chosen = states_equal(neighbor, next_state)
            is_min_f = (new_f == min_f_in_neighbors)
            
            marker = ""
            if is_chosen:
                marker = " ← CHỌN (trong solution path)"
            elif is_min_f and "A*" in algorithm:
                marker = " ← f(n) min trong neighbors này"
            
            print(f"  Kề {idx}: g(n)={new_g}, h(n)={new_h}, f(n)={new_f}{marker}")
            for row in neighbor:
                print(f"    {row}")
            print()
    
    # In bước cuối (đích)
    final_state = path[-1]
    g = len(path) - 1
    h = manhattan_distance(final_state)
    f = g + h
    
    print(f"\n{'='* 80}")
    print(f"BƯỚC {len(path)}/{len(path)-1}: ĐẠT ĐÍCH - g(n)={g}, h(n)={h}, f(n)={f}")
    print(f"{'='* 80}")
    print("\nTrạng thái đích:")
    for row in final_state:
        print(f"  {row}")
    print(f"\n{'='* 80}\n")

# ==================================================
# THUẬT TOÁN TÌM KIẾM
# ==================================================
def greedy_bfs(start: List[List[int]]) -> Tuple[List[List[List[int]]], int, int]:
    """
    Greedy Best-First Search
    Priority = h(n) only (không có g(n))
    Lưu tuple state trong heap để tránh dùng counter
    """
    pq = []
    visited: Set[Tuple] = set()

    # push trạng thái ban đầu (dùng tuple thay vì list)
    start_t = to_tuple(start)
    heapq.heappush(
        pq,
        (manhattan_distance(start), start_t, [])
    )

    nodes_expanded = 0
    nodes_generated = 1

    while pq:
        h, current_t, path = heapq.heappop(pq)
        
        # nếu đã thăm → bỏ
        if current_t in visited:
            continue

        # chuyển tuple về list để xử lý
        current = [list(row) for row in current_t]

        # nếu đạt đích
        if states_equal(current, GOAL):
            return path + [current], nodes_expanded, nodes_generated

        visited.add(current_t)
        nodes_expanded += 1

        # sinh các trạng thái kề
        for neighbor in get_neighbors(current):
            n_t = to_tuple(neighbor)
            if n_t not in visited:
                heapq.heappush(
                    pq,
                    (manhattan_distance(neighbor), n_t, path + [current])
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
    - Lưu tuple state trong heap để tránh dùng counter
    
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
    
    # Priority dựa vào f(n) = g(n) + h(n)
    priority = 0 + h0  # f = g + h
    
    # Lưu (f, g, state_tuple, path) trong heap
    heapq.heappush(pq, (priority, 0, start_t, []))
    g_score[start_t] = 0
    
    nodes_expanded = 0
    nodes_generated = 1
    
    while pq:
        f, g, current_t, path = heapq.heappop(pq)
        
        # Kiểm tra đã thăm chưa
        if current_t in visited:
            continue
        
        # Chuyển tuple về list để xử lý
        current = [list(row) for row in current_t]
        
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
            
            heapq.heappush(
                pq,
                (priority, new_g, n_t, path + [current])
            )
            nodes_generated += 1
    
    return [], nodes_expanded, nodes_generated

# ==================================================
# IN LỜI GIẢI
# ==================================================
def print_solution(path: List[List[List[int]]], nodes_expanded: int, nodes_generated: int):
    if not path:
        print("❌ Không tìm thấy lời giải!")
        return
    
    steps = len(path) - 1
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
    
    print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
    print(f"   • Số bước di chuyển: {steps}")
    print(f"   • Số nút được mở rộng (explored): {nodes_expanded}")
    print(f"   • Số nút được sinh ra (generated): {nodes_generated}")    
    print("\n" + "=" * 80)

# ==================================================
# MAIN
# ==================================================
def main():
    import sys
    import io
    
    # Fix encoding for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Trạng thái ban đầu A: 7 2 4 / 5 _ 6 / 8 3 1
    start = [
        [7, 2, 4],
        [5, 0, 6],
        [8, 3, 1]
    ]
    
    print("=" * 80)
    print("BÀI TOÁN 8 Ô TRƯỢT (8-PUZZLE)")
    print("=" * 80)
    
    print("\n📌 TRẠNG THÁI BAN ĐẦU (A):")
    print_puzzle_box(start)
    print(print_manhattan_detail(start))
    print(f"Manhattan distance đến đích: {manhattan_distance(start)}")
    
    print("\n📌 TRẠNG THÁI ĐÍCH (B):")
    print_puzzle_box(GOAL)
    
    # =====================================================================
    # THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH
    # =====================================================================
    print("\n" + "=" * 80)
    print("THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH (Greedy BeFS)")
    print("=" * 80)
    print("📖 Chiến lược: Priority = h(n) = Manhattan distance")
    print("   - Chỉ xem xét heuristic, bỏ qua chi phí đã đi")
    print("   - Chọn trạng thái gần đích nhất theo heuristic")
    
    path_greedy, nodes_greedy, gen_greedy = greedy_bfs(start)
    
    print_solution(path_greedy, nodes_greedy, gen_greedy)
    
    # =====================================================================
    # THUẬT TOÁN 2: A* SEARCH
    # =====================================================================
    print("\n" + "=" * 80)
    print("THUẬT TOÁN 2: A* SEARCH")
    print("=" * 80)
    print("📖 Chiến lược: Priority = f(n) = g(n) + h(n)")
    print("   - g(n): Chi phí thực tế từ điểm bắt đầu (số bước đã đi)")
    print("   - h(n): Ước lượng chi phí đến đích (Manhattan distance)")
    print("   - Kết hợp cả chi phí đã đi và ước lượng còn lại")
    
    path_astar, nodes_astar, gen_astar = astar_search(start)
    
    print_solution(path_astar, nodes_astar, gen_astar)

if __name__ == "__main__":
    main()
