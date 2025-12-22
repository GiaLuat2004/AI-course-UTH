import heapq
from typing import Tuple, List, Set
import time

# Trạng thái đích B: _ 1 2 / 3 4 5 / 6 7 8
GOAL = (0, 1, 2, 3, 4, 5, 6, 7, 8)

def manhattan_distance(state: Tuple[int, ...]) -> int:
    """
    Tính khoảng cách Manhattan từ trạng thái hiện tại đến trạng thái đích.
    Manhattan distance = tổng khoảng cách hàng + tổng khoảng cách cột của mỗi ô.
    """
    distance = 0
    for i, value in enumerate(state):
        if value == 0:  # Bỏ qua ô trống
            continue
        # Vị trí hiện tại
        current_row, current_col = divmod(i, 3)
        # Vị trí đích của value
        goal_index = GOAL.index(value)
        goal_row, goal_col = divmod(goal_index, 3)
        # Cộng khoảng cách Manhattan
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def get_neighbors(state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Tạo các trạng thái kề bằng cách di chuyển ô trống lên/xuống/trái/phải.
    """
    blank_index = state.index(0)
    row, col = divmod(blank_index, 3)
    neighbors_list = []
    
    # Thử 4 hướng: lên, xuống, trái, phải
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        # Kiểm tra trong phạm vi bàn cờ 3x3
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_blank_index = new_row * 3 + new_col
            # Tạo trạng thái mới bằng cách hoán đổi ô trống với ô kề
            new_state = list(state)
            new_state[blank_index], new_state[new_blank_index] = \
                new_state[new_blank_index], new_state[blank_index]
            neighbors_list.append(tuple(new_state))
    
    return neighbors_list

def search(start: Tuple[int, ...], algorithm: str = 'astar') -> Tuple[List[Tuple[int, ...]], int]:
    """
    Tìm kiếm đường đi từ trạng thái ban đầu đến trạng thái đích.
    
    Args:
        start: Trạng thái ban đầu
        algorithm: 'astar' hoặc 'greedy'
    
    Returns:
        (path, nodes_expanded): Đường đi và số nút được mở rộng
    """
    # Priority queue: (priority, counter, g_cost, state, path)
    # counter để đảm bảo thứ tự khi priority bằng nhau
    counter = 0
    priority_queue = []
    h = manhattan_distance(start)
    
    if algorithm == 'astar':
        priority = 0 + h  # f = g + h
    else:  # greedy
        priority = h  # chỉ dùng h
    
    heapq.heappush(priority_queue, (priority, counter, 0, start, []))
    counter += 1
    
    visited: Set[Tuple[int, ...]] = set()
    nodes_expanded = 0
    
    while priority_queue:
        _, _, g_cost, current_state, path = heapq.heappop(priority_queue)
        
        # Kiểm tra đã đạt đích chưa
        if current_state == GOAL:
            return path + [current_state], nodes_expanded
        
        # Bỏ qua nếu đã thăm
        if current_state in visited:
            continue
        
        visited.add(current_state)
        nodes_expanded += 1
        
        # Mở rộng các trạng thái kề
        for neighbor in get_neighbors(current_state):
            if neighbor not in visited:
                new_g = g_cost + 1
                h = manhattan_distance(neighbor)
                
                if algorithm == 'astar':
                    priority = new_g + h  # f = g + h
                else:  # greedy
                    priority = h  # chỉ dùng h
                
                heapq.heappush(priority_queue, 
                             (priority, counter, new_g, neighbor, path + [current_state]))
                counter += 1
    
    return [], nodes_expanded  # Không tìm thấy lời giải

def print_state(state: Tuple[int, ...]) -> None:
    """In trạng thái dạng lưới 3x3."""
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print(' '.join('_' if x == 0 else str(x) for x in row))

def print_solution(path: List[Tuple[int, ...]], algorithm_name: str, 
                   nodes_expanded: int, exec_time: float) -> None:
    """In chi tiết lời giải."""
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ THUẬT TOÁN: {algorithm_name}")
    print(f"{'='*70}")
    
    if not path:
        print("❌ Không tìm thấy lời giải!")
        return
    
    steps = len(path) - 1
    print(f"✓ Số bước di chuyển: {steps}")
    print(f"✓ Số nút được mở rộng: {nodes_expanded}")
    print(f"✓ Thời gian thực thi: {exec_time:.4f} giây")
    
    print(f"\n{'─'*70}")
    print("CHI TIẾT CÁC BƯỚC DI CHUYỂN:")
    print(f"{'─'*70}")
    
    for i, state in enumerate(path):
        if i == 0:
            print(f"\nTrạng thái ban đầu:")
        elif i == len(path) - 1:
            print(f"\nBước {i}: Đạt đích!")
        else:
            print(f"\nBước {i}:")
        print_state(state)
        if i < len(path) - 1:
            print(f"  h(n) = {manhattan_distance(state)}")

def verify_solution(path: List[Tuple[int, ...]]) -> bool:
    """Kiểm tra tính hợp lệ của lời giải."""
    if not path:
        return False
    
    # Kiểm tra trạng thái cuối có phải là đích không
    if path[-1] != GOAL:
        print("❌ Lỗi: Trạng thái cuối không phải là đích!")
        return False
    
    # Kiểm tra mỗi bước có hợp lệ không (chỉ di chuyển ô trống 1 bước)
    for i in range(len(path) - 1):
        current = path[i]
        next_state = path[i + 1]
        
        # Đếm số ô khác nhau
        diff_count = sum(1 for j in range(9) if current[j] != next_state[j])
        
        if diff_count != 2:  # Phải có đúng 2 ô khác nhau (ô trống và ô bị hoán đổi)
            print(f"❌ Lỗi: Bước {i+1} không hợp lệ! Có {diff_count} ô thay đổi.")
            return False
    
    print("✓ Lời giải hợp lệ!")
    return True

def main():
    # Trạng thái ban đầu A: 7 2 4 / 5 _ 6 / 8 3 1
    start_state = (7, 2, 4, 5, 0, 6, 8, 3, 1)
    
    print("="*70)
    print("BÀI TOÁN 8 Ô TRƯỢT (8-PUZZLE)")
    print("="*70)
    
    print("\n📌 TRẠNG THÁI BAN ĐẦU (A):")
    print_state(start_state)
    print(f"Manhattan distance đến đích: {manhattan_distance(start_state)}")
    
    print("\n📌 TRẠNG THÁI ĐÍCH (B):")
    print_state(GOAL)
    
    # =====================================================================
    # THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH
    # =====================================================================
    print("\n" + "="*70)
    print("THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH (Greedy BeFS)")
    print("="*70)
    print("Chiến lược: Chỉ sử dụng hàm heuristic h(n) = Manhattan distance")
    print("Ưu điểm: Tìm kiếm nhanh, mở rộng ít nút")
    print("Nhược điểm: Không đảm bảo tìm được lời giải tối ưu")
    
    start_time = time.time()
    path_greedy, nodes_greedy = search(start_state, algorithm='greedy')
    time_greedy = time.time() - start_time
    
    print_solution(path_greedy, "Greedy Best-First Search", nodes_greedy, time_greedy)
    
    print("\n" + "─"*70)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI GREEDY BeFS:")
    print("─"*70)
    verify_solution(path_greedy)
    
    # =====================================================================
    # THUẬT TOÁN 2: A* SEARCH
    # =====================================================================
    print("\n" + "="*70)
    print("THUẬT TOÁN 2: A* SEARCH")
    print("="*70)
    print("Chiến lược: Sử dụng f(n) = g(n) + h(n)")
    print("  - g(n): Chi phí thực tế từ điểm bắt đầu")
    print("  - h(n): Ước lượng chi phí đến đích (Manhattan distance)")
    print("Ưu điểm: Đảm bảo tìm được lời giải tối ưu (đường đi ngắn nhất)")
    print("Nhược điểm: Có thể mở rộng nhiều nút hơn Greedy BeFS")
    
    start_time = time.time()
    path_astar, nodes_astar = search(start_state, algorithm='astar')
    time_astar = time.time() - start_time
    
    print_solution(path_astar, "A* Search", nodes_astar, time_astar)
    
    print("\n" + "─"*70)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI A*:")
    print("─"*70)
    verify_solution(path_astar)
    
    # =====================================================================
    # SO SÁNH KẾT QUẢ
    # =====================================================================
    print("\n" + "="*70)
    print("SO SÁNH VÀ ĐÁNH GIÁ KẾT QUẢ")
    print("="*70)
    
    if path_greedy and path_astar:
        steps_greedy = len(path_greedy) - 1
        steps_astar = len(path_astar) - 1
        
        print(f"\n{'Tiêu chí':<30} {'Greedy BeFS':>15} {'A* Search':>15}")
        print("─"*70)
        print(f"{'Số bước di chuyển':<30} {steps_greedy:>15} {steps_astar:>15}")
        print(f"{'Số nút được mở rộng':<30} {nodes_greedy:>15} {nodes_astar:>15}")
        print(f"{'Thời gian thực thi (s)':<30} {time_greedy:>15.4f} {time_astar:>15.4f}")
        
        print("\n" + "─"*70)
        print("KẾT LUẬN:")
        print("─"*70)
        
        if steps_astar <= steps_greedy:
            print(f"✓ A* tìm được lời giải TỐI ƯU với {steps_astar} bước")
            if steps_astar < steps_greedy:
                improvement = ((steps_greedy - steps_astar) / steps_greedy) * 100
                print(f"✓ A* ngắn hơn Greedy BeFS {steps_greedy - steps_astar} bước ({improvement:.1f}% tốt hơn)")
        
        if nodes_greedy < nodes_astar:
            reduction = ((nodes_astar - nodes_greedy) / nodes_astar) * 100
            print(f"✓ Greedy BeFS hiệu quả hơn về không gian, mở rộng ít hơn {nodes_astar - nodes_greedy} nút ({reduction:.1f}%)")
        
        if time_greedy < time_astar:
            print(f"✓ Greedy BeFS nhanh hơn {time_astar - time_greedy:.4f} giây")
        
        print(f"\n💡 Với heuristic Manhattan distance:")
        print(f"   - A* đảm bảo tìm được đường đi ngắn nhất")
        print(f"   - Greedy BeFS có thể nhanh hơn nhưng không đảm bảo tối ưu")
        print(f"   - Cả hai thuật toán đều sử dụng heuristic admissible (không overestimate)")

if __name__ == "__main__":
    main()
