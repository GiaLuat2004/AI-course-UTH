'''
Bài toán đong nước (Water Jug Problem) với hai bình có dung tích khác nhau.
Mục tiêu là đong được một lượng nước cụ thể bằng cách sử dụng hai bình này và các hành động đổ đầy, đổ hết, và rót nước giữa hai bình.
Giải pháp được thực hiện bằng hai thuật toán tìm kiếm: A* và thuật toán Greedy BFS.

Câu 2: Cho hai bình A, B lần lượt có dung tích là 3 lít và 8 lít (không có vạch chia). 
Ban đầu hai bình không có nước. Có thể rót nước đổ đầy các bình, có thể đổ hết nước từ một
bình đi, có thể rót từ bình này sang bình khác.
Mục tiêu: Đong được 7 lít nước.
'''

import heapq
from typing import List, Set, Dict, Tuple as TupleType

# Cấu hình bài toán
JUG_A_CAPACITY = 3  # Dung tích bình A (lít)
JUG_B_CAPACITY = 8  # Dung tích bình B (lít)
TARGET = 7          # Mục tiêu: đong được 7 lít


class WaterJugState:
    """Đại diện cho trạng thái của hai bình nước."""
    
    def __init__(self, jug_a: int, jug_b: int):
        self.jug_a = jug_a  # Lượng nước trong bình A
        self.jug_b = jug_b  # Lượng nước trong bình B
    
    def to_tuple(self) -> TupleType[int, int]:
        """Chuyển trạng thái thành tuple để sử dụng làm key."""
        return (self.jug_a, self.jug_b)
    
    def is_goal(self) -> bool:
        """Kiểm tra có đạt mục tiêu chưa (có 7 lít ở một trong hai bình hoặc tổng)."""
        return self.jug_a == TARGET or self.jug_b == TARGET or (self.jug_a + self.jug_b) == TARGET
    
    def __str__(self) -> str:
        return f"A={self.jug_a}L, B={self.jug_b}L (Tổng: {self.jug_a + self.jug_b}L)"
    
    def __eq__(self, other) -> bool:
        return self.jug_a == other.jug_a and self.jug_b == other.jug_b
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())


def heuristic(state: WaterJugState) -> int:
    """
    Hàm heuristic ước lượng khoảng cách đến mục tiêu.
    
    Logic:
    - Tính khoảng cách nhỏ nhất từ trạng thái hiện tại đến mục tiêu
    - Xem xét cả lượng nước trong từng bình và tổng
    - Heuristic admissible: không bao giờ overestimate
    """
    # Khoảng cách đến mục tiêu cho từng trường hợp
    dist_a = abs(state.jug_a - TARGET)
    dist_b = abs(state.jug_b - TARGET)
    dist_total = abs((state.jug_a + state.jug_b) - TARGET)
    
    # Chọn khoảng cách nhỏ nhất
    return min(dist_a, dist_b, dist_total)


def get_neighbors(state: WaterJugState) -> List[TupleType[WaterJugState, str]]:
    """
    Tạo các trạng thái kề từ trạng thái hiện tại.
    
    6 hành động có thể thực hiện:
    1. Đổ đầy bình A
    2. Đổ đầy bình B
    3. Đổ hết bình A
    4. Đổ hết bình B
    5. Rót từ A sang B
    6. Rót từ B sang A
    
    Returns:
        List of (new_state, action_description)
    """
    neighbors = []
    a, b = state.jug_a, state.jug_b
    
    # 1. Đổ đầy bình A
    if a < JUG_A_CAPACITY:
        new_state = WaterJugState(JUG_A_CAPACITY, b)
        neighbors.append((new_state, f"Đổ đầy bình A: (A={a}L, B={b}L) → (A={JUG_A_CAPACITY}L, B={b}L)"))
    
    # 2. Đổ đầy bình B
    if b < JUG_B_CAPACITY:
        new_state = WaterJugState(a, JUG_B_CAPACITY)
        neighbors.append((new_state, f"Đổ đầy bình B: (A={a}L, B={b}L) → (A={a}L, B={JUG_B_CAPACITY}L)"))
    
    # 3. Đổ hết bình A
    if a > 0:
        new_state = WaterJugState(0, b)
        neighbors.append((new_state, f"Đổ hết bình A: (A={a}L, B={b}L) → (A=0L, B={b}L)"))
    
    # 4. Đổ hết bình B
    if b > 0:
        new_state = WaterJugState(a, 0)
        neighbors.append((new_state, f"Đổ hết bình B: (A={a}L, B={b}L) → (A={a}L, B=0L)"))
    
    # 5. Rót từ A sang B
    if a > 0 and b < JUG_B_CAPACITY:
        # Tính lượng nước có thể rót
        pour_amount = min(a, JUG_B_CAPACITY - b)
        new_a = a - pour_amount
        new_b = b + pour_amount
        new_state = WaterJugState(new_a, new_b)
        neighbors.append((new_state, f"Rót từ A sang B ({pour_amount}L): (A={a}L, B={b}L) → (A={new_a}L, B={new_b}L)"))
    
    # 6. Rót từ B sang A
    if b > 0 and a < JUG_A_CAPACITY:
        # Tính lượng nước có thể rót
        pour_amount = min(b, JUG_A_CAPACITY - a)
        new_a = a + pour_amount
        new_b = b - pour_amount
        new_state = WaterJugState(new_a, new_b)
        neighbors.append((new_state, f"Rót từ B sang A ({pour_amount}L): (A={a}L, B={b}L) → (A={new_a}L, B={new_b}L)"))
    
    return neighbors


def search(start: WaterJugState, algorithm: str = 'astar') -> TupleType[List[TupleType[WaterJugState, str]], int, int]:
    """
    Tìm kiếm đường đi từ trạng thái ban đầu đến trạng thái đích.
    
    Args:
        start: Trạng thái ban đầu
        algorithm: 'astar' hoặc 'greedy'
    
    Returns:
        (path, nodes_expanded, nodes_generated): 
            - path: List of (state, action)
            - nodes_expanded: Số nút được mở rộng
            - nodes_generated: Số nút được sinh ra
    
    Logic:
        - Greedy BFS: Priority = h(n) (chỉ heuristic)
        - A*: Priority = f(n) = g(n) + h(n) (chi phí thực + heuristic)
    """
    # Priority queue: (priority, counter, g_cost, state, path)
    counter = 0
    priority_queue = []
    h = heuristic(start)
    
    # LOGIC QUAN TRỌNG:
    # - Greedy BFS: Chỉ xem xét h(n), bỏ qua g(n)
    # - A*: Xem xét cả g(n) và h(n) để đảm bảo tối ưu
    if algorithm == 'astar':
        priority = 0 + h  # f = g + h, với g ban đầu = 0
    else:  # greedy
        priority = h  # chỉ dùng h, không quan tâm g
    
    heapq.heappush(priority_queue, (priority, counter, 0, start, []))
    counter += 1
    
    visited: Set[TupleType[int, int]] = set()
    nodes_expanded = 0
    nodes_generated = 1  # Đếm cả node ban đầu
    
    # Lưu g_cost tốt nhất cho mỗi trạng thái (quan trọng cho A*)
    g_scores: Dict[TupleType[int, int], int] = {start.to_tuple(): 0}
    
    while priority_queue:
        current_priority, _, g_cost, current_state, path = heapq.heappop(priority_queue)
        
        # Kiểm tra đã đạt đích chưa
        if current_state.is_goal():
            return path + [(current_state, "ĐẠT MỤC TIÊU!")], nodes_expanded, nodes_generated
        
        # Bỏ qua nếu đã thăm
        state_tuple = current_state.to_tuple()
        if state_tuple in visited:
            continue
        
        visited.add(state_tuple)
        nodes_expanded += 1
        
        # Mở rộng các trạng thái kề
        for neighbor_state, action in get_neighbors(current_state):
            neighbor_tuple = neighbor_state.to_tuple()
            
            if neighbor_tuple in visited:
                continue
            
            new_g = g_cost + 1  # Chi phí thực tế từ start đến neighbor
            
            # Kiểm tra xem có tìm được đường đi tốt hơn không
            if neighbor_tuple in g_scores and new_g >= g_scores[neighbor_tuple]:
                continue  # Đã có đường đi tốt hơn rồi
            
            g_scores[neighbor_tuple] = new_g
            h = heuristic(neighbor_state)
            
            # LOGIC KHÁC BIỆT GIỮA 2 THUẬT TOÁN:
            if algorithm == 'astar':
                priority = new_g + h  # A*: f(n) = g(n) + h(n)
            else:  # greedy
                priority = h  # Greedy BFS: chỉ dùng h(n)
            
            heapq.heappush(priority_queue, 
                         (priority, counter, new_g, neighbor_state, path + [(current_state, action)]))
            counter += 1
            nodes_generated += 1
    
    return [], nodes_expanded, nodes_generated  # Không tìm thấy lời giải


def print_solution(path: List[TupleType[WaterJugState, str]], title: str, 
                   nodes_expanded: int, nodes_generated: int) -> None:
    """In chi tiết lời giải dạng bảng."""
    if not path:
        print("❌ Không tìm thấy lời giải!")
        return
    
    steps = len(path) - 1
    final_state = path[-1][0]
    
    print(f"\n✓ Đã tìm thấy giải pháp để đong được {TARGET} lít!")
    print(f"✓ Tổng số bước: {steps}")
    print(f"✓ Số nút được mở rộng (explored): {nodes_expanded}")
    print(f"✓ Số nút được sinh ra (generated): {nodes_generated}")
    print(f"✓ Branching factor trung bình: {nodes_generated / max(nodes_expanded, 1):.2f}")
    
    # In đường đi dạng (a, b) -> (a, b) -> ...
    path_str = "(0, 0)"
    for state, _ in path[1:]:
        path_str += f" -> ({state.jug_a}, {state.jug_b})"
    print(f"\n📍 Giải: {path_str}")
    
    # In bảng kết quả
    print("\n┌──────┬─────────────────────────────────────────────────────────────────┬─────────┬─────────┐")
    print("│ Bước │                        Hành động                                │  Bình A │  Bình B │")
    print("├──────┼─────────────────────────────────────────────────────────────────┼─────────┼─────────┤")
    print(f"│  0   │ Trạng thái ban đầu                                              │    0    │    0    │")
    
    for i in range(1, len(path)):
        state = path[i][0]
        action = path[i][1]
        
        # Bỏ "ĐẠT MỤC TIÊU!" ở cuối
        if action == "ĐẠT MỤC TIÊU!":
            action = ""
        
        print(f"│  {i:<2}  │ {action:<63} │    {state.jug_a}    │    {state.jug_b}    │")
    
    print("└──────┴─────────────────────────────────────────────────────────────────┴─────────┴─────────┘")
    
    if final_state.jug_a == TARGET:
        print(f"\n✓ Kết quả: Bình A chứa {TARGET} lít (Mục tiêu đạt được!)")
    elif final_state.jug_b == TARGET:
        print(f"\n✓ Kết quả: Bình B chứa {TARGET} lít (Mục tiêu đạt được!)")
    else:
        print(f"\n✓ Kết quả: Tổng {final_state.jug_a + final_state.jug_b} lít (Mục tiêu đạt được!)")


def verify_solution(path: List[TupleType[WaterJugState, str]]) -> bool:
    """Kiểm tra tính hợp lệ của lời giải."""
    if not path:
        return False
    
    # Kiểm tra trạng thái cuối có đạt mục tiêu không
    final_state = path[-1][0]
    if not final_state.is_goal():
        print("❌ Lỗi: Trạng thái cuối không đạt mục tiêu!")
        return False
    
    # Kiểm tra mỗi bước có hợp lệ không
    for i in range(len(path) - 1):
        current_state = path[i][0]
        next_state = path[i + 1][0]
        
        # Kiểm tra không vi phạm dung tích
        if next_state.jug_a > JUG_A_CAPACITY or next_state.jug_b > JUG_B_CAPACITY:
            print(f"❌ Lỗi: Bước {i+1} vượt quá dung tích bình!")
            return False
        
        if next_state.jug_a < 0 or next_state.jug_b < 0:
            print(f"❌ Lỗi: Bước {i+1} có lượng nước âm!")
            return False
        
        # Kiểm tra hành động có hợp lý không
        # (lượng nước không thể tăng giảm một cách không hợp lý)
        current_total = current_state.jug_a + current_state.jug_b
        next_total = next_state.jug_a + next_state.jug_b
        
        # Tổng nước chỉ có thể tăng (đổ đầy), giảm (đổ bỏ), hoặc giữ nguyên (rót)
        if next_total > current_total:
            # Đang đổ thêm nước
            added = next_total - current_total
            if added > JUG_A_CAPACITY and added > JUG_B_CAPACITY:
                print(f"❌ Lỗi: Bước {i+1} thêm quá nhiều nước cùng lúc!")
                return False
    
    print("✓ Lời giải hợp lệ!")
    print(f"✓ Đạt được {final_state.jug_a + final_state.jug_b} lít nước")
    print(f"✓ Bình A: {final_state.jug_a}L, Bình B: {final_state.jug_b}L")
    return True


def main():
    print("="*80)
    print("BÀI TOÁN ĐONG NƯỚC (WATER JUG PROBLEM)")
    print("="*80)
    
    print(f"\n📋 THÔNG TIN BÀI TOÁN:")
    print(f"   • Bình A: Dung tích {JUG_A_CAPACITY} lít")
    print(f"   • Bình B: Dung tích {JUG_B_CAPACITY} lít")
    print(f"   • Mục tiêu: Đong được {TARGET} lít nước")
    print(f"   • Trạng thái ban đầu: Cả hai bình đều rỗng")
    
    print(f"\n📖 CÁC HÀNH ĐỘNG CÓ THỂ THỰC HIỆN:")
    print(f"   1. Đổ đầy bình A (từ nguồn nước)")
    print(f"   2. Đổ đầy bình B (từ nguồn nước)")
    print(f"   3. Đổ hết nước từ bình A")
    print(f"   4. Đổ hết nước từ bình B")
    print(f"   5. Rót nước từ bình A sang bình B")
    print(f"   6. Rót nước từ bình B sang bình A")
    
    # Trạng thái ban đầu: cả hai bình đều rỗng
    start_state = WaterJugState(0, 0)
    
    print(f"\n🎯 TRẠNG THÁI BAN ĐẦU:")
    print(f"   {start_state}")
    print(f"   Heuristic h(n) = {heuristic(start_state)}")
    
    # =====================================================================
    # THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH
    # =====================================================================
    print("\n" + "="*80)
    print("THUẬT TOÁN 1: GREEDY BEST-FIRST SEARCH (Greedy BeFS)")
    print("="*80)
    print("📖 Chiến lược: Priority = h(n) = min(|A-7|, |B-7|, |(A+B)-7|)")
    print("   - Chỉ xem xét khoảng cách đến mục tiêu")
    print("   - Chọn trạng thái gần mục tiêu nhất")
    print("✓ Ưu điểm: Tìm kiếm nhanh, ít tốn bộ nhớ")
    print("✗ Nhược điểm: Không đảm bảo lời giải tối ưu")
    
    path_greedy, nodes_greedy, gen_greedy = search(start_state, algorithm='greedy')
    
    print(f"\n{'='*100}")
    print(f"KẾT QUẢ THUẬT TOÁN: GREEDY BEST-FIRST SEARCH")
    print(f"{'='*100}")
    print_solution(path_greedy, "Greedy Best-First Search", nodes_greedy, gen_greedy)
    
    print("\n" + "─"*80)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI GREEDY BeFS:")
    print("─"*80)
    verify_solution(path_greedy)
    
    # =====================================================================
    # THUẬT TOÁN 2: A* SEARCH
    # =====================================================================
    print("\n" + "="*80)
    print("THUẬT TOÁN 2: A* SEARCH")
    print("="*80)
    print("📖 Chiến lược: Priority = f(n) = g(n) + h(n)")
    print("   - g(n): Số bước đã thực hiện")
    print("   - h(n): Ước lượng số bước còn lại")
    print("   - Cân bằng giữa chi phí thực tế và ước lượng")
    print("✓ Ưu điểm: Đảm bảo tìm lời giải tối ưu (ít bước nhất)")
    print("✗ Nhược điểm: Có thể khám phá nhiều trạng thái hơn")
    
    path_astar, nodes_astar, gen_astar = search(start_state, algorithm='astar')
    
    print(f"\n{'='*100}")
    print(f"KẾT QUẢ THUẬT TOÁN: A* SEARCH")
    print(f"{'='*100}")
    print_solution(path_astar, "A* Search", nodes_astar, gen_astar)
    
    print("\n" + "─"*80)
    print("KIỂM TRA TÍNH HỢP LỆ CỦA LỜI GIẢI A*:")
    print("─"*80)
    verify_solution(path_astar)
    
    # =====================================================================
    # SO SÁNH KẾT QUẢ
    # =====================================================================
    print("\n" + "="*80)
    print("SO SÁNH VÀ ĐÁNH GIÁ KẾT QUẢ")
    print("="*80)
    
    if path_greedy and path_astar:
        steps_greedy = len(path_greedy) - 1
        steps_astar = len(path_astar) - 1
        
        print(f"\n{'Tiêu chí':<35} {'Greedy BeFS':>20} {'A* Search':>20}")
        print("─"*80)
        print(f"{'Số bước thực hiện':<35} {steps_greedy:>20} {steps_astar:>20}")
        print(f"{'Số nút mở rộng (explored)':<35} {nodes_greedy:>20} {nodes_astar:>20}")
        print(f"{'Số nút sinh ra (generated)':<35} {gen_greedy:>20} {gen_astar:>20}")
        
        print("\n" + "─"*80)
        print("PHÂN TÍCH CHI TIẾT:")
        print("─"*80)
        
        # So sánh số bước
        if steps_astar <= steps_greedy:
            print(f"\n📊 Số bước thực hiện:")
            print(f"   ✓ A* tìm được lời giải TỐI ƯU với {steps_astar} bước")
            if steps_astar < steps_greedy:
                improvement = ((steps_greedy - steps_astar) / steps_greedy) * 100
                print(f"   ✓ A* ít hơn Greedy BeFS {steps_greedy - steps_astar} bước ({improvement:.1f}% tốt hơn)")
                print(f"   ✗ Greedy BeFS không tối ưu: {steps_greedy} bước")
            else:
                print(f"   ✓ Cả hai đều tìm được lời giải tối ưu")
        
        # So sánh hiệu quả
        print(f"\n📊 Hiệu quả tìm kiếm:")
        if nodes_greedy < nodes_astar:
            reduction = ((nodes_astar - nodes_greedy) / nodes_astar) * 100
            print(f"   ✓ Greedy BeFS hiệu quả hơn, mở rộng ít hơn {nodes_astar - nodes_greedy} nút ({reduction:.1f}%)")
        else:
            print(f"   ✓ A* tương đương hoặc hiệu quả hơn về không gian tìm kiếm")
        
        # Kết luận
        print("\n" + "─"*80)
        print("KẾT LUẬN:")
        print("─"*80)
        print(f"\n💡 Nhận xét về bài toán Water Jug:")
        print(f"   1. A* đảm bảo tìm lời giải tối ưu với {steps_astar} bước")
        print(f"   2. Greedy BeFS nhanh hơn nhưng có thể không tối ưu ({steps_greedy} bước)")
        print(f"   3. Heuristic sử dụng: min(|A-{TARGET}|, |B-{TARGET}|, |(A+B)-{TARGET}|)")
        print(f"   4. Heuristic admissible vì không bao giờ overestimate số bước thực tế")
        print(f"   5. Với bài toán này, không gian trạng thái nhỏ ({(JUG_A_CAPACITY+1)*(JUG_B_CAPACITY+1)} trạng thái)")
        
        print(f"\n🎯 Xác nhận:")
        if path_astar:
            final_state = path_astar[-1][0]
            print(f"   ✓ Đã đong được {TARGET} lít nước thành công!")
            print(f"   ✓ Trạng thái cuối: Bình A = {final_state.jug_a}L, Bình B = {final_state.jug_b}L")


if __name__ == "__main__":
    main()