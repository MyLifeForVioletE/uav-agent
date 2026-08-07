"""
子任务分配器：将子任务分配到具体UAV
"""
import sys
from core.fleet_config import SubTask, UAVConfig


class TaskAllocator:
    """子任务分配器：基于任务类型 × UAV能力进行匹配分配"""
    
    def __init__(self):
        pass
    
    def allocate(self, sub_tasks: list[SubTask], uav_configs: list[UAVConfig]) -> dict[str, list[str]]:
        """
        分配子任务到UAV
        
        策略：
        1. 基于任务类型和UAV能力进行匹配
        2. 考虑负载均衡
        3. 考虑依赖关系
        
        Returns:
            {uav_id: [task_ids]} 分配映射
        """
        assignments: dict[str, list[str]] = {uav.uav_id: [] for uav in uav_configs}
        
        # 构建任务依赖图
        task_map = {t.task_id: t for t in sub_tasks}
        
        # 按依赖拓扑排序（优先分配无依赖的任务）
        sorted_tasks = self._topological_sort(sub_tasks)
        
        for task in sorted_tasks:
            best_uav = self._find_best_uav(task, uav_configs, assignments, task_map)
            if best_uav:
                assignments[best_uav.uav_id].append(task.task_id)
                task.assigned_uav = best_uav.uav_id
                sys.stderr.write(f"[Allocator] 分配任务 {task.task_id} -> {best_uav.uav_id}\n")
                sys.stderr.flush()
            else:
                sys.stderr.write(f"[Allocator] 无法分配任务 {task.task_id}\n")
                sys.stderr.flush()
        
        return assignments
    
    def _topological_sort(self, sub_tasks: list[SubTask]) -> list[SubTask]:
        """拓扑排序：按依赖关系排序任务"""
        task_map = {t.task_id: t for t in sub_tasks}
        visited = set()
        result = []
        
        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map.get(task_id)
            if task:
                for prereq in task.prerequisite_tasks:
                    visit(prereq)
                result.append(task)
        
        for task in sub_tasks:
            visit(task.task_id)
        
        return result
    
    def _find_best_uav(
        self, 
        task: SubTask, 
        uav_configs: list[UAVConfig],
        assignments: dict[str, list[str]],
        task_map: dict[str, SubTask]
    ) -> UAVConfig | None:
        """为任务找到最佳UAV"""
        candidates = []
        
        for uav in uav_configs:
            score = self._score_match(task, uav, assignments, task_map)
            if score > 0:
                candidates.append((uav, score))
        
        if not candidates:
            return None
        
        # 按分数排序，选择最高分
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _score_match(
        self, 
        task: SubTask, 
        uav: UAVConfig,
        assignments: dict[str, list[str]],
        task_map: dict[str, SubTask]
    ) -> float:
        """计算任务与UAV的匹配分数"""
        score = 0.0
        
        # 1. 角色匹配
        if task.assigned_uav_role:
            if uav.role.value == task.assigned_uav_role:
                score += 10.0
            else:
                return 0  # 角色不匹配则排除
        
        # 2. 任务关键词匹配
        task_text = (task.task_name + task.goal).lower()
        role_keywords = {
            "scout": ["侦察", "侦察", "探测", "监视"],
            "jammer": ["干扰", "压制", "电磁"],
            "relay": ["中继", "通信", "转发"],
            "leader": ["指挥", "规划", "协调"],
        }
        
        keywords = role_keywords.get(uav.role.value, [])
        for keyword in keywords:
            if keyword in task_text:
                score += 5.0
                break
        
        # 3. 负载均衡（分配任务少的UAV优先）
        current_load = len(assignments.get(uav.uav_id, []))
        score -= current_load * 2.0
        
        # 4. 依赖关系：避免循环依赖
        for prereq_id in task.prerequisite_tasks:
            prereq_task = task_map.get(prereq_id)
            if prereq_task and prereq_task.assigned_uav == uav.uav_id:
                # 自己依赖自己的任务，加分（减少跨机通信）
                score += 3.0
        
        return max(score, 0.0)
