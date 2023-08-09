from dataclasses import dataclass, field
import numpy as np

@dataclass
class Superpoint:
    label: int
    box_index: int
    points: list
    avg_point_feature: list = field(default_factory=list)
    is_labelled: bool = False
    informativeness: float = 0.0
    uncertainty: float = 0.0
    class_diversity: float = 0.0
    diversity: float = 0.0
    budget: int = 1
    
    def add_point(self, p):
        self.budget += 1
        self.points.append(p)
        
    def get_class_label(self):
        all_labels = [p.c_label for p in self.points]
        return max(all_labels, key=all_labels.count)
        
    def get_avg_feature(self):
        return self.avg_point_feature
    
        # avg_feature = np.average([p.feature for p in self.points], axis=0)
        # return avg_feature / np.linalg.norm(avg_feature)
    
    def update_informativeness(self, alpha, beta, gamma):
        self.informativeness = alpha * self.uncertainty + beta * self.diversity + gamma * self.class_diversity
        
    def penalise(self, penalty):
        self.informativeness -= penalty
        
    
@dataclass
class Point:
    c_label: int
    feature: list
    probabilities: list