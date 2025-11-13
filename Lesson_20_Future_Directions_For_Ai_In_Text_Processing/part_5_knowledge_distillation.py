import torch
import torch.nn as nn

# Teacher and Student models
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 10 inputs, 2 outputs

    def forward(self, x):
        return self.fc(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Smaller model

    def forward(self, x):
        return self.fc(x)

# Distillation Loss
def distillation_loss(student_output, teacher_output, temperature):
    return nn.KLDivLoss()(nn.functional.log_softmax(student_output / temperature, dim=1),
                           nn.functional.softmax(teacher_output / temperature, dim=1)) * (temperature ** 2)

teacher = TeacherModel()
student = StudentModel()

# Example output from teacher (predicted values)
data = torch.rand(1, 10)  # Random input
with torch.no_grad():
    teacher_output = teacher(data)

# During the distillation process
student_output = student(data)
loss = distillation_loss(student_output, teacher_output, temperature=2)
print(f"Knowledge Distillation Loss: {loss.item()}")
