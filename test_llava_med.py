# Save this as test_deepspeed.py
import deepspeed

def simple_model_engine():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    engine = deepspeed.initialize(model=model, model_parameters=model.parameters())[0]
    print("Deepspeed engine is initialized.")

if __name__ == "__main__":
    simple_model_engine()
