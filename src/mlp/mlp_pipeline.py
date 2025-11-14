from mlp.mlp_model import MLPClassifier

def make_mlp_pipeline(input_dim: int):
    mlp ={
    'large': MLPClassifier(input_dim=input_dim, hidden_dim=[128, 64, 32], dropout=0.5),
    'medium': MLPClassifier(input_dim=input_dim, hidden_dim=[64, 32, 16], dropout=0.5),
    'small': MLPClassifier(input_dim=input_dim, hidden_dim=[32, 16, 8], dropout=0.5),
    'deep': MLPClassifier(input_dim=input_dim, hidden_dim=[128, 64, 32, 16, 8, 4, 2], dropout=0.5),
    }
    return mlp