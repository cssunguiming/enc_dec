import argparse

def parsers():
    parser = argparse.ArgumentParser(description='Set Config')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1, type=float) 
    parser.add_argument('--d_model', default=100, type=int)
    parser.add_argument('--d_ff', default=400, type=int)
    parser.add_argument('--n_layers', default=6, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--head_n', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--label_smoothing', default=True, type=bool)
    parser.add_argument('--n_warmup_steps', default=4000, type=int)
    args = parser.parse_args()
    return args