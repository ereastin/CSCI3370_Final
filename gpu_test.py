import torch

def main:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    print(f'Device: {dev}')

if __name__ == '__main__':
    main()

