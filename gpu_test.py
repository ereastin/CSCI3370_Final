import torch

def main():
    print('anything')
    if torch.cuda.is_available():
        print(f'Using {torch.cuda.device_count()} GPUs')
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    #print(f'Device: {dev}')

if __name__ == '__main__':
    main()

