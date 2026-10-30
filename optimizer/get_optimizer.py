import torch.optim as optim

def get_optimizer(op_name, net, lr, beta_1, beta_2):
    if op_name == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta_1, beta_2))
    elif op_name == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif op_name == "rmsprop":
        optimizer = optim.RMSprop(net.parameters())      
    elif op_name == "adamw":
        optimizer = optim.AdamW(net.parameters())
    print(optimizer)
    return optimizer
