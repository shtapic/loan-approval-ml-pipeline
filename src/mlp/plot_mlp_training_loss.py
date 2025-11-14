from matplotlib import pyplot as plt

def plot_mlp_training_loss(mlp, train_loss_list):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (10, 8))  
    for i, (name, model) in enumerate(mlp.items()):
        r = i // 2   
        c = i % 2   
        axs[r, c].plot(train_loss_list[name], label=name)
        axs[r, c].set_title(f'Training Loss for {name} MLP', fontsize=12)
        axs[r, c].legend(loc='best', frameon=False, fontsize=12)
        axs[r, c].grid(True, linestyle='--', alpha=0.6)
        
    fig.supxlabel('Эпохи', fontsize=12)
    fig.supylabel('Значение loss', fontsize=12)
    plt.show()
