#IMPORT
from core import *
#IMPORT


def kFolds_plot(k,datas):
    folds_title = str(k) + ' k-Folds Accuracy'
    #prints kfolds plots
    plt.plot(datas)
    plt.ylabel('accuracy')
    plt.xlabel('folds')
    plt.xticks(np.arange(k),np.arange(1,k+1))
    plt.title(folds_title)
    plt.savefig(folds_title)


if __name__=="__main__":
    a = []
    args = sys.argv
    k_ind = args.index('--k') + 1
    k = int(args[k_ind])

    for i in range(k):
        inp = input("Angka ke-"+str(i)+' : ')
        a.append(float(inp))

    kFolds_plot(k,a)