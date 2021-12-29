
import matplotlib.pyplot as plt
import xgboost as XGBR

def DrawTree(bst,f1):
    graph = XGBR.to_graphviz(bst, num_trees=0)
    graph.render(filename=f"./Predict/Log/Png/dot-{f1}.dot")

    # 画图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["figure.dpi"]=600
    plt.rcParams["figure.figsize"]=(15,15)
    pic=XGBR.plot_importance(
        bst,
        ax=None,
        height=0.2,
        xlim=None,
        ylim=None,
        title="Feature importance",
        xlabel="F score",
        ylabel="Features",
        fmap="",
        importance_type="weight",
        max_num_features=None,
        grid=True,
        show_values=True,
    )
    plt.savefig(f"./Predict/Log/Png/result-{f1}-weight.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()
    plt.cla()
    pic=XGBR.plot_importance(
        bst,
        ax=None,
        height=0.2,
        xlim=None,
        ylim=None,
        title="Feature importance",
        xlabel="F score",
        ylabel="Features",
        fmap="",
        importance_type="gain",
        max_num_features=None,
        grid=True,
        show_values=True,
    )
    plt.savefig(f"./Predict/Log/Png/result-{f1}-gain.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()
    plt.cla()
    pic=XGBR.plot_importance(
        bst,
        ax=None,
        height=0.2,
        xlim=None,
        ylim=None,
        title="Feature importance",
        xlabel="F score",
        ylabel="Features",
        fmap="",
        importance_type="cover",
        max_num_features=None,
        grid=True,
        show_values=True,
    )
    plt.savefig(f"./Predict/Log/Png/result-{f1}-cover.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()
    plt.cla()

    # pic2=XGBR.plot_tree(bst, fmap="", rankdir=None, ax=None)
    # fig = plt.gcf()
    # fig.set_size_inches(100, 100)
    # fig.savefig(f"./Predict/Log/Png/tree-{f1}.jpg", bbox_inches="tight")
    # plt.close()
    # plt.clf()
    # plt.cla()

