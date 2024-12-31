from sklearn.svm import SVC
import numpy as np

def guard_learning(trace, svm_kernel = 'poly'):
    mode_num = len(trace[0]['labels_num'])

    mode_transition_metric = np.eye(mode_num)
    for i in range(len(trace)):
        begin = trace[i]['labels_trace'][0]
        for j in trace[i]['labels_trace']:
            end = trace[i]['labels_trace'][j]
            if mode_transition_metric[begin - 1, end - 1] != 1:
                mode_transition_metric[begin - 1, end - 1] = 1
            begin = end

    mode_transition_list = []

    for i in range(mode_num):
        temp = []
        for j in range(mode_num):
            if i != j and mode_transition_metric[i][j] == 1:
                temp.append(j + 1)
        mode_transition_list.append(temp)

    models = {}
    train_data = {}

    for i in range(mode_num):
        models[str(i + 1)] = []
        train_data[str(i + 1)] = [[[], []] for _ in range(len(mode_transition_list[i]))]

    for i in range(len(trace)):
        for j in range(len(trace[i]['labels_trace'])):
            mode_now = trace[i]['labels_trace'][j]

            x_now = trace[i]['x'][:,trace[i]['chpoints'][j] - 1:trace[i]['chpoints'][j + 1] - 1].T
            label_now = np.zeros((x_now.shape[0], 1))

            for k in range(len(mode_transition_list[mode_now - 1])):
                train_data[str(mode_now)][k][0].append(x_now)
                train_data[str(mode_now)][k][1].append(label_now)

            x_chp = trace[i]['x'][:, trace[i]['chpoints'][j + 1] - 1].T

            if j + 1 != len((trace[i]['labels_trace'])):
                mode_next = trace[i]['labels_trace'][j + 1]
                train_data[str(mode_now)][mode_transition_list[mode_now - 1].index(mode_next)][0].append(x_chp)
                train_data[str(mode_now)][mode_transition_list[mode_now - 1].index(mode_next)][1].append(np.ones((1,1)))

            else:
                for k in range(len(mode_transition_list[mode_now - 1])):
                    train_data[str(mode_now)][k][0].append(x_chp)
                    train_data[str(mode_now)][k][1].append(np.zeros((1,1)))

    for mm in range(mode_num):
        for nn in range(len(train_data[str(mm + 1)])):
            train_data[str(mm + 1)][nn][0] = np.vstack((train_data[str(mm + 1)][nn][0]))
            train_data[str(mm + 1)][nn][1] = np.vstack((train_data[str(mm + 1)][nn][1]))

    for mm in range(mode_num):
        for nn in range(len(mode_transition_list[mm])):
            svm_model = SVC(kernel = svm_kernel, class_weight={0: 1, 1: 1})
            svm_model.fit(train_data[str(mm + 1)][nn][0], train_data[str(mm + 1)][nn][1])
            models[str(mm + 1)].append(svm_model)



    import matplotlib.pyplot as plt


    X = np.array(train_data['1'][0][0])
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 500),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 500))
    X_train = X
    y_train = train_data['1'][0][1]
    # 计算每个点的决策函数值
    svc =  models['1'][0]
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

    # 绘制支持向量
    # plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='r',
    #             label='Support Vectors')

    # 绘制训练数据
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='autumn', marker='o', edgecolors='k',
                label='Training Data')


    plt.title('SVM Decision Boundary with Polynomial Kernel')
    plt.legend()
    plt.show()

    return models