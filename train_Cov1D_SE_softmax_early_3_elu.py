import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.optimizers import Adam, SGD
from keras.layers import Input, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, Dropout, Flatten, Dense, BatchNormalization, Activation, Concatenate, Reshape, Multiply
from keras.models import Model, load_model
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from scipy import interp


# 定义密集卷积块中单个卷积层
def conv_factory(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=3,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

# 定义transition层
def transition(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    return x

# 定义密集卷积块
def denseblock(x, concat_axis, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    for i in range(layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=concat_axis)(list_feature_map)
        filters = filters + growth_rate
    return x, filters

# 定义压缩-激发层
def squeeze_excitation_layer(x, out_dim, ratio):
    squeeze = GlobalAveragePooling1D()(x)
    excitation = Dense(units=out_dim // ratio, activation="relu",
                       use_bias=False, kernel_initializer='he_normal')(squeeze)
    excitation = Dense(units=out_dim, activation="sigmoid",
                       kernel_initializer='he_normal')(excitation)
    excitation = Reshape((1, out_dim))(excitation)
    return Multiply()([x, excitation])  # 给通道加权
    # excitation = tf.reshape(excitation, [-1, out_dim, 1, 1])
    # excitation = keras.layers.Lambda(lambda excitation_: tf.reshape(excitation_, [-1, out_dim, 1, 1]))(excitation)
    # scale = x * excitation
    # return scale

# 构建模型
def build_model(windows=16, concat_axis=-1, denseblocks=4, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.2, weight_decay=1e-4):
    input_1 = Input(shape=(2*windows+1, 21))
    input_2 = Input(shape=(2*windows+1, 5))
    input_3 = Input(shape=(2*windows+1, 8))
    # 模块化网络一
    x_1 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_1)
    # Add denseblocks
    filters_1 = filters
    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                    filters=filters_1, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition
        x_1 = transition(x_1, concat_axis=concat_axis, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add squeeze-excitation
        #x_1 = Activation('elu')(x_1)
        #x_1 = squeeze_excitation_layer(x_1, filters_1, 16)
    # The last denseblock
    # Add denseblock
    x_1, filters_1 = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                filters=filters_1, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)"""
    x_1 = Activation('elu')(x_1)
    # Add squeeze-excitation
    x_1 = squeeze_excitation_layer(x_1, filters_1, 16)
    x_1 = GlobalAveragePooling1D()(x_1)
    ##################################################################################
    # 模块化网络二
    x_2 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_2)
    # Add denseblocks
    filters_2 = filters
    for i in range(denseblocks - 1):
        # Add denseblock
        x_2, filters_2 = denseblock(x_2, concat_axis=concat_axis, layers=layers,
                                    filters=filters_2, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition
        x_2 = transition(x_2, concat_axis=concat_axis, filters=filters_2,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add squeeze-excitation
        #x_2 = Activation('elu')(x_2)
        #x_2 = squeeze_excitation_layer(x_2, filters_2, 16)
    # The last denseblock
    # Add denseblock
    x_2, filters_2 = denseblock(x_2, concat_axis=concat_axis, layers=layers,
                                filters=filters_2, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_2 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_2)"""
    x_2 = Activation('elu')(x_2)
    # Add squeeze-excitation
    x_2 = squeeze_excitation_layer(x_2, filters_2, 16)
    x_2 = GlobalAveragePooling1D()(x_2)
    ##################################################################################
    # 模块化网络三
    x_3 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_3)
    # Add denseblocks
    filters_3 = filters
    for i in range(denseblocks - 1):
        # Add denseblock
        x_3, filters_3 = denseblock(x_3, concat_axis=concat_axis, layers=layers,
                                    filters=filters_3, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition
        x_3 = transition(x_3, concat_axis=concat_axis, filters=filters_3,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add squeeze-excitation
        #x_3 = Activation('elu')(x_3)
        #x_3 = squeeze_excitation_layer(x_3, filters_3, 16)
    # The last denseblock
    # Add denseblock
    x_3, filters_3 = denseblock(x_3, concat_axis=concat_axis, layers=layers,
                                filters=filters_3, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_3 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_3)"""
    x_3 = Activation('elu')(x_3)
    # Add squeeze-excitation
    x_3 = squeeze_excitation_layer(x_3, filters_3, 16)
    x_3 = GlobalAveragePooling1D()(x_3)
    ##################################################################################
    # 级联三个模块的结果
    x = Concatenate(axis=-1)([x_1, x_2, x_3])
    # 全连接层进行预测
    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[x], name="DenseBlock")
    optimizer = Adam(lr=1e-4, epsilon=1e-8)
    #optimizer = SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# 说明： 性能评估函数
# 输入： predictions 预测结果，Y_test 实际标签，verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 输出： [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr] 验证指标结果
def perform_eval_1(predictions, Y_test, verbose=0):
    #class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    #R_ = np.uint8(Y_test)
    #R = np.asarray(R_)
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP

    # 计算各项指标
    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  # TP/(TP+FN)
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  # TN/(TN+FP)
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  # (TP+TN)/(TP+TN+FP+FN)
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  # TP/(TP+FP)
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  # 2*TP/(2*TP+FP+FN)
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt((CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]

# 说明： 性能评估函数
# 输入： predictions 预测结果，Y_test 实际标签，verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 输出： [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr] 验证指标结果
def perform_eval_2(predictions, Y_test, verbose=0):
    #class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    #R_ = np.uint8(Y_test)
    #R = np.asarray(R_)
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP

    # 计算各项指标
    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  # TP/(TP+FN)
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  # TN/(TN+FP)
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  # (TP+TN)/(TP+TN+FP+FN)
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  # TP/(TP+FP)
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  # 2*TP/(2*TP+FP+FN)
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt((CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]

# 说明： 实验结果保存到文件
# 输入： 文件标识符和结果
# 输出： 无
def write_res_1(filehandle, res, fold=0):
    filehandle.write("Fold: " + str(fold) + " ")
    filehandle.write("Sn(Recall): %s Sp: %s Acc: %s Pre(PPV): %s F1: %s MCC: %s G-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.4f}".format(res[0]),
                      "{:.4f}".format(res[1]),
                      "{:.4f}".format(res[2]),
                      "{:.4f}".format(res[3]),
                      "{:.4f}".format(res[4]),
                      "{:.4f}".format(res[5]),
                      "{:.4f}".format(res[6]),
                      "{:.4f}".format(res[7]),
                      "{:.4f}".format(res[8]))
                     )
    filehandle.flush()
    return

# 说明： 实验结果保存到文件
# 输入： 文件标识符和结果
# 输出： 无
def write_res_2(filehandle, res):
    filehandle.write("Sn(Recall): %s Sp: %s Acc: %s Pre(PPV): %s F1: %s MCC: %s G-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.4f}".format(res[0]),
                      "{:.4f}".format(res[1]),
                      "{:.4f}".format(res[2]),
                      "{:.4f}".format(res[3]),
                      "{:.4f}".format(res[4]),
                      "{:.4f}".format(res[5]),
                      "{:.4f}".format(res[6]),
                      "{:.4f}".format(res[7]),
                      "{:.4f}".format(res[8]))
                     )
    filehandle.flush()
    return

# 说明： loss-epoch，acc-epoch曲线作图函数
def figure(history, K_FOLD, fold):
    def show_train_history(train_history, train_metrics, validation_metrics):
        plt.plot(train_history.history[train_metrics])
        plt.plot(train_history.history[validation_metrics])
        plt.title('Train History')
        plt.grid(True)  # 设置网格线
        plt.ylabel(train_metrics)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')  # 设置图例位置

    # 画图显示训练过程
    def plt_fig(history, K_FOLD, fold):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        show_train_history(history, 'acc', 'val_acc')
        plt.subplot(1, 2, 2)
        show_train_history(history, 'loss', 'val_loss')
        plt.savefig("./picture/%d折交叉第%d折-elu.png" % (K_FOLD, fold))  # 保存图片
        plt.close()

    return plt_fig(history, K_FOLD, fold)


if __name__ == '__main__':

    # 超参数设置
    BATCH_SIZE = 1000
    K_FOLD = 10
    N_EPOCH = 2000
    WINDOWS = 25

    # 打开保存结果的文件
    res_file = open("./result/cv/yan_res_Cov1D_SE_softmax_early_3_elu.txt", "w", encoding='utf-8')
    # 创建空列表，保存每折的结果
    res = []
    # 交叉验证开始
    tprs = []
    # pres = []
    aurocs = []
    # auprs = []
    mean_fpr = np.linspace(0, 1, 100)
    # mean_rec = np.linspace(0, 1, 100)
    plt.figure(figsize=(12, 4))
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    # 分层交叉验证
    for fold in range(K_FOLD):

        # 从文件读取序列片段（训练+验证，阳性+阴性）
        f_r_train = open("./dataset/human/after_CD-HIT(0.4)/train/%s/cv_10/Acetylation_Pos_Neg_train-%d.txt" % (str(2 * WINDOWS + 1), fold), "r", encoding='utf-8')
        f_r_test = open("./dataset/human/after_CD-HIT(0.4)/train/%s/cv_10/Acetylation_Pos_Neg_test-%d.txt" % (str(2 * WINDOWS + 1), fold), "r", encoding='utf-8')

        # 训练序列片段构建
        train_data = f_r_train.readlines()

        # 预测序列片段构建
        test_data = f_r_test.readlines()

        # 关闭文件
        f_r_train.close()
        f_r_test.close()

        # 数据编码
        from information_coding import one_hot, Phy_Chem_Inf_2, Structure_Inf_1

        # one_hot编码序列片段
        train_X_1, train_Y = one_hot(train_data, windows=WINDOWS)
        train_Y = to_categorical(train_Y, num_classes=2)
        test_X_1, test_Y = one_hot(test_data, windows=WINDOWS)
        test_Y = to_categorical(test_Y, num_classes=2)
        # 理化属性信息
        train_X_2 = Phy_Chem_Inf_2(train_data, windows=WINDOWS)
        test_X_2 = Phy_Chem_Inf_2(test_data, windows=WINDOWS)
        # 蛋白质结构信息
        train_X_3 = Structure_Inf_1(train_data, windows=WINDOWS)
        test_X_3 = Structure_Inf_1(test_data, windows=WINDOWS)

        # 引入模型
        model = build_model(windows=WINDOWS)
        # 打印模型
        model.summary()

        # 训练模型
        print("fold:", str(fold))
        # 早停
        history = model.fit(x=[train_X_1, train_X_2, train_X_3], y=train_Y, batch_size=BATCH_SIZE, epochs=N_EPOCH, shuffle=True, class_weight={0: 1.0, 1: 8.9}, callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')], verbose=2, validation_data=([test_X_1, test_X_2, test_X_3], test_Y))
        # 无早停
        # history = model.fit(x=[train_X_1, train_X_2], y=train_Y, batch_size=BATCH_SIZE, epochs=N_EPOCH, shuffle=True, class_weight={0: 1.0, 1: 8.9}, verbose=1, validation_data=([test_X_1, test_X_2], test_Y))

        # 得到预测结果
        predictions = model.predict(x=[test_X_1, test_X_2, test_X_3], verbose=0)

        # 验证预测结果
        res = perform_eval_1(predictions, test_Y, verbose=1)

        # 将结果写入文件
        write_res_1(res_file, res, fold)

        # 画loss-epoch，acc-epoch曲线
        figure(history, K_FOLD, fold)

        # 画ROC曲线
        R = np.asarray(np.uint8([sublist[1] for sublist in test_Y]))
        plt.subplot(1, 2, 1)
        fpr, tpr, auc_thresholds = metrics.roc_curve(y_true=R, y_score=np.asarray(predictions)[:, 1], pos_label=1)
        # 计算AUROC
        auroc_score = metrics.auc(fpr, tpr)
        # interp：插值，把结果添加到tprs列表中
        aurocs.append(auroc_score)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        # 画图，只需要plt.plot(fpr, tpr)，变量auc_score只是记录auc的值，通过auc()函数计算
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (fold, auroc_score))

        # 画PR曲线
        plt.subplot(1, 2, 2)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true=R, probas_pred=np.asarray(predictions)[:, 1], pos_label=1)
        # 计算AUPR
        aupr_score = metrics.auc(recall, precision)
        # interp：插值，把结果添加到pres列表中
        # auprs.append(aupr_score)
        # interp_pre = interp(mean_rec, recall, precision)
        # interp_pre[0] = 0.0
        # pres.append(interp_pre)
        # 画图，只需要plt.plot(fpr, tpr)，变量auc_score只是记录auc的值，通过auc()函数计算
        plt.plot(recall, precision, lw=1, alpha=0.3, label='PR fold %d(area=%0.4f)' % (fold, aupr_score))

        # 保存训练好的模型(既保存了模型图结构，又保存了模型参数)
        model.save('./model/yan_model_Cov1D_SE_softmax_early_3_elu_fold%d.h5' % fold)

    # 画第一个子图
    plt.subplot(1, 2, 1)
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auroc, std_auroc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # plt.set(xlim=[-0.05, 1.05], xlabel='False Positive Rate', ylim=[-0.05, 1.05], ylabel='True Positive Rate', title="Receiver operating characteristic curves")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.title('Receiver operating characteristic curves', fontsize=10)
    plt.legend(loc="lower right")

    # 画第二个子图
    plt.subplot(1, 2, 2)
    # mean_pre = np.mean(pres, axis=0)
    # mean_pre[-1] = 1.0
    # mean_aupr = metrics.auc(mean_rec, mean_pre)
    # std_aupr = np.std(auprs)
    # plt.plot(mean_rec, mean_pre, color='b', label=r'Mean PR (AUPR = %0.3f $\pm$ %0.3f)' % (mean_aupr, std_aupr), lw=2, alpha=.8)
    # std_pre = np.std(pres, axis=0)
    # pres_upper = np.minimum(mean_pre + std_pre, 1)
    # pres_lower = np.maximum(mean_pre - std_pre, 0)
    # plt.fill_between(mean_rec, pres_lower, pres_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # plt.set(xlim=[-0.05, 1.05], xlabel='Recall', ylim=[-0.05, 1.05], ylabel='Precision', title="Precision-Recall curves")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.ylabel('Precision', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.title('Precision-Recall curves', fontsize=10)
    plt.legend(loc="upper right")
    plt.savefig("./result/cv/%d折交叉-elu.png" % (K_FOLD))
    plt.close()

    # 关闭文件
    res_file.close()
    ##########################################################################################
    time.sleep(1)
    # 计算10折交叉验证结果每项指标的均值
    sn = []
    sp = []
    acc = []
    pre = []
    f1 = []
    mcc = []
    gmean = []
    auroc = []
    aupr = []

    f_r = open("./result/cv/yan_res_Cov1D_SE_softmax_early_3_elu.txt", "r", encoding='utf-8')
    lines = f_r.readlines()

    for line in lines:
        x = line.split()
        sn.append(float(x[3]))
        sp.append(float(x[5]))
        acc.append(float(x[7]))
        pre.append(float(x[9]))
        f1.append(float(x[11]))
        mcc.append(float(x[13]))
        gmean.append(float(x[15]))
        auroc.append(float(x[17]))
        aupr.append(float(x[19]))

    mean_sn = np.mean(sn)
    mean_sp = np.mean(sp)
    mean_acc = np.mean(acc)
    mean_pre = np.mean(pre)
    mean_f1 = np.mean(f1)
    mean_mcc = np.mean(mcc)
    mean_gmean = np.mean(gmean)
    mean_auroc = np.mean(auroc)
    mean_aupr = np.mean(aupr)

    std_sn = np.std(sn)
    std_sp = np.std(sp)
    std_acc = np.std(acc)
    std_pre = np.std(pre)
    std_f1 = np.std(f1)
    std_mcc = np.std(mcc)
    std_gmean = np.std(gmean)
    std_auroc = np.std(auroc)
    std_aupr = np.std(aupr)

    print("mean_sn:", "{:.4f}".format(mean_sn), "mean_sp:", "{:.4f}".format(mean_sp), "mean_acc:", "{:.4f}".format(mean_acc),
          "mean_pre:", "{:.4f}".format(mean_pre), "mean_f1:", "{:.4f}".format(mean_f1), "mean_mcc", "{:.4f}".format(mean_mcc),
          "mean_gmean:", "{:.4f}".format(mean_gmean), "mean_auroc:", "{:.4f}".format(mean_auroc), "mean_aupr:", "{:.4f}".format(mean_aupr))
    print("std_sn:", "{:.4f}".format(std_sn), "std_sp:", "{:.4f}".format(std_sp), "std_acc:", "{:.4f}".format(std_acc),
          "std_pre:", "{:.4f}".format(std_pre), "std_f1:", "{:.4f}".format(std_f1), "std_mcc", "{:.4f}".format(std_mcc),
          "std_gmean:", "{:.4f}".format(std_gmean), "std_auroc:", "{:.4f}".format(std_auroc), "std_aupr:", "{:.4f}".format(std_aupr))

    f_r.close()
