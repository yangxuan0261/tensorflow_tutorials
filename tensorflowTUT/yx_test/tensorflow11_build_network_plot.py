import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def add_layer(inputs,in_size,out_size,activation_function=None): #activation_function=None线性函数
    print("in_size:", in_size)
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #Weight中都是随机变量
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #biases推荐初始值不为0
    Wx_plus_b = tf.matmul(inputs,Weights)+biases #inputs*Weight+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def test_activation_function():
    #创建数据x_data，y_data
    x_data = np.linspace(-1,1,300)[:,np.newaxis] #[-1,1] 区间，300个单位，np.newaxis增加维度， 这个也就是表示 300x1 的矩阵
    noise = np.random.normal(0,0.05,x_data.shape) #噪点
    y_data = np.square(x_data)-0.5+noise

    xs = tf.placeholder(tf.float32,[None,1]) # 不限定行，限定为1列
    ys = tf.placeholder(tf.float32,[None,1])
    #三层神经，输入层（1个神经元），隐藏层（10神经元），输出层（1个神经元）
    l1 = add_layer(xs,1,10,activation_function=tf.nn.relu) #输入层
    prediction = add_layer(l1,10,1,activation_function=None) #隐藏层

    #predition值与y_data差别, 和下面的 train_step 就是 真实值与预测值 参与训练，调参w，然后就可以 run 张量 prediction ，并喂上数据 x_data(测试的运算已经不需要真实值 y_data 参与运算了) 来获取训练后的 预测值 prediction_value
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) #square()平方,sum()求和,mean()平均值

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #0.1学习效率,minimize(loss)减小loss误差

    init = tf.global_variables_initializer() # 初始化所有变量的 "句柄"
    sess = tf.Session()
    sess.run(init) #先执行init

    #可视化
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    plt.ion() #不让show() block
    plt.show()

    #训练1k次
    print("--- train 1000 times")
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            try:
                ax.lines.remove(lines[0]) #lines建一个抹除一个
            except Exception:
                pass
            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data})) #输出loss值
            #可视化
            prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5) # 画一条线，x_data X轴，prediction_value Y轴，'r-'红线，lw=5线宽5
            plt.pause(0.2) #暂停0.1秒
    sess.close()
    print("--- prediction over")

def main():
    test_activation_function()
    # os.system("pause")
    pass

if __name__ == '__main__': main()
