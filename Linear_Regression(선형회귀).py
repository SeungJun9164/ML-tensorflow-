import tensorflow as tf
import numpy as np

# # hypothesis = W * x + b : 가설
#
# x_data = [1, 2, 3, 4, 5]
# y_data = [1, 2, 3, 4, 5]
#
# W = tf.Variable(2.9) # Variable : 메로리에 저장하는 변수(값 지정)
# b = tf.Variable(0.5)
#
# hypothesis = W * x_data + b
#
# # reduce_mean : 특정 차원을 제거하고 평균을 구하는 함수
# # reduce_mean(x) : 변수 x가 가리키는 배열 전체 원소의 평균을 구한다
# # reduce_mean(x, 0) : 열 단위로 평균 / reduce_mean(x, 1) : 행 단위로 평균
# # square : 제곱을 계산
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
#
# # Gradient descent : 경사하강법 / minimize cost를 찾기 위한 알고리즘
#
# # Learning_rate initialize
# learning_rate = 0.01
#
# # Gradient descent
# # with _ as _ : 파일을 열때 사용
# # with문 안에 있는 문장들의 변수(W, b) 변화를 tape 이라는 변수에 저장
# with tf.GradientTape() as tape:
#     hypothesis = W * x_data + b
#     cost = cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
# # gradient : 경사도 값(cost에 대해 변수들의 대한 기울기(미분) 값을 구함)
# W_grad, b_grad = tape.gradient(cost, [W, b])
#
# # assign_sub : A -= B
# # learning_rate : grad 값을 얼만큼 반영할 것인지(값이 크면 많이 반영)
# W.assign_sub(learning_rate * W_grad)
# b.assign_sub(learning_rate * b_grad)
#
# # Gradient descent full code
#
# # Data
# x_data = [1, 2, 3, 4, 5]
# y_data = [1, 2, 3, 4, 5]
#
# # W, b initialize
# W = tf.Variable(2.9)
# b = tf.Variable(0.5)
#
# # W, b update
# for i in range(100):
#     # Gradient descent
#     with tf.GradientTape() as tape:
#         hypothesis = W * x_data + b
#         cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#     W_grad, b_grad = tape.gradient(cost, [W, b])
#     W.assign_sub(learning_rate * W_grad)
#     b.assign_sub(learning_rate * b_grad)
#     if i % 10 == 0:
#       print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
#
# print()
#
# # predict
# print(W * 5 + b)
# print(W * 2.5 + b)
#
# # Simplified hypothesis
# # hypothesis = W * x_data
#
# # 파이썬 코드
# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])
#
# def cost_func(W, X, Y):
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2
#     return c / len(X) # c를 계속 더한후 X의 길이만큼 나눔(평균을 구한 것)
#
# # linspace : 특정 구간애서 출력한 데이터의 개수를 설정하여, 동일한 각격으로 숫자 출력
# for feed_W in np.linspace(-3, 5, num=15):
#     curr_cost = cost_func(feed_W, X, Y)
#     print("{:6.2f} | {:10.5f}".format(feed_W, curr_cost))

# # tensorflow 코드
# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])
#
# def cost_func(W, X, Y):
#     hypothesis = X * W
#     return tf.reduce_mean(tf.square(hypothesis - Y))
#
# W_values = np.linspace(-3, 5, num=15)
# cost_values = []
#
# for feed_W in W_values:
#     curr_cost = cost_func(feed_W, X, Y)
#     cost_values.append(curr_cost)
#     print("{:6.2f} | {:10.5f}".format(feed_W, curr_cost))
#
# # Gradient descent
tf.set_random_seed(0)

x_data = [1., 2., 3., 4.]
y_date = [1., 3., 5., 7.]

W = tf.Variable(tf.random_normal([1], -100., 100.))

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))

# Multi-variable linear regression : 다중 선형회귀
x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.]

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

learnig_rate = 0.000001

for i in range(1000+1):
    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

        w1.assign_sub(learnig_rate * w1_grad)
        w2.assign_sub(learnig_rate * w2_grad)
        w3.assign_sub(learnig_rate * w3_grad)
        b.assign_sub(learnig_rate * b_grad)

        if i % 50 == 0:
            print("{:5} | {:12.4f}".format(i, cost.numpy()))

# Matrix
data = np.array([
   # x1,  x2,  x3,  y
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.],
], dtype=np.float32)

# slice data
# data[:, :-1] : 행렬을 나타내는 것으로 [행, 열]을 나타낸다.
# data[:, :-1] : :앞뒤에 아무것도 없으면 처음부터 끝까지를 나타내고 :-1 은 뒤에서 마지막 한개를 제외한 모든 열을 선택
# [-1] : 마지막 열만 선택
X = data[:, :-1]
y = data[:, [-1]]

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

learnig_rate = 0.000001

def predict(X):
    return tf.matmul(X, W) + b # matmul : matrix multiply(행렬 곱)

n_epochs = 2000
for i in range(n_epochs+1):
    with tf.GradientTape as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

        W_grad, b_grad = tape.gradient(cost, [W, b])

        W.assign_sub(learnig_rate * W_grad)
        b.assign_sub(learnig_rate * b_grad)

        if i % 100 == 0:
            print("{:5} | {:10.4f}".format((i, cost.numpy())))