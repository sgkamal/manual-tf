import numpy as np
import sklearn
class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        for node in input_nodes:
            node.output_nodes.append(self)
        _default_graph.operations.append(self)
    def compute(self):
        pass
class add(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return(x_var + y_var)
class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return(x_var * y_var)
class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return(x_var.dot(y_var))

class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)
class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)

class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    def set_as_default(self):
        global _default_graph
        _default_graph = self

    def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first , then Ax + b).
    """

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return(nodes_postorder)

class Session():
    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                # OPERATION
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if type(node.output) == list:
                node.output = np.array(node.output)
        return(operation.output)
g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)
x = Placeholder()
y = multiply(A,x)
z = add(y,b)
sess = Session()
result = sess.run(operation = z, feed_dict = {x:100000})
#print(result)

class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])
    def compute(self, z_val):
        return(1/(1+np.exp(-z_val)))

# from sklearn.datasets import make_blobs
# data = make_blobs(n_samples = 50, n_features = 2, centers = 2, random_state = 75)
# features = data[0]
# labels = data[1]
# plt.scatter(features[:,0], features[:,1], c=labels, cmap='coolwarm')
# x = np.linspace(0,11,10)
# y = -x + 5
# g = Graph()
# g.set_as_default()
# x = Placeholder()
# w = Variable([1,1])
# b = Variable(-5)
# z = add(matmul(w,x),b)
# a = Sigmoid(z)
# sess = Session()
# result = sess.run(operation = a, feed_dict={x:[8,10]})
# print(result)
#
# import tensorflow as tf
# print(tf.__version__)
# hello = tf.constant("Hello ")
# world = tf.constant("World")
# with tf.Session() as sess:
#     result = sess.run(hello+world)
# print(result)
# a = tf.constant(10)
# b = tf.constant(20)
# with tf.Session() as sess:
#     result = sess.run(a + b)
#
# const = tf.constant(10)
# fill_mat = tf.fill((4,4), 10)
# myzeros = tf.zeros((4,4))
# myones = tf.ones((4,4))
# myrandn = tf.random_normal((4,4),mean= 0,stddev = 1.0)
# myrandu = tf.random_uniform((4,4), minval=0, maxval=1)
# my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]
# with tf.Session() as sess:
#     for op in sess.run(my_ops):
#         print(op)
#         print('\n')
# a = tf.constant([[1,2],[3,4]])
# b = tf.constant([[10],[100]])
# with tf.Session() as sess:
#     print(sess.run(tf.matmul(a,b)))
#
# n1 = tf.constant(1)
# n2 = tf.constant(2)
# n3 = n1 + n2
# with tf.Session() as sess:
#     result = sess.run(n3)
# print(result)
# print(tf.get_default_graph())
# g = tf.Graph()
# print(g)
# g1 = tf.get_default_graph()
# print(g1)
# g2 = tf.Graph()
#
# my_tensor = tf.random_uniform((4,4),0,1)
# my_var = tf.Variable(initial_value=my_tensor)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print((sess.run(my_var)))
# ph = tf.placeholder(tf.float32)
#
# np.random.seed(101)
# tf.set_random_seed(101)
#
# rand_a = np.random.uniform(0,100,(5,5))
# rand_b = np.random.uniform(0,100,(5,1))
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# add_op = a + b
# mul_op = a * b
# with tf.Session() as sess:
#     add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
#     print(add_result)
#     print('\n')
#     mult_result = sess.run(mul_op, feed_dict = {a:rand_a, b:rand_b})
#     print(mult_result)
#
# n_features = 10
# n_dense_neurons = 3
#
# x = tf.placeholder(tf.float32, (None, n_features))
