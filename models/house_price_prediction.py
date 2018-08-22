import tensorflow as tf
import numpy as np
import math
import matplotlib

# This is macOS specific,..grr
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_houses = 100
np.random.seed(17)

# Step 1 -  Data Preparation

# Generate random data for house sizes, this may be input from file.
house_sizes=np.random.randint(low=1000, high=50000, size=num_houses)

# Generate house prices based on sizes.
house_prices=house_sizes*np.random.randint(low=10, high=15, size=num_houses)+np.random.randint(low=1000, high=1500, size=num_houses)


def plot(house_sizes,house_prices):
    plt.plot(house_sizes, house_prices, "bx")
    plt.ylabel("House Price")
    plt.xlabel("House Size")
    plt.show()


# Lets plot size vs price of house
# plot(house_sizes,house_prices )


def normalize(array):
    return (array - array.mean())/array.std()


# Take 75% of values for training.
num_train_samples_size = math.floor(num_houses * 0.75)

# 75% is training data
train_house_sizes=np.asanyarray(house_sizes[:num_train_samples_size])
train_house_prices=np.asanyarray(house_prices[:num_train_samples_size])

# normalize training data
train_house_sizes_norm = normalize(train_house_sizes)
train_house_prices_norm = normalize(train_house_prices)

# 25% is test data
test_house_sizes=np.asanyarray(house_sizes[num_train_samples_size:])
test_house_prices=np.asanyarray(house_prices[num_train_samples_size:])

# normalize test data
test_house_sizes_norm = normalize(test_house_sizes)
test_house_prices_norm = normalize(test_house_prices)

# Step 2 -  Create Predict function
# Set up TF placeholders
tf_house_sizes = tf.placeholder(dtype="float",name="house_size")
tf_house_prices = tf.placeholder(dtype="float",name="house_price")

tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

# Price predict function
tf_price_predict = tf.add(tf.multiply(tf_size_factor,tf_house_sizes),tf_price_offset)


# Error function which we want to reduce
tf_cost = tf.reduce_sum(tf.pow(tf_price_predict-tf_house_prices,2)/(2 * num_train_samples_size))
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


# Step 3 -  Train model

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    log_every_itr = 2
    num_training_itr = 50

    for itr in range(num_training_itr):

        # Feed all training data
        for(x,y) in zip(train_house_sizes_norm,train_house_prices_norm):
            session.run(optimizer,feed_dict={tf_house_sizes:x,tf_house_prices:y})

            # Display current cost
            if(itr+1)%log_every_itr == 0:
                c = session.run(tf_cost,feed_dict={tf_house_sizes:train_house_sizes_norm , tf_house_prices:train_house_prices_norm})
                print("Iteration No # ", '%04d'% (itr+1) , "cost", "{:.9f}".format(c),\
                      "size_factor" , session.run(tf_size_factor), \
                      "price offset", session.run(tf_price_offset))

    print("Optimization finished")
    training_cost = session.run(tf_cost,
                    feed_dict={tf_house_sizes: train_house_sizes_norm, tf_house_prices: train_house_prices_norm})
    print("training_cost", "{:.9f}".format(c), \
          "size_factor", session.run(tf_size_factor), \
          "price offset", session.run(tf_price_offset))

    # Step 4 -  Predict price for given size of house
    for size in range(1000):
        print("Price for house of size", "%04d"%size," = ", "%.9f" % session.run(tf_price_predict, feed_dict={tf_house_sizes: size}))














