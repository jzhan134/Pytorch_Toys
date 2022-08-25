'''
#!Stochastic gradient descent
    g = grad(Loss)
    lr = const
    w = w - lr * g
'''

'''
#!Stochastic gradient descent with moment
    g = grad(Loss)
    b = moment * b + (1 - moment) * g #* b0 = 0
    lr = const
    w = w - lr * b
'''

'''
#!Stochastic gradient descent with moment, dampening, and weight decay
    g = grad(Loss) + weight_decay * w # * weight decay is L2 penalty
    b = moment * b + (1 - moment) * (1 - dampening) * g #* b0 = 0
    lr = const
    w = w - lr * b
'''

'''
#!Adagrad (adaptive gradient)
    g = grad(Loss)
    w = w - lr / sqrt(sum(s^2))* g
    #* learning rate is corrected by the sum of 
    #* second moment of gradients
    #* only the gradients of the same parameter are added together
    #? learning rate will decay to zero
'''

'''
#!Adadelta
    g = grad(Loss)
    s = beta * s + (1 - beta) * g^2 
    w = w - lr / sqrt(s) * g
    #* learning rate is now corrected by the 
    #* moving average of the second moment of gradient
    #? learning rate is dynamic based on a period of histories.
'''

'''
#!Adam (combine SGD with moment and Adadelta)
    g = grad(Loss)
    v = beta1 * v + (1 - beta1) * g
    v = v / (1 - beta1) # * bias correction
    s = beta2 * s + (1 - beta2) * g^2 
    s = s / (1 - beta2) # * correct for bias
    w = w - lr / sqrt(s) * v
'''