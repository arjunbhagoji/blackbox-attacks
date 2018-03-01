import numpy as np
import tensorflow as tf
import keras.backend as K
import os
from mnist import data_mnist, set_mnist_flags, load_model
from tf_utils import tf_test_error_rate, batch_eval
from keras.utils import np_utils
from attack_utils import gen_grad

import time
from os.path import basename
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool 

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

K.set_learning_phase(0)

RANDOM = True
BATCH_SIZE = 100
CLIP_MIN = 0
CLIP_MAX = 1
PARALLEL_FLAG = False

IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

def wb_write_out(eps, white_box_error, wb_norm):
    if RANDOM is False:
        print('Fraction of targets achieved (white-box) for {}: {}'.format(target, white_box_error))
    else:
        print('Fraction of targets achieved (white-box): {}'.format(white_box_error))
    return

def est_write_out(eps, success, avg_l2_perturb, X_adv=None):
    if RANDOM is False:
        print('Fraction of targets achieved (query-based) with {} for {}: {}'.format(target_model_name, target, success))
    else:
        print('Fraction of targets achieved (query-based): {}'.format(success)) 
    return

def pca_components(X, dim):
    X = X.reshape((len(X), dim))
    pca = PCA(n_components=dim)
    pca.fit(X)

    U = (pca.components_).T
    U_norm = normalize(U, axis=0)

    return U_norm[:,:args.num_comp]


def xent_est(prediction, x, x_plus_i, x_minus_i, curr_target):
    pred_plus = K.get_session().run([prediction], feed_dict={x: x_plus_i})[0]
    pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]
    pred_minus = K.get_session().run([prediction], feed_dict={x: x_minus_i})[0]
    pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]
    single_grad_est = (pred_plus_t - pred_minus_t)/args.delta

    return single_grad_est/2.0

def CW_est(logits, x, x_plus_i, x_minus_i, curr_sample, curr_target):
    curr_logits = K.get_session().run([logits], feed_dict={x: curr_sample})[0]
    # So that when max is taken, it returns max among classes apart from the
    # target
    curr_logits[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
    max_indices = np.argmax(curr_logits, 1)
    logit_plus = K.get_session().run([logits], feed_dict={x: x_plus_i})[0]
    logit_plus_t = logit_plus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_plus_max = logit_plus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_minus = K.get_session().run([logits], feed_dict={x: x_minus_i})[0]
    logit_minus_t = logit_minus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_minus_max = logit_minus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_t_grad_est = (logit_plus_t - logit_minus_t)/args.delta
    logit_max_grad_est = (logit_plus_max - logit_minus_max)/args.delta

    return logit_t_grad_est/2.0, logit_max_grad_est/2.0

def overall_grad_est(j, logits, prediction, x, curr_sample, curr_target, 
                        p_t, random_indices, num_groups, U=None):
    basis_vec = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

    if PCA_FLAG == False:
        if j != num_groups-1:
            curr_indices = random_indices[j*args.group_size:(j+1)*args.group_size]
        elif j == num_groups-1:
            curr_indices = random_indices[j*args.group_size:]
        row = curr_indices/IMAGE_COLS
        col = curr_indices % IMAGE_COLS
        for i in range(len(curr_indices)):
            basis_vec[:, row[i], col[i]] = 1.

    elif PCA_FLAG == True:
        basis_vec[:] = U[:,j].reshape((1, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        # basis_vec = np.sign(basis_vec)

    x_plus_i = np.clip(curr_sample + args.delta * basis_vec, CLIP_MIN, CLIP_MAX)
    x_minus_i = np.clip(curr_sample - args.delta * basis_vec, CLIP_MIN, CLIP_MAX)

    if args.loss_type == 'cw':
        logit_t_grad_est, logit_max_grad_est = CW_est(logits, x, x_plus_i,
                                        x_minus_i, curr_sample, curr_target)
        if '_un' in args.method:
            single_grad_est = logit_t_grad_est - logit_max_grad_est
        else:
            single_grad_est = logit_max_grad_est - logit_t_grad_est
    elif args.loss_type == 'xent':
        single_grad_est = xent_est(prediction, x, x_plus_i, x_minus_i, curr_target)

    return single_grad_est

def spsa(prediction, logits, x, curr_sample, curr_target, p_t, dim):
    grad_est = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS,
                         NUM_CHANNELS))
    logits_np = K.get_session().run([logits], feed_dict={x: curr_sample})[0]
    perturb_vec = np.random.normal(size=dim*BATCH_SIZE).reshape((BATCH_SIZE, dim))
    for i in range(BATCH_SIZE):
        perturb_vec[i,:] = perturb_vec[i,:]/np.linalg.norm(perturb_vec[i,:])
    # perturb_vec = perturb_vec/np.linalg.norm(perturb_vec)
    perturb_vec = perturb_vec.reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

    x_plus_i = np.clip(curr_sample + args.delta * perturb_vec, CLIP_MIN, CLIP_MAX)
    x_minus_i = np.clip(curr_sample - args.delta * perturb_vec, CLIP_MIN, CLIP_MAX)

    if args.loss_type == 'cw':
        logit_t_grad_est, logit_max_grad_est = CW_est(logits, x, x_plus_i,
                                        x_minus_i, curr_sample, curr_target)
        if '_un' in args.method:
            single_grad_est = logit_t_grad_est - logit_max_grad_est
        else:
            single_grad_est = logit_max_grad_est - logit_t_grad_est
    elif args.loss_type == 'xent':
        single_grad_est = xent_est(prediction, x, x_plus_i, x_minus_i, curr_target)

    for i in range(BATCH_SIZE):
        grad_est[i] = single_grad_est[i]/perturb_vec[i]
    # Getting gradient of the loss
    if args.loss_type == 'xent':
        loss_grad = -1.0 * grad_est/p_t[:, None, None, None]
    elif args.loss_type == 'cw':
        logits_np_t = logits_np[np.arange(BATCH_SIZE), list(curr_target)].reshape(BATCH_SIZE)
        logits_np[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
        max_indices = np.argmax(logits_np, 1)
        logits_np_max = logits_np[np.arange(BATCH_SIZE), list(max_indices)].reshape(BATCH_SIZE)
        logit_diff = logits_np_t - logits_np_max
        if '_un' in args.method:
            zero_indices = np.where(logit_diff + args.conf < 0.0)
        else:
            zero_indices = np.where(-logit_diff + args.conf < 0.0)
        grad_est[zero_indices[0]] = np.zeros((len(zero_indices), IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        loss_grad = grad_est

    return loss_grad


def finite_diff_method(prediction, logits, x, curr_sample, curr_target, p_t, dim, U=None):
    grad_est = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS,
                         NUM_CHANNELS))
    logits_np = K.get_session().run([logits], feed_dict={x: curr_sample})[0]
    if PCA_FLAG == False:
        random_indices = np.random.permutation(dim)
        num_groups = dim / args.group_size
    elif PCA_FLAG == True:
        num_groups = args.num_comp
        random_indices = None

    if PARALLEL_FLAG == True:

        j_list = range(num_groups)

        #Creating partial function with single argument
        partial_overall_grad_est = partial(overall_grad_est, logits=logits,
            prediction=prediction, x=x, curr_sample=curr_sample, 
            curr_target=curr_target, p_t=p_t, random_indices=random_indices, num_groups=num_groups, U=U)

        #Creating pool of threads 
        pool = ThreadPool(3)
        all_grads = pool.map(partial_overall_grad_est, j_list)

        print(len(all_grads))

        pool.close() 
        pool.join()

        for j in j_list:
            # all_grads.append(partial_overall_grad_est(j))
            if PCA_FLAG == False:
                if j != num_groups-1:
                    curr_indices = random_indices[j*args.group_size:(j+1)*args.group_size]
                elif j == num_groups-1:
                    curr_indices = random_indices[j*args.group_size:]
                row = curr_indices/IMAGE_COLS
                col = curr_indices % IMAGE_COLS
            for i in range(len(curr_indices)):
                grad_est[:, row[i], col[i]] = all_grads[j].reshape((BATCH_SIZE,1))

    else:
        for j in range(num_groups):
            single_grad_est = overall_grad_est(j, logits, prediction, x, curr_sample, curr_target, 
                        p_t, random_indices, num_groups, U)
            if PCA_FLAG == False:
                if j != num_groups-1:
                    curr_indices = random_indices[j*args.group_size:(j+1)*args.group_size]
                elif j == num_groups-1:
                    curr_indices = random_indices[j*args.group_size:]
                row = curr_indices/IMAGE_COLS
                col = curr_indices % IMAGE_COLS
                for i in range(len(curr_indices)):
                    grad_est[:, row[i], col[i]] = single_grad_est.reshape((BATCH_SIZE,1))
            elif PCA_FLAG == True:
                basis_vec = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
                basis_vec[:] = U[:,j].reshape((1, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
                grad_est += basis_vec*single_grad_est[:,None,None,None]

    # Getting gradient of the loss
    if args.loss_type == 'xent':
        loss_grad = -1.0 * grad_est/p_t[:, None, None, None]
    elif args.loss_type == 'cw':
        logits_np_t = logits_np[np.arange(BATCH_SIZE), list(curr_target)].reshape(BATCH_SIZE)
        logits_np[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
        max_indices = np.argmax(logits_np, 1)
        logits_np_max = logits_np[np.arange(BATCH_SIZE), list(max_indices)].reshape(BATCH_SIZE)
        logit_diff = logits_np_t - logits_np_max
        if '_un' in args.method:
            zero_indices = np.where(logit_diff + args.conf < 0.0)
        else:
            zero_indices = np.where(-logit_diff + args.conf < 0.0)
        grad_est[zero_indices[0]] = np.zeros((len(zero_indices), IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        loss_grad = grad_est

    return loss_grad

def estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, delta=None):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    U = None
    X_adv = np.zeros((BATCH_SIZE*BATCH_EVAL_NUM, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    if PCA_FLAG == True:
        U = pca_components(X_test, dim)
    for i in range(BATCH_EVAL_NUM):
        if i % 10 ==0:
            print('Batch no.: {}, {}'.format(i, eps))
        curr_sample = X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))
        curr_sample_ini = X_test_ini[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))

        curr_target = targets[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        curr_prediction = K.get_session().run([prediction], feed_dict={x: curr_sample})[0]

        p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

        if 'query_based' in args.method:
            loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                        curr_target, p_t, dim, U)

        # Getting signed gradient of loss
        if args.norm == 'linf':
            normed_loss_grad = np.sign(loss_grad)
        elif args.norm == 'l2':
            grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis = 1)
            indices = np.where(grad_norm != 0.0)
            normed_loss_grad = np.zeros_like(curr_sample)
            normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

        eps_mod = eps - args.alpha
        if args.loss_type == 'xent':
            if '_un' in args.method:
                x_adv = np.clip(curr_sample + eps_mod * normed_loss_grad, 0, 1)
            else:
                x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, 0, 1)
        elif args.loss_type == 'cw':
            x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, 0, 1)


        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv-curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        X_adv[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = x_adv.reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = K.get_session().run([prediction], feed_dict={x: x_adv})[0]
        success += np.sum(np.argmax(adv_prediction,1) == curr_target)

    success = 100.0 * float(success)/(BATCH_SIZE*BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb/BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb, X_adv)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    return

def estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, beta):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    U = None
    X_adv = np.zeros((BATCH_SIZE*BATCH_EVAL_NUM, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    if PCA_FLAG == True:
        U = pca_components(X_test, dim)
    for i in range(BATCH_EVAL_NUM):
        if i % 10 ==0:
            print('Batch no.: {}, {}'.format(i, eps))
        curr_sample = X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))
        curr_sample_ini = X_test_ini[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))

        curr_target = targets[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        eps_mod = eps - args.alpha

        for j in range(args.num_iter):
            if j % 10 == 0:
                print ('Num_iter:{}'.format(j))
            curr_prediction = K.get_session().run([prediction], feed_dict={x: curr_sample})[0]

            p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

            if 'query_based' in args.method:
                loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                            curr_target, p_t, dim, U)
            elif 'spsa' in args.method:
                loss_grad = spsa(prediction, logits, x, curr_sample,
                                            curr_target, p_t, dim)
                # print loss_grad.shape

            # Getting signed gradient of loss
            if args.norm == 'linf':
                normed_loss_grad = np.sign(loss_grad)
            elif args.norm == 'l2':
                grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis = 1)
                indices = np.where(grad_norm != 0.0)
                normed_loss_grad = np.zeros_like(curr_sample)
                normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

            if args.loss_type == 'xent':
                if '_un' in args.method:
                    x_adv = np.clip(curr_sample + beta * normed_loss_grad, 0, 1)
                else:
                    x_adv = np.clip(curr_sample - beta * normed_loss_grad, 0, 1)
            elif args.loss_type == 'cw':
                x_adv = np.clip(curr_sample - beta * normed_loss_grad, 0, 1)
            r = x_adv-curr_sample_ini
            r = np.clip(r, -eps, eps)
            curr_sample = curr_sample_ini + r

            logits_curr = K.get_session().run([logits], feed_dict={x: curr_sample})[0]
            logits_curr_t = logits_curr[np.arange(BATCH_SIZE), list(curr_target)].reshape(BATCH_SIZE)
            logits_curr[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
            max_indices = np.argmax(logits_curr, 1)
            logits_curr_max = logits_curr[np.arange(BATCH_SIZE), list(max_indices)].reshape(BATCH_SIZE)
            loss = logits_curr_t - logits_curr_max
            # print loss

        x_adv = np.clip(curr_sample, 0, 1)
        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv-curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = K.get_session().run([prediction], feed_dict={x: x_adv})[0]
        X_adv[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = x_adv.reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1))
        success += np.sum(np.argmax(adv_prediction,1) == curr_target)

    success = 100.0 * float(success)/(BATCH_SIZE*BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb/BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb, X_adv)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    return


def white_box_fgsm(prediction, target_model, x, logits, y, X_test, X_test_ini, Y_test_uncat, targets, targets_cat, eps, dim):
    time1 = time.time()
    #Get gradient from model
    if args.loss_type == 'xent':
        grad = gen_grad(x, logits, y)
    elif args.loss_type == 'cw':
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if '_un' in args.method:
            loss = tf.maximum(0.0,real-other+args.conf)
        else:
            loss = tf.maximum(0.0,other-real+args.conf)
        grad = K.gradients(loss, [x])[0]

    # normalized gradient
    if args.norm == 'linf':
        normed_grad = K.sign(grad)
    elif args.norm == 'l2':
        normed_grad = K.l2_normalize(grad, axis = (1,2,3))

    # Multiply by constant epsilon
    scaled_grad = (eps - args.alpha) * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if args.loss_type == 'xent':
        if '_un' in args.method:
            adv_x_t = K.stop_gradient(x + scaled_grad)
        else:
            adv_x_t = K.stop_gradient(x - scaled_grad)
    elif args.loss_type == 'cw':
        adv_x_t = K.stop_gradient(x - scaled_grad)

    adv_x_t = K.clip(adv_x_t, CLIP_MIN, CLIP_MAX)

    X_test_ini_slice = X_test_ini[:BATCH_SIZE*BATCH_EVAL_NUM]
    targets_cat_mod = targets_cat[:BATCH_SIZE*BATCH_EVAL_NUM]
    targets_mod = targets[:BATCH_SIZE*BATCH_EVAL_NUM]

    X_adv_t = np.zeros_like(X_test_ini_slice)

    for i in range(BATCH_EVAL_NUM):
        X_test_slice = X_test[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        targets_cat_slice = targets_cat[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        X_adv_t[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)] = K.get_session().run([adv_x_t], feed_dict={x: X_test_slice, y: targets_cat_slice})[0]

    adv_pred_np = K.get_session().run([prediction], feed_dict={x: X_adv_t})[0]
    pred_np = K.get_session().run([prediction], feed_dict={x: X_test_slice})[0]

    # _, _, white_box_error = tf_test_error_rate(target_model, x, X_adv_t, targets_cat_mod)
    white_box_error = 100.0 * np.sum(np.argmax(adv_pred_np,1) != targets_mod) / adv_pred_np.shape[0]
    benign_error = 100.0 * np.sum(np.argmax(pred_np,1) != Y_test_uncat) / pred_np.shape[0]

    print('Benign error: {}'.format(benign_error))

    if '_un' not in args.method:
        white_box_error = 100.0 - white_box_error

    wb_norm = np.mean(np.linalg.norm((X_adv_t-X_test_ini_slice).reshape(BATCH_SIZE*BATCH_EVAL_NUM, dim), axis=1))
    print('Average white-box l2 perturbation: {}'.format(wb_norm))
    time2= time.time()
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    wb_write_out(eps, white_box_error, wb_norm)

    return

def white_box_fgsm_iter(prediction, target_model, x, logits, y, X_test, X_test_ini, targets, targets_cat, eps, dim, beta):
    #Get gradient from model
    if args.loss_type == 'xent':
        grad = gen_grad(x, logits, y)
    elif args.loss_type == 'cw':
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if '_un' in args.method:
            loss = tf.maximum(0.0,real-other+args.conf)
        else:
            loss = tf.maximum(0.0,other-real+args.conf)
        grad = K.gradients(loss, [x])[0]

    # normalized gradient
    if args.norm == 'linf':
        normed_grad = K.sign(grad)
    elif args.norm == 'l2':
        normed_grad = K.l2_normalize(grad, axis = (1,2,3))

    # Multiply by constant epsilon
    scaled_grad = beta * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if args.loss_type == 'xent':
        if '_un' in args.method:
            adv_x_t = K.stop_gradient(x + scaled_grad)
        else:
            adv_x_t = K.stop_gradient(x - scaled_grad)
    elif args.loss_type == 'cw':
        adv_x_t = K.stop_gradient(x - scaled_grad)

    adv_x_t = K.clip(adv_x_t, CLIP_MIN, CLIP_MAX)

    X_test_ini_mod = X_test_ini[:BATCH_SIZE*BATCH_EVAL_NUM]
    targets_cat_mod = targets_cat[:BATCH_SIZE*BATCH_EVAL_NUM]
    targets_mod = targets[:BATCH_SIZE*BATCH_EVAL_NUM]

    X_adv_t = np.zeros_like(X_test_ini_mod)

    for i in range(BATCH_EVAL_NUM):
        X_test_slice = X_test[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        X_test_ini_slice = X_test_ini[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        targets_cat_slice = targets_cat[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        X_adv_curr = X_test_slice
        for k in range(args.num_iter):
            X_adv_curr = K.get_session().run([adv_x_t], feed_dict={x: X_adv_curr, y: targets_cat_slice})[0]
            r = X_adv_curr - X_test_ini_slice
            r = np.clip(r, -eps, eps)
            X_adv_curr = X_test_ini_slice + r
        X_adv_t[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)] = np.clip(X_adv_curr, CLIP_MIN, CLIP_MAX)

    adv_pred_np = K.get_session().run([prediction], feed_dict={x: X_adv_t})[0]

    # _, _, white_box_error = tf_test_error_rate(target_model, x, X_adv_t, targets_cat_mod)
    white_box_error = 100.0 * np.sum(np.argmax(adv_pred_np,1) != targets_mod) / adv_pred_np.shape[0]
    if '_un' not in args.method:
        white_box_error = 100.0 - white_box_error

    wb_norm = np.mean(np.linalg.norm((X_adv_t-X_test_ini_mod).reshape(BATCH_SIZE*BATCH_EVAL_NUM, dim), axis=1))
    print('Average white-box l2 perturbation: {}'.format(wb_norm))

    wb_write_out(eps, white_box_error, wb_norm)

    return

def main(target_model_name, target=None):
    np.random.seed(0)
    tf.set_random_seed(0)

    x = K.placeholder((None,
                       IMAGE_ROWS,
                       IMAGE_COLS,
                       NUM_CHANNELS))

    y = K.placeholder((None, NUM_CLASSES))

    dim = int(IMAGE_ROWS*IMAGE_COLS)

    _, _, X_test_ini, Y_test = data_mnist()
    print('Loaded data')

    Y_test_uncat = np.argmax(Y_test,axis=1)

    # target model for crafting adversarial examples
    target_model = load_model(target_model_name)
    target_model_name = basename(target_model_name)

    logits = target_model(x)
    prediction = K.softmax(logits)

    sess = tf.Session()
    print('Creating session')

    if '_un' in args.method:
        targets = np.argmax(Y_test[:BATCH_SIZE*BATCH_EVAL_NUM], 1)
    elif RANDOM is False:
        targets = np.array([target]*(BATCH_SIZE*BATCH_EVAL_NUM))
    elif RANDOM is True:
        targets = []
        allowed_targets = list(range(NUM_CLASSES))
        for i in range(BATCH_SIZE*BATCH_EVAL_NUM):
            allowed_targets.remove(Y_test_uncat[i])
            targets.append(np.random.choice(allowed_targets))
            allowed_targets = list(range(NUM_CLASSES))
        # targets = np.random.randint(10, size = BATCH_SIZE*BATCH_EVAL_NUM)
        targets = np.array(targets)
        print targets
    targets_cat = np_utils.to_categorical(targets, NUM_CLASSES).astype(np.float32)

    if args.norm == 'linf':
        # eps_list = list(np.linspace(0.025, 0.1, 4))
        # eps_list.extend(np.linspace(0.15, 0.5, 8))
        eps_list = [0.3]
        if "_iter" in args.method:
            eps_list = [0.3]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 2.0, 5))
        eps_list.extend(np.linspace(2.5, 9.0, 14))
        # eps_list = [5.0]
    print(eps_list)

    random_perturb = np.random.randn(*X_test_ini.shape)

    if args.norm == 'linf':
        random_perturb_signed = np.sign(random_perturb)
        X_test = np.clip(X_test_ini + args.alpha * random_perturb_signed, CLIP_MIN, CLIP_MAX)
    elif args.norm == 'l2':
        random_perturb_unit = random_perturb/np.linalg.norm(random_perturb.reshape(curr_len,dim), axis=1)[:, None, None, None]
        X_test = np.clip(X_test_ini + args.alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

    for eps in eps_list:
        if '_iter' in args.method:
            white_box_fgsm_iter(prediction, target_model, x, logits, y, X_test, X_test_ini, Y_test_uncat, targets, targets_cat, eps, dim, args.beta)
            estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, args.beta)
        else:
            white_box_fgsm(prediction, target_model, x, logits, y, X_test, X_test_ini, Y_test_uncat, targets, targets_cat, eps, dim)
            # estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--method", choices=['query_based', 'spsa_iter',
                        'query_based_un', 'spsa_un_iter', 'query_based_un_iter','query_based_iter'], default='query_based_un')
    parser.add_argument("--delta", type=float, default=0.01,
                        help="local perturbation")
    parser.add_argument("--norm", type=str, default='linf',
                            help="Norm to use for attack")
    parser.add_argument("--loss_type", type=str, default='cw',
                            help="Choosing which type of loss to use")
    parser.add_argument("--conf", type=float, default=0.0,
                                help="Strength of CW sample")
    parser.add_argument("--alpha", type=float, default=0.0,
                            help="Strength of random perturbation")
    parser.add_argument("--group_size", type=int, default=1,
                            help="Number of features to group together")
    parser.add_argument("--num_comp", type=int, default=None,
                            help="Number of pca components")
    parser.add_argument("--num_iter", type=int, default=40,
                            help="Number of iterations")
    parser.add_argument("--beta", type=int, default=0.01,
                            help="Step size per iteration")


    args = parser.parse_args()

    if '_un' in args.method:
        RANDOM = True
        PCA_FLAG=False
    if args.num_comp != 784:
        PCA_FLAG = True

    if '_iter' in args.method:
        BATCH_EVAL_NUM = 10
    else:
        BATCH_EVAL_NUM = 10

    # target_model_name = basename(args.target_model)

    set_mnist_flags()

    if RANDOM is False:
        for i in range(NUM_CLASSES):
            main(args.target_model, i)
    elif RANDOM is True:
        main(args.target_model)
