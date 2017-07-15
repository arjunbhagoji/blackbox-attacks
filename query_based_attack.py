import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from tf_utils import tf_test_error_rate, batch_eval
from keras.utils import np_utils
from attack_utils import gen_grad
from matplotlib import image as img
import time
from os.path import basename


from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

RANDOM = True
BATCH_SIZE = 100
BATCH_EVAL_NUM = 100
CLIP_MIN = 0
CLIP_MAX = 1

def wb_img_save(adv_pred_np, targets, eps, X_adv_t):
    img_count = 0
    for k in range(30):
        adv_label_wb = np.argmax(adv_pred_np[k].reshape(1, FLAGS.NUM_CLASSES),1)
        if adv_label_wb[0] != targets[k]:
            img.imsave( 'images/'+args.method+'/'+args.norm+'/{}_wb_{}_{}_{}.png'.format(target_model_name,
                adv_label_wb, targets[k], eps),
                X_adv_t[k].reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)*255, cmap='gray')
            img_count += 1
        if img_count >=10:
            return
    return

def est_img_save(i, adv_prediction, curr_target, eps, x_adv):
    img_count = 0
    if i==0:
        for k in range(30):
            adv_label = np.argmax(adv_prediction[k].reshape(1, FLAGS.NUM_CLASSES),1)
            if adv_label[0] != curr_target[k]:
                img.imsave( 'images/'+args.method+'/'+args.norm+'/{}_{}_{}_{}_{}.png'.format(target_model_name,
                    adv_label, curr_target[k], eps, args.delta),
                    x_adv[k].reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)*255, cmap='gray')
                img_count += 1
            if img_count >= 10:
                return
    return

def wb_write_out(eps, white_box_error, wb_norm):
    if RANDOM is False:
        ofile = open('output_data/'+args.method+'_'+args.norm+'_based_classwise'+str(eps)+'_'+str(target_model_name)+'.txt', 'a')
        ofile.write('{}'.format(white_box_error))
        print('Fraction of targets achieved (white-box) for {}: {}'.format(target, white_box_error))
    else:
        ofile = open('output_data/wb_'+target_model_name+'_'+args.norm+'_attack.txt','a')
        ofile.write('{} {} {} \n'.format(eps, white_box_error, wb_norm))
        print('Fraction of targets achieved (white-box): {}'.format(white_box_error))

    return

def est_write_out(eps, success, avg_l2_perturb):
    if RANDOM is False:
        ofile = open('output_data/'+args.method+'_'+args.norm+'_classwise'+str(eps)+'_'+str(target_model_name)+'.txt', 'a')
        ofile.write(' {} \n'.format(success))
        print('Fraction of targets achieved (query-based) with {} for {}: {}'.format(target_model_name, target, success))
    else:
        ofile = open('output_data/'+args.method+'_'+args.norm+'_'+target_model_name+'_attack.txt','a')
        ofile.write('{} {} {} {}\n'.format(args.delta, eps, success, avg_l2_perturb))
        print('Fraction of targets achieved (query-based): {}'.format(success))
    ofile.close()
    return

def estimated_grad_attack(X_test, x, targets, prediction, eps, dim):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    for i in range(BATCH_EVAL_NUM):
        if i % 10 ==0:
            print('{}, {}'.format(i, eps))
        curr_sample = X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))

        curr_target = targets[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        curr_prediction = K.get_session().run([prediction], feed_dict={x: curr_sample, K.learning_phase(): 0})[0]

        p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

        if 'query_based' in args.method:
            grad_est = finite_diff_method(prediction, x, curr_sample, curr_target, dim)
        elif 'one_shot' in args.method:
            grad_est = one_shot_method(prediction, x, curr_sample, curr_target)
        # Getting gradient of the loss
        loss_grad = -1.0 * grad_est/p_t[:, None, None, None]
        # Getting signed gradient of loss
        if args.norm == 'linf':
            normed_loss_grad = np.sign(loss_grad)
        elif args.norm == 'l2':
            grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis = 1)
            indices = np.where(grad_norm != 0.0)
            normed_loss_grad = np.zeros_like(curr_sample)
            normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

        if '_un' in args.method:
            x_adv = np.clip(curr_sample + eps * normed_loss_grad, 0, 1)
        else:
            x_adv = np.clip(curr_sample - eps * normed_loss_grad, 0, 1)

        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv-curr_sample).reshape(BATCH_SIZE, dim), axis=1)
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = K.get_session().run([prediction], feed_dict={x: x_adv, K.learning_phase(): 0})[0]
        success += np.sum(np.argmax(adv_prediction,1) == curr_target)

        est_img_save(i, adv_prediction, curr_target, eps, x_adv)


    success = 100.0 * float(success)/(BATCH_SIZE*BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb/BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    return


def white_box_fgsm(prediction, target_model, x, logits, y, X_test, targets, targets_cat, eps, dim):

    #Get gradient from model
    grad = gen_grad(x, logits, y)

    # normalized gradient
    if args.norm == 'linf':
        normed_grad = K.sign(grad)
    elif args.norm == 'l2':
        normed_grad = K.l2_normalize(grad, axis = (1,2,3))

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if '_un' in args.method:
        adv_x_t = K.stop_gradient(x + scaled_grad)
    else:
        adv_x_t = K.stop_gradient(x - scaled_grad)

    adv_x_t = K.clip(adv_x_t, CLIP_MIN, CLIP_MAX)

    X_test_slice = X_test[:BATCH_SIZE*BATCH_EVAL_NUM]

    X_adv_t = batch_eval([x, y], [adv_x_t], [X_test_slice, targets_cat])[0]

    adv_pred_np = K.get_session().run([prediction], feed_dict={x: X_adv_t, K.learning_phase(): 0})[0]

    wb_img_save(adv_pred_np, targets, eps, X_adv_t)

    white_box_error = tf_test_error_rate(target_model, x, X_adv_t, targets_cat)
    if '_un' not in args.method:
        white_box_error = 100.0 - white_box_error

    wb_norm = np.mean(np.linalg.norm((X_adv_t-X_test_slice).reshape(BATCH_SIZE*BATCH_EVAL_NUM, dim), axis=1))
    print('Average white-box l2 perturbation: {}'.format(wb_norm))

    wb_write_out(eps, white_box_error, wb_norm)

    return


def finite_diff_method(prediction, x, curr_sample, curr_target, dim):
    grad_est = np.zeros((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    for j in range(dim):
        basis_vec = np.zeros((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
        row = int(j/FLAGS.IMAGE_COLS)
        col = int(j % FLAGS.IMAGE_COLS)
        basis_vec[:, row, col] = 1.
        x_plus_i = np.clip(curr_sample + args.delta * basis_vec, CLIP_MIN, CLIP_MAX)
        x_minus_i = np.clip(curr_sample - args.delta * basis_vec, CLIP_MIN, CLIP_MAX)
        pred_plus = K.get_session().run([prediction], feed_dict={x: x_plus_i,
                                                    K.learning_phase(): 0})[0]
        pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]
        pred_minus = K.get_session().run([prediction], feed_dict={x: x_minus_i,
                                                    K.learning_phase(): 0})[0]
        pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]
        single_grad_est = (pred_plus_t - pred_minus_t)/args.delta
        grad_est[:, row, col] = single_grad_est.reshape((BATCH_SIZE,1))

    return grad_est

def one_shot_method(prediction, x, curr_sample, curr_target):
    grad_est = np.zeros((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    DELTA = np.random.randint(2, size=(BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    np.place(DELTA, DELTA==0, -1)

    y_plus = np.clip(curr_sample + args.delta * DELTA, CLIP_MIN, CLIP_MAX)
    pred_plus = K.get_session().run([prediction], feed_dict={x: y_plus, K.learning_phase(): 0})[0]
    pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]

    y_minus = np.clip(curr_sample - args.delta * DELTA, CLIP_MIN, CLIP_MAX)
    pred_minus = K.get_session().run([prediction], feed_dict={x: y_minus, K.learning_phase(): 0})[0]
    pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]

    num_est = (pred_plus_t - pred_minus_t)
    grad_est = num_est[:, None, None, None]/(args.delta * DELTA)

    return grad_est


def main(target_model_name, target=None):
    np.random.seed(0)
    tf.set_random_seed(0)

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    dim = int(FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS)

    _, _, X_test, Y_test = data_mnist()
    print('Loaded data')

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
        targets = np.random.randint(10, size = BATCH_SIZE*BATCH_EVAL_NUM)
    targets_cat = np_utils.to_categorical(targets, FLAGS.NUM_CLASSES).astype(np.float32)

    if args.norm == 'linf':
        eps_list = list(np.linspace(0.0, 0.1, 5))
        eps_list.extend(np.linspace(0.2, 0.5, 7))
        # eps_list = [0.3]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 9.0, 28))
        # eps_list = [5.0]

    for eps in eps_list:
        white_box_fgsm(prediction, target_model, x, logits, y, X_test, targets,
                        targets_cat, eps, dim)

        estimated_grad_attack(X_test, x, targets, prediction, eps, dim)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--method", choices=['query_based', 'one_shot',
                        'query_based_un', 'one_shot_un'], default='query_based')
    parser.add_argument("--delta", type=float, default=0.01,
                        help="local perturbation")
    parser.add_argument("--norm", type=str, default='linf',
                            help="Norm to use for attack")
    args = parser.parse_args()

    target_model_name = basename(args.target_model)

    set_mnist_flags()

    if '_un' in args.method:
        RANDOM = True

    if RANDOM is False:
        ofile = open('output_data/'+args.method+'_classwise'+str(eps)+
                        '_'+str(target_model_name)+'.txt', 'a')
        ofile.write('{} \n'.format(args.delta))
        for i in range(FLAGS.NUM_CLASSES):
            main(args.target_model, i)
    elif RANDOM is True:
        main(args.target_model)