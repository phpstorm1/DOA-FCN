import os.path
import sys
import glob
import json

import numpy as np
import tensorflow as tf
import random
import math
from scipy.io import wavfile
from matplotlib import pyplot as plt
from functools import reduce

import model
import input_data
import eval


def main(_):

    json_dir = './config.json'
    with open(json_dir) as config_json:
        config = json.load(config_json)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    phase_specs = tf.placeholder(tf.float32, shape=[None, config['context_window_width'], 129, 4], name='phase_specs')
    ground_truth_doa_label = tf.placeholder(tf.float32,
                                            shape=[None, config['dim_direction_label']],
                                            name='ground_truth_input')

    model_settings = model.create_model_settings(
        dim_direction_label=config['dim_direction_label'],
        sample_rate=config["sample_rate"],
        win_len=config['win_len'],
        win_shift=config['win_shift'],
        nDFT=config['nDFT'],
        context_window_width=config['context_window_width'])

    with tf.variable_scope('CNN'):
        predict_logits = model.doa_cnn(phase_specs=phase_specs, model_settings=model_settings, is_training=True)
    CNN_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')

    print('-' * 80)
    print('CNN vars')
    nparams = 0
    for v in CNN_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=ground_truth_doa_label, logits=predict_logits)

    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(ground_truth_doa_label, 1), predictions=tf.argmax(predict_logits, 1))
    pc_acc, pc_acc_op = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(ground_truth_doa_label, 1),
                                                           predictions=tf.argmax(predict_logits, 1),
                                                           num_classes=config['dim_direction_label'])
    tf.summary.scalar('cross_entropy', mean_cross_entropy)
    tf.summary.scalar('class_accuracy', acc_op)
    tf.summary.histogram('per_class_accuracy', pc_acc_op)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('train'), tf.control_dependencies(extra_update_ops):
        adam = tf.train.AdamOptimizer(config['Adam_learn_rate'])
        # rms = tf.train.RMSPropOptimizer(config['Adam_learn_rate'])
        train_step = adam.minimize(cross_entropy, global_step=global_step, var_list=CNN_vars)
        # train_step = rms.minimize(cross_entropy, global_step=global_step, var_list=CNN_vars)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config['summaries_dir'], sess.graph)

    # tf.global_variables_initializer().run()
    start_step = 1

    tf.global_variables_initializer().run()
    init_local_variable = tf.local_variables_initializer()
    init_local_variable.run()

    if config['start_checkpoint']:
        model.load_variables_from_checkpoint(sess, config['start_checkpoint'])
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, config['train_dir'], 'model.pbtxt')

    # find training files
    training_data_dir = config['training_data_dir']
    training_file_list = glob.glob(os.path.join(training_data_dir, "*.wav"))

    training_speech_dir = config['training_speech_dir']
    training_speech_list = glob.glob(os.path.join(training_speech_dir,"**", "*.wav"), recursive=True)

    rir_data_dir = config['rir_data_dir']
    rir_file_list = glob.glob(os.path.join(rir_data_dir, "*.wav"))

    reverb = config['reverb']
    reverb.sort()

    room_index = config['room_idx']
    room_index.sort()

    # find testing files
    testing_file_list = glob.glob(os.path.join(config['testing_data_dir'], "*.wav"))

    if not len(training_file_list):
        Exception("No wav files found at " + training_data_dir)
    if not len(rir_file_list):
        Exception("No wav files found at " + rir_data_dir)

    tf.logging.info("Number of training wav files: %d", len(training_file_list))

    # Training loop.
    how_many_training_steps = config['how_many_training_steps']
    for training_step in range(start_step, int(how_many_training_steps + 1)):

        training_file_idx = random.randint(0, len(training_file_list)-1)
        # rir_idx = random.randint(0, len(rir_file_list)-1)
        # rir_idx = training_step % (1+config['direction_range'][1])
        rir_idx = training_step % len(rir_file_list)

        training_filename = training_file_list[training_file_idx]
        rir_filename = rir_file_list[rir_idx]

        reverb_percent = int(rir_filename.split('reverb_')[1].split('Percent_')[0])
        if reverb_percent == 75 or reverb_percent == 65:
            if random.randint(0, 1):
                speech_file_idx = random.randint(0, len(training_speech_list)-1)
                training_filename = training_speech_list[speech_file_idx]

        reverb_wav, training_phase_specs = input_data.get_input_specs(training_filename,
                                                                      rir_filename,
                                                                      config['win_len'],
                                                                      config['win_shift'],
                                                                      config['nDFT'],
                                                                      config['context_window_width'],
                                                                      config['max_wav_length'])
        num_frames = training_phase_specs.shape[0]

        training_doa_label = input_data.get_direction_label(rir_filename, config['dim_direction_label'], config['direction_range'], num_frames)

        training_summary, training_cross_entropy, _, _ = sess.run(
            [merged_summaries, mean_cross_entropy, train_step, init_local_variable],
            feed_dict={phase_specs:training_phase_specs, ground_truth_doa_label: training_doa_label})

        print("Step: ", training_step, " "*10,
              "cross entropy: ", format(training_cross_entropy, '.5f'), " "*10,
              "rir: ", format(reverb_percent, '2.0f'), " "*10,
              "training file: ", os.path.basename(training_filename))
        train_writer.add_summary(training_summary, training_step)

        # Save the model checkpoint periodically.
        if training_step % config['save_step_interval'] == 0 or training_step == how_many_training_steps:
            checkpoint_path = os.path.join(config['train_dir'], 'model.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    set_size = len(testing_file_list)
    tf.logging.info('testing set size=%d', set_size)

    doa_per_reverb = int(max(config['direction_range']) - min(config['direction_range']) + 1)
    test_acc = np.zeros([len(testing_file_list), doa_per_reverb, len(reverb), len(room_index)])
    test_adj_acc = np.zeros([len(testing_file_list), doa_per_reverb, len(reverb), len(room_index)])
    test_frame_acc = np.zeros([len(testing_file_list), doa_per_reverb, len(reverb), len(room_index)])
    test_adj_frame_acc = np.zeros([len(testing_file_list), doa_per_reverb, len(reverb), len(room_index)])
    for testing_file_idx, testing_file in enumerate(testing_file_list):
        print("testing file:", os.path.basename(testing_file))
        for rir_file_idx, rir_file in enumerate(rir_file_list):
            rir_filename = os.path.basename(rir_file)
            degree = int(rir_filename.split('angle_')[1].split('deg_')[0])
            reverb_percent = int(rir_filename.split('reverb_')[1].split('Percent_')[0])
            room_num = int(rir_filename.split('_ROOM')[1].split('.wav')[0])

            if reverb_percent not in reverb or room_num not in room_index:
                continue
            reverb_idx = reverb.index(reverb_percent)
            room_idx = room_index.index(room_num)

            reverb_wav, testing_phase_specs = input_data.get_input_specs(testing_file,
                                                                         rir_file,
                                                                         config['win_len'],
                                                                         config['win_shift'],
                                                                         config['nDFT'],
                                                                         config['context_window_width'],
                                                                         config['max_wav_length'])
            num_frames = testing_phase_specs.shape[0]

            testing_doa_label = input_data.get_direction_label(rir_file,
                                                               config['dim_direction_label'],
                                                               config['direction_range'], num_frames)

            logits, class_acc, _ = sess.run(
                [predict_logits, acc_op, init_local_variable],
                feed_dict={phase_specs: testing_phase_specs, ground_truth_doa_label: testing_doa_label})

            adjacent_class = 2
            how_many_previous_frame = 15
            testing_predict = eval.get_label_from_logits(logits)
            adj_acc = eval.eval_adjacent_accuracy(testing_predict, testing_doa_label, adjacent_class)
            frame_acc = eval.eval_frame_accuracy(testing_predict, testing_doa_label, how_many_previous_frame)
            adj_frame_acc =eval.eval_joint_deg_frame(testing_predict, testing_doa_label, adjacent_class, how_many_previous_frame)
            test_acc[testing_file_idx, degree, reverb_idx, room_idx] = class_acc
            test_adj_acc[testing_file_idx, degree, reverb_idx, room_idx] = adj_acc
            test_frame_acc[testing_file_idx, degree, reverb_idx, room_idx] = frame_acc
            test_adj_frame_acc[testing_file_idx, degree, reverb_idx, room_idx] = adj_frame_acc

            print("degree:", format(degree, '5.1f'), " "*6,
                  "reverb:", format(reverb_percent, '5.0f'), " "*6,
                  "room:", format(room_num, '5.0f'), " "*6,
                  "acc:", format(class_acc, '5.5f'), " "*6,
                  "adj acc:", format(adj_acc, '5.5f'), " "*6,
                  "frame acc:", format(frame_acc, '5.5f'), " "*6,
                  "adj frame acc:", format(adj_frame_acc, '5.5f'))
    print("overall acc:", np.mean(test_acc))
    print("overall adj_acc:", np.mean(test_adj_acc))
    print("overall frame_acc:", np.mean(test_frame_acc))
    print("overall adj frame acc:", np.mean(test_adj_frame_acc))
    print("-"*30)
    print("Degree accuracy")
    print(format("deg", '10.10s'),
          format("acc", '10.10s'),
          format("deg acc", '15.10s'),
          format("frame acc", '15.10s'),
          format("deg frame acc", "15.10s"))
    for i in range(doa_per_reverb):
        print(format(i, '.1f'), " "*5,
              format(np.mean(test_acc[:, i, :]), '.4f'), " "*6,
              format(np.mean(test_adj_acc[:, i, :]), '.4f'), " "*6,
              format(np.mean(test_frame_acc[:, i, :]), '.4f'), " "*6,
              format(np.mean(test_adj_frame_acc[:, i, :]), '.4f'))

    deg_idx = range(doa_per_reverb)

    print("-" * 30)

    for room in range(len(room_index)):
        for i in range(len(reverb)):
            print("reverb: ", reverb[i])
            print("room: ", room_index[room])
            print("acc:", np.mean(test_acc[:, :, i, room]))
            print("adj_acc:", np.mean(test_adj_acc[:, :, i, room]))
            print("frame_acc:", np.mean(test_frame_acc[:, :, i, room]))
            print("adj frame acc:", np.mean(test_adj_frame_acc[:, :, i, room]))

            for j in range(doa_per_reverb):
                print(format(j, '.1f'), " " * 5,
                      format(np.mean(test_acc[:, j, i, room]), '.4f'), " " * 6,
                      format(np.mean(test_adj_acc[:, j, i, room]), '.4f'), " " * 6,
                      format(np.mean(test_frame_acc[:, j, i, room]), '.4f'), " " * 6,
                      format(np.mean(test_adj_frame_acc[:, j, i, room]), '.4f'))

            print("-" * 30)

    for room in range(len(room_index)):
        for i in range(len(reverb)):
            plt.figure(i)
            plt.plot(deg_idx, np.mean(test_adj_acc[:, :, i, room], axis=0), '.')
            plt.yscale('linear')
            plt.xlabel('degree')
            plt.ylabel('accuracy')
            plt.title('room ' + str(room_index[room]) + ', reverb ' + str(reverb[i]) + ' percent')
            plt.grid(True)
            filename = 'room_' + str(room_index[room]) + '_reverb_' + str(reverb[i]) + '_acc.png'
            fig_save_path = os.path.join('./figures', 'v4',os.path.basename(config['testing_data_dir']))
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            filename = os.path.join(fig_save_path, filename)
            plt.savefig(filename)
            plt.show()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
