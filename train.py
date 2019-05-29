# -*-coding:utf-8-*-
import tensorflow as tf
from model import VDSR, EDSR
import datetime
from data_loader import DataLoader
import os

MODEL = 'VSER'
TRAIN_DIR = 'output/{}/model'.format(MODEL)        #保存的模型
LOG_DIR = 'output/{}/log'.format(MODEL)
SCALE = 4
BATCH_SIZE = 64
SHUFFLE_NUM = 20000
PREFETCH_NUM = 10000
MAX_TRAIN_STEP = 50000
LR_BOUNDS = [45000]
LR_VALS = [1e-4, 1e-5]
SAVE_PER_STEP = 2000
TRAIN_PNG_PATH = 'DIV2K/DIV2K_train_HR'
TRAIN_TFRECORD_PATH = 'DIV2K/tfrecords'
DATA_lOADER_MODE = 'RAW'   # 'TFRECORD' OR 'RAW'
DEVICE_MODE = 'GPU'
DEVICE_GPU_ID = '0'

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.paht.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if DEVICE_MODE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICE'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICE'] = DEVICE_GPU_ID

# 模型恢复
def restore_session_from_checkpoint(sess, saver):
    #tf.train.latest.checkpoint()自动获得最后一次保存的模型路径
    checkpoint = tf.train.latest_checkpoint(TRAIN_DIR)
    if checkpoint:
        saver.restore(sess, checkpoint)
        return True
    else:
        return False

if MODEL == 'VDSR':
    model = VDSR(scale = SCALE)
else:
    model = EDSR(scale = SCALE)

data_loader = DataLoader(data_dir=TRAIN_PNG_PATH,
                         batch_size = PATCH_SIZE,
                         shuffle_num = SHUFFLE_NUM,
                         prefetch_num = PREFETCH_NUM,
                         scale = SCALE)

if DATA_LOADER_MODE == 'TFRECORD':
    if len(os.listdir(TRAIN_TFRECORD_PATH)) == 0:
        data_loader.gen_tfrecords(TRAIN_TFRECORD_PATH)
    lrs, bics, gts = data_loader.read_tfrecords(TRAIN_TFRECORD_PATH)
else:
    lrs, bics, gts = data_loader.read_pngs()

res = model(lrs, bics)
with tf.name_scope('train'):
    global_step = tf.Variable(0, trainale=False, name='global_step')
    mse_loss = tf.reduce_sum(tf.square(res - gts)) / BATCH_SIZE
    reg_loss = tf.losses.get_regularization_loss()
    loss = mse_loss + reg_loss

    #手调学习率
    learning_rate = tf.train.piecewise_constant(global_step, LR_BOUNDS, LR_VALS)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)

with tf.name_scope('summaries'):
    tf.summary.scalar('learning_rate', learining_rate)
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('loss', loss)

    tf.summary.image('lr', lrs, 1)
    tf.summary.image('bic', model.bic, 1)
    tf.summary.image('out', tf.clip_by_value(res, 0, 1), 1)
    #tf.clip_by_value(V, min, max)截取V使其处于min与max之间
    tf.summary.image('gt', gts, 1)

    summary_op = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=500)
config = tf.ConfigProto()
config.gpu_option.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

restore_session_from_checkpoint(sess, saver)

start_time = data_time.datetime.now()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

while True:
    _, loss_value, step = sess.run([train_op, loss, global_step])
    if step % 20 == 0:
        end_time = datetime.datetime.now()
        print('[{}] Step:{}, loss:{}'.format(
            end_time - start_time, step, loss_value
            ))
        summary_value = sess.run(summary_op)
        writer.add_summary(summary_value, step)
        start_time = end_time
    if step % SAVER_PER_STEP == 0:
        saver.save(sess, os.path.join(TRAIN_DIR, 'checkpoint.ckpt'), global_step = step)
    if step >= MAX_TRAIN_STEP:
        print('Done train')
        break
    
    

