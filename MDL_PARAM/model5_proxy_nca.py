# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:37:15 2016

@author: mingzhang

PVA NET
"""

"""
Contains the definition of the Inception Resnet V2 architecture.		
As described in http://arxiv.org/abs/1602.07261.		
Inception-v4, Inception-ResNet and the Impact of Residual Connections		
on Learning		
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi		

note: image size must be:(299, 299)
datashape = (1, 3, 299, 299)
"""
import sys
sys.path.insert(0, 'distribution/')


import numpy as np
import mxnet as mx
import time
import cPickle
import custom_layers
import logging


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True, namepre='', args=None):
  if args is None:
    weight = mx.sym.Variable(namepre+'_weight')
    bias = mx.sym.Variable(namepre+'_bias')
    gamma = mx.sym.Variable(namepre+'_gamma')
    beta = mx.sym.Variable(namepre+'_beta')
    args = {'weight':weight, 'bias':bias}
  else:
    weight = args['weight']
    bias = args['bias']
    gamma = args['gamma']
    beta = args['beta']
  
  conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, weight=weight, bias=bias, name=namepre+'_conv')
  bn = mx.symbol.BatchNorm(data=conv, gamma=gamma, beta=beta, name=namepre+'_bn')
  act = bn
  if with_act:
      act = mx.symbol.Activation(data=bn, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def stem(data, namepre='', args=None):
  if args is None:
    args = {'conv1a_3_3':None, 'conv2a_3_3':None, 'conv2b_3_3':None, 'conv3b_1_1':None, 'conv4a_3_3':None}
  conv1a_3_3, args['conv1a_3_3'] = ConvFactory(data=data, num_filter=32,
                           kernel=(3, 3), stride=(2, 2), namepre=namepre+'_conv1a_3_3', args=args['conv1a_3_3'])
  conv2a_3_3, args['conv2a_3_3'] = ConvFactory(conv1a_3_3, 32, (3, 3), namepre=namepre+'_conv2a_3_3', args=args['conv2a_3_3'])
  conv2b_3_3, args['conv2b_3_3'] = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1), namepre=namepre+'_conv2b_3_3', args=args['conv2b_3_3'])
  maxpool3a_3_3 = mx.symbol.Pooling(
      data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_maxpool3a_3_3')
  conv3b_1_1, args['conv3b_1_1'] = ConvFactory(maxpool3a_3_3, 80, (1, 1), namepre=namepre+'_conv3b_1_1', args=args['conv3b_1_1'])
  conv4a_3_3, args['conv4a_3_3'] = ConvFactory(conv3b_1_1, 192, (3, 3), namepre=namepre+'_conv4a_3_3', args=args['conv4a_3_3'])

  return conv4a_3_3, args 


def reductionA(conv4a_3_3, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv2_0':None, 'tower_conv2_1':None, 'tower_conv2_2':None, 'tower_conv3_1':None}
  maxpool5a_3_3 = mx.symbol.Pooling(
      data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_maxpool5a_3_3')

  tower_conv, args['tower_conv'] = ConvFactory(maxpool5a_3_3, 96, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(maxpool5a_3_3, 48, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])

  tower_conv2_0, args['tower_conv2_0'] = ConvFactory(maxpool5a_3_3, 64, (1, 1), namepre=namepre+'_tower_conv2_0', args=args['tower_conv2_0'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])

  tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
      3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg', name=namepre+'_tower_pool3_0')
  tower_conv3_1, args['tower_conv3_1'] = ConvFactory(tower_pool3_0, 64, (1, 1), namepre=namepre+'_tower_conv3_1', args=args['tower_conv3_1'])
  tower_5b_out = mx.symbol.Concat(
      *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
  return tower_5b_out, args 


def reductionB(net, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_pool = mx.symbol.Pooling(net, kernel=(
      3, 3), stride=(2, 2), pool_type='max', name=namepre+'_tower_pool')
  net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])

  return net, args


def reductionC(net, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv0_1':None, 'tower_conv1':None, 'tower_conv1_1':None, 'tower_conv2':None, 'tower_conv2_1':None, 'tower_conv2_2':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv0_1, args['tower_conv0_1'] = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv0_1', args=args['tower_conv0_1'])
  tower_conv1, args['tower_conv1'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv1', args=args['tower_conv1'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv2, args['tower_conv2'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv2', args=args['tower_conv2'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])
  tower_pool = mx.symbol.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_tower_pool')
  net = mx.symbol.Concat(*[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
  return net, args


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv2_0':None, 'tower_conv2_1':None, 'tower_conv2_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv2_0, args['tower_conv2_0'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv2_0', args=args['tower_conv2_0'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])

  net += scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 129, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])
  net += scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])
  net += scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def repeat(inputs, repetitions, layer, *ltargs, **kwargs):
  outputs = inputs
  namepre = kwargs['namepre']
  args = kwargs['args']
  if args is None:
    args = {}
    for i in xrange(repetitions):
      argname='repeat_'+str(i)
      args[argname] = None
  for i in range(repetitions):
    kwargs['namepre'] = namepre+'_'+str(i)
    argname='repeat_'+str(i)
    kwargs['args'] = args[argname]
#    print ltargs
#    print kwargs
    outputs, args[argname] = layer(outputs, *ltargs, **kwargs)

  return outputs, args


def create_inception_resnet_v2_attr(data, namepre='', args=None):
  if args is None:
    args = {'stem':None, 'reductionA':None, 'repeat_block35':None, 'reductionB':None, 
            'repeat_block17':None, 'reductionC':None, 'repeat_block8':None, 
            'final_block8':None, 'final_conv':None, 'finalfc':None}

  stem_net, args['stem']= stem(data, namepre=namepre+'_stem', args=args['stem'])

  reduceA, args['reductionA'] = reductionA(stem_net, namepre=namepre+'_reductionA', args=args['reductionA'])

  repeat_block35, args['repeat_block35'] = repeat(reduceA, 2, block35, scale=0.17, input_num_channels=320, namepre=namepre+'_repeat_block35', args=args['repeat_block35'])


  reduceB, args['reductionB'] = reductionB(repeat_block35, namepre=namepre+'_reductionB', args=args['reductionB'])

  repeat_block17, args['repeat_block17'] = repeat(reduceB, 4, block17, scale=0.1, input_num_channels=1088, namepre=namepre+'_repeat_block17', args=args['repeat_block17'])

  reduceC, args['reductionC'] = reductionC(repeat_block17, namepre=namepre+'_reductionC', args=args['reductionC'])

  repeat_block8, args['repeat_block8'] = repeat(reduceC, 2, block8, scale=0.2, input_num_channels=2080, namepre=namepre+'_repeat_block8', args=args['repeat_block8'])
  final_block8, args['final_block8'] = block8(repeat_block8, with_act=False, input_num_channels=2080, namepre=namepre+'_final_block8', args=args['final_block8'])

  final_conv, args['final_conv'] = ConvFactory(final_block8, 1536, (1, 1), namepre=namepre+'_final_conv', args=args['final_conv'])
  final_pool = mx.symbol.Pooling(final_conv, kernel=(8, 8), global_pool=True, pool_type='avg', name=namepre+'_final_pool')
  final_flatten = mx.symbol.Flatten(final_pool, name=namepre+'_final_flatten')

  drop1 = mx.sym.Dropout(data=final_flatten, p=0.5, name=namepre+'_dropout1')

  if args['finalfc'] is None:
    args['finalfc'] = {}
    args['finalfc']['weight'] = mx.sym.Variable(namepre+'_fc1_weight')
    args['finalfc']['bias'] = mx.sym.Variable(namepre+'_fc1_bias')
    
  reid_fc1 = mx.sym.FullyConnected(data=drop1, num_hidden=43, name=namepre+"_fc1", 
                                   weight=args['finalfc']['weight'], bias=args['finalfc']['bias']) 
  reid_act = mx.sym.Activation(data=reid_fc1, act_type='sigmoid', name=namepre+'_fc1_sigmod')
#  reid_act = mx.sym.Activation(data=reid_fc1, act_type='sigmod', name=namepre+'_fc1_relu')
#  net =  reid_act
  net = reid_act
#  net = final_flatten

  return net, args
def create_inception_resnet_v2(data, namepre='', args=None):
  if args is None:
    args = {'stem':None, 'reductionA':None, 'repeat_block35':None, 'reductionB':None, 
            'repeat_block17':None, 'reductionC':None, 'repeat_block8':None, 
            'final_block8':None, 'final_conv':None, 'finalfc':None}

  stem_net, args['stem']= stem(data, namepre=namepre+'_stem', args=args['stem'])

  reduceA, args['reductionA'] = reductionA(stem_net, namepre=namepre+'_reductionA', args=args['reductionA'])

  repeat_block35, args['repeat_block35'] = repeat(reduceA, 2, block35, scale=0.17, input_num_channels=320, namepre=namepre+'_repeat_block35', args=args['repeat_block35'])


  reduceB, args['reductionB'] = reductionB(repeat_block35, namepre=namepre+'_reductionB', args=args['reductionB'])

  repeat_block17, args['repeat_block17'] = repeat(reduceB, 4, block17, scale=0.1, input_num_channels=1088, namepre=namepre+'_repeat_block17', args=args['repeat_block17'])

  reduceC, args['reductionC'] = reductionC(repeat_block17, namepre=namepre+'_reductionC', args=args['reductionC'])

  repeat_block8, args['repeat_block8'] = repeat(reduceC, 2, block8, scale=0.2, input_num_channels=2080, namepre=namepre+'_repeat_block8', args=args['repeat_block8'])
  final_block8, args['final_block8'] = block8(repeat_block8, with_act=False, input_num_channels=2080, namepre=namepre+'_final_block8', args=args['final_block8'])

  final_conv, args['final_conv'] = ConvFactory(final_block8, 1536, (1, 1), namepre=namepre+'_final_conv', args=args['final_conv'])
  final_pool = mx.symbol.Pooling(final_conv, kernel=(8, 8), global_pool=True, pool_type='avg', name=namepre+'_final_pool')
  final_flatten = mx.symbol.Flatten(final_pool, name=namepre+'_final_flatten')

  drop1 = mx.sym.Dropout(data=final_flatten, p=0.5, name=namepre+'_dropout1')

  if args['finalfc'] is None:
    args['finalfc'] = {}
    args['finalfc']['weight'] = mx.sym.Variable(namepre+'_fc1_weight')
    args['finalfc']['bias'] = mx.sym.Variable(namepre+'_fc1_bias')
    
  reid_fc1 = mx.sym.FullyConnected(data=drop1, num_hidden=128, name=namepre+"_fc1", 
                                   weight=args['finalfc']['weight'], bias=args['finalfc']['bias']) 
#  reid_act = mx.sym.Activation(data=reid_fc1, act_type='tanh', name=namepre+'_fc1_relu')

  net = reid_fc1
#  net = final_flatten

  return net, args


def create_reid4_net(batch_size, proxy_num):
  data0 = mx.sym.Variable('data')
  proxy_yM = mx.sym.Variable('proxy_yM')
  proxy_Z = mx.sym.Variable(name='proxy_Z_weight', 
                       shape=(proxy_num, 128), dtype=np.float32)
  proxy_ZM = mx.sym.Variable('proxy_ZM')
  args_all = None
  reid_feature, args_all = create_inception_resnet_v2(data0, namepre='part1', args=args_all)

  features = mx.sym.SliceChannel(reid_feature, axis=0, num_outputs=batch_size, name='features_slice')
  proxy_yMs = mx.sym.SliceChannel(proxy_yM, axis=0, num_outputs=batch_size, name='proxy_yM_slice')
  proxy_ZMs = mx.sym.SliceChannel(proxy_ZM, axis=0, num_outputs=batch_size, name='proxy_ZM_slice')
  proxy_ncas = []
  min_value = 10**-36
#  norm_value = (84**0.5)/2
#  norm_value = np.log(2.0**128)/4
  useSquare = True
  useHing = False
  logging.info('useSquare:' + str(useSquare) + ', useHing:' + str(useHing))
  if useSquare:
    norm_value = np.log((2.0**126)/proxy_num)/2 #2.0**126 is near the maxinum value of float32
  else:
    norm_value = np.log((2.0**126)/proxy_num)/(2*np.sqrt(128)) #2.0**126 is near the maxinum value of float32, 128 is the featnum
  logging.info('norm_value:' + str(norm_value))
  
  #norm
  if True:
    proxy_Znorm = mx.sym.sum_axis(proxy_Z**2, axis=1)
#    print proxy_Znorm.name
    proxy_Znorm = mx.sym.sqrt(proxy_Znorm) + min_value 
#    print proxy_Znorm.name
    proxy_Znorm = mx.sym.Reshape(proxy_Znorm, shape=(-2, 1))
    proxy_Z = mx.sym.broadcast_div(proxy_Z, proxy_Znorm) * norm_value
#    print proxy_Z.name

  for bi in xrange(batch_size):
    one_feat = features[bi]
    
    #norm 
    if True:
      one_feat_norm = mx.sym.sqrt(mx.sym.sum(one_feat**2)) + min_value
      one_feat_norm = mx.sym.Reshape(one_feat_norm, shape=(-2, 1))
      one_feat = mx.sym.broadcast_div(one_feat, one_feat_norm) * norm_value
    
    one_proxy_yM = proxy_yMs[bi]
    one_proxy_ZM = proxy_ZMs[bi]

    tzM = mx.sym.Reshape(one_proxy_ZM, shape=(-1,))
    z = mx.sym.broadcast_minus(one_feat, proxy_Z)
    if useSquare:
      z = mx.sym.square(z)
      z = mx.sym.sum_axis(z, axis=1)
      z = mx.sym.sqrt(z)
    else:
      z = mx.sym.abs(z)
      z = mx.sym.sum_axis(z, axis=1)
    z = -z

    tyM = mx.sym.Reshape(one_proxy_yM, shape=(-1, 1))
    one_proxy_y = mx.sym.broadcast_mul(tyM, proxy_Z)
    one_proxy_y = mx.sym.sum_axis(one_proxy_y, axis=0)
#    print 'one_proxy_y:', one_proxy_y.name
    one_feat = mx.sym.Reshape(one_feat, shape=(-1,))
    y = one_feat - one_proxy_y
    if useSquare:
      y = mx.sym.square(y)
      y = mx.sym.sum(y)
      y = mx.sym.sqrt(y)
    else:
      y = mx.sym.abs(y)
      y = mx.sym.sum(y)
    y = -y

#    print 'z:', z.name, 'y:', y.name
    z_y = mx.sym.broadcast_minus(z, y)
#    print z_y.name
    z_y = mx.sym.exp(z_y) * tzM
    z_y = mx.sym.sum(z_y)
    z_y = z_y + min_value
    one_proxy_nca = mx.sym.log(z_y)
    
    proxy_ncas.append(one_proxy_nca)

  proxy_nca = mx.sym.Concat(*proxy_ncas, dim=0)
  if useHing:
    reid_net = mx.sym.maximum(-1.0, proxy_nca)
  else:
   # reid_net = proxy_nca
    reid_net = mx.sym.maximum(-87.0, proxy_nca) #avoid the nan, exp(-87) is near the mininum of float
  reid_net = mx.sym.MakeLoss(reid_net, name='proxy_nca_loss')

#  print args_all
  return reid_net
def create_attr_net(batch_size, attr_num):
    data = mx.sym.Variable('data')
    cross_label = mx.sym.Variable('label')
    args_all = None
    attr_feature, args_all = create_inception_resnet_v2_attr(data, namepre = 'part1', args =  args_all)
#    attr_feature = mx.sym.FullyConnected(data=attr_feature, num_hidden=attr_num, name = 'fc_cls')
    out = (attr_feature-cross_label)**2
    out = mx.sym.sum(out)
    attr_net = mx.sym.MakeLoss(out)
    """ 
    features = mx.sym.SliceChannel(attr_feature, axis = 0, num_outputs =  batch_size, name = 'features_slice')
    
    cross_labels = mx.sym.SliceChannel(cross_label, axis = 0, num_outputs = batch_size, name = 'labels_slice')
    cross_loss = []
    for bi in xrange(batch_size):
        one_feat = features[bi]
        one_cross_label = cross_labels[bi]
        
        out = one_cross_label*mx.sym.log(one_feat + 1e-6) + (1.0 - one_cross_label)*mx.sym.log(1+(1e-6) - one_feat)
        out = mx.sym.sum(out)
        cross_loss.append(out)
    cross_loss = mx.sym.Concat(*cross_loss, dim = 0)
    attr_net = mx.sym.MakeLoss(cross_loss)
    """ 
    return  attr_net
        #if True:
         #   one_feat_norm = mx.sym.sqrt(mx.sym(onefeat**2))
         #   one_feat_norm = mx.sym.Reshape(one_feat_norm, shape = (-2, 1))
          #  one_feat = mx.sym.broadcast_div(one_feat, one_feat_norm)
       # out1 = mx.sym.broadcast_mul(one_cross_label, mx.sym.log(one_feat)) 
       # out2 =  
#        out_one =  one_cross_label*mx.sym.log(one_feat+1e-6)  + (1.0-one_cross_label)*(1.0-mx.sym.log(one_feat)
#        out_one = mx.sym.sum_axis(out_one, axis = 1)
#        cross_loss.append(out_one)
#    cross_loss = mx.sym.Concat(*cross_loss, dim = 0)

#    attr_net = mx.sym.MakeLoss(attr_net, name = 'cross_loss')
#    if True:
#      attr_net_exec =  attr_net.simple_bind(ctx = ctxdev)
def CreateModel_Cross_Entropy(ctx, batch_size, clsnum):
    print "creating network crossentropy..."
    attr_net = create_attr_net(batch_size, clsnum)
    return attr_net
def CreateModel_Color2(ctx, batch_size, proxy_num, imagesize):
  print 'creating network model2_proxy_nca...'
  reid_net = create_reid4_net(batch_size, proxy_num)

  return reid_net 


def create_reid_nsoftmax_net(batch_size, proxy_num):
  data0 = mx.sym.Variable('data')
  proxy_yM = mx.sym.Variable('proxy_yM')
  proxy_Z = mx.sym.Variable(name='proxy_Z_weight',
                                 shape=(proxy_num, 128), dtype=np.float32)
  args_all = None
  feat_final, args_all = create_inception_resnet_v2(data0, namepre='part1', args=args_all)
#
  proxy_ncas = []
  margin_ncas = []
  dists = []
  oth_dists = []
  min_value =10**-36
  norm_value = 32
  norm_value = (norm_value)**0.5
  logging.info('norm_value:%f, min_value:%e', norm_value, min_value)

  #norm
  znorm_loss = None
  if norm_value>0:
#    proxy_Z = mx.sym.L2Normalization(proxy_Z) * norm_value
#    feat_final = mx.sym.L2Normalization(feat_final) * norm_value
    proxy_Znorm = mx.sym.sum_axis(proxy_Z**2, axis=1)
    proxy_Znorm = mx.sym.sqrt(proxy_Znorm) + min_value

    #znorm_loss = mx.sym.square(proxy_Znorm - norm_value)
    znorm_loss = mx.sym.abs(proxy_Znorm - 1.0)
    znorm_loss = mx.sym.sum(znorm_loss)#/proxy_num
    znorm_loss = mx.sym.MakeLoss(znorm_loss)

    proxy_Znorm = mx.sym.Reshape(proxy_Znorm, shape=(-2, 1))
    proxy_Z = mx.sym.broadcast_div(proxy_Z, proxy_Znorm) * norm_value

    feat_finalnorm = mx.sym.sum_axis(feat_final**2, axis=1)
    feat_finalnorm = mx.sym.sqrt(feat_finalnorm) + min_value
    feat_finalnorm = mx.sym.Reshape(feat_finalnorm, shape=(-2, 1))
    feat_final = mx.sym.broadcast_div(feat_final, feat_finalnorm) * norm_value

  softlayer = mx.sym.FullyConnected(data=feat_final, weight=proxy_Z, num_hidden=proxy_num, no_bias=True)

  proxy_yMs = mx.sym.SliceChannel(proxy_yM, axis=0, num_outputs=batch_size, squeeze_axis=True, name='proxy_yM_slice')

  softmaxact_o = mx.sym.SoftmaxActivation(softlayer, name='softmaxact_o')

  softmaxacts_o = mx.sym.SliceChannel(softmaxact_o, axis=0, num_outputs=batch_size, name='softmaxact_slice_o')
  costvals_o = []
  for bi in xrange(batch_size):
    oneact = softmaxacts_o[bi]
    oney = proxy_yMs[bi]
    oneact = mx.sym.Reshape(oneact, shape=(-1, 1))
    onecost = mx.sym.Take(data=oneact, index=oney) + min_value
    costvals_o.append(onecost)

  costall_o = mx.sym.Concat(*costvals_o, dim=0)
  costall_o = -mx.sym.log(costall_o)
  reid_net = mx.sym.MakeLoss(costall_o)
  if znorm_loss is not None:
    reid_net = mx.sym.Group([reid_net, znorm_loss])

  return reid_net


def CreateModel_Color_NSoftmax(ctx, batch_size, proxy_num, imagesize):
  print 'creating network normalized softmax network...'

  reid_net = create_reid_nsoftmax_net(batch_size, proxy_num)

  return reid_net 



def draw_inception_renet_v2():
  featdim = 128
  proxy_num = 1000
  batch_size = 4
  args_all = None
  data0 = mx.sym.Variable('data0')
  reid_net, args_all = create_inception_resnet_v2(data0, namepre='part1', args=args_all)
  #darw net
  datashape = (batch_size, 3, 200, 80)
  net = reid_net.simple_bind(ctx=mx.gpu(0), data0=datashape)
  print net.output_dict
  #graph = mx.visualization.plot_network(reid_net, shape={'data':datashape, 'proxy_y':yshape, 'proxy_Z':Zshape, 'proxy_M':Mshape})
  #graph = mx.visualization.plot_network(reid_net)
  #graph.render('inception_renet_v2_proxy_nca') 


def CreateModel_Color_Split_test():
   data1 = mx.sym.Variable('part1_data')
   args_all = None
   reid_feature, args_all = create_inception_resnet_v2(data1, namepre='part1', args=args_all)
   if True:
     feature_norm = mx.sym.sqrt(mx.sym.sum_axis(reid_feature**2, axis=1))
     feature_norm = mx.sym.sqrt(mx.sym.sum_axis(reid_feature**2))
     feature_norm = mx.sym.Reshape(feature_norm, shape=(-2, 1))
     reid_feature = mx.sym.broadcast_div(reid_feature, feature_norm) 
#     reid_feature =  feature_norm
#     reid_feature =  reid_feature
   feature1 = mx.sym.Variable('feature1_data')
   feature2 = mx.sym.Variable('feature2_data')
#   absdiff = mx.sym.sum(mx.sym.abs(feature1-feature2), axis=1)
   absdiff = mx.sym.abs(feature1-feature2)
   return reid_feature, absdiff 


def CreateModel_Color_Split_test2(batch_size=1, featdim=128):
   data1 = mx.sym.Variable('part1_data')
   args_all = None
   reid_feature, args_all = create_inception_resnet_v2(data1, namepre='part1', args=args_all)
   if True:
     feature_norm = mx.sym.sqrt(mx.sym.sum_axis(reid_feature**2, axis=1))
     feature_norm = mx.sym.Reshape(feature_norm, shape=(-2, 1))
     reid_feature = mx.sym.broadcast_div(reid_feature, feature_norm) * 40
   print '-------------------'
   feature1 = mx.sym.Variable('feature1_data', shape=(1, featdim))
   feature2 = mx.sym.Variable('feature2_data', shape=(batch_size, featdim))
#   feature1 = mx.sym.Reshape(feature1, shape=(1, -1))
   diff = mx.sym.broadcast_minus(feature1, feature2)
   absdiff = mx.sym.abs(diff)
   return reid_feature, absdiff 


def CreateModel_Color_Separate(ctx, batch_size, proxy_num, imagesize):
  print 'creating network color separate...'
  args_all = None
  reid_feature, args_all = create_inception_resnet_v2(data0, namepre='part1', args=args_all)  
  reid_loss =  proxy_nca_loss_layer(batch_size)
  return reid_feature, reid_loss


def CreateModel_Color_predict():
   data1 = mx.sym.Variable('data')
   args_all = None
   reid_feature, args_all = create_inception_resnet_v2(data1, namepre='part1', args=args_all)
   return reid_feature 
def Createmodel_Resnet_Test(num, **kwargs):
  num_layers = num
  if num_layers >= 101:
    filter_list = [64, 256, 512, 1024, 2048]
    bottle_neck = True
  else:
    filter_list = [64, 64, 128, 256, 512]
    bottle_neck = False
  if num_layers == 18:
    units = [2, 2, 2, 2]
  elif num_layers == 34:
    units = [3, 4, 6, 3]
  elif num_layers == 49:
    units = [3, 4, 14, 3]
  elif num_layers == 50:
    units = [3, 4, 14, 3]
  elif num_layers == 74:
    units = [3, 6, 24, 3]

  else:
    raise ValueError('no experiments done on num_layers {}, you can do  it'.format(num_layers))
  num_stages = 4
  reid_feature = resnet(units = units, num_stages = num_stages, filter_list = filter_list,
                    num_classes = 128, bottle_neck = bottle_neck, **kwargs)
#  reid_feature = reid_feature/100
  if True:
    # feature_norm = mx.sym.sqrt(mx.sym.sum_axis(reid_feature**2, axis=1))
     feature_norm = mx.sym.sqrt(mx.sym.sum_axis(reid_feature**2))
     feature_norm = mx.sym.Reshape(feature_norm, shape=(-2, 1))
     reid_feature = mx.sym.broadcast_div(reid_feature, feature_norm)
#     reid_feature =  feature_norm
#     reid_feature =  reid_feature
  feature1 = mx.sym.Variable('feature1_data')
  feature2 = mx.sym.Variable('feature2_data')
# absdiff = mx.sym.sum(mx.sym.abs(feature1-feature2), axis=1)
  absdiff = mx.sym.abs(feature1-feature2)
  return reid_feature, absdiff


def  CreateModel_Resnet(ctx, batch_size,  proxy_num, imagesize, num_layers, **kwargs):
  print 'create network  in Resnet  format.. reid'
  if num_layers >= 101:
    filter_list = [64, 256, 512, 1024, 2048]
    bottle_neck = True
  else:
    filter_list = [64, 64, 128, 256, 512]
    bottle_neck = False
  if num_layers == 18:
    units = [2, 2, 2, 2]
  elif num_layers == 34:
    units = [3, 4, 6, 3]
  elif num_layers == 49:
    units = [3, 4, 14, 3]
  elif num_layers == 50:
    units = [3, 4, 14, 3]
  elif num_layers == 74:
    units = [3, 6, 24, 3]
  
  else:
    raise ValueError('no experiments done on num_layers {}, you can do  it'.format(num_layers))
  num_stages = 4
  reid_feature = resnet(units = units, num_stages = num_stages, filter_list = filter_list,
                        num_classes = proxy_num, bottle_neck = bottle_neck, **kwargs)
  proxy_yM = mx.sym.Variable('proxy_yM')
  proxy_ZM = mx.sym.Variable('proxy_ZM')
  proxy_Z = mx.sym.Variable(name = 'proxy_Z_weight', shape= (proxy_num, 128), dtype=np.float32)
  
  features = mx.sym.SliceChannel(reid_feature, axis=0, num_outputs=batch_size, name='features_slice')
  proxy_yMs = mx.sym.SliceChannel(proxy_yM, axis=0, num_outputs=batch_size, name='proxy_yM_slice')
  proxy_ZMs = mx.sym.SliceChannel(proxy_ZM, axis=0, num_outputs=batch_size, name='proxy_ZM_slice')
  proxy_ncas = []
  min_value = 10**-36
#  norm_value = (84**0.5)/2
#  norm_value = np.log(2.0**128)/4
  useSquare = True
  useHing = False
  logging.info('useSquare:' + str(useSquare) + ', useHing:' + str(useHing))
  if useSquare:
    norm_value = np.log((2.0**126)/proxy_num)/2 #2.0**126 is near the maxinum value of float32
  else:
    norm_value = np.log((2.0**126)/proxy_num)/(2*np.sqrt(128)) #2.0**126 is near the maxinum value of float32, 128 is the featnum
  logging.info('norm_value:' + str(norm_value))

  #norm
  if True:
    proxy_Znorm = mx.sym.sum_axis(proxy_Z**2, axis=1)
    proxy_Znorm = mx.sym.sqrt(proxy_Znorm) + min_value
    proxy_Znorm = mx.sym.Reshape(proxy_Znorm, shape=(-2, 1))
    proxy_Z = mx.sym.broadcast_div(proxy_Z, proxy_Znorm) * norm_value

  for bi in xrange(batch_size):
    one_feat = features[bi]

    #norm 
    if True:
      one_feat_norm = mx.sym.sqrt(mx.sym.sum(one_feat**2)) + min_value
      one_feat_norm = mx.sym.Reshape(one_feat_norm, shape=(-2, 1))
      one_feat = mx.sym.broadcast_div(one_feat, one_feat_norm) * norm_value

    one_proxy_yM = proxy_yMs[bi]
    one_proxy_ZM = proxy_ZMs[bi]

    tzM = mx.sym.Reshape(one_proxy_ZM, shape=(-1,))
    z = mx.sym.broadcast_minus(one_feat, proxy_Z)
    if useSquare:
      z = mx.sym.square(z)
      z = mx.sym.sum_axis(z, axis=1)
      z = mx.sym.sqrt(z)
    else:
      z = mx.sym.abs(z)
      z = mx.sym.sum_axis(z, axis=1)
    z = -z

    tyM = mx.sym.Reshape(one_proxy_yM, shape=(-1, 1))
    one_proxy_y = mx.sym.broadcast_mul(tyM, proxy_Z)
    one_proxy_y = mx.sym.sum_axis(one_proxy_y, axis=0)
#    print 'one_proxy_y:', one_proxy_y.name
    one_feat = mx.sym.Reshape(one_feat, shape=(-1,))
    y = one_feat - one_proxy_y
    if useSquare:
      y = mx.sym.square(y)
      y = mx.sym.sum(y)
      y = mx.sym.sqrt(y)
    else:
      y = mx.sym.abs(y)
      y = mx.sym.sum(y)
    y = -y

    z_y = mx.sym.broadcast_minus(z, y)
    z_y = mx.sym.exp(z_y) * tzM
    z_y = mx.sym.sum(z_y)
    z_y = z_y + min_value
    one_proxy_nca = mx.sym.log(z_y)

    proxy_ncas.append(one_proxy_nca)

  proxy_nca = mx.sym.Concat(*proxy_ncas, dim=0)
  if useHing:
    reid_net = mx.sym.maximum(-1.0, proxy_nca)
  else:
   # reid_net = proxy_nca
    reid_net = mx.sym.maximum(-87.0, proxy_nca) #avoid the nan, exp(-87) is near the mininum of float
  reid_net = mx.sym.MakeLoss(reid_net, name='proxy_nca_loss')
  return  reid_net
def resnet(units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'relu') 
    data0 = mx.sym.Variable('data')
     
    num_unit =  len(units)
    body = Conv(data=data0, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
      body = residual_unit(body, filter_list[i+1], (2, 2), False,
        name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      for j in range(units[i]-1):
        body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i+1, j+2),
          bottle_neck=bottle_neck, **kwargs)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
 #   body = mx.symbol.Dropout(data=body, p=0.5)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=128, name='pre_fc1')
    return fc1

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
  return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)    

def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    use_se = kwargs.get('version_se', 0)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'relu')
    #print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn4 = mx.symbol.broadcast_mul(bn4, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn4 + shortcut

    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn3 + shortcut
     
def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type=='prelu':
      pass
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body
if __name__ == '__main__':
  draw_inception_renet_v2()



