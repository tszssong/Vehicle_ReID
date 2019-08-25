import sys
import numpy as np
import logging
import mxnet as mx 
import DataGenerator as dg 
import operator
import os

class PersonReID_Proxy_Batch_Plate_Mxnet_Iter2(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, 
               proxy_num, featdim, proxy_batchsize, proxy_num_batch, repeat_times=4, num_proxy_batch_max=0.0):
    super(PersonReID_Proxy_Batch_Plate_Mxnet_Iter2, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.datas_batch = {} 
    self.datas_batch['data'] = mx.nd.zeros(data_shapes[0], dtype=np.float32)
    self.datas_batch['databuffer'] = np.zeros(data_shapes[0], dtype=np.float32)
    self.labels_batch = {}
    self.labels_batch['proxy_yM'] = mx.nd.zeros(label_shapes[0], dtype=np.float32)
    self.labels_batch['proxy_ZM'] = mx.nd.zeros(label_shapes[1], dtype=np.float32)
    self.cur_batch = 0
    self.datalist = dg.get_datalist2(datafn)
    self.datalen = len(self.datalist)
    self.labeldict = dict(self._provide_label)
    self.proxy_batchsize = proxy_batchsize
    self.proxy_num_batch = proxy_num_batch 
    self.rndidx_list = None 
    self.num_batches = None#self.proxy_batchsize / label_shapes[0][0]
    self.batch_personids = []
    self.batch_infos = []
    self.num_proxy_batch = self.datalen / self.proxy_batchsize
    self.num_proxy_batch_max = num_proxy_batch_max
    self.cur_proxy_batch = 0
    self.big_epoch = 0
    self.proxy_num = proxy_num
    self.featdim = featdim
    self.proxy_Z_fn = './proxy_Z.params1'
    proxy_Ztmp = np.random.rand(self.proxy_num, self.featdim)-0.5
    self.proxy_Z = proxy_Ztmp.astype(np.float32) 
    if os.path.exists(self.proxy_Z_fn):
      tmpZ = mx.nd.load(self.proxy_Z_fn)
      self.proxy_Z = tmpZ[0].asnumpy()
      print self.proxy_num, tmpZ[0].shape[0]
      assert(self.proxy_num==tmpZ[0].shape[0])
      print 'load proxy_Z from', self.proxy_Z_fn
      
    proxy_Z_ptmp = np.random.rand(self.proxy_num_batch, self.featdim)-0.5
    self.proxy_Z_p = proxy_Z_ptmp.astype(np.float32)
    self.proxy_Z_map = np.zeros(self.proxy_batchsize, dtype=np.int32)-1
    self.personidnum = 0
    self.total_proxy_batch_epoch = 0
    self.repeat_times = repeat_times
    self.do_reset()

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.batch_personids = []
    self.batch_infos = []
    pass

  def proxy_updateset(self, proxy_Z_p_new):
    num = np.sum(self.proxy_Z_map>-1)
    p_Z = proxy_Z_p_new.asnumpy()
    self.proxy_Z_p[:] = p_Z
    for i in xrange(num):
      personid = self.proxy_Z_map[i]
      self.proxy_Z[personid] = p_Z[i]
    savename = self.proxy_Z_fn 
    mx.nd.save(savename, [mx.nd.array(self.proxy_Z)])
    print 'save proxy_Z into file', savename    #, num, self.caridnum, p_Z[:self.caridnum].sum()#, a#, a[a<0], a[a>0]
    pass

  def do_reset(self):
    self.cur_batch = 0        
    self.batch_carids = []
    self.batch_infos = []
    if self.total_proxy_batch_epoch == 0 \
       or self.cur_proxy_batch == self.num_proxy_batch \
       or (self.num_proxy_batch_max > 0.0 \
       and self.cur_proxy_batch > self.num_proxy_batch * self.num_proxy_batch_max):
      self.cur_proxy_batch = 0 
      self.big_epoch += 1
      self.rndidx_list = np.random.permutation(self.datalen)
      print 'permutation....................'

    self.proxy_datalist = []
    personids = {}
    self.proxy_Z_map[:] = -1
    prndidxs = np.random.permutation(self.proxy_batchsize)
    for i in xrange(self.proxy_batchsize):
      pidx = prndidxs[i]
      pxyi = self.cur_proxy_batch * self.proxy_batchsize + pidx
      idx = self.rndidx_list[pxyi]
      onedata = self.datalist[idx] 
    
      parts    = onedata.split('*')
      path     = parts[ 0]
      son      = parts[ 1]
      personid = parts[-1]
      if not personids.has_key(personid):
          personids[personid] = len(personids)
      if len(personids)>self.proxy_num_batch:
        logging.info('arriving max number proxy_num_batch:%d[%d/%d]...', len(personids), i+1, self.proxy_batchsize)
        break
      ori_id  = int(personid)
      proxyid = personids[personid] 
      self.proxy_Z_p[proxyid] = self.proxy_Z[ori_id]
      self.proxy_Z_map[proxyid] = ori_id
      proxy_str = '%s,%s,%s,%s'%(path, son, personid, str(proxyid))
      self.proxy_datalist.append(proxy_str)
    realnum = i+1
    self.num_batches = len(self.proxy_datalist) / self.batch_size
    self.personidnum = len(personids)
    print 'got another proxy batch to train(%d/%d/%d/%d, %d/%d) [big_epoch=%d]...'%(\
         self.personidnum, self.proxy_batchsize, realnum, self.datalen, self.cur_proxy_batch+1,\
         self.num_proxy_batch, self.big_epoch)
    sys.stdout.flush()
    self.total_proxy_batch_epoch += 1
    if self.total_proxy_batch_epoch%self.repeat_times==0:
      self.cur_proxy_batch += 1
    return self.personidnum, self.proxy_Z_p

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label


  def next(self):
    if self.cur_batch < self.num_batches:
      datas, labels, personids, infos = dg.get_data_label_proxy_batch_plate_mxnet_threads( \
                                            self._provide_data,  self.datas_batch,   \
                                            self._provide_label, self.labels_batch,  \
                                            self.proxy_datalist, self.cur_batch, self.personidnum) 
      self.batch_personids = personids
      self.batch_infos = infos
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration



if __name__=='__main__':
  print 'testing DataIter.py...'
  data_shape = (4, 3, 200, 200)
  proxy_yM_shape = (4, 4000)
  proxy_ZM_shape = (4, 4000)
  datafn_list = ['/home/chuanruihu/list_dir/Person_train.list']
  total_proxy_num =  60000
  featdim = 128
  proxy_batch = 4000 
  #  num_batches = 10
#  pair_part1_shape = (32, 3, 128, 128)
#  pair_part2_shape = (32, 3, 128, 128)
#  label_shape = (pair_part1_shape[0],)
# data_iter = CarReID_Iter(['part1_data', 'part2_data'], [pair_part1_shape, pair_part2_shape],
#                      ['label'], [label_shape], get_pairs_data_label,
#                      num_batches)
  data_iter = PersonReID_Proxy_Batch_Plate_Mxnet_Iter2(['data'], [data_shape], ['proxy_yM','proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape],datafn_list, total_proxy_num, featdim, proxy_batch, 1)
  for d in data_iter:
     print  d.data
     print  d.label
#    dks = d.data.keys()
#    lks = d.label.keys()
#    print dks[0], ':', d.data[dks[0]].asnumpy().shape, '   ', lks[0], ':', d.label[lks[0]].asnumpy().shape






