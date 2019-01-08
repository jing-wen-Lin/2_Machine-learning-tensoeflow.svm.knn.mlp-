"""
使用TensorFlow Hub中的圖像模塊進行簡單的傳輸學習。
此示例顯示如何基於計算圖像特徵向量的任何TensorFlow Hub模塊訓練圖像分類器。
使用在ImageNet上訓練的Inception V3計算的特徵向量。

頂層接收每個圖像的2048維向量（假設初始V3）作為輸入。
我們在此表示的頂部訓練softmax層。
如果softmax層包含N個標籤，則這對應於學習偏差和權重的N + 2048 * N模型參數。

這是一個示例，假設您有一個包含類命名子文件夾的文件夾，每個子文件夾都包含每個標籤的圖像。
子文件夾名稱很重要，因為它們定義了應用於每個圖像的標籤，但文件名本身並不重要。

您可以將image_dir參數替換為包含圖像子文件夾的任何文件夾。 每個圖像的標籤取自其所在子文件夾的名稱。
這將生成一個新的模型文件，可以由任何TensorFlow程序加載和運行，例如label_image示例代碼。
此腳本將使用高度準確但相對較大且速度較慢的Inception V3模型體系結構。

運行Mobilenet的浮點版本：這些儀表化模型可以通過TensorFlow Lite轉換為完全量化的移動模型。
有不同的Mobilenet型號可供選擇，有各種文件
大小和延遲選項。
    - 第一個數字可以是'100'，'075'，'050'或'025'來控制數字
     神經元（隱藏層的激活）; 權重的數量（因此
     在某種程度上，文件大小和速度）縮小與平方
     分數。
    - 第二個數字是輸入圖像大小。 你可以選擇'224'，'192'，
     '160'或'128'，尺寸更小，速度更快。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # 每分類最大圖片數max-image-per-class~134M
#要使用的hub模型，就是module_name
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'


#hub模型中要使用的量化操作節點名(使用TF-Lite進行量化)
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

#讀取圖片列表
def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """從文件系統構建訓練圖像列表。
   分析圖像目錄中的子文件夾，將它們拆分為穩定培訓，測試和驗證集，並返回數據結構描述每個標籤及其路徑的圖像列表。
  ARGS：
     image_dir：包含圖像子文件夾的文件夾的字符串路徑。
     testing_percentage：要為測試保留的圖像的整數百分比。
     validation_percentage：為驗證保留的圖像的整數百分比。
     
  Returns：
     OrderedDict，包含每個標籤子文件夾的條目，帶有圖像在每個標籤內分成培訓，測試和驗證集。
     項目的順序定義了類索引。
  """
  if not tf.gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = collections.OrderedDict()#有序字典，匹配顺序到labels
  sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir)) #獲得槍械類型文件夾列表，第一個是根目錄。
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir: #跳過根目錄
      is_root_dir = False
      continue
    extensions = sorted(set(os.path.normcase(ext)  
    #Windows上的Smash案例。
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg']))
    file_list = []
    dir_name = os.path.basename(sub_dir)#得到槍械分類的名字falsegun，Knife，realgun...
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)#獲取所有圖片路徑
      file_list.extend(tf.gfile.Glob(file_glob)) #將所有圖片路徑加入到file_list  
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []#用於訓練的圖片路徑
    testing_images = []#用於測試的圖片路徑
    validation_images = []#用於驗證的圖片路徑
    for file_name in file_list:
      base_name = os.path.basename(file_name)#得到文件名fg1.jpg...
"""在決定將圖像放入哪個集合時，我們希望忽略文件名中“_nohash_”之後的任何內容，
   數據集創建者可以對彼此接近變化的照片進行分組。
"""    
""" 這看起來有點神奇，但我們需要確定此文件是否應該進入訓練，測試或驗證集，
    並且我們希望將現有文件保留在同一組中，即使隨後添加了更多文件。
    要做到這一點，我們需要一種基於文件名本身的穩定決策方式，
    因此我們對其進行 hash 處理，然後使用它來生成我們用來分配它的概率值。
"""   
      hash_name = re.sub(r'_nohash_.*$', '', file_name)#得到圖片全路径
      hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest() #轉40位hash
      percentage_hash = ((int(hash_name_hashed, 16) % #轉超長的整數
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS)) #放缩到0~100
      if percentage_hash < validation_percentage:#驗證集
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage): #測試集+驗證集
        testing_images.append(base_name)
      else:
        training_images.append(base_name)#訓練集
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

#獲取圖片全路徑的函數
def get_image_path(image_lists, label_name, index, image_dir, category):
  """返回給定索引處標籤的圖像路徑。
  ARGS:
   image_lists：每個標籤的訓練圖像的OrderedDict。
   label_name：我們想要獲取圖像的標籤字符串。
   index：我們想要的圖像的Int偏移量。 這將由模塊化
   標籤的可用圖像數量，因此可以任意大。
   image_dir：包含訓練的子文件夾的根文件夾字符串圖片。
   category：用於從中提取圖像的名稱字符串 - 訓練，測試或驗證。
  Returns：
   文件系統路徑字符串到滿足請求參數的圖像。
  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)#避免超出長度范圍
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path
"""
生成瓶頸文件Bottleneck路徑的函數
Bottleneck是對原始眾多圖片數據進行整理後的數據文件，能夠更加方便的被tensorflow調用來訓練、檢驗或預測使用。
具體創建方法在下面會詳解，這裏只是先生成一個存放瓶頸文件的目錄。
"""
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, module_name):
  """返回給定索引處標籤的瓶頸文件的路徑。
  ARGS：
     image_lists：每個標籤的訓練圖像的OrderedDict。
     label_name：我們想要獲取圖像的標籤字符串。
     index：我們想要的圖像的整數偏移量。這將由模塊化標籤的可用圖像數量，因此可以任意大。
     bottleneck_dir：保存瓶頸值緩存文件的文件夾字符串。
     category：用於從中提取圖像的名稱字符串 - 訓練，測試或驗證。
     module_name：正在使用的映像模塊的名稱。
  Returns:
    文件系統路徑字符串到滿足請求參數的圖像。
  """
  module_name = (module_name.replace('://', '~') #URL方案。
                 .replace('/', '~')  #URL和Unix路徑。
                 .replace(':', '~').replace('\\', '~'))  #Windows路徑。
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + module_name + '.txt'#為了簡化路徑，我們臨時使用字符串module_name而不是變量


"""
  從hub.ModuleSpec生成圖和模型的函數
  因為我們並不是完全從頭開始訓練模型，而是在hub的某個模型的基礎上進行再訓練，
  所以首先我們要從hub上拉取模型信息，並從中恢覆出原計算圖graph的一些張量參數以便於使用。
"""
def create_module_graph(module_spec):
  """
  創建一個圖表並將Hub Module加載到其中。
  ARGS：
     module_spec：正在使用的映像模塊的hub.ModuleSpec。
  Returns:
     graph：創建的tf.Graph。
     bottleneck_tensor：模塊輸出的瓶頸值。
     resized_input_tensor：輸入圖像，按模塊的預期調整大小。
     wants_quantization：一個布爾值，模塊是否已經使用偽量化操作進行了檢測。
 """

  height, width = hub.get_expected_image_size(module_spec)
  with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    bottleneck_tensor = m(resized_input_tensor)
    wants_quantization = any(node.op in FAKE_QUANT_OPS
                             for node in graph.as_graph_def().node)#any任何一個為真即為真
  return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


"""
提取圖片瓶頸值的函數
上面的代碼我們從hub獲得了模型，並提取到了graph計算圖，
這個函數將使用計算圖中的兩個張量進行運算，先對圖片進行調整尺寸轉為張量，然後提取它的瓶頸張量值。
"""
#使用inception-v3處理圖片獲取特徵向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """在圖像上運行推斷以提取“瓶頸”摘要圖層。

  ARGS：
     sess：當前活動的TensorFlow會話。
     image_data：原始JPEG數據的字符串。
     image_data_tensor：圖中的輸入數據層。
     decoding_image_tensor：初始圖像大小調整和預處理的輸出。
     resized_input_tensor：識別圖的輸入節點。
     bottleneck_tensor：最終softmax之前的圖層。
  Returns:
     Numpy數組瓶頸值。
  """
  #首先解碼JPEG圖像，調整其大小，然後重新縮放像素值。
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  #然後通過識別網絡運行它。
  
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)#去掉冗余的數組嵌套，簡化形狀
  return bottleneck_values

#確保目錄路徑存在，不存在就創建
def ensure_dir_exists(dir_name):
  """
  確保磁盤上存在該文件夾。

  ARGS：
     dir_name：我們要創建的文件夾的路徑字符串。
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

#創建一個瓶頸文件
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not tf.gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path) #獲取圖片文件全路徑
  image_data = tf.gfile.FastGFile(image_path, 'rb').read() #獲取文件原數據
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor) #從圖片生成瓶頸值
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)#將獲得的bottleneck值用逗號連接成字符串寫入文件

#取得或創建瓶頸文件數據，如果沒有就創建它。返回由bottleneck層產生的圖片的numpy array數組
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name):

  """
  檢索或計算圖像的瓶頸值。
  如果磁盤上存在瓶頸數據的緩存版本，則返回該信息，否則計算數據並將其保存到磁盤以備將來使用。
  ARGS：
    sess：當前活動的TensorFlow會話。
    image_lists：每個標籤的訓練圖像的OrderedDict。
    label_name：我們想要獲取圖像的標籤字符串。
    index：我們想要的圖像的整數偏移量。這將由模塊化標籤的可用圖像數量，因此可以任意大。
    image_dir：包含訓練的子文件夾的根文件夾字符串
    圖片。
    category：設置為從中提取圖像的名稱字符串 - 培訓，測試，
    或驗證。
    bottleneck_dir：保存瓶頸值緩存文件的文件夾字符串。
    jpeg_data_tensor：將加載的jpeg數據輸入的張量。
    decoding_image_tensor：解碼和調整圖像大小的輸出。
    resized_input_tensor：識別圖的輸入節點。
    bottleneck_tensor：瓶頸值的輸出張量。
    module_name：正在使用的映像模塊的名稱。
  Returns：
    由圖像的瓶頸層產生的Numpy數組值。
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']#獲取槍械武器分類名如'falsegun'
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)#確保路徑文件夾存在
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, module_name)
  if not os.path.exists(bottleneck_path):#如果文件不存在就創建文件
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:#讀取瓶頸文件
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False #遇到錯誤
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:#如果出錯就重建瓶頸文件
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    #允許異常在這裡傳播，因為它們不應該在新創建後發生

    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values

#確保所有的training、testing、validation要用的bottleneck文件都已經被緩存
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
  """
  因為在訓練過程中，對同一個圖片會反覆多次讀取(不對圖像進行扭曲處理的話)，
  如果我們對圖片bottleneck緩存就能大大提高效率。
  我們將用這個函數檢查所有圖片進行計算並保存。
  這個函數其實也只是循環調用前面的get_or_create_bottleneck函數。

  ARGS：
    sess：當前活動的TensorFlow會話。
    image_lists：每個標籤的訓練圖像的OrderedDict。
    image_dir：包含訓練的子文件夾的根文件夾字符串
    圖片。
    bottleneck_dir：保存瓶頸值緩存文件的文件夾字符串。
    jpeg_data_tensor：從文件輸入jpeg數據的張量。
    decoding_image_tensor：解碼和調整圖像大小的輸出。
    resized_input_tensor：識別圖的輸入節點。
    bottleneck_tensor：圖的倒數第二個輸出層。
    module_name：正在使用的映像模塊的名稱。

  返回：
    沒有。
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]#針對每一個分類，比如falsegun
      for index, unused_base_name in enumerate(category_list): #創建索引
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')#每100張輸出一次提示
"""
運行過程中會隔一會（處理100張）輸出一行提示。整個過程可能需要十幾分鐘或更久，
全部完成後會在/bottlenecks/文件夾下增加每個危險槍械類別的文件夾並且裏面包含了很多很多txt文件。
"""
#隨機獲取所有種類中隨機bottleneck數據列表、對應的label_index和圖片文件路徑列表,
#how_many數量小於等於0時候獲取全部
def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name):

  """
  檢索緩存圖像的瓶頸值。

  如果未應用任何失真，則此函數可以直接從磁盤檢索緩存的瓶頸值以獲取圖像。 
  它從指定的類別中選擇一組隨機圖像。

  ARGS：
    sess：當前的TensorFlow會話。
    image_lists：每個標籤的訓練圖像的OrderedDict。
    how_many：如果是正數，將選擇此大小的隨機樣本。
    如果是否定的，將檢索所有瓶頸。
    category：要從中拉出的名稱字符串 -  training，testing或
    驗證。
    bottleneck_dir：保存瓶頸值緩存文件的文件夾字符串。
    image_dir：包含訓練的子文件夾的根文件夾字符串
    圖片。
    jpeg_data_tensor：將jpeg圖像數據輸入的圖層。
    decoding_image_tensor：解碼和調整圖像大小的輸出。
    resized_input_tensor：識別圖的輸入節點。
    bottleneck_tensor：CNN圖的瓶頸輸出層。
    module_name：正在使用的映像模塊的名稱。

  返回：
    瓶頸陣列列表，它們相應的基本事實和
    相關文件名。
  """
  class_count = len(image_lists.keys()) #有多少種槍械分類
  bottlenecks = []
  ground_truths = []#槍械分類索引號
  filenames = [] #圖片文件路徑列表
  if how_many >= 0:
    #檢索隨機的瓶頸樣本。
    for unused_i in range(how_many): #随機一種槍械如falsegun
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)#每種類最大數量，如果超過後面會自動取餘數
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)#圖片路徑
      bottleneck = get_or_create_bottleneck(#讀取bottleneck文件數據
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, module_name)
      bottlenecks.append(bottleneck)
      ground_truths.append(label_index)
      filenames.append(image_name)
  else:
    #檢索所有瓶頸。
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name)
        bottlenecks.append(bottleneck)
        ground_truths.append(label_index)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

"""
如果我們使用變形的圖片進行訓練，比如裁剪、放縮、翻轉的圖片，我們需要針對每個圖片重新計算整個模型，
所以我們不能使用原來緩存的圖片bottleneck數據，
我們需要使用另外的變形計算圖來運行得到新的變形bottleneck數據，然後再把它投入到整個計算圖進行訓練。
"""
#隨機獲取變形的瓶頸數據，返回bottlenecks數組和對應的label_index數組
def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """
  在扭曲之後檢索訓練圖像的瓶頸值。

  因為我們可能會多次讀取相同的圖像（如果我們正在訓練像作物，鱗片或翻轉這樣的扭曲，我們必須重新計算每個圖像的完整模型，因此我們不能使用緩存的瓶頸值。
  相反，我們找到所請求類別的隨機圖像，通過失真圖運行它們，然後是完整圖形以獲得每個圖像的瓶頸結果。

  ARGS：
    sess：當前的TensorFlow會話。
    image_lists：每個標籤的訓練圖像的OrderedDict。
    how_many：要返回的整數瓶頸值。
    category：要獲取的圖像集的名稱字符串 - 培訓，測試，
    或驗證。
    image_dir：包含訓練的子文件夾的根文件夾字符串
    圖片。
    input_jpeg_tensor：我們將圖像數據提供給的輸入層。
    distorted_image：失真圖的輸出節點。
    resized_input_tensor：識別圖的輸入節點。
    bottleneck_tensor：CNN圖的瓶頸輸出層。

  返回：
    瓶頸陣列列表及其相應的基本事實。
  """
  class_count = len(image_lists.keys()) #有幾種槍械分類
  bottlenecks = []#變形後的瓶頸數據
  ground_truths = []#標簽編號
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)#隨機一個槍械分類
    label_name = list(image_lists.keys())[label_index]#falsegun
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)#隨機一張圖
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not tf.gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()#沒有參數傳遞jpeg_data進來，要重新讀取文件
    #請注意，在對圖像發送運行推理之前，我們將distorted_image_data實現為numpy數組。
    #這涉及2個內存副本，可能在其他實現中進行了優化。
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    bottlenecks.append(bottleneck_values)
    ground_truths.append(label_index)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
 """
  是否從輸入標誌啟用任何失真。

  ARGS：
     flip_left_right：Boolean是否水平隨機鏡像圖像。
     random_crop：整數百分比設置周圍使用的總保證金
     裁剪框。
     random_scale：縮放比例的整數百分比。
     random_brightness：整數範圍，用於隨機乘以像素值。

  返回：
     布林數，指示是否應該應用任何失真。
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))
  
#生成兩個變形操作ops的函數input_jpeg_tensor,distorted_image_tensor
#注意，只是生成一個grah並返回需要運行這個graph的兩個feed_dict入口
def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, module_spec):
  """
  創建應用指定扭曲的操作。
  
  生成變形圖片操作ops的函數add_input_distortions
  在訓練的過程中我們對圖片進行一些變形（裁切、放縮、翻轉或調整亮度），可以利用有限數量的圖片模擬更多的真實情況，進而有效改進模型。

  裁剪
  ~~~~~~~~
   通過將邊界框放置在完整圖像中的隨機位置來完成裁剪。
   cropping參數控制該框相對於輸入圖像的大小。 如果它為零，則該框與輸入的大小相同，並且不執行裁剪。 
   如果值為50％，則裁剪框將為輸入的寬度和高度的一半。 
   在圖中它看起來像這樣：
  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  縮放
  ~~~~~~~
    縮放很像裁剪，除了邊界框始終居中並且其大小在給定範圍內隨機變化。 
    例如，如果比例百分比為零，則邊界框與輸入的大小相同，並且不應用縮放。 
    如果它是50％，那麼邊界框將在寬度和高度的一半與全尺寸之間的隨機範圍內。

  ARGS：
     flip_left_right：Boolean是否水平隨機鏡像圖像。
     random_crop：整數百分比設置周圍使用的總保證金
     裁剪框。
     random_scale：縮放比例的整數百分比。
     random_brightness：整數範圍，用於隨機乘以像素值。
     圖形。
     module_spec：正在使用的映像模塊的hub.ModuleSpec。

  返回：
     jpeg輸入層和失真結果張量。
  """
   
  input_height, input_width = hub.get_expected_image_size(module_spec)#獲取已有模型中的寬高要求
  input_depth = hub.get_num_image_channels(module_spec)#獲取模型中圖片通道深度數
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput') #feed_dict輸入口
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth) #讀取圖片數據
  #從uint8的全範圍轉換到float32的範圍[0,1]。
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)#數據類型轉換
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0) #升維
  #對圖片數據進行裁切和放縮
  margin_scale = 1.0 + (random_crop / 100.0)#參數範圍0~100
  resize_scale = 1.0 + (random_scale / 100.0)#參數範圍0~100
  margin_scale_value = tf.constant(margin_scale)#轉為張量
  resize_scale_value = tf.random_uniform(shape=[],
                                         minval=1.0,
                                         maxval=resize_scale)#轉為張量
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
  #對圖片進行翻轉
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  #調整圖片亮度
  brightness_min = 1.0 - (random_brightness / 100.0)#random_brightness參數範圍0~100
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(shape=[],
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def variable_summaries(var):
  """將大量摘要附加到Tensor（用於TensorBoard可視化）。"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


#添加最終再訓練操作ops，一個softmax層一個dense層
#注意這裏的quantize_layer為真假值，但必須與create_module_graph(module_spec)得到的wants_quantization一致，否則出錯
def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          quantize_layer, is_training):
  """
  添加一個新的softmax和完全連接的層用於訓練和評估。
  我們需要重新訓練頂層來識別新的分類，這個函數將向圖添加一些操作，隨著一些變量保存權重，然後為所有反向傳播設置梯度變化。
  這個函數將為訓練和計算添加新的SOFTMAX和全連接層（密集層）。
  ARGS：
    class_count：我們嘗試的事物類別的整數
        認識。
    final_tensor_name：生成結果的新最終節點的名稱字符串。
    bottleneck_tensor：主CNN圖的輸出。
    quantize_layer：Boolean，指定新添加的層是否應該是
        使用TF-Lite進行量化。
    is_training：Boolean，指定新添加的圖層是否用於訓練
        或評估。

  返回：
    訓練和交叉熵結果的張量，以及張量的張量
    瓶頸輸入和地面實況輸入。
  """
  batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
  assert batch_size is None, '我們希望針對任意批次大小進行計算'
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[batch_size, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [batch_size], name='GroundTruthInput')
    
  #組織下面的操作使他們在Tensorboard中可見
  layer_name = 'final_retrain_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal( #正態截取，上下不超過0.001*2
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

  # tf.contrib.quantize函數重寫圖形以進行量化。
  #導入的模型圖已經被重寫，因此在調用這些重寫時，只會轉換新添加的最終圖層。
  if quantize_layer:
    if is_training:
      tf.contrib.quantize.create_training_graph()#自動重寫graph量子化，僅新增的layer被變換，訓練用
    else:
      tf.contrib.quantize.create_eval_graph() #預測用

  tf.summary.histogram('activations', final_tensor)
  
  if not is_training: #對於預測，不需要添加損失函數或優化器，所以返回兩個None
    return None, None, bottleneck_input, ground_truth_input, final_tensor

  with tf.name_scope('cross_entropy'):#平均交叉熵作為損失函數
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)#梯度漸變優化函數
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)

#插入評價精確度的操作，返回元組(evaluation step, prediction)
def add_evaluation_step(result_tensor, ground_truth_tensor):
  """
  插入我們評估結果準確性所需的操作。
  評價方法需要輸入新的圖片特徵數據和對應的標籤，
  這裡的函數接著上面的再訓練函數得到的final_tensor，ground_truth_input，作為新的輸入口，實現評價功能。
  ARGS：
     result_tensor：生成結果的新最終節點。
     ground_truth_tensor：我們提供地面實況數據的節點
    成。

  返回：
     （評估步驟，預測）的元組。
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1) #獲取axis=1維度的最大值，即預測結果
      correct_prediction = tf.equal(prediction, ground_truth_tensor)#預測與標簽是否相等
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#跨維度
  #計算平均數
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction

#執行最終評估運算
def run_final_eval(train_session, module_spec, class_count, image_lists,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):
  """
  使用測試數據集在eval圖上運行最終評估。
  
  從集線器讀取模型，創建圖以及相關操作入口，添加各種操作（訓練操作，圖片解碼等）
  ，讀取瓶頸數據，然後開始運作，使用變形扭曲的瓶頸數據或者緩存的，sess.run運行訓練操作。
  同時注意總結信息的保存和檢查站模型保存。

  ARGS：
     train_session：下面有張量的訓練圖的會話。
     module_spec：正在使用的映像模塊的hub.ModuleSpec。
     class_count：類的數量
     image_lists：每個標籤的訓練圖像的OrderedDict。
     jpeg_data_tensor：將jpeg圖像數據輸入的圖層。
     decoding_image_tensor：解碼和調整圖像大小的輸出。
     resized_image_tensor：識別圖的輸入節點。
     bottleneck_tensor：CNN圖的瓶頸輸出層。
  """
  test_bottlenecks, test_ground_truth, test_filenames = (
      get_random_cached_bottlenecks(train_session, image_lists,
                                    FLAGS.test_batch_size,
                                    'testing', FLAGS.bottleneck_dir,
                                    FLAGS.image_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor,
                                    bottleneck_tensor, FLAGS.tfhub_module))
   #創建評估會話
  (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
   prediction) = build_eval_session(module_spec, class_count)
  #隨機獲取bottleneck
  test_accuracy, predictions = eval_session.run(
      [evaluation_step, prediction],
      feed_dict={
          bottleneck_input: test_bottlenecks,
          ground_truth_input: test_ground_truth
      })
  #運行評估！
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                  (test_accuracy * 100, len(test_bottlenecks)))

  if FLAGS.print_misclassified_test_images:
    tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
    for i, test_filename in enumerate(test_filenames):
      if predictions[i] != test_ground_truth[i]:
        tf.logging.info('%70s  %s' % (test_filename,
                                      list(image_lists.keys())[predictions[i]]))

#創建和恢覆評價會話（沒有訓練操作），用於導出結果
#返回一個包含評價圖的會話，以及相關其他張量和操作
def build_eval_session(module_spec, class_count):
  """  
  從存儲的訓練圖檢查點文件讀取變量，恢復到評價圖，並利用上面的函數add_evaluation_step添加評估操作。

  ARGS：
     module_spec：正在使用的映像模塊的hub.ModuleSpec。
     class_count：類的數量

  返回：
     包含已恢復的eval圖的Eval會話。
     瓶頸輸入，基礎事實，評估步驟和預測張量。
  """
  #如果量化，我們需要為導出創建正確的eval圖。
  eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
      create_module_graph(module_spec))

  eval_sess = tf.Session(graph=eval_graph)
  with eval_graph.as_default(): #添加新的導出層
    (_, _, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=False)
    #把訓練圖的值恢覆到評價圖
    tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
    #添加評估操作
    evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                      ground_truth_input)

  return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
          evaluation_step, prediction)


def save_graph_to_file(graph_file_name, module_spec, class_count):
  """將圖形保存到文件，必要時創建有效的量化圖形。"""
  sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
  graph = sess.graph

  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

  with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())


def prepare_file_system():
  #設置我們為TensorBoard寫入摘要的目錄
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return

#添加操作，執行jpeg解碼和調整大小 ，返回兩個張量jpeg_data_tensor,decoded_image_tensor
def add_jpeg_decoding(module_spec):
  """
  添加執行JPEG解碼和調整大小的操作。

  ARGS：
     module_spec：正在使用的映像模塊的hub.ModuleSpec。

  返回：
     節點的張量將JPEG數據輸入到輸出中
       預處理步驟。
  """
  input_height, input_width = hub.get_expected_image_size(module_spec)
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  #從全範圍的uint8轉換為float32的範圍[0,1]。
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)#擴充形狀的維度
  resize_shape = tf.stack([input_height, input_width]) #通過合並提升維度
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)  #放縮圖像尺寸
  return jpeg_data, resized_image


def export_model(module_spec, class_count, saved_model_dir):
  """
  出口服務模式。

  ARGS：
     module_spec：正在使用的映像模塊的hub.ModuleSpec。
     class_count：類的數量。
     saved_model_dir：保存導出的模型和變量的目錄。
  """
  #SavedModel應該保存eval圖。
  sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
  with sess.graph.as_default() as graph:
    tf.saved_model.simple_save(
        sess,
        saved_model_dir,
        inputs={'image': in_image},
        outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
    )


def main(_):
  #需要確保日誌輸出可見。
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.image_dir:
    tf.logging.error('Must set flag --image_dir.')
    return -1

  #準備可在培訓期間使用的必要目錄
  prepare_file_system()

  #查看文件夾結構，並創建所有圖像的列表。
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')
    return -1

  #看看命令行標誌是否意味著我們正在應用任何扭曲。
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  #設置預先訓練的圖形。
  module_spec = hub.load_module_spec(FLAGS.tfhub_module)
  graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
      create_module_graph(module_spec))

  #添加我們將要訓練的新圖層。
  with graph.as_default():
    (train_step, cross_entropy, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=True)

  with tf.Session(graph=graph) as sess:
    #初始化所有權重：模塊到其預訓練值，以及新添加的再訓練層到隨機初始值。
    init = tf.global_variables_initializer()
    sess.run(init)

    #設置圖像解碼子圖。
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

    if do_distort_images:

      #我們將應用扭曲，因此建立我們需要的操作。
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, module_spec)
    else:
        
      #我們將確保計算出“瓶頸”圖像摘要並將其緩存在磁盤上。
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, FLAGS.tfhub_module)

   
    #創建評估新圖層準確性所需的操作。
    evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

    #合併所有摘要並將它們寫出到summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

  #創建一個訓練保護程序，用於在導出模型時將值恢復為eval圖。
    
    train_saver = tf.train.Saver()

    #在命令行上按照請求運行多個週期的訓練。
    for i in range(FLAGS.how_many_training_steps):
       #獲取一批輸入瓶頸值，每次應用失真時計算新鮮，或者從磁盤上存儲的緩存中獲取。
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             FLAGS.tfhub_module)
      #將瓶頸和基礎事實輸入圖表，然後運行訓練步驟。 使用`merged`操作捕獲TensorBoard的培訓摘要。

      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      #經常打印圖表的訓練程度。
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))
       #TODO：使用eval圖表來避免量化
       #移動平均線由驗證集更新，但在實踐中這會產生可忽略的差異。
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.tfhub_module))
        
        #運行驗證步驟並使用`merged`操作捕獲TensorBoard的培訓摘要。
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))

      #存儲中間結果
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        #如果我們要進行中間保存，請保存訓練圖的檢查點，以恢復到eval圖。
        train_saver.save(sess, CHECKPOINT_NAME)
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(intermediate_file_name, module_spec,
                           class_count)

    #訓練結束後，強制最後一次保存訓練檢查站。
    train_saver.save(sess, CHECKPOINT_NAME)

    #我們已完成所有培訓，因此對我們以前沒有使用的一些新圖像進行最終測試評估。
    run_final_eval(sess, module_spec, class_count, image_lists,
                   jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                   bottleneck_tensor)

    #寫出訓練有素的圖形和標籤，權重存儲為常量。
    tf.logging.info('Save final result to : ' + FLAGS.output_graph)
    if wants_quantization:
      tf.logging.info('The model is instrumented for quantization with TF-Lite')
    save_graph_to_file(FLAGS.output_graph, module_spec, class_count)
    with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

    if FLAGS.saved_model_dir:
      export_model(module_spec, class_count, FLAGS.saved_model_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help=
      """
      要測試多少圖像。 該測試集僅使用一次，以在訓練完成後評估模型的最終準確度。 
      值為-1會導致使用整個測試集，從而在運行期間獲得更穩定的結果。
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help=
      """評估批次中要使用的圖像數量。 
      此驗證集的使用頻率遠高於測試集，並且是模型在培訓期間準確程度的早期指標。
      值為-1會導致使用整個驗證集，這會在訓練迭代中產生更穩定的結果，但在大型訓練集上可能會更慢。
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help=
      """
      是否打印出所有錯誤分類的測試圖像的列表。
      """,
      action='store_true'
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='將瓶頸圖層值緩存為文件的路徑。'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help=
      """
      再訓練圖中輸出分類圖層的名稱
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""
      是否水平地隨機翻轉一半訓練圖像。
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""
      確定隨機裁剪訓練圖像的餘量的百分比。
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""
      一個百分比決定隨機放大多少訓練圖像的大小。
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""
      確定將訓練圖像輸入像素向上或向下隨機乘以多少的百分比。
      """
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
      使用哪個TensorFlow Hub模塊。對於一些公開可用的模塊。
      """)
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='',
      help='在哪裡保存導出的圖形。')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)