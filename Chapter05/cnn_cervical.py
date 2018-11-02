import numpy as np
import os
from sklearn.utils import shuffle
from PIL import Image
from PIL import ImageFile
import matplotlib.image as mpimg
from itertools import count
import scipy.misc as sci
import scipy.ndimage.interpolation as scizoom
import random
import time
from multiprocessing.pool import ThreadPool
from keras.models import Model
import math
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, concatenate
from keras.layers.normalization import BatchNormalization


ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
************************************************Data Preprocessing*************************************************
'''


#label count
def labelsCount(labels):
    labelCountDic = dict()
    for label in labels:
        if label in labelCountDic:
            labelCountDic[label] += 1
        else:
            labelCountDic[label] = 1
    return labelCountDic


# Read file paths and labels where labels are the type of cervical(1, 2, 3)
def readFilePaths(dir, no_labels=False, label_type=None):
    filePaths = []
    labels = []
    labels2Nums = dict()
    numLabels = None

    for dirName, subDirs, Files in os.walk(dir):
        if len(subDirs) > 0:
            numLabels = len(subDirs)
            for i, subDir in enumerate(subDirs):
                labels2Nums[subDir] = i
        else:
            cervicalType = dirName.split('/')[-1]

        for img in Files:
            if '.jpg' in img.lower():
                filePaths.append(os.path.join(dirName, img))
                if no_labels:
                    labels.append(img)
                elif type(label_type) is int:
                    labels.append(label_type)
                else:
                    labels.append(labels2Nums[cervicalType])

    if type(numLabels) is int:
        return filePaths, labels, numLabels
    else:
        return filePaths, labels, None


# Image resizing is important considering memory footprint, but its important to maintain key characteristics that will
# preserve the key features.
def resizeImage(imgPath, maxSize=(256,256,3), savePath=None, addFlip=False):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(imgPath)

    # set aspect ratio
    if type(img) == type(np.array([])):
        img = Image.fromarray(img)
    img.thumbnail(maxSize, Image.ANTIALIAS)
    tmpImage = (np.random.random(maxSize)*255).astype(np.uint8)
    resizedImg = Image.fromarray(tmpImage)
    resizedImg.paste(img,((maxSize[0]-img.size[0])//2, (maxSize[1]-img.size[1])//2))

    if savePath:
        resizedImg.save(savePath)

    if addFlip:
        flip = resizedImg.transpose(Image.FLIP_LEFT_RIGHT)
        if savePath:
            splitPath = savePath.split('/')
            flip_path = '/'.join(splitPath[:-1] + ['flipped_'+splitPath[-1]])
            flip.save(flip_path)
        return np.array(resizedImg, dtype=np.float32), np.array(flip,dtype=np.float32)
    return np.array(resizedImg, dtype=np.float32)


# process cervical dataset
def processCervicalData():
    # image resizing
    imgPaths = []
    labels = []
    trainingDirs = ['/deeplearning-keras/ch05/data/train']
    for dir in trainingDirs:
        newFilePaths, newLabels, numLabels = readFilePaths(dir)
        if len(newFilePaths) > 0:
            imgPaths += newFilePaths
            labels += newLabels

    imgPaths, labels = shuffle(imgPaths, labels)
    labelCount = labelsCount(labels)

    type1Count = labelCount[0]
    type2Count = labelCount[1]
    type3Count = labelCount[2]

    print("Count of type1 : ", type1Count)
    print("Count of type2 : ", type2Count)
    print("Count of type3 : ", type3Count)
    print("Total Number of data samples: " + str(len(imgPaths)))
    print("Number of Classes: " + str(numLabels))

    newShape = [(256,256,3)]
    destDir = ['/deeplearning-keras/ch05/data/resized_imgs']

    for newImgShape, destFolder in zip(newShape,destDir):
        for i, path,label in zip(count(),imgPaths,labels):
            split_path = path.split('/')
            newPath = 'size'+str(newImgShape[0])+'_'+split_path[-1]
            newPath = '/'.join([destFolder]+split_path[8:-1]+[newPath])
            add_flip = True
            if label == 1:
                add_flip = False

            # Used to exclude corrupt data
            try:
                resizeImage(path, maxSize=newImgShape, savePath=newPath, addFlip=add_flip)
            except OSError:
                print("Error at path " + path)


'''
************************************************Cervical Training*************************************************
'''


# get steps
def getSteps(n_samples,batch_size,n_augs=1):
    n_samples = n_samples*(n_augs+1)
    steps_per_epoch = n_samples//batch_size + 1
    if n_samples % batch_size == 0:
        train_steps_per_epoch = n_samples//batch_size
    return steps_per_epoch


# one hot encoding
def oneHotEncode(labels, n_classes):
    one_hots = []
    for label in labels:
        one_hot = [0]*n_classes
        if label >= len(one_hot):
            print("Labels out of bounds\nCheck your n_classes parameter")
            return
        one_hot[label] = 1
        one_hots.append(one_hot)
    return np.array(one_hots,dtype=np.float32)


# split dataset for training and validation
def getSplitData(csv_file_path):
    paths = []
    labels = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            paths.append(split_line[0])
            labels.append(int(split_line[1]))
    return paths,labels


# save paths into csv's
def savePaths(csv_file_path, paths, labels):
    with open(csv_file_path, 'w') as csv_file:
        for path,label in zip(paths,labels):
            csv_file.write(path + ',' + str(label) + '\n')


# generate images
def image_generator(file_paths, labels, batch_size, resize_dims=None, randomly_augment=False,rand_order=True):
    if randomly_augment:
        batch_size = int(batch_size/2) # maintains batch size despite image additions
        aug_paths = file_paths
        aug_labels = labels
    else:
        aug_paths, aug_labels = [], []

    while True:
        if rand_order:
            file_paths,labels = shuffle(file_paths,labels)
            aug_paths, aug_labels = shuffle(aug_paths, aug_labels)
        for batch in range(0, len(file_paths), batch_size):
            rpaths = []
            rlabels = []
            if randomly_augment:
                rpaths = aug_paths[batch:batch+batch_size]
                rlabels = aug_labels[batch:batch+batch_size]
            images, batch_labels = convertImages(file_paths[batch:batch+batch_size],
                                                  labels[batch:batch+batch_size],
                                                  resize_dims=resize_dims,
                                                  rpaths=rpaths,
                                                  rlabels=rlabels)
            yield images, batch_labels


def rotate(image, angle, ones=None, random_fill=True, color_range=255):
    if not random_fill:
        return sci.imrotate(image, angle).astype(np.float32)
    elif ones == None:
        ones = sci.imrotate(np.ones_like(image),angle)
    rot_image = sci.imrotate(image, angle).astype(np.float32)
    edge_filler = np.random.random(rot_image.shape).astype(np.float32)*color_range
    rot_image[ones[:,:,:]!=1] = edge_filler[ones[:,:,:]!=1]
    return rot_image


def translate(img, row_amt, col_amt, color_range=255):
    translation = np.random.random(img.shape).astype(np.float32)*color_range
    if row_amt > 0:
        if col_amt > 0:
            translation[row_amt:,col_amt:] = img[:-row_amt,:-col_amt]
        elif col_amt < 0:
            translation[row_amt:,:col_amt] = img[:-row_amt,-col_amt:]
        else:
            translation[row_amt:,:] = img[:-row_amt,:]
    elif row_amt < 0:
        if col_amt > 0:
            translation[:row_amt,col_amt:] = img[-row_amt:,:-col_amt]
        elif col_amt < 0:
            translation[:row_amt,:col_amt] = img[-row_amt:,-col_amt:]
        else:
            translation[:row_amt,:] = img[-row_amt:,:]
    else:
        if col_amt > 0:
            translation[:,col_amt:] = img[:,:-col_amt]
        elif col_amt < 0:
            translation[:,:col_amt] = img[:,-col_amt:]
        else:
            return img.copy()
    return translation.astype(img.dtype)


def random_zoom(image, max_zoom=1/6., allow_out_zooms=False):
    color_range = 255
    if allow_out_zooms:
        zoom_factor = 1 + (random.random()-.5)*max_zoom*2
    else:
        zoom_factor = 1 + random.random()*max_zoom
    while zoom_factor == 1:
        zoom_factor = 1 + (random.random()-0.5)*max_zoom

    # scipy's zoom function returns different size array
    # The following code ensures the zoomed image has same pixel size as initial image
    img_height, img_width = image.shape[:2]
    zoomed_h = round(img_height*zoom_factor)
    zoomed_w = round(img_width*zoom_factor)
    diff_h = abs(zoomed_h-img_height)
    diff_w = abs(zoomed_w-img_width)
    start_row = round(diff_h/2)
    start_col = round(diff_w/2)

    # Zoom in on image
    if zoom_factor > 1:
        end_row = start_row+img_height
        end_col = start_col+img_width
        zoom_img = scizoom.zoom(image,(zoom_factor,zoom_factor,1),output=np.uint8)[start_row:end_row,
                                                               start_col:end_col]
    # Zoom out on image
    elif zoom_factor < 1:
        temp = scizoom.zoom(image,(zoom_factor,zoom_factor,1),output=np.uint8)
        temp_height, temp_width = temp.shape[:2]
        zoom_img = np.random.random(image.shape)*color_range # Random pixels instead of black space for out zoom
        zoom_img[start_row:start_row+temp_height,
                 start_col:start_col+temp_width] = temp[:,:]

    return zoom_img.astype(np.float32)


# random augment - returns a randomly rotated, translated, or scaled copy of an image
def randomAugment(image, rotation_limit=45, shift_limit=10,
                    zoom_limit=1/3., random_fill=True):
    augmentation_type = random.randint(0,4)

    # Rotation
    if augmentation_type >= 0 and augmentation_type <= 1:
        random_angle = random.randint(-rotation_limit,rotation_limit)
        while random_angle == 0:
            random_angle = random.randint(-rotation_limit,rotation_limit)
        aug_image = rotate(image,random_angle,random_fill=random_fill)

    # Translation
    elif augmentation_type >= 2 and augmentation_type <=3:
        row_shift = random.randint(-shift_limit, shift_limit)
        col_shift = random.randint(-shift_limit, shift_limit)
        aug_image = translate(image,row_shift,col_shift)

    # Scale
    else:
        aug_image = random_zoom(image,max_zoom=zoom_limit)

    return aug_image


# convert randoms
def convertRandoms(paths, labels, resize_dims=None, warp_ok=False):
    images = []
    for i,path in enumerate(paths):
        try:
            if resize_dims and not warp_ok:
                img = resizeImage(path, maxsizes=resize_dims)
            else:
                img = mpimg.imread(path)
                if resize_dims:
                    img = sci.imresize(img, resize_dims)

            img = randomAugment(img)

        except OSError:
            # Uses augmented version of next image in list
            if i == 0:
                if resize_dims and not warp_ok:
                    img = resizeImage(paths[i+1],maxsizes=resize_dims)
                else:
                    img = mpimg.imread(paths[i+1])
                    if resize_dims:
                        img = sci.imresize(img, resize_dims)
                img = randomAugment(img)
                labels[i] = labels[i+1]

            # Uses most recent original image
            elif i > 0:
                img = randomAugment(images[-1])
                labels[i] = labels[i-1]

        images.append(img)

    return images, labels


# convert images
def convertImages(paths, labels, resize_dims=None, warp_ok=False, rpaths=[], randomly_augment=True, rlabels=[]):
    if len(rpaths) > 0 and len(rlabels) > 0:
        # pool = Pool(processes=1)
        # result = pool.apply_async(convert_randoms, (rpaths,rlabels,resize_dims))
        rand_imgs, rand_labels = convertRandoms(rpaths,rlabels,resize_dims)

    images = []
    for i,path in enumerate(paths):
        try:
            if resize_dims and not warp_ok:
                img = resizeImage(path)
            else:
                img = mpimg.imread(path)
                if resize_dims:
                    img = sci.imresize(img, resize_dims)

        except OSError:
            # Uses augmented version of next image in list
            if i == 0:
                if resize_dims and not warp_ok:
                    img = resizeImage(paths[i+1])
                else:
                    img = mpimg.imread(paths[i+1])
                    if resize_dims:
                        img = sci.imresize(img, resize_dims)
                img = randomAugment(img)
                labels[i] = labels[i+1]

            # Uses most recent original image
            elif i > 0:
                sub_index = -1
                if randomly_augment:
                    sub_index = -2
                img = randomAugment(images[sub_index])
                labels[i] = labels[i-sub_index]

        images.append(img)

    if len(rpaths) > 0 and len(rlabels) > 0:
        # result.wait()
        # rand_imgs, rand_labels = result.get()
        images = images+rand_imgs
        labels = np.concatenate([labels,rand_labels],axis=0)
        return np.array(images,dtype=np.float32), labels
    return np.array(images,dtype=np.float32), labels


# cnn model
def convModel(first_conv_shapes=[(4,4),(3,3),(5,5)], conv_shapes=[(3,3),(5,5)], conv_depths=[12,12,11,8,8], dense_shapes=[100,50,3], image_shape=(256,256,3), n_labels=3):
    stacks = []
    pooling_filter = (2,2)
    pooling_stride = (2,2)

    inputs = Input(shape=image_shape)
    zen_layer = BatchNormalization()(inputs)

    for shape in first_conv_shapes:
        stacks.append(Conv2D(conv_depths[0], shape, padding='same', activation='elu')(zen_layer))
    layer = concatenate(stacks,axis=-1)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pooling_filter,strides=pooling_stride,padding='same')(layer)
    layer = Dropout(0.05)(layer)

    for i in range(1,len(conv_depths)):
        stacks = []
        for shape in conv_shapes:
            stacks.append(Conv2D(conv_depths[i],shape,padding='same',activation='elu')(layer))
        layer = concatenate(stacks,axis=-1)
        layer = BatchNormalization()(layer)
        layer = Dropout(i*10**-2+.05)(layer)
        layer = MaxPooling2D(pooling_filter,strides=pooling_stride, padding='same')(layer)

    layer = Flatten()(layer)
    fclayer = Dropout(0.1)(layer)

    for i in range(len(dense_shapes)-1):
        fclayer = Dense(dense_shapes[i], activation='elu')(fclayer)
        fclayer = BatchNormalization()(fclayer)

    outs = Dense(dense_shapes[-1], activation='softmax')(fclayer)

    return inputs, outs


def max_index(array):
    # ** Returns index of maximum value in an array **
    max_i = 0
    for j in range(1,len(array)):
        if array[j] > array[max_i]: max_i = j
    return max_i


# get confidence for the predicted class
def confid(predictions,conf):
    for i,prediction in enumerate(predictions):
        max_i = max_index(prediction)
        predictions[i][max_i] = conf
        for j in range(len(prediction)):
            if j != max_i:
                predictions[i][j] = (1-conf)/(len(prediction)-1)
    return predictions


#  save predictions into csv
def save(csv_file_path, names, predictions, header=None):
    with open(csv_file_path, 'w') as f:
        f.write(header+'\n')
        for name,logit in zip(names,predictions):
            f.write(name+',')
            for i,element in enumerate(logit):
                if i == logit.shape[0]-1: f.write(str(element)+'\n')
                else: f.write(str(element)+',')


# train a cnn model
def cervicalTraining():
    resizedImageDir = ['/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/data/resized_imgs/train']

    imagePaths = []
    labels = []
    for i, resizedPath in enumerate(resizedImageDir):
        new_paths, new_labels, n_classes = readFilePaths(resizedPath)
        if len(new_paths) > 0:
            imagePaths += new_paths
            labels += new_labels

    imagePaths, labels = shuffle(imagePaths, labels)

    trainCSV = '/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/csvs/train_set.csv'
    validCSV = '/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/csvs/valid_set.csv'

    training_portion = .8
    split_index = int(training_portion * len(imagePaths))
    X_train_paths, y_train = imagePaths[:split_index], labels[:split_index]
    X_valid_paths, y_valid = imagePaths[split_index:], labels[split_index:]

    print("Train size: ")
    print(len(X_train_paths))
    print("Valid size: ")
    print(len(X_valid_paths))

    savePaths(trainCSV, X_train_paths, y_train)
    savePaths(validCSV, X_valid_paths, y_valid)

    train_csv = 'csvs/train_set.csv'
    valid_csv = 'csvs/valid_set.csv'

    X_train_paths, y_train = getSplitData(train_csv)
    X_valid_paths, y_valid = getSplitData(valid_csv)
    n_classes = max(y_train) + 1

    y_train = oneHotEncode(y_train, n_classes)
    y_valid = oneHotEncode(y_valid, n_classes)

    batch_size = 110
    add_random_augmentations = False
    resize_dims = None
    n_train_samples = len(X_train_paths)
    train_steps_per_epoch = getSteps(n_train_samples, batch_size, n_augs=1)
    n_valid_samples = len(X_valid_paths)
    valid_steps_per_epoch = getSteps(n_valid_samples, batch_size, n_augs=0)
    train_generator = image_generator(X_train_paths, y_train, batch_size,
                                      resize_dims=resize_dims,
                                      randomly_augment=add_random_augmentations)
    valid_generator = image_generator(X_valid_paths, y_valid, batch_size,
                                      resize_dims=resize_dims, rand_order=False)

    '''
    modeling
    '''
    n_classes = 3
    image_shape = (256, 256, 3)

    first_conv_shapes = [(4, 4), (3, 3), (5, 5)]
    conv_shapes = [(3, 3), (5, 5)]
    conv_depths = [12, 12, 11, 8, 8]
    dense_shapes = [100, 50, n_classes]

    inputs, outs = convModel(first_conv_shapes, conv_shapes, conv_depths, dense_shapes, image_shape, n_classes)

    model = Model(inputs=inputs, outputs=outs)

    learning_rate = .0001
    for i in range(20):
        if i > 4:
            learning_rate = .00001  # Anneals the learning rate
        adam_opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
        history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
                                      validation_data=valid_generator, validation_steps=valid_steps_per_epoch,
                                      max_queue_size=1)
        model.save('/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/models/model.h5')

    '''
    get predictions
    '''
    data_path = '/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/data/test'
    model_path = '/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/models/model.h5'

    resize_dims = (256, 256, 3)
    test_divisions = 20  # Used for segmenting image evaluation in threading
    batch_size = 100  # Batch size used for keras predict function

    ins, outs = convModel()
    model = Model(inputs=ins, outputs=outs)
    model.load_weights(model_path)
    test_paths, test_labels, _ = readFilePaths(data_path, no_labels=True)
    print(str(len(test_paths)) + ' testing images')

    pool = ThreadPool(processes=1)
    portion = len(test_paths) // test_divisions + 1  # Number of images to read in per pool

    async_result = pool.apply_async(convertImages, (test_paths[0 * portion:portion * (0 + 1)],
                                                     test_labels[0 * portion:portion * (0 + 1)], resize_dims))

    total_base_time = time.time()
    test_imgs = []
    predictions = []
    for i in range(1, test_divisions + 1):
        base_time = time.time()

        print("Begin set " + str(i))
        while len(test_imgs) == 0:
            test_imgs, _ = async_result.get()
        img_holder = test_imgs
        test_imgs = []

        if i < test_divisions:
            async_result = pool.apply_async(convertImages, (test_paths[i * portion:portion * (i + 1)],
                                                             test_labels[0 * portion:portion * (0 + 1)],
                                                             resize_dims))

        predictions.append(model.predict(img_holder, batch_size=batch_size, verbose=0))
        print("Execution Time: " + str((time.time() - base_time) / 60) + 'min\n')

    predictions = np.concatenate(predictions, axis=0)
    print("Total Execution Time: " + str((time.time() - total_base_time) / 60) + 'mins')

    conf = .95  # Prediction confidence
    savePredictions = '/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch05/predictions.csv'
    predictions = confid(predictions, conf)
    header = 'image_name,Type_1,Type_2,Type_3'
    save(savePredictions, test_labels, predictions, header)


if __name__ == '__main__':
    processCervicalData()
    cervicalTraining()
