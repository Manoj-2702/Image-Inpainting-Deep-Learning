import cv2 as cv
import os
import numpy as np
import time


# ! [CropLayenr]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        # self.ystart = (inputShape[2] - targetShape[2]) / 2
        # self.xstart = (inputShape[3] - targetShape[3]) / 2
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def hed(net, start_paths, target_paths):
    width = 256
    height = 256
    for start_path_i in range(len(start_paths)):
        s_path = start_paths[start_path_i]
        t_path = target_paths[start_path_i]
        if not os.path.exists(t_path):
            os.makedirs(t_path)
        image_lists = [os.path.join(s_path, i) for i in os.listdir(s_path)]
        size = len(image_lists)

        for img_i, img_path in enumerate(image_lists):
            if '.jpg' not in img_path.lower() and '.png' not in img_path.lower():
                continue
            if img_i % 10 == 0:
                print(f'{t_path} finish {img_i}/{size}.')
            frame = cv.imread(img_path)

            inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(width, height),mean=(104.00698793, 116.66876762, 122.67891434),swapRB=False, crop=False)
            net.setInput(inp)
            out = net.forward()
            out = out[0, 0]
            out = cv.resize(out, (frame.shape[1], frame.shape[0]))
            out = out * 255
            cv.imwrite(os.path.join(t_path, img_path[img_path.rfind('\\')+1:]), out.astype('uint8'))
            time.sleep(0.05)
            
    return


def flist(paths, outputs):
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    for path_i, path in enumerate(paths):
        output = outputs[path_i]
        images = []
        for root, dirs, files in os.walk(path):
            print('loading ' + root)
            for file in files:
                if os.path.splitext(file)[1].upper() in ext:
                    images.append(os.path.join(root, file))
        
        images = sorted(images)
        np.savetxt(output, images, fmt='%s')
        
    return



if __name__ == '__main__':
    # ! [CropLayer]

    # ! [Register]
    cv.dnn_registerLayer('Crop', CropLayer)
    # ! [Register]

    # Load the model.
    prototxt_path = 'deploy.prototxt'
    caffemodel_path = 'hed_pretrained_bsds.caffemodel'
    net = cv.dnn.readNet(cv.samples.findFile(prototxt_path), cv.samples.findFile(caffemodel_path))

    start_paths = ['training/cat_train', 'training/cat_test_original', 'training/cat_val']
    target_paths = ['training/cat_edges_train', 'training/cat_edges_test', 'training/cat_edges_val']
    hed(net, start_paths, target_paths)
    outputs = ['datasets/cat_edges_train.flist', 'datasets/cat_edges_test.flist', 'datasets/cat_edges_val.flist']
    flist(target_paths, outputs)
