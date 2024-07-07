import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

lzz_flag = 1
import cv2
import concurrent.futures
import numpy as np
import os
import cv2
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
JSONUSElist =[]
def compute_flow_batch(frames_np):
    # 初始化光流数组
    flows = np.zeros((frames_np.shape[0], frames_np.shape[2], frames_np.shape[3], 2), dtype=np.float32)
    flowsbefore = np.zeros((frames_np.shape[0], frames_np.shape[2], frames_np.shape[3], 2), dtype=np.float32)
    # 对批次中的每对连续帧计算光流
    for idx in range(frames_np.shape[0] - 1):
        frame1 = np.moveaxis(frames_np[idx], 0, -1)  # C, H, W -> H, W, C
        frame2 = np.moveaxis(frames_np[idx + 1], 0, -1)  # C, H, W -> H, W, C

        # 转换为灰度图像
        frame1_gray = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flows[idx] = flow
        # flowsbefore[idx + 1] = flow
    return flows, flowsbefore

def save_images(data, seg_all, save_folder, name):
    # 确保保存文件夹存在
    save_folder = os.path.join(save_folder, name)
    os.makedirs(save_folder, exist_ok=True)
    
    # 循环保存每张图像
    for i in range(data.shape[2]):
        im = (data[0, :, i, :, :] * 255).transpose(1, 2, 0).astype(np.uint8)
        filepath = os.path.join(save_folder, f"{i}.png")
        cv2.imwrite(filepath, im)
    # 可视化标签
    keys = [i for i in range(48)]
    values = list(seg_all[0])
    plt.plot(keys, values, color='blue', marker='o', markersize=3)
    plt.title('cardiac cycle')
    plt.xlabel('frame')
    plt.ylabel('y')
    plt.axis([0,keys[-1],-1,1.1])
    # 保存为SVG格式的矢量图
    plt.savefig(os.path.join(save_folder, "label.png"), format='png')
    return 0

def letterbox(im, new_shape=(192, 192), color=(0,0,0), auto=False, scaleFill=False, scaleup=True, stride=1):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    im = np.array(im)
    im = im.transpose(1,2,0)
    # im shape  (H, W, C)
    # filepath = r"/root/lzz2/cardiac_cycle/mamba/U-Mamba/data/showorg.png"
    # cv2.imwrite(filepath, im)
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # filepath = r"/root/lzz2/cardiac_cycle/mamba/U-Mamba/data/show224.png"
    # cv2.imwrite(filepath, im)
    im = im.transpose(2,0,1)
    # im shape (C,H,W)
    return im

# Assuming 'data' is the list of arrays and 'letterbox' is the letterbox function

def process_item(nowdata):
    return letterbox(nowdata)

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        global lzz_flag
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        if lzz_flag == 1:
            seg_all = np.zeros(self.seg_shape, dtype=np.float16)
        case_properties = []
        needimg = data_all.shape[2]
        needshape = (data_all.shape[1], data_all.shape[3], data_all.shape[4])
        if lzz_flag == 1:
            for j, i in enumerate(selected_keys):
                # oversampling foreground will improve stability of model training, especially if many patches are empty
                # (Lung for example)
                force_fg = self.get_do_oversample(j)
                datapath, segpath, properties = self._data.load_case(i)
                # 根据json中的 -1  截取掉无法使用的帧
                with open(segpath, encoding='utf-8') as f:
                    frame = json.load(f)
                    annotations = frame.get("annotations", {})
                    # jsonkeys = [int(key) for key in annotations.keys()]
                    jsonvalues = list(annotations.values())
                    keepindex = list(np.where(np.array(jsonvalues) != -1)[0])
                jsonstart = keepindex[0]
                jsonend = keepindex[-1]
                # 记录本次选取帧的起始  方便取json对应的值
                annostart = 0
                annoend = 0
                #  读取视频
                cap = cv2.VideoCapture(datapath)
                if not cap.isOpened():
                    print(f"无法打开视频文件：{datapath}")
                    return None
                # 获取视频的总帧数
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = min(frame_count, jsonend + 1)
                jsonuse = frame_count - max(jsonstart, 1)
                # print("jsonuse: ",jsonuse)
                # print(jsonuse)
                # global JSONUSElist
                # JSONUSElist.append(jsonuse)
                # print("min(JSONUSElist): ",JSONUSElist)
                # print("min(JSONUSElist): ",min(JSONUSElist))
                frames = []
                # if needimg < frame_count - 1:
                if needimg < jsonuse:
                    # startindex = 1
                    startindex = max(jsonstart, 1)
                    endindex = frame_count - needimg
                    split_start = random.randint(startindex, endindex)
                    annostart = split_start
                    annoend = split_start + needimg
                    # 跳到起始帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, split_start)
                    for i in range(split_start, split_start + needimg):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                else:
                    # startindex = 1
                    startindex = max(jsonstart, 1)
                    split_start = startindex
                    # 跳到起始帧
                    annostart = split_start
                    annoend = frame_count
                    cap.set(cv2.CAP_PROP_POS_FRAMES, split_start)
                    for i in range(split_start, frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                data = np.array(frames)
                # 释放视频对象
                cap.release()
                #  ************************************************* 处理image  *************************************************
                assert np.min(data)>= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
                assert np.max(data_all) <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                        ". Your images do not seem to be RGB images"
                data = data / 255.
                data = np.transpose(data, (0,3,1,2))
                # print(newdata)
                # newdata = []
                # for nowdata in data:
                #     nowdata = letterbox(nowdata)
                #     newdata.append(nowdata)
                # newdata = np.stack(newdata, axis=0)
                # Create a ThreadPoolExecutor with the desired number of threads
                num_threads = 4  # Adjust the number of threads as per your requirements
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
                # Process the items in parallel using the ThreadPoolExecutor
                newdata = []
                with executor:
                    futures = [executor.submit(process_item, nowdata) for nowdata in data]
                    for future in concurrent.futures.as_completed(futures):
                        nowdata = future.result()
                        newdata.append(nowdata)
                # Convert the processed data list to a NumPy array
                needtap = needimg - len(newdata)
                if needtap > 0:
                    for i in range(needtap):
                        zeros_array = np.zeros(needshape)
                        newdata.append(zeros_array)
                newdata = np.transpose(np.stack(newdata, axis=0), (1,0,2,3))
                #  ************************************************* 加上光流  *************************************************
                flowflag = 0
                if flowflag == 1:
                    video_frames_tensor = np.transpose(newdata, (1,0,2,3)) * 255
                    flows,flowsbefore = compute_flow_batch(video_frames_tensor)
                    newdata = np.concatenate((np.transpose(flows, (3,0,1,2)), newdata), axis=0)
                    # newdata = np.concatenate((np.transpose(flowsbefore, (3,0,1,2)), np.transpose(flows, (3,0,1,2)), newdata), axis=0)
                #  ************************************************* 加上光流  *************************************************
                data_all[j] = newdata
                #  ************************************************* 处理regression  *************************************************
                # 读取json
                f = open(segpath, encoding='utf-8')
                nowseg = json.load(f)
                annotations = nowseg["annotations"]
                seg = []
                for i in range(annostart, annoend):
                    seg.append(np.array([annotations[str(i)]]))
                # seg不补齐
                if needtap > 0:
                    needarray = np.array([2.0])
                    for i in range(needtap):
                        seg.append(needarray)
                newseg = np.array(seg, dtype=np.float16)
                seg_all[j] = newseg
                #  ************************************************* 处理properties  ************************************************
                case_properties.append(properties)
            # 示例用法
            # save_folder = "/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/nnUNet_preprocessed/Dataset701_AbdomenCT/show_org"  # 保存图像的文件夹路径
            # if selected_keys[0] == "20190411_162835_161":
            #     print(annostart, "              ", annoend)
            #     save_images(data_all, seg_all, save_folder, selected_keys[0])
            if len(np.where(seg_all == -1)[0]):
                print("有label为0！！！")
            return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}
    def generate_train_batch_npy(self):
        global lzz_flag
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        if lzz_flag == 1:
            seg_all = np.zeros(self.seg_shape, dtype=np.float16)
        case_properties = []
        needimg = data_all.shape[2]
        needshape = (data_all.shape[1], data_all.shape[3], data_all.shape[4])
        if lzz_flag == 1:
            for j, i in enumerate(selected_keys):
                # oversampling foreground will improve stability of model training, especially if many patches are empty
                # (Lung for example)
                force_fg = self.get_do_oversample(j)
                data, seg, properties = self._data.load_case(i)
                #  ************************************************* 处理image  *************************************************
                assert np.min(data)>= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
                assert np.max(data_all) <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                        ". Your images do not seem to be RGB images"
                data = data / 255.
                data = np.transpose(data, (1,0,3,2))
                # print(newdata)
                # newdata = []
                # for nowdata in data:
                #     nowdata = letterbox(nowdata)
                #     newdata.append(nowdata)
                # newdata = np.stack(newdata, axis=0)
                # Create a ThreadPoolExecutor with the desired number of threads
                num_threads = 4  # Adjust the number of threads as per your requirements
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
                # Process the items in parallel using the ThreadPoolExecutor
                newdata = []
                with executor:
                    futures = [executor.submit(process_item, nowdata) for nowdata in data]
                    for future in concurrent.futures.as_completed(futures):
                        nowdata = future.result()
                        newdata.append(nowdata)
                # Convert the processed data list to a NumPy array
                needtap = needimg - len(newdata)
                if needtap > 0:
                    for i in range(needtap):
                        zeros_array = np.zeros(needshape)
                        newdata.append(zeros_array)
                newdata = np.transpose(np.stack(newdata, axis=0), (1,0,2,3))
                data_all[j] = newdata
                #  ************************************************* 处理regression  *************************************************
                # seg不补齐
                if needtap > 0:
                    seg = seg.tolist()
                    needarray = np.array([2.0])
                    for i in range(needtap):
                        seg.append(needarray)
                newseg = np.array(seg, dtype=np.float16)
                seg_all[j] = newseg
                #  ************************************************* 处理properties  ************************************************
                case_properties.append(properties)
            # 示例用法
            # save_folder = "/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/show/org"  # 保存图像的文件夹路径
            # save_images(data_all, save_folder, selected_keys[0])
            return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

class nnUNetDataLoader3D_org(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        global lzz_flag
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # lzzappend
            if lzz_flag == 1:
                data = np.repeat(data, 3, axis=0)
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
